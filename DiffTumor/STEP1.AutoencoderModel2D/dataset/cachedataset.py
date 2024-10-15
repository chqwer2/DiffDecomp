from copy import deepcopy
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data.dataset import Dataset
from monai.data.iterable_dataset import IterableDataset
from monai.data.utils import iter_patch
from monai.transforms import apply_transform
from monai.utils import NumpyPadMode, deprecated_arg, ensure_tuple, first, look_up_option
from monai.data import PatchIter

class PatchIterd:
    """
    Dictionary-based wrapper of :py:class:`monai.data.PatchIter`.
    Return a patch generator for dictionary data and the coordinate, Typically used
    with :py:class:`monai.data.GridPatchDataset`.
    Suppose all the expected fields specified by `keys` have same shape.

    Args:
        keys: keys of the corresponding items to iterate patches.
        patch_size: size of patches to generate slices for, 0/None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        pad_opts: other arguments for the `np.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    coords_key = "patch_coords"
    original_spatial_shape_key = "original_spatial_shape"
    start_pos_key = "start_pos"

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        start_pos: Sequence[int] = (),
        mode: str = NumpyPadMode.WRAP,
        **pad_opts,
    ):
        self.keys = ensure_tuple(keys)
        self.patch_iter = PatchIter(patch_size=patch_size, start_pos=start_pos, mode=mode, **pad_opts)

    def __call__(self, datas: Mapping[Hashable, np.ndarray]):
        for data in datas:
            d = dict(data)  # A bug introduce two 
            original_spatial_shape = d[first(self.keys)].shape[1:]
            
            # Filter Zero Slices
            filter = torch.any(torch.any(d[first(self.keys)][0], dim=0), dim=0)
            for key in self.keys:
                d[key] = d[key][..., filter]
            original_spatial_shape = d[first(self.keys)].shape[1:]
            
            # print("new original_spatial_shape", original_spatial_shape)
            
            for patch in zip(*[self.patch_iter(d[key]) for key in self.keys]):
                coords = patch[0][1]  # use the coordinate of the first item
                ret = {k: v[0] for k, v in zip(self.keys, patch)}
                # fill in the extra keys with unmodified data
                for k in set(d.keys()).difference(set(self.keys)):
                    ret[k] = deepcopy(d[k])
                # also store the `coordinate`, `spatial shape of original image`, `start position` in the dictionary
                ret[self.coords_key] = coords
                ret[self.original_spatial_shape_key] = original_spatial_shape
                ret[self.start_pos_key] = self.patch_iter.start_pos
                yield ret, coords
