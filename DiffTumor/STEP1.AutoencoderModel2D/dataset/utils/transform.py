from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    EnsureChannelFirstd,
    RandZoomd, 
    SqueezeDimd,
    RandCropByLabelClassesd,
)
import monai
import numpy as np
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()


class LoadImaged_BodyMap(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    # def label_transfer(self, lbl_dir, shape):
    #     organ_lbl = np.zeros(shape)
        
    #     if os.path.exists(lbl_dir + 'liver' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'liver' + '.nii.gz')
    #         organ_lbl[array > 0] = 1
    #     if os.path.exists(lbl_dir + 'pancreas' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'pancreas' + '.nii.gz')
    #         organ_lbl[array > 0] = 2
    #     if os.path.exists(lbl_dir + 'kidney_left' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'kidney_left' + '.nii.gz')
    #         organ_lbl[array > 0] = 3
    #     if os.path.exists(lbl_dir + 'kidney_right' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'kidney_right' + '.nii.gz')
    #         organ_lbl[array > 0] = 3
    #     if os.path.exists(lbl_dir + 'liver_tumor' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'liver_tumor' + '.nii.gz')
    #         organ_lbl[array > 0] = 4
    #     if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'pancreas_tumor' + '.nii.gz')
    #         organ_lbl[array > 0] = 5
    #     if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
    #         array, mata_infomation = self._loader(lbl_dir + 'kidney_tumor' + '.nii.gz')
    #         organ_lbl[array > 0] = 6

    #     return organ_lbl, mata_infomation
    
    
    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        
        # d: {'image': '/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00456-000/BraTS-GLI-00456-000-t1c.nii.gz', 
        #     'label': '/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00456-000/BraTS-GLI-00456-000-seg.nii.gz', 
        #     'name': 'BraTS-GLI-00456-000'}
        
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            
            try:
                data = self._loader(d[key], reader)
            except:
                print("failed:", d['name'])
                
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        
        d['label'], d['label_meta_dict'] = self._loader(d['label'])
        # print("d = ", d.keys())
        
        return d


class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        # print('file_name', d['name'])
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        return d
    

def get_transforms(args):

    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["image", "aux"]),
            AddChanneld(keys=["image", "aux", "label"]),
            Orientationd(keys=["image", "aux", "label"], axcodes="RAS"),
            
            Spacingd(
                keys=["image", "aux", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest"),
            ), # process h5 to here
            
            ScaleIntensityRanged(
                keys=["image", "aux"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            
            SpatialPadd(keys=["image", "aux", "label"], 
                        spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                        mode=["minimum", "minimum", "constant"]),
            
            RandCropByPosNegLabeld(
                keys=["image", "aux", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                pos=20,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            RandRotate90d(
                keys=["image", "aux", "label"],
                prob=0.10,
                max_k=3,
            ),
            
            EnsureChannelFirstd(keys=["image", "aux","seg"]),
            ToTensord(keys=["image", "aux", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "aux"]),
            AddChanneld(keys=["image", "aux", "label"]),
            Orientationd(keys=["image", "aux", "label"], axcodes="RAS"),
            
            # ValueError: Sequence must have length 3, got 2.
            Spacingd(
                keys=["image", "aux", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "bilinear", "nearest"),
            ), 
            ScaleIntensityRanged(
                keys=["image", "aux"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "aux", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                        mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "aux", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=0,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),
            ToTensord(keys=["image", "aux", "label"]),
        ]
    )
    
    return train_transforms, val_transforms



