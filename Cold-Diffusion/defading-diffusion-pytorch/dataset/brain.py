# Dataloader for abdominal images
import glob
import numpy as np
from .utils import niftiio as nio
from .utils import transform_utils as trans
from .utils.abd_dataset_utils import get_normalize_op
from .utils.transform_albu import get_albu_transforms, get_resize_transforms

import torch
import os
from pdb import set_trace
from multiprocessing import Process
from .basic import BasicDataset


LABEL_NAME = ["bg", "NCR", "ED", "ET"]

def normalize(data, mean, stddev, eps=0.0):
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data (torch.Tensor): Input data to be normalized.
        mean (float): Mean value.
        stddev (float): Standard deviation.
        eps (float, default=0.0): Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(data, eps=0.0):
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data (torch.Tensor): Input data to be normalized
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class BrainDataset(BasicDataset):
    def __init__(self, mode, base_dir, image_size, 
                 nclass, domains, aux_modality, **kwargs):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        self.dataset_key = "brain"
        transforms = get_albu_transforms(mode)
        if isinstance(domains, str):
            domains = [domains]
            
        super(BrainDataset, self).__init__(image_size, mode, 
                                           transforms, 
                                           base_dir, 
                                           domains, aux_modality, 
                                           nclass=nclass, 
                                           LABEL_NAME=LABEL_NAME, 
                                           filter_non_labeled=True, 
                                           **kwargs)
        
        self.original_size = [384, 384] # down sample
        self.use_size      = [192, 192] # down scale x2
        self.crop_size     = [160, 160] # crop size
        
    def hwc_to_chw(self,img):
        img = np.float32(img)
        img = np.transpose(img, (2, 0, 1))   # [C, H, W]
        img = torch.from_numpy( img.copy() )
        return img

    def perform_trans(self, img, mask, aux):
        
        T = self.albu_transform if self.is_train else self.test_resizer
        buffer = T(image = img, mask=mask, image2=aux)     # [0 - 255]
        img, mask, aux = buffer['image'], buffer['mask'], buffer['image2']
        if len(mask.shape) == 2:
            mask = mask[..., None]
        
        if self.is_train:
            img, mask, aux = self.get_patch_from_img(img, mask, aux, crop_size=self.crop_size)  # 192
        
        return img, mask, aux

        
    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]  # numpy

        # ----------------------- Extract Slice -----------------------
        img, mask, aux  = curr_dict["img"], curr_dict["lb"], curr_dict["aux"]     # H, W, C, [0 - 255]
        domain, pid     = curr_dict["domain"], curr_dict["pid"]
        mean, std       = curr_dict['mean'], curr_dict['std']

        # max, min = img.max(), img.min()
        std    = 1 if std < 1e-3 else std
        
        # img = (img - mean) / std
        ### 对input image和target image都做(x-mean)/std的归一化操作
        img, img_mean, img_std = normalize_instance(img, eps=1e-11)
        aux, aux_mean, aux_std = normalize_instance(aux, eps=1e-11)

        ### clamp input to ensure training stability.
        img = np.clip(img, -6, 6)
        aux = np.clip(aux, -6, 6)


        # img = (img - img.min()) / (img.max() - img.min())  # [0 - 1]
        # if aux.max() - aux.min() > 1e-3:
        #     aux = (aux - aux.min()) / (aux.max() - aux.min())  # [0 - 1]
       
       
        mask = mask[..., 0] 
        img, mask, aux = self.perform_trans(img, mask, aux)
        img, mask, aux = map(lambda arr: self.hwc_to_chw(arr), [img, mask, aux])

        # img = torch.clamp(img, 0, 1)
        # aux = torch.clamp(aux, 0, 1)

        # img = (img - img.min()) / (img.max() - img.min())  # [0 - 1]
        # aux = (aux - aux.min()) / (aux.max() - aux.min())  # [0 - 1]
        
        if self.tile_z_dim > 1 and self.input_window == 1 and  self.num_channels == 3 : 
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3

        data = {"img": img, "lb": mask, "aux": aux,
                "img_mean": img_mean, "img_std": img_std,
                "aux_mean": aux_mean, "aux_std": aux_std,

                "is_start": curr_dict["is_start"],
                "is_end": curr_dict["is_end"],
                "nframe": np.int32(curr_dict["nframe"]),
                "scan_id": curr_dict["scan_id"],
                "z_id": curr_dict["z_id"],
                "file_id": curr_dict["file_id"]
                }

        return data



