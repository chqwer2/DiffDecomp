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
from .BasicDataset import BasicDataset


LABEL_NAME = ["bg", "NCR", "ED", "ET"]

class BrainDataset(BasicDataset):
    def __init__(self, mode, transforms, base_dir, domains, resolution=192, **kwargs):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        self.dataset_key = "brain"
        super(BrainDataset, self).__init__(mode, transforms, base_dir, domains, LABEL_NAME=LABEL_NAME,
                                           filter_non_labeled=True, fineSize=resolution,  **kwargs)
        self.original_size = [384, 384] # down sample
        self.use_size      = [192, 192] # down scale x2, 192 should be 
        
        # self.crop_size     = [160, 160] # crop size
    def hwc_to_chw(self,img):
        img = np.float32(img)
        img = np.transpose(img, (2, 0, 1))   # [C, H, W]
        img = torch.from_numpy( img.copy() )
        return img

    def perform_trans(self, img, mask):
        
        T = self.albu_transform if self.is_train else self.test_resizer
        buffer = T(image = img, mask=mask)     # [0 - 255]
        img, mask = buffer['image'], buffer['mask']

        return img, mask
    
    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]  # numpy

        # ----------------------- Extract Slice -----------------------
        img, mask  = curr_dict["img"],  curr_dict["lb"]     # H, W, C, [0 - 255]
        # mean, std  = curr_dict['mean'], curr_dict['std']
        slice      = int(curr_dict['slice'] * 100 + 0.5)
        
        # std    = 1 if std < 1e-3 else std
        
        mean = img.mean()
        std  = img.std()
        std    = 1 if std < 1e-3 else std
        
        img = (img - mean) / std
        
        img = (img - img.min()) / (img.max() - img.min())
        img, mask = self.perform_trans(img, mask)
        img, mask = map(lambda arr: self.hwc_to_chw(arr), [img, mask])

        img = (img - img.min()) / (img.max() - img.min())
        
        if self.tile_z_dim > 1 and self.input_window == 1: 
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3

        return img, slice, index


def set_brain_dataset(mode, basedir, modality, norm_func = None,
                idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, pseudo = False, chunksize=200, **kwargs):

    norm_func = None if mode=="train" else norm_func
    tr_func = None if mode == "train" else trans.transform_with_label(trans.tr_aug)

    return BrainDataset(idx_pct = idx_pct,
                            mode = mode,
                            pseudo = pseudo,
                            domains = modality,
                            transforms = tr_func,
                            base_dir = basedir,
                            extern_norm_fn = norm_func,
                            tile_z_dim = tile_z_dim,
                            chunksize  = chunksize, **kwargs)
