from monai.transforms import apply_transform
from .utils.transform import get_transforms
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from glob import glob
sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset

IMAGE_NII = "ct.nii.gz"


class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)
    


def get_loader(args, splits=[0.7, 0.1, 0.2]):
    """_summary_

    Args:
        args (arguments): input params
        splits (list, optional): train/val/test splition ratio. Defaults to [0.7, 0.1, 0.2].

    Returns:
        dataloader
    """
    
    data_path = args.data_root_path
    modality  = args.data_modality     # t2w, t1c, t1n, t2f
    
    train_transforms, val_transforms = get_transforms(args)

    # ------------  Process Data List ------------  
    # ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
    # |--BraTS-GLI-00xxx-00x   # id-longitudinal  ? 
        # |--BraTS-GLI-00xxx-00x-seg.nii.gz

    seg_list = glob(os.path.join(data_path, '*/*seg.nii.gz'))
    img_list = [seg.replace('seg.nii.gz', f'{modality}.nii.gz') for seg in seg_list]
    patient_list = [os.path.basename(seg).replace('-seg.nii.gz', '') for seg in seg_list]
    
    data_dicts = [{'image': image, 'label': label, 'name': name}    
                    for image, label, name in zip(img_list, seg_list, patient_list)]
    
    if args.phase == 'train':   
        data_dicts = data_dicts[:int(splits[0] * len(data_dicts))]
        transform = train_transforms
    elif args.phase == 'validation':
        data_dicts = data_dicts[int(splits[0] * len(data_dicts)): int(splits[0] + splits[1] * len(data_dicts))]
        transform = val_transforms
    elif args.phase == 'test':
        data_dicts = data_dicts[int((splits[0] + splits[1]) * len(data_dicts)):]
        transform = val_transforms
        
    print(f'{args.phase} len - {len(data_dicts)}')
    
    # ------------ Format dataset ------------  
    if args.cache_dataset:
        if args.uniform_sample and args.phase == 'train':
            dataset = UniformCacheDataset(data=data_dicts, transform=transform, 
                                                cache_rate=args.cache_rate, datasetkey=args.datasetkey)
        else:
            dataset = CacheDataset(data=data_dicts, transform=transform, cache_rate=args.cache_rate)
    else:
        if args.uniform_sample and args.phase == 'train':
            dataset = UniformDataset(data=data_dicts, transform=transform, 
                                            datasetkey=args.datasetkey)
        else:
            dataset = Dataset(data=data_dicts, transform=transform)
            
            
    if args.phase == 'train':
        sampler = DistributedSampler(dataset=dataset, even_divisible=True, shuffle=True) if args.dist else None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=args.num_workers, 
                                collate_fn=list_data_collate, sampler=sampler)
        return loader, sampler, len(dataset)
    else:  
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    
    return loader, transform, len(dataset)
    
    

def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()