
# Dataloader for abdominal images
import glob
import numpy as np
from .utils import niftiio as nio
from .utils import transform_utils as trans
from .utils.abd_dataset_utils import get_normalize_op
from .utils.transform_albu import get_albu_transforms, get_resize_transforms
import copy
import random, cv2, os
import torch.utils.data as torch_data
import math
import itertools
from pdb import set_trace
from multiprocessing import Process
import albumentations as A
from tqdm import tqdm




class BasicDataset(torch_data.Dataset):
    def __init__(self, mode, transforms, base_dir, domains: list, pseudo = False, 
                 idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, extern_norm_fn = None, LABEL_NAME=["bg", "fore"],
                 filter_non_labeled=False, use_diff_axis_view=False, chunksize=200, fineSize=192):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        super(BasicDataset, self).__init__()
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.pseudo = pseudo
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct
        self.fineSize = fineSize
        self.albu_transform = get_albu_transforms()
        # self.test_resizer   = get_resize_transforms((opt.fineSize, opt.fineSize))
        self.fake_interpolate    = True # True
        self.use_diff_axis_view  = use_diff_axis_view
        self.filter_non_labeled  = filter_non_labeled
        self.input_window = 1  

        self.resizer = A.Compose([
            A.Resize(fineSize, fineSize, interpolation=cv2.INTER_NEAREST)
        ], p=1.0, additional_targets={'image2': 'image', "mask2":"mask"})

        self.img_pids = {}
        for _domain in self.domains: # load file names
            if "BraTS" in _domain:
                self.img_pids[_domain] = sorted([ fid.split("-")[-2] for fid in
                                     glob.glob(self._base_dir + "/" + _domain + "/img/*.nii.gz") ],
                                     key = lambda x: int(x))
            else:
                self.img_pids[_domain] = sorted([fid.split("_")[-1].split(".nii.gz")[0] for fid in
                                                 glob.glob(self._base_dir + "/" + _domain + "/img/*.nii.gz")],
                                                key=lambda x: int(x))
        self.scan_ids = self.__get_scanids(mode, idx_pct) # train val test split in terms of patient ids
        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids len = ' + \
              f'{[len(self.scan_ids[_dm]) for _dm in self.scan_ids.keys()]}')

        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) # image files names according to self.scan_ids
        if self.is_train:
            self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        elif mode == 'test': # Source domain test
            self.pid_curr_load = self.scan_ids
        elif mode == 'test_all':
            # Choose this when being used as a target domain testing set. Liu et al.
            self.pid_curr_load = self.scan_ids
        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [itm['img_fid'] for _, itm in
                                                                   self.sample_list[self.domains[0]].items() ])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
            assert len(self.domains) == 1, 'for now we only support one normalization function for the entire set'
            self.normalize_op = extern_norm_fn


        # load to memory
        # self.sample_list All
        self.actual_dataset = None
        self.chunksize  = chunksize
        self.chunk_id = 0
        self.chunk_pool, self.current_chunk = {}, {}
        for _domain, item in self.sample_list.items():
            self.chunk_pool[_domain] = list(item.keys())

        chunk, status = self.next_chunk(self.sample_list)
        self.actual_dataset = self.__read_dataset(chunk, status)
        self.size = len(self.actual_dataset)     # 2D

    def update_chunk(self):
        chunk, status = self.next_chunk(self.sample_list)
        self.actual_dataset = self.__read_dataset(chunk, status)

    def __get_scanids(self, mode, idx_pct):
        """
        index by domains given that we might need to load multi-domain data
        idx_pct: [0.7 0.1 0.2] for train val test. with order te val tr
        """
        tr_ids      = {}
        val_ids     = {}
        te_ids      = {}
        te_all_ids  = {}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            te_ids[_domain]     = self.img_pids[_domain][: te_size]
            val_ids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_ids[_domain]     = self.img_pids[_domain][te_size + val_size: ]
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]   ))

        if self.phase == 'train':
            return tr_ids
        elif self.phase == 'val':
            return val_ids
        elif self.phase == 'test':
            return te_ids
        elif self.phase == 'test_all':
            return te_all_ids

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            domain_dir = os.path.join(self._base_dir, _domain)
            print("=== reading domains from:", domain_dir)
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}
                if "BraTS" in _domain:

                    _img_fid = os.path.join(domain_dir, 'img', f'{_domain[:-4]}-{curr_id}-000.nii.gz')
                    if not self.pseudo:
                        _lb_fid  = os.path.join(domain_dir, 'seg', f'{_domain[:-4]}-{curr_id}-000.nii.gz')
                    else:
                        _lb_fid  = os.path.join(domain_dir, 'seg', f'{_domain[:-4]}-{curr_id}-000.nii.gz.npy')  # npy

                    # _sam_fid = os.path.join(domain_dir, 'seg', f'sam_{curr_id}.npy')

                else:
                    _img_fid = os.path.join(domain_dir, 'img', f'img_{curr_id}.nii.gz')
                    if not self.pseudo:
                        _lb_fid = os.path.join(domain_dir, 'seg', f'seg_{curr_id}.nii.gz')
                    else:
                        _lb_fid = os.path.join(domain_dir, 'seg', f'pseudo_{curr_id}.nii.gz.npy')  # npy

                    # _sam_fid = os.path.join(domain_dir, 'seg', f'sam_{curr_id}.npy')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                # curr_dict["sam_fid"] = _sam_fid
                out_list[_domain][str(curr_id)] = curr_dict

        print("=== search sample num:", len(out_list))
        return out_list

    def filter_with_label(self, img, lb):
        # H, W, C, filter zero
        if self.phase == "train":
            filter = np.any(np.any(img, axis=0), axis=0)
            img, lb = img[..., filter], lb[..., filter]

            if self.filter_non_labeled:
                if self.dataset_key == "knee":
                    filter2 = np.any(np.any(lb == 2, axis=0), axis=0)
                    filter4 = np.any(np.any(lb == 4, axis=0), axis=0)
                    filter = filter2 + filter4
                else:
                    filter = np.any(np.any(lb, axis=0), axis=0)

                # img, lb = img[..., filter], lb[..., filter]
        return img, lb

    def __read_dataset(self, chunk, status):
        """
        Read the dataset into memory
        """

        out_list = []
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset
        for _domain, _curr_chunk in tqdm(chunk.items()):  # .items()
            domain_ids = 0
            if status[_domain] != 3:
                print(f"==== update dataset for: {_domain} w/ status = , {status[_domain]}")

            for scan_id in _curr_chunk:
                domain_ids += 1
                itm = self.sample_list[_domain][scan_id]
                if scan_id not in self.pid_curr_load[_domain]:
                    continue  # Pass

                if (status[_domain] == 0) or \
                   (status[_domain] == 1 and domain_ids > self.chunksize // 2) or \
                   (status[_domain] == 2 and domain_ids <= self.chunksize // 2):
                    # print(f"status = {status[_domain]}, length= {len(self.actual_dataset)}, sample = {glb_idx}")
                    size = self.actual_dataset[glb_idx]['size']
                    out_list.extend(self.actual_dataset[glb_idx: glb_idx + size])
                    glb_idx += size
                    continue

                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img_original = np.float32(img)
                img = img_original.copy()

                # img, self.mean, self.std = self.normalize_op(img)
                _, mean, std = self.normalize_op(img)
                if not self.pseudo:
                    lb = nio.read_nii_bysitk(itm["lbs_fid"])
                else:
                    uncertainty_thr = 0.05  #  0.05
                    lb_cache = np.load(itm["lbs_fid"], allow_pickle=True).item()
                    lb = lb_cache['pseudo'].cpu().numpy()              # "pseudo": curr_pred, "score":curr_score , "uncertainty"
                    uncertainty = lb_cache['uncertainty'].cpu().numpy()   # Z, C, H, W
                    uncertainty = np.float32(uncertainty)
                    
                    new_lb = np.zeros_like(lb)
                    for cls in range(self.nclass - 1):
                        un_mask = (uncertainty[:, cls+1] < uncertainty_thr ) * (cls+1)
                        new_lb[lb == (cls+1)] = un_mask[lb == (cls+1)]

                    lb = new_lb

                lb_original = np.float32(lb)
                lb = lb_original.copy()

                # -> H, W, C
                img, lb = map(lambda arr: np.transpose(arr, (1, 2, 0)), [img, lb])
                assert img.shape[-1] == lb.shape[-1], f"ASSERT {img.shape} = {lb.shape}"

                # Resize:
                if img.shape[1] != self.fineSize:
                    # H, W, C
                    res = self.resizer(image=img, mask=lb)
                    img, lb = res['image'], res['mask']

                prt_cache = f" {_domain} stat ({domain_ids}/{len(_curr_chunk)}): shape={img.shape}, max={img.max()}, min={img.min()}"

                # filter ...
                # img, lb = self.filter_with_label(img, lb)

                out_list, glb_idx = self.add_to_list(glb_idx, out_list, img, lb,
                                                     mean, std, _domain,
                                                     scan_id, itm["img_fid"])

                if (domain_ids) % (len(_curr_chunk) // 2) == 0:
                    print(prt_cache + f", filtered shape={img.shape}")

                # Add various axis view !!!
                if self.phase == "train" and self.use_diff_axis_view:
                    # C, W, H
                    img, lb = img_original, lb_original
                    # Resize:
                    if img.shape[1] != self.fineSize:
                        res = self.resizer(image=img, mask=lb) # assume H, W, (C)<-
                        img, lb = res['image'], res['mask']
                    
                    img, lb = self.filter_with_label(img, lb)
                    
                    out_list, glb_idx = self.add_to_list(glb_idx, out_list, img, lb,
                                                     mean, std, _domain, scan_id, itm["img_fid"])

                del img, lb, img_original, lb_original

        del self.actual_dataset
        return out_list
    
    def next_chunk(self, all_samples):
        # 0 No update, 1 First half, 2 Second half, 3 All updates
        status = {}
        self.last_chunk = copy.deepcopy(self.current_chunk)
        for _domain, _sample_list in tqdm(all_samples.items()):
            # Put all in
            if not self.is_train or len(_sample_list) < self.chunksize:
                self.current_chunk[_domain] = _sample_list
                if _domain not in self.last_chunk:
                    status[_domain] = 3  # all
                else:
                    status[_domain] = 0  # not updates
                continue

            # chunksize
            random.shuffle(self.chunk_pool[_domain])
            if _domain not in self.last_chunk:
                status[_domain] = 3
                self.current_chunk[_domain] = self.chunk_pool[_domain][:self.chunksize]
                self.chunk_pool[_domain]    = self.chunk_pool[_domain][self.chunksize:]

            else:
                status[_domain] = self.chunk_id//2 + 1  # 1, 2
                candidate                = self.chunk_pool[_domain][:self.chunksize // 2]
                self.chunk_pool[_domain] = self.chunk_pool[_domain][self.chunksize // 2:]
                if status[_domain] == 1:
                    self.current_chunk[_domain][:self.chunksize // 2] = candidate
                else:
                    self.current_chunk[_domain][self.chunksize // 2:] = candidate


            if _domain in self.last_chunk:
                self.chunk_pool[_domain] = self.chunk_pool[_domain] + self.last_chunk[_domain]
            self.chunk_id += 1

        return self.current_chunk, status


    def add_to_list(self, glb_idx, out_list, img, lb, mean, std, _domain, scan_id, file_id):
        # now start writing everthing in
        c = 3
        
        # np.any(np.any(lb == 2, axis=0), axis=0)
        
        for ii in range(img.shape[-1]):
            is_end   = False
            is_start   = False
            if ii == 0:
                is_start = True
                # write the beginning frame
                if self.input_window == 3:
                    _img = img[..., 0: c].copy()
                    _img[..., 1] = _img[..., 0]
                elif self.input_window == 1:
                    _img = img[..., 0: 0 + 1].copy()
               
                
            elif ii < img.shape[-1] - 1:
                if self.input_window == 3:
                    _img = img[..., ii -1: ii + 2].copy()
                elif self.input_window == 1:
                    _img = img[..., ii: ii + 1].copy()
                    
            else:
                is_end = True
                if self.input_window == 3:
                    _img = img[..., ii-2: ii + 1].copy()
                    _img[..., 0] = _img[..., 1]
                elif self.input_window == 1:
                    _img = img[..., ii: ii+ 1].copy()
            
            _lb  = lb[..., ii: ii + 1]
            
            # Filter vacant label
            # filter = np.any(np.any(lb, axis=0), axis=0)
            if not np.any(np.any(_lb, axis=0), axis=0):
                continue
            
            out_list.append(
                       {"img": _img, "lb":_lb,  "size": img.shape[-1],
                        "mean":mean, "std":std, "slice": ii/ img.shape[-1],
                        "is_start": is_start, "is_end": is_end,
                        "domain": _domain, "nframe": img.shape[-1],
                        "scan_id": _domain + "_" + scan_id,
                        "pid": scan_id, "file_id": file_id, "z_id":ii})
            glb_idx += 1

 
        
        return out_list, glb_idx


    def get_patch_from_img(self, img_H, img_L, img_L2, crop_size=[320, 320], zslice_dim=2):
        # --------------------------------
        # randomly crop the patch
        # --------------------------------

        H, W, _ = img_H.shape
        rnd_h = random.randint(0, max(0, H - crop_size[0]))
        rnd_w = random.randint(0, max(0, W - crop_size[1]))

        # image = torch.index_select(image, 0, torch.tensor([1]))
        if zslice_dim == 2:
            patch_H = img_H[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
            patch_L = img_L[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
            patch_L2 = img_L2[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
        elif zslice_dim == 0:
            patch_H = img_H[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]
            patch_L = img_L[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]
            patch_L2 = img_L2[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]

        return patch_H, patch_L, patch_L2


    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)
