import math
import copy
import torch
from torch import nn
import torch.nn.functional as func
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

# from einops import rearrange

import torchgeometry as tgm
import os
import errno
from PIL import Image
# from pytorch_msssim import ssim
import cv2
import numpy as np
import imageio

# from torch.utils.tensorboard import SummaryWriter
from .degradation import get_fade_kernels, get_ksu_kernel, apply_ksu_kernel, apply_tofre, apply_to_spatial
from dataset import Dataset, Dataset_Aug1, BrainDataset
from .st_branch_model.lpips import LPIPS
try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def cycle(dl):
    while True:
        for inputs in dl:
            yield inputs


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            diffusion_type,
            restore_fn,
            *,
            image_size,
            device_of_kernel,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            kernel_std=0.1,
            initial_mask=11,
            fade_routine='Incremental',
            sampling_routine='default',
            discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.restore_fn = restore_fn

        # self.backbone = diffusion_type.split('_')[0]

        self.device_of_kernel = device_of_kernel
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.kernel_std = kernel_std
        self.initial_mask = initial_mask
        self.fade_routine = fade_routine
        self.backbone = diffusion_type.split('_')[0]
        self.degradation_type = diffusion_type.split('_')[1]

        if self.degradation_type == 'fade':
            self.fade_kernels = get_fade_kernels(fade_routine, self.num_timesteps, image_size, kernel_std, initial_mask)
            # print("=== self.fade_kernels shape = ", self.fade_kernels.shape)  # [5, 256, 256]

        elif self.degradation_type == "kspace":
            self.get_new_kspace()
            # print("=== self.kspace_kernels shape = ", self.kspace_kernels.shape)   # [5, 256, 256]
        else:
            raise NotImplementedError()

        self.sampling_routine = sampling_routine
        self.discrete = discrete

        # Frequency Loss
        if self.backbone == 'twobranch' or self.backbone == 'twounet':
            self.amploss = self.restore_fn.amploss  # .to(self.device, non_blocking=True)
            self.phaloss = self.restore_fn.phaloss  # .to(self.device, non_blocking=True)


        self.lpips = LPIPS().eval().cuda()  # .to(self.device, non_blocking=True)

        self.use_fre_loss = True
        self.update_kernel = False
        self.use_lpips = True
        self.clamp_every_sample = True # Stride
        self.use_fre_noise = True

    # if _MRIDOWN == "4X":
    #     mask_type_str, center_fraction, MRIDOWN = "random", 0.1, 4
    # elif _MRIDOWN == "8X":
    #     mask_type_str, center_fraction, MRIDOWN = "equispaced", 0.04, 8
    #
    # ff = create_mask_for_mask_type(mask_type_str, [center_fraction], [MRIDOWN]) ## 0.2 fo

    def get_new_kspace(self):
        self.kspace_kernels = get_ksu_kernel(self.num_timesteps, self.image_size)
        self.kspace_kernels = torch.stack(self.kspace_kernels).squeeze(1)


    def get_kspace_kernels(self, index, rand_kernels):
        if rand_kernels is not None:
            k = torch.stack([rand_kernels[:, index]], 1)
        else:
            k = torch.stack([self.kspace_kernels[index]], 1)
        return k

    @torch.no_grad()
    def sample(self, batch_size=16, faded_recon_sample=None, aux=None, t=None):
        rand_kernels = None
        sample_device = faded_recon_sample.device
        if self.degradation_type == 'fade':
            if 'Random' in self.fade_routine:
                rand_kernels = []
                rand_x = torch.randint(0, self.image_size + 1, (batch_size,), device=sample_device).long()
                rand_y = torch.randint(0, self.image_size + 1, (batch_size,), device=sample_device).long()
                for i in range(batch_size):
                    rand_kernels.append(torch.stack(
                        [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                         rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
                rand_kernels = torch.stack(rand_kernels)

        elif self.degradation_type == 'kspace':
            if self.update_kernel:
                self.get_new_kspace()
            rand_kernels = []

            for i in range(batch_size ):
                rand_kernels.append(torch.stack(
                    [self.kspace_kernels[j][:self.image_size,
                     : self.image_size] for j in range(len(self.kspace_kernels))]))

            rand_kernels = torch.stack(rand_kernels)

        if t is None:
            t = self.num_timesteps

        for i in range(t):
            with torch.no_grad():
                if self.degradation_type == 'fade':
                    if rand_kernels is not None:
                        faded_recon_sample = torch.stack([rand_kernels[:, i].to(sample_device),
                                                          rand_kernels[:, i].to(sample_device),
                                                          rand_kernels[:, i].to(sample_device)],
                                                         1) * faded_recon_sample
                    else:
                        faded_recon_sample = self.fade_kernels[i].to(sample_device) * faded_recon_sample

                elif self.degradation_type == 'kspace':
                    if rand_kernels is not None:
                        k = torch.stack([rand_kernels[:, i]], 1)
                    else:
                        k = torch.stack([self.kspace_kernels[i]], 1)

                    faded_recon_sample = apply_ksu_kernel(faded_recon_sample, k)

        return_k = k

        if self.discrete:
            faded_recon_sample = (faded_recon_sample + 1) * 0.5
            faded_recon_sample = (faded_recon_sample * 255)
            faded_recon_sample = faded_recon_sample.int().float() / 255
            faded_recon_sample = faded_recon_sample * 2 - 1

        xt = faded_recon_sample
        direct_recons = None
        recon_sample = None
        all_recons = []
        all_masks = []

        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            if self.backbone == "unet":
                recon_sample = self.restore_fn(faded_recon_sample, step)
            elif self.backbone == "twounet":
                recon_sample = self.restore_fn(faded_recon_sample, aux, k, step)

            elif self.backbone == "twobranch":
                recon_sample, recon_fre = self.restore_fn(faded_recon_sample, aux, step)
                recon_sample = recon_sample // 2 + recon_fre // 2

            if direct_recons is None:
                direct_recons = recon_sample
                # recon_cache   = recon_sample

            if self.degradation_type == 'fade':
                if self.sampling_routine == 'default':
                    for i in range(t - 1):
                        with torch.no_grad():
                            if rand_kernels is not None:
                                recon_sample = torch.stack([rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device)], 1) * recon_sample
                            else:
                                recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample
                    faded_recon_sample = recon_sample

                elif self.sampling_routine == 'x0_step_down':
                    for i in range(t):
                        with torch.no_grad():
                            recon_sample_sub_1 = recon_sample
                            if rand_kernels is not None:
                                recon_sample = torch.stack([rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device)], 1) * recon_sample
                            else:
                                recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample

                    faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1

            elif self.degradation_type == 'kspace':
                # faded_recon_sample = recon_sample

                if self.sampling_routine == 'default':
                    with torch.no_grad():
                        if t >=1:
                            k = self.get_kspace_kernels(t - 1, rand_kernels)
                            recon_sample = apply_ksu_kernel(recon_sample, k)

                    faded_recon_sample = recon_sample


                elif self.sampling_routine == 'x0_step_down':
                    all_recons.append(recon_sample)
                    if t <= 1:
                        if t == 1:
                            recon_sample_sub_1 = recon_sample
                            k = self.get_kspace_kernels(0, rand_kernels)

                            recon_sample = apply_ksu_kernel(recon_sample, k)
                            faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1

                        else:
                            faded_recon_sample = recon_sample
                    else:
                        with torch.no_grad():
                            k = self.get_kspace_kernels(t - 2, rand_kernels)
                            recon_sample_sub_1 = apply_ksu_kernel(recon_sample, k)

                            k = self.get_kspace_kernels(t - 1, rand_kernels)
                            recon_sample = apply_ksu_kernel(recon_sample, k)

                        faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1
                        if self.clamp_every_sample:
                            faded_recon_sample = faded_recon_sample.clamp(-1, 1)

                elif self.sampling_routine == 'x0_step_down_fre':
                    all_recons.append(recon_sample)
                    if t <= 1:
                        faded_recon_sample = recon_sample

                    else:
                        k_full = self.get_kspace_kernels(- 1, rand_kernels)
                        faded_recon_sample_fre, _ = apply_tofre(faded_recon_sample, k_full)


                        with torch.no_grad():
                            if t > 1:
                                kt_sub_1 = self.get_kspace_kernels(t - 2, rand_kernels)
                            else:
                                kt_sub_1 = torch.ones_like(k_full).to(sample_device)

                            kt = self.get_kspace_kernels(t - 1, rand_kernels)  # last one

                            recon_sample_sub_1_fre, kt_sub_1 = apply_tofre(recon_sample, kt_sub_1)
                            recon_sample_fre, kt = apply_tofre(recon_sample, kt)


                        # Mask Region...
                        k_mask = (kt_sub_1 - kt).cuda()  # Stride
                        fre_amend = (recon_sample_sub_1_fre * kt_sub_1 - recon_sample_fre * kt)
                        faded_recon_sample_fre =  faded_recon_sample_fre  + fre_amend

                        faded_recon_sample = apply_to_spatial(faded_recon_sample_fre)

                        # Strange black stripe
                    if self.clamp_every_sample:
                        faded_recon_sample =faded_recon_sample.clamp(-1, 1)


            recon_sample = faded_recon_sample

            # all_masks.append()
            t -= 1

        all_recons = torch.stack(all_recons)
        print("all_recons shape: ", all_recons.shape)

        return xt, direct_recons, recon_sample, return_k, all_recons

    @torch.no_grad()
    def all_sample(self, batch_size=16, faded_recon_sample=None, aux=None, t=None, times=None):
        # TODO
        print("Running into all_sample...")
        rand_kernels = None
        sample_device = faded_recon_sample.device
        if self.degradation_type == 'fade':
            if 'Random' in self.fade_routine:
                rand_kernels = []
                rand_x = torch.randint(0, self.image_size + 1, (batch_size,), device=faded_recon_sample.device).long()
                rand_y = torch.randint(0, self.image_size + 1, (batch_size,), device=faded_recon_sample.device).long()
                for i in range(batch_size, ):
                    rand_kernels.append(torch.stack(
                        [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                         rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
                rand_kernels = torch.stack(rand_kernels)

        elif self.degradation_type == 'kspace':
            rand_kernels = []
            rand_x = torch.randint(0, self.image_size + 1, (batch_size,), device=faded_recon_sample.device).long()

            for i in range(batch_size, ):
                rand_kernels.append(torch.stack(
                    [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                     : self.image_size] for j in range(len(self.fade_kernels))]))
            rand_kernels = torch.stack(rand_kernels)

        if t is None:
            t = self.num_timesteps
        if times is None:
            times = t

        for i in range(t):
            with torch.no_grad():
                if self.degradation_type == 'fade':
                    if 'Random' in self.fade_routine:
                        faded_recon_sample = torch.stack([rand_kernels[:, i].to(sample_device),
                                                          rand_kernels[:, i].to(sample_device),
                                                          rand_kernels[:, i].to(sample_device)], 1) * faded_recon_sample
                    else:
                        faded_recon_sample = self.fade_kernels[i].to(sample_device) * faded_recon_sample
                elif self.degradation_type == 'kspace':
                    if rand_kernels is not None:
                        # print(f"kspace randkeynel k={rand_kernels[:, i].shape}, x={x.shape}")
                        k = torch.stack([rand_kernels[:, i]], 1)
                        faded_recon_sample = apply_ksu_kernel(faded_recon_sample, k)
                    else:
                        # print(f"kspace k={self.kspace_kernels[i].shape}, x={x.shape}")
                        k = self.kspace_kernels[i]
                        faded_recon_sample = apply_ksu_kernel(faded_recon_sample, k)

        if self.discrete:
            faded_recon_sample = (faded_recon_sample + 1) * 0.5
            faded_recon_sample = (faded_recon_sample * 255)
            faded_recon_sample = faded_recon_sample.int().float() / 255
            faded_recon_sample = faded_recon_sample * 2 - 1

        x0_list = []
        xt_list = []

        while times:
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            if self.backbone == "unet":
                recon_sample = self.restore_fn(faded_recon_sample, step)
            elif self.backbone == "twounet":
                recon_sample = self.restore_fn(faded_recon_sample, aux, k, step)

            elif self.backbone == "twobranch":
                recon_sample, recon_fre = self.restore_fn(faded_recon_sample, aux, step)
                recon_sample = recon_sample // 2 + recon_fre // 2
            x0_list.append(recon_sample)

            if self.degradation_type == 'fade':
                if self.sampling_routine == 'default':
                    for i in range(times - 1):
                        with torch.no_grad():
                            if rand_kernels is not None:
                                recon_sample = torch.stack([rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device),
                                                            rand_kernels[:, i].to(sample_device)], 1) * recon_sample
                            else:
                                recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample
                    faded_recon_sample = recon_sample

                elif self.sampling_routine == 'x0_step_down':
                    for i in range(t):
                        with torch.no_grad():
                            recon_sample_sub_1 = recon_sample
                            if rand_kernels is not None:

                                recon_sample = apply_ksu_kernel(recon_sample, rand_kernels[i])
                            else:
                                recon_sample = apply_ksu_kernel(recon_sample, self.kspace_kernels[i])

                    faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1

            elif self.degradation_type == 'kspace':
                # faded_recon_sample = recon_sample
                if self.sampling_routine == 'default':
                    for i in range(t - 1):
                        with torch.no_grad():
                            if rand_kernels is not None:
                                k = torch.stack([rand_kernels[:, i]], 1)
                                recon_sample = apply_ksu_kernel(recon_sample, k)
                            else:
                                recon_sample = apply_ksu_kernel(recon_sample, self.kspace_kernels[i])

                    faded_recon_sample = recon_sample

                elif self.sampling_routine == 'x0_step_down':
                    for i in range(t):
                        with torch.no_grad():
                            recon_sample_sub_1 = recon_sample
                            if rand_kernels is not None:
                                k = torch.stack([rand_kernels[:, i]], 1)
                                recon_sample = apply_ksu_kernel(recon_sample, k)
                            else:
                                recon_sample = apply_ksu_kernel(recon_sample, self.kspace_kernels[i])

                    faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1


            xt_list.append(faded_recon_sample)
            times -= 1

        return x0_list, xt_list

    def q_sample(self, x_start, t):
        with torch.no_grad():
            rand_kernels = None
            
        if self.degradation_type == 'fade':
            if 'Random' in self.fade_routine:
                rand_kernels = []
                rand_x = torch.randint(0, self.image_size + 1, (x_start.size(0),), device=x_start.device).long()
                rand_y = torch.randint(0, self.image_size + 1, (x_start.size(0),), device=x_start.device).long()
                for i in range(x_start.size(0),):
                    rand_kernels.append(torch.stack(
                        [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                         rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
                rand_kernels = torch.stack(rand_kernels)

        elif self.degradation_type == 'kspace':
            if self.update_kernel:
                self.get_new_kspace()
            rand_kernels = []

            for i in range(x_start.size(0),):
                rand_kernels.append(torch.stack(
                    [self.kspace_kernels[j][:self.image_size,
                     : self.image_size] for j in range(len(self.kspace_kernels))]))

            rand_kernels = torch.stack(rand_kernels)

        max_iters = torch.max(t)
        all_fades = []
        x = x_start
        for i in range(max_iters + 1):
            with torch.no_grad():
                if self.degradation_type == 'fade':
                    if rand_kernels is not None:
                        k = torch.stack([rand_kernels[:, i],
                                         rand_kernels[:, i],
                                         rand_kernels[:, i]], 1)
                    else:
                        k = self.fade_kernels[i]   # fade k=torch.Size([24, 3, 128, 128]),

                    x = k * x

                elif self.degradation_type == 'kspace':
                    # fade k=torch.Size([24, 128, 128]), x=torch.Size([24, 1, 128, 128])
                    if rand_kernels is not None:
                        k = torch.stack([rand_kernels[:, i]], 1)
                    else:
                        k = torch.stack([self.kspace_kernels[i]], 1)

                    x = apply_ksu_kernel(x, k, self.use_fre_noise)
                all_fades.append(x)

        all_fades = torch.stack(all_fades)  # Fade, all_fades shape: torch.Size([5, 24, 3, 128, 128])

        choose_fade = []
        # print("debug: t = ", t), [1, 4, 3, 3, 3, 2, 3, 4, 4, 2, 0, 4]
        for step in range(t.shape[0]):
            if step != -1:
                choose_fade.append(all_fades[t[step], step])
            else:
                choose_fade.append(x_start[step])
        choose_fade = torch.stack(choose_fade)

        if self.discrete:
            choose_fade = (choose_fade + 1) * 0.5
            choose_fade = (choose_fade * 255)
            choose_fade = choose_fade.int().float() / 255
            choose_fade = choose_fade * 2 - 1

        return choose_fade, k

    def reconstruct_loss(self, x_start, x_recon):
        if self.loss_type == 'l1':
            loss = (x_start - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = func.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()
        return loss


    def p_losses(self, x_start, aux, t):
        self.debug_print = False
        x_mix, k = self.q_sample(x_start=x_start, t=t)

        if self.debug_print:
            self.debug_print = False

        if self.backbone == 'unet':
            x_recon = self.restore_fn(x_mix, t)
            loss = self.reconstruct_loss(x_start, x_recon)

        elif self.backbone == 'twounet':
            x_recon = self.restore_fn(x_mix, aux, k, t)
            loss = self.reconstruct_loss(x_start, x_recon)

            # LPIPS
            if self.use_lpips:
                lpips_weight = 0.1
                lpips_loss = self.lpips(x_recon, x_start).mean()
                loss += lpips_weight * lpips_loss


            if self.use_fre_loss:  # NAN
                fft_weight = 0.01
                amp = self.amploss(x_recon, x_start)
                # pha = self.phaloss(x_recon, x_start)

                loss += fft_weight * amp



        elif self.backbone == 'twobranch':
            x_recon, x_recon_fre = self.restore_fn(x_mix, aux, t)

            loss_spatial = self.reconstruct_loss(x_start, x_recon)
            loss_freq = self.reconstruct_loss(x_start, x_recon_fre)
            loss = loss_spatial + loss_freq

            if np.random.rand() < 0.01:
                print("loss_spatial:", loss_spatial, "loss_freq:", loss_freq)

            # LPIPS
            if self.use_lpips:
                lpips_weight = 0.1
                lpips_loss = self.lpips(x_recon, x_start).mean()
                loss += lpips_weight * lpips_loss

                # print("lpips_loss:", lpips_loss)

            # if self.use_fre_loss:
            #     fft_weight = 0.01
            #     amp = self.amploss(x_recon_fre, x_start)
            #     pha = self.phaloss(x_recon_fre, x_start)
            #
            #     loss += fft_weight * amp

        return loss

    def forward(self, x1, x2=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.degradation_type == 'fade':
            self.fade_kernels = self.fade_kernels.to(device)

        elif self.degradation_type == 'kspace':
            self.kspace_kernels = self.kspace_kernels.to(device)

        loss = self.p_losses(x1, x2, t, *args, **kwargs)
        max_norm = 1.0  # Maximum norm for gradients
        torch.nn.utils.clip_grad_norm_(self.restore_fn.parameters(), max_norm)

        return loss



class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=700000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            load_path=None,
            dataset=None,
            shuffle=True,
            domain=None,
            aux_modality=None,
            num_channels=1,
            debug=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every if not debug else 10

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.module.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'train':
            print(dataset, "DA used")
            self.ds = Dataset_Aug1(folder, image_size)
        elif dataset.lower() == 'brain':
            print(dataset, "Brain DA used")
            # mode, base_dir, image_size, nclass, domains, aux_modality,
            self.ds = BrainDataset("train", folder, image_size, 4,
                                   debug=debug,
                                   domains=domain,
                                   num_channels=num_channels,
                                   aux_modality=aux_modality)  # mode, base_dir, domains:
        else:
            print(dataset)
            self.ds = Dataset(folder, image_size)


        self.dl = cycle(
            data.DataLoader(self.ds,
                            batch_size=train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=16,
                            drop_last=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed for mixed precision training on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        os.makedirs(results_folder, exist_ok=True)
        self.results_folder = Path(results_folder)
        # self.results_folder.mkdir(exist_ok=True)
        self.lpips = LPIPS().eval().cuda()

        self.reset_parameters()

        if load_path is not None:
            self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        model_data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(model_data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        model_data = torch.load(load_path)

        self.step = model_data['step']
        self.model.load_state_dict(model_data['model'])
        self.ema_model.load_state_dict(model_data['ema'])

    @staticmethod
    def add_title(path, title):
        img1 = cv2.imread(path)

        black = [0, 0, 0]
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        # writer = SummaryWriter()

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                # inputs = next(self.dl).cuda()
                last_model_state = self.model.state_dict()
                optimizer_state = self.opt.state_dict()

                data_dict = next(self.dl)

                img = data_dict['img'].cuda()
                # Add some Gaussian
                sigma = torch.rand(1).item() * 0.1
                img = img + torch.randn_like(img) * sigma

                aux = data_dict['aux'].cuda()

                # loss = self.model(inputs)
                loss = torch.mean(self.model(img, aux))
                if torch.isnan(loss).any():
                    print(f"NaN encountered in epoch {self.step}. Reverting model.")
                    self.model.load_state_dict(last_model_state)  # Revert model
                    self.opt.load_state_dict(optimizer_state)  # Revert optimizer
                    continue  # Skip the rest of this training step


                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)


            if (self.step + 1) % (min(self.train_num_steps // 100 + 1, 100)) == 0:
                print(f'{self.step + 1}: {u_loss}')


            # writer.add_scalar("Loss/train", loss.item(), self.step)
            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)


            self.opt.step()

            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # TEST and SAVE
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                batches = self.batch_size
                data_dict = next(self.dl)  # .cuda()

                og_img = data_dict['img'].cuda()
                aux = data_dict['aux'].cuda()

                # xt, direct_recons, all_images = self.ema_model.sample(batch_size=batches, faded_recon_sample=og_img)
                xt, direct_recons, all_images, return_k, all_recons = self.ema_model.module.sample(
                                                 batch_size=batches,
                                                 faded_recon_sample=og_img,
                                                 aux=aux)


                og_img = (og_img + 1) * 0.5
                aux = (aux + 1) * 0.5
                all_images = ((all_images + 1) * 0.5 ) .clamp_(0, 1)
                all_recons = ((all_recons + 1) * 0.5) .clamp_(0, 1)
                direct_recons = ((direct_recons + 1) * 0.5) .clamp_(0, 1)
                xt = (xt + 1) * 0.5

                # print("DEBUG - og_img shape: ", og_img.shape, og_img.max(), og_img.min())
                # print("DEBUG - all_images shape: ", all_images.shape, all_images.max(), all_images.min())
                # print("DEBUG - direct_recons shape: ", direct_recons.shape, direct_recons.max(), direct_recons.min())
                # print("DEBUG - xt shape: ", xt.shape, xt.max(), xt.min())
                # print("DEBUG - aux shape: ", aux.shape, aux.max(), aux.min())
                # print("DEBUG - return_k shape: ", return_k.shape, return_k.max(), return_k.min())
                # print("DEBUG - return_sample shape: ", return_sample.shape, return_sample.max(), return_sample.min())

                # 24, 1, 128, 128
                # Calculate SSIM and PSNR, LPIPS
                from skimage.metrics import structural_similarity as ssim
                from skimage.metrics import peak_signal_noise_ratio as psnr

                img_ = all_images.cpu().permute(0, 2, 3, 1).numpy()[..., 0]
                og_img_ = og_img.cpu().permute(0, 2, 3, 1).numpy()[..., 0]
                img_ = np.clip(img_, 0, 1)

                ssim_ = ssim(img_, og_img_, multichannel=False, data_range=1.0).mean()
                psnr_ = psnr(img_, og_img_, data_range=1).mean()

                lpips = self.lpips(all_images, og_img).mean()

                print("=== Final Metrics: SSIM: ", ssim_, " PSNR: ", psnr_, " LPIPS: ", lpips)

                img_ = direct_recons.cpu().permute(0, 2, 3, 1).numpy()[..., 0]
                img_ = np.clip(img_, 0, 1)

                ssim_ = ssim(img_, og_img_, multichannel=False, data_range=1.0).mean()
                psnr_ = psnr(img_, og_img_, data_range=1).mean()

                lpips = self.lpips(direct_recons, og_img).mean()

                print("=== first step Metrics: SSIM: ", ssim_, " PSNR: ", psnr_, " LPIPS: ", lpips)


                os.makedirs(self.results_folder, exist_ok=True)
                # utils.save_image(xt, str(self.results_folder / f'{self.step}-xt-Noise.png'), nrow=6)
                # utils.save_image(all_images, str(self.results_folder / f'{self.step}-full_recons.png'),
                #                  nrow=6)
                # utils.save_image(direct_recons,
                #                  str(self.results_folder / f'{self.step}-sample-direct_recons.png'), nrow=6)
                # utils.save_image(og_img, str(self.results_folder / f'{self.step}-img.png'), nrow=6)
                # utils.save_image(aux, str(self.results_folder / f'{self.step}-aux.png'), nrow=6)

                return_k = return_k.cuda()

                combine = torch.cat((return_k, xt,
                                     all_images, direct_recons, og_img, aux), 2)

                print("combine shape: ", combine.shape)
                utils.save_image(combine, str(self.results_folder / f'{self.step}-combine.png'), nrow=6)
                # all_recons # SHape 50, 24, 1, 128, 128

                # all_recon = all_recons[:, 0] # 50, 1, 128, 128
                # Ensure all_recons is on the CPU
                all_recons = all_recons.cpu()
                all_recons = torch.cat(list(all_recons), dim=-1)

                s = all_recons.shape[-2]
                repeats = all_recons.shape[3] // og_img.shape[3]  # Calculate repeat factor
                # tensor_small = tensor_small.repeat(1, 1, 1, repeats)
                og_img = og_img.cpu()
                all_recons_residual = all_recons - og_img.repeat(1, 1, 1, repeats)
                # all_recons[:, :, :, s:]
                all_recons_residual_2 = all_recons[:, :, :, s:] - all_recons[:, :, :, :-s]
                padding = torch.zeros_like(all_recons_residual[:, :, :, :s // 2])
                all_recons_residual_2 = torch.cat([padding, all_recons_residual_2, padding], dim=-1)

                all_recons = torch.cat([all_recons, all_recons_residual_2, all_recons_residual], dim=-2)

                utils.save_image(all_recons, str(self.results_folder / f'{self.step}-all_recons.png'),
                                 nrow=1)


                acc_loss = acc_loss / (self.save_and_sample_every + 1)
                print(f'Mean of last {self.step}: {acc_loss}, save to :', str(self.results_folder / f'{self.step}-combine.png'))

                acc_loss = 0

                self.save()

            self.step += 1

        print('training completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def test_with_mixup(self, extra_path):
        batches = self.batch_size
        og_img_1 = next(self.dl).cuda()
        og_img_2 = next(self.dl).cuda()
        og_img = (og_img_1 + og_img_2) / 2

        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img_1 = (og_img_1 + 1) * 0.5
        utils.save_image(og_img_1, str(self.results_folder / f'og1-{extra_path}.png'), nrow=6)

        og_img_2 = (og_img_2 + 1) * 0.5
        utils.save_image(og_img_2, str(self.results_folder / f'og2-{extra_path}.png'), nrow=6)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)
            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        frame_one = frames_0[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), format="GIF", append_images=frames_0,
                       save_all=True, duration=100, loop=0)

        frame_one = frames_t[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), format="GIF", append_images=frames_t,
                       save_all=True, duration=100, loop=0)

    def test_from_random(self, extra_path):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        og_img = og_img * 0.9
        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t_names = []
        frames_0_names = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'))

        frames_0 = []
        frames_t = []
        for i in range(len(x0_list)):
            print(i)
            frames_0.append(imageio.imread(frames_0_names[i]))
            frames_t.append(imageio.imread(frames_t_names[i]))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def controlled_direct_reconstruct(self, extra_path):
        batches = self.batch_size
        torch.manual_seed(0)
        og_img = next(self.dl).cuda()
        xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, faded_recon_sample=og_img)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'sample-og-{extra_path}.png'), nrow=6)

        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-recon-{extra_path}.png'), nrow=6)

        direct_recons = (direct_recons + 1) * 0.5
        utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{extra_path}.png'), nrow=6)

        xt = (xt + 1) * 0.5
        utils.save_image(xt, str(self.results_folder / f'sample-xt-{extra_path}.png'),
                         nrow=6)

        self.save()

    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end is not None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = blurred_samples

        cnt = 0
        while cnt < all_samples.shape[0]:
            og_x = all_samples[cnt: cnt + 50]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=og_img.shape[0],
                                                         faded_recon_sample=og_img,
                                                         times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = xt_list[0].to('cpu')
            deblurry_imgs = x0_list[-1].to('cpu')
            direct_deblurry_imgs = x0_list[0].to('cpu')

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(1, 3 // direct_deblurry_imgs.shape[1], 1, 1)

            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

                if sanity_check:
                    folder = './sanity_check/'
                    create_folder(folder)

                    san_imgs = og_img[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-og.png'), nrow=6)

                    san_imgs = blurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-xt.png'), nrow=6)

                    san_imgs = deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-recons.png'), nrow=6)

                    san_imgs = direct_deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-direct-recons.png'), nrow=6)

            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)

            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)

            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)

            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat((direct_deblurred_samples, direct_deblurry_imgs), dim=0)

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        fid_blur = fid_func(samples=[original_sample, blurred_samples])
        rmse_blur = torch.sqrt(torch.mean((original_sample - blurred_samples) ** 2))
        ssim_blur = ssim(original_sample, blurred_samples, data_range=1, size_average=True)
        print(f'The FID of blurry images with original image is {fid_blur}')
        print(f'The RMSE of blurry images with original image is {rmse_blur}')
        print(f'The SSIM of blurry images with original image is {ssim_blur}')

        fid_deblur = fid_func(samples=[original_sample, deblurred_samples])
        rmse_deblur = torch.sqrt(torch.mean((original_sample - deblurred_samples) ** 2))
        ssim_deblur = ssim(original_sample, deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of deblurred images with original image is {fid_deblur}')
        print(f'The RMSE of deblurred images with original image is {rmse_deblur}')
        print(f'The SSIM of deblurred images with original image is {ssim_deblur}')

        print(f'Hence the improvement in FID using sampling is {fid_blur - fid_deblur}')

        fid_direct_deblur = fid_func(samples=[original_sample, direct_deblurred_samples])
        rmse_direct_deblur = torch.sqrt(torch.mean((original_sample - direct_deblurred_samples) ** 2))
        ssim_direct_deblur = ssim(original_sample, direct_deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of direct deblurred images with original image is {fid_direct_deblur}')
        print(f'The RMSE of direct deblurred images with original image is {rmse_direct_deblur}')
        print(f'The SSIM of direct deblurred images with original image is {ssim_direct_deblur}')

        print(f'Hence the improvement in FID using direct sampling is {fid_blur - fid_direct_deblur}')

    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches,
                                                         faded_recon_sample=og_img,
                                                         times=s_times)
            og_img = (og_img + 1) * 0.5

            for j in range(og_img.shape[0]//3):
                original = og_img[j: j + 1]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = x0_list[0][j: j + 1]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = x0_list[-1][j: j + 1]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                blurry_image = xt_list[0][j: j + 1]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)

                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def paper_showing_diffusion_images(self, s_times=None):

        cnt = 0
        to_show = [0, 1, 2, 4, 8, 16, 32, 64, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

        for i in range(100):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

            for k in range(xt_list[0].shape[0]):
                lst = []

                for j in range(len(xt_list)):
                    x_t = xt_list[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(xt_list)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(xt_list)-j}_{cnt}.png')
                    if j in to_show:
                        lst.append(x_t)

                x_0 = x0_list[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                lst.append(x_0)
                im_h = cv2.hconcat(lst)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)
                cnt += 1

    def test_from_data_save_results(self):
        batch_size = 100
        dl = data.DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                             drop_last=True)

        all_samples = None

        for i, img in enumerate(dl, 0):
            print(i)
            print(img.shape)
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

            # break

        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1

        orig_folder = f'{self.results_folder}_orig/'
        create_folder(orig_folder)

        blur_folder = f'{self.results_folder}_blur/'
        create_folder(blur_folder)

        d_deblur_folder = f'{self.results_folder}_d_deblur/'
        create_folder(d_deblur_folder)

        deblur_folder = f'{self.results_folder}_deblur/'
        create_folder(deblur_folder)

        cnt = 0
        while cnt < all_samples.shape[0]:
            print(cnt)
            og_x = all_samples[cnt: cnt + 32]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=og_img.shape[0], faded_recon_sample=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = xt_list[0].to('cpu')
            deblurry_imgs = x0_list[-1].to('cpu')
            direct_deblurry_imgs = x0_list[0].to('cpu')

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(1, 3 // direct_deblurry_imgs.shape[1], 1, 1)

            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)

            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)

            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)

            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat((direct_deblurred_samples, direct_deblurry_imgs), dim=0)

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        for i in range(blurred_samples.shape[0]):
            utils.save_image(original_sample[i], f'{orig_folder}{i}.png', nrow=1)
            utils.save_image(blurred_samples[i], f'{blur_folder}{i}.png', nrow=1)
            utils.save_image(deblurred_samples[i], f'{deblur_folder}{i}.png', nrow=1)
            utils.save_image(direct_deblurred_samples[i], f'{d_deblur_folder}{i}.png', nrow=1)