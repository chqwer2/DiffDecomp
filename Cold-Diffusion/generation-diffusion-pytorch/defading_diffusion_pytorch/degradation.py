import torchgeometry as tgm
import torch
import numpy as np


# ---------------------------
# Fade kernels
# ---------------------------

def get_fade_kernel(dims, std):
    fade_kernel = tgm.image.get_gaussian_kernel2d(dims, std)
    fade_kernel = fade_kernel / torch.max(fade_kernel)
    fade_kernel = torch.ones_like(fade_kernel) - fade_kernel
    fade_kernel = fade_kernel[1:, 1:]
    return fade_kernel


def get_kernels_with_schedule(timesteps, size, kernel_std, initial_mask, reverse):
    faded_kernels = []
    reverse_kernels = []
    kers = torch.ones((1, size, size))

    for i in range(timesteps):
        reverse_kernels.append(kers)
        kernel = get_fade_kernel((size + 1, size + 1), (kernel_std * (i + initial_mask), kernel_std * (i + initial_mask)))
        kers = kers * kernel
        faded_kernels.append(kers)
        
    if reverse:
        reverse_kernels.reverse()
        return torch.stack(reverse_kernels)
    else:
        return torch.stack(faded_kernels)


# ---------------------------
# Kspace kernels
# ---------------------------
from torch.fft import *
# Cartesian Mask Support
from .ksutils import RandomMaskFunc, EquispacedMaskFractionFunc
# Gaussain 2D Mask Support

# from utils.utils_kspace_undersampling import utils_undersampling_pattern, utils_radial_spiral_undersampling

def get_mask_func(ksu_mask_type, af, cf):
    if ksu_mask_type == 'cartesian_regular':
        return EquispacedMaskFractionFunc(center_fractions=[cf], accelerations=[af])
    elif ksu_mask_type == 'cartesian_random':
        return RandomMaskFunc(center_fractions=[cf], accelerations=[af])
    
    # elif ksu_mask_type == 'gaussian_2d':
    #     raise NotImplementedError
    #     return utils_undersampling_pattern.cs_generate_pattern_2d
    # elif ksu_mask_type == 'radial_add':
    #     return utils_radial_spiral_undersampling.generate_mask_add
    # elif ksu_mask_type == 'radial_sub':
    #     return utils_radial_spiral_undersampling.generate_mask_sub
    # elif ksu_mask_type == 'spiral_add':
    #     return utils_radial_spiral_undersampling.generate_mask_add
    # elif ksu_mask_type == 'spiral_sub':
    #     return utils_radial_spiral_undersampling.generate_mask_sub
    else:
        raise NotImplementedError


def get_ksu_mask( ksu_mask_type, af, cf, pe, fe, seed=0):
    
        mask_func = get_mask_func(ksu_mask_type, af, cf)

        if ksu_mask_type in ['cartesian_regular', 'cartesian_random']:

            mask, num_low_freq = mask_func((1, pe, 1), seed=seed)  # mask (torch): (1, pe, 1)
            mask = mask.permute(0, 2, 1).repeat(1, fe, 1)  # mask (torch): (1, pe, 1) --> (1, 1, pe) --> (1, fe, pe)

        elif ksu_mask_type == 'gaussian_2d':
            mask, _ = mask_func(resolution=(fe, pe), accel=af, sigma=100, seed=seed)  # mask (numpy): (fe, pe)
            mask = torch.from_numpy(mask[np.newaxis, :, :])  # mask (torch): (fe, pe) --> (1, fe, pe)

        elif ksu_mask_type in ['radial_add', 'radial_sub', 'spiral_add', 'spiral_sub']:
            sr = 1 / af
            mask = mask_func(mask_type=ksu_mask_type, mask_sr=sr, res=pe, seed=seed)  # mask (numpy): (pe, pe)
            mask = torch.from_numpy(mask[np.newaxis, :, :])  # mask (torch): (pe, pe) --> (1, pe, pe)

        else:
            raise NotImplementedError

        return mask
    
    
# ksu_masks = get_ksu_kernels()
# (C, H, W) --> (B, C, H, W)
# ksu_mask = ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
# img = ksu(x_start=x_start, mask=ksu_mask)

def get_ksu_kernel(timesteps, image_size, ksu_routine="LogSamplingRate", 
                   ksu_mask_type="cartesian_random"):
    masks = []
    ksu_mask_pe = ksu_mask_fe  = image_size   # , ksu_mask_pe=320, ksu_mask_fe=320
    ksu_mask_fe
    if ksu_routine == 'LinearSamplingRate':
        # Generate the sampling rate list with torch.linspace, reversed, and skip the first element
        sr_list = torch.linspace(start=0.01, end=1, steps=timesteps + 1).flip(0)

        for sr in sr_list:
            af = 1 / sr
            cf = sr * 0.32
            masks.append(get_ksu_mask(ksu_mask_type, af, cf, pe=ksu_mask_pe, fe=ksu_mask_fe))

    elif ksu_routine == 'LogSamplingRate':
        # Generate the sampling rate list with torch.logspace, reversed, and skip the first element
        sr_list = torch.logspace(start=-2, end=0, steps=timesteps + 1).flip(0)

        for sr in sr_list:
            af = 1 / sr
            cf = sr * 0.32
            masks.append(get_ksu_mask(ksu_mask_type, af, cf, pe=ksu_mask_pe, fe=ksu_mask_fe))

    elif ksu_mask_type == 'gaussian_2d':
        raise NotImplementedError("Gaussian 2D mask type is not implemented.")

    else:
        raise NotImplementedError(f'Unknown k-space undersampling routine {ksu_routine}')

    # Return masks, excluding the first one
    return masks[1:]


def apply_ksu_kernel(x_start, mask, pixel_range='-1_1'):
        if pixel_range == '0_1':
            pass
        
        elif pixel_range == '-1_1':
            # x_start (-1, 1) --> (0, 1)
            x_start = (x_start + 1) / 2
            
        elif pixel_range == 'complex':
            x_start = torch.complex(x_start[:, :1, ...], x_start[:, 1:, ...])
            
        else:
            raise ValueError(f"Unknown pixel range {pixel_range}.")

        fft = fftshift(fft2(x_start))
        fft = fft * mask
        x_ksu = ifft2(ifftshift(fft))

        if pixel_range == '0_1':
            x_ksu = torch.abs(x_ksu)
            
        elif pixel_range == '-1_1':
            x_ksu = torch.abs(x_ksu)
            # x_ksu (0, 1) --> (-1, 1)
            x_ksu = x_ksu * 2 - 1
            
        elif pixel_range == 'complex':
            x_ksu = torch.concat((x_ksu.real, x_ksu.imag), dim=1)
        else:
            raise ValueError(f"Unknown pixel range {pixel_range}.")

        return x_ksu
