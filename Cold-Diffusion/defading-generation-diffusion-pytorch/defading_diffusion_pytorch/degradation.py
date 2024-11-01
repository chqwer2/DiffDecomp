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

# ksu_masks = get_ksu_kernels()
# (C, H, W) --> (B, C, H, W)
# ksu_mask = ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
# img = ksu(x_start=x_start, mask=ksu_mask)

def get_ksu_kernel(timesteps, ksu_routine="LogSamplingRate", ksu_mask_type="cartesian_random", ksu_mask_pe=320, ksu_mask_fe=320):
    masks = []

    if ksu_routine == 'LinearSamplingRate':
        # Generate the sampling rate list with torch.linspace, reversed, and skip the first element
        sr_list = torch.linspace(start=0.01, end=1, steps=timesteps + 1).flip(0)

        for sr in sr_list:
            af = 1 / sr
            cf = sr * 0.32
            masks.append(get_ksu_kernel(ksu_mask_type, af, cf, pe=ksu_mask_pe, fe=ksu_mask_fe))

    elif ksu_routine == 'LogSamplingRate':
        # Generate the sampling rate list with torch.logspace, reversed, and skip the first element
        sr_list = torch.logspace(start=-2, end=0, steps=timesteps + 1).flip(0)

        for sr in sr_list:
            af = 1 / sr
            cf = sr * 0.32
            masks.append(get_ksu_kernel(ksu_mask_type, af, cf, pe=ksu_mask_pe, fe=ksu_mask_fe))

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
