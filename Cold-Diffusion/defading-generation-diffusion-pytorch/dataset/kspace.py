import torch
from torch import nn
import os
import cv2
import gc
import numpy as np
from scipy.io import *
from scipy.fftpack import *



# Fourier Transform
def fft_map(x):
    fft_x = torch.fft.fftn(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag

    return fft_x_real, fft_x_imag


def undersample_kspace(x, mask, is_noise, noise_level, noise_var):
    
    # d.1.0.complex --> d.1.1.complex
    # WARNING: This function only take x (H, W), not x (H, W, 1)
    # x (H, W) & x (H, W, 1) return different results
    # x (H, W): after fftshift, the low frequency is at the center.
    # x (H, W, 1): after fftshift, the low frequency is NOT at the center.
    # use abd(fft) to visualise the difference

    fft = fft2(x)
    fft = fftshift(fft)
    fft = fft * mask

    if is_noise:
        raise NotImplementedError
        fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)

    fft = ifftshift(fft)
    x = ifft2(fft)

    return x