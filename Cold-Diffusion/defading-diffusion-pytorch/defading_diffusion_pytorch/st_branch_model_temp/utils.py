""" Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import warnings
import torch
import imageio

import math
import numpy as np
import skvideo.io

import sys
import pdb as pdb_original
import SimpleITK as sitk
import logging
from torch import nn
import torch.nn.functional as F

import imageio.core.util
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight


def save_video_grid(video, fname, nrow=None, fps=6):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    ## skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    #print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class AMPLoss(nn.Module):
    def __init__(self, epsilon=1e-8, norm='ortho'):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()
        self.epsilon = epsilon  # To prevent division by zero
        self.norm = norm  # Normalization for FFT

    def forward(self, x, y):
        # Validate inputs
        # if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        #     raise ValueError("Input contains NaN or Inf values")

        # Perform FFT and compute magnitudes
        x_fft = torch.fft.rfft2(x, norm=self.norm)
        y_fft = torch.fft.rfft2(y, norm=self.norm)

        x_mag = torch.clamp(torch.abs(x_fft), min=self.epsilon)  # Clamp to avoid zeros
        y_mag = torch.clamp(torch.abs(y_fft), min=self.epsilon)  # Clamp to avoid zeros

        x_phase = torch.angle(x_fft)
        y_phase = torch.angle(y_fft)

        # Compute L1 loss between magnitudes
        return self.cri(x_mag, y_mag) + self.cri(x_phase, y_phase)


class PhaLoss(nn.Module):
    def __init__(self, epsilon=1e-8, norm='ortho'):
        super(PhaLoss, self).__init__()
        self.cri = nn.L1Loss()
        self.epsilon = epsilon  # To prevent undefined phase for zero magnitudes
        self.norm = norm  # Normalization for FFT

    def forward(self, x, y):
        # Validate inputs
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Input contains NaN or Inf values")

        # Perform FFT
        x_fft = torch.fft.rfft2(x, norm=self.norm)
        y_fft = torch.fft.rfft2(y, norm=self.norm)

        # Compute phase
        x_phase = torch.angle(x_fft)
        y_phase = torch.angle(y_fft)

        # Compute L1 loss between phases
        return self.cri(x_phase, y_phase)
