import torchgeometry as tgm
import torch


# ---------------------------
# Fade kernels
# ---------------------------

def get_fade_kernel(dims, std):
    fade_kernel = tgm.image.get_gaussian_kernel2d(dims, std)
    fade_kernel = fade_kernel / torch.max(fade_kernel)
    fade_kernel = torch.ones_like(fade_kernel) - fade_kernel
    fade_kernel = fade_kernel[1:, 1:]
    return fade_kernel


def get_reverse_kernels_with_schedule(timesteps, size, kernel_std, initial_mask):
    faded_kernels = []
    kers = torch.ones((1, size, size))

    for i in range(timesteps):
        faded_kernels.append(kers)
        kernel = get_fade_kernel((size + 1, size + 1), (kernel_std * (i + initial_mask), kernel_std * (i + initial_mask)))
        kers = kers * kernel

    faded_kernels.reverse()
    return torch.stack(faded_kernels)



# ---------------------------
# Kspace kernels
# ---------------------------



