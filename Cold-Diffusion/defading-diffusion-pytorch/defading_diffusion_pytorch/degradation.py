# ---------------------------
# Fade kernels
# ---------------------------
import cv2


def get_fade_kernel(dims, std):
    fade_kernel = tgm.image.get_gaussian_kernel2d(dims, std)
    fade_kernel = fade_kernel / torch.max(fade_kernel)
    fade_kernel = torch.ones_like(fade_kernel) - fade_kernel
    # if device_of_kernel == 'cuda':
    #     fade_kernel = fade_kernel.cuda()
    fade_kernel = fade_kernel[1:, 1:]
    return fade_kernel

def get_fade_kernels(fade_routine, num_timesteps, image_size, kernel_std,initial_mask):
    kernels = []
    for i in range(num_timesteps):
        if fade_routine == 'Incremental':
            kernels.append(get_fade_kernel((image_size + 1, image_size + 1),
                                                (kernel_std * (i + initial_mask),
                                                 kernel_std * (i + initial_mask))))
        elif fade_routine == 'Constant':
            kernels.append(get_fade_kernel(
                (image_size + 1, image_size + 1),
                (kernel_std, kernel_std)))
        elif fade_routine == 'Random_Incremental':
            kernels.append(get_fade_kernel((2 * image_size + 1, 2 * image_size + 1),
                                                (kernel_std * (i + initial_mask),
                                                 kernel_std * (i + initial_mask))))
    return torch.stack(kernels)


import torchgeometry as tgm
import torch
import numpy as np

# ---------------------------
# Kspace kernels
# ---------------------------
from torch.fft import *
# Cartesian Mask Support
try:
    from .ksutils import RandomMaskFunc, EquispacedMaskFractionFunc
except:
    from ksutils import RandomMaskFunc, EquispacedMaskFractionFunc

# Gaussain 2D Mask Support
# from utils.utils_kspace_undersampling import utils_undersampling_pattern, utils_radial_spiral_undersampling


# cartesian_regular
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






# af (Acceleration Factor), cf (Center Fraction)
# pe (Phase Encoding), fe (Frequency Encoding)
def get_ksu_mask(ksu_mask_type, af, cf, pe, fe, seed=0):
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

    # print("return mask = ", mask.shape)
    return mask


# ksu_masks = get_ksu_kernels()
# (C, H, W) --> (B, C, H, W)
# ksu_mask = ksu_masks[t].repeat(batch_size, 1, 1, 1).to(device)
# img = ksu(x_start=x_start, mask=ksu_mask)

def get_ksu_kernel(timesteps, image_size,
                   ksu_routine="LogSamplingRate",
                   ksu_mask_type="cartesian_random",
                   accelerated_factor=4):
    masks = []
    ksu_mask_pe = ksu_mask_fe = image_size  # , ksu_mask_pe=320, ksu_mask_fe=320
    # ksu_mask_fe
    if ksu_routine == 'LinearSamplingRate':
        # Generate the sampling rate list with torch.linspace, reversed, and skip the first element
        sr_list = torch.linspace(start=1/accelerated_factor, end=1, steps=timesteps + 1).flip(0)
        # Start from 0.01

        for sr in sr_list:
            af = 1 / sr  # * accelerated_factor           # acceleration factor
            cf = sr * 0.32
            masks.append(get_ksu_mask(ksu_mask_type, af, cf, pe=ksu_mask_pe, fe=ksu_mask_fe))

    elif ksu_routine == 'LogSamplingRate':
        # Generate the sampling rate list with torch.logspace, reversed, and skip the first element
        sr_list = torch.logspace(start=-torch.log10(torch.tensor(accelerated_factor)),
                                 end=0, steps=timesteps + 1).flip(0)
        # start=-2

        # print("sr_list length: ", sr_list.shape, sr_list)
        sr_list = sr_list #  accelerated_factor

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

class high_fre_mask:
    def __init__(self):
        self.mask_cache = {}

    def __call__(self, H, W):
        if (H, W) in self.mask_cache:
            return self.mask_cache[(H, W)]
        center_x, center_y = H // 2, W // 2
        radius = 30  # 影响的频率范围半径

        high_freq_mask = torch.ones(H, W)
        for i in range(H):
            for j in range(W):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                    high_freq_mask[i, j] = 0.0
        self.mask_cache[(H, W)] = high_freq_mask
        return high_freq_mask

high_fre_mask_cls = high_fre_mask()

def apply_ksu_kernel(x_start, mask, use_fre_noise=False, pixel_range='-1_1'):
    fft, mask = apply_tofre(x_start, mask, pixel_range)


    # try:

    if use_fre_noise:
        fft = torch.fft.fftshift(fft)  # 将低频分量移到中心

        fft_magnitude = torch.abs(fft)  # 幅度
        fft_phase = torch.angle(fft)  # 相位
        # print("fft shape: ", fft.shape, fft_magnitude.shape, fft_phase.shape)
        _, _, H, W = fft.shape

        high_freq_mask = high_fre_mask_cls(H, W).to(fft.device)
        high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0).repeat(fft.shape[0], 1, 1, 1)

        noise_magnitude = torch.randn_like(fft_magnitude) * 0.1 * fft_magnitude.mean()

        # fft = fft * mask
        fft_noisy_magnitude = fft_magnitude * mask + noise_magnitude * high_freq_mask * (1 - mask)
        fft_noisy_magnitude = torch.clamp(fft_noisy_magnitude, min=0.0)

        fft = fft_noisy_magnitude * torch.exp(1j * fft_phase)
    else:
        fft = fft * mask

    # except:
    #     print("Error in transforming fft:", fft.shape, mask.shape)


    x_ksu = apply_to_spatial(fft, pixel_range)

    return x_ksu


def apply_tofre(x_start, mask, pixel_range='-1_1'):
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
    mask = mask.to(fft.device)
    return fft, mask



def apply_to_spatial(fft, pixel_range='-1_1'):

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







if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    masks = get_ksu_kernel(10, 128)

    image_size = 128
    batch_size = 1

    img = plt.imread("/Users/haochen/Documents/GitHub/DiffDecomp/Cold-Diffusion/generation-diffusion-pytorch/defading_diffusion_pytorch/assets/img.png")
    img = cv2.resize(img, (128, 128))

    img = np.transpose(img, (2, 0, 1))
    img = img[0]
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).unsqueeze(0).float()
    print(" img shape: ", img.shape, img.max(), img.min())

    rand_kernels = []
    rand_x = torch.randint(0, image_size + 1, (batch_size,)).long()
    print("rand_x shape:", rand_x.shape, rand_x)

    masked_img = []

    for m in masks:
        m = m.unsqueeze(0)
        img = apply_ksu_kernel(img, m, pixel_range='0_1')
        masked_img.append(img)
        print("-- k shape: ", m.shape)
        print("-- img shape: ", img.shape)

    masks = np.concatenate(masks, axis=-1)[0]
    masked_img = (torch.concat(masked_img, dim=-1).numpy() + 1) * 0.5
    masked_img = np.transpose(masked_img, (0, 2, 3, 1))[0, ..., 0]
    # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

    print(" masked_img shape: ", masked_img.shape)
    print(" mask shape: ", masks.shape)

    img = np.concatenate([masks, masked_img], axis=0)

    plt.figure(figsize=(100, 10))
    plt.imshow(img)      # (1, 128, 1280)
    plt.show()

    print("Second stage...")

    import matplotlib.pyplot as plt
    import numpy as np

    image_size = 128
    batch_size = 1
    t = 50
    kspace_kernels = get_ksu_kernel(t, image_size)   # 2 *
    kspace_kernels = torch.stack(kspace_kernels).squeeze(1)

    img = plt.imread(
        "/Users/haochen/Documents/GitHub/DiffDecomp/Cold-Diffusion/generation-diffusion-pytorch/defading_diffusion_pytorch/assets/img.png")
    img = cv2.resize(img, (128, 128))

    img = np.transpose(img, (2, 0, 1))
    img = img[0]
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).unsqueeze(0).float()
    print(" img shape: ", img.shape, img.max(), img.min())

    rand_kernels = []
    rand_x = torch.randint(0, image_size + 1, (batch_size,)).long()

    print("rand_x shape:", rand_x.shape, rand_x)

    for i in range(batch_size):
        print("kspace_kernels[j] shape = ", kspace_kernels[i].shape, rand_x[i])
        # k = kspace_kernels[j].clone()
        rand_kernels.append(torch.stack(
            [kspace_kernels[j][:  image_size,  # rand_x[i]:rand_x[i] +
             : image_size] for j in
             range(len(kspace_kernels))]))

    rand_kernels = torch.stack(rand_kernels)

    # rand_kernels shape: torch.Size([24, 5, 128, 128])
    print("=== rand_kernels: ", rand_kernels.shape, kspace_kernels[0].shape)

    masked_img = []
    masks = []
    for i in range(t):
        k = torch.stack([rand_kernels[:, i]], 1)[0]
        masks.append(k)
        print("-- k shape: ", k.shape)
        print("-- img shape: ", img.shape)

        img = apply_ksu_kernel(img, k, pixel_range='0_1')
        masked_img.append(img)

    masks = np.concatenate(masks, axis=-1)[0]
    masked_img = (torch.concat(masked_img, dim=-1).numpy() + 1) * 0.5
    masked_img = np.transpose(masked_img, (0, 2, 3, 1))[0, ..., 0]
    # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

    print(" masked_img shape: ", masked_img.shape)
    print(" mask shape: ", masks.shape)

    img = np.concatenate([masks, masked_img], axis=0)

    plt.figure(figsize=(100, 10))
    plt.imshow(img)  # (1, 128, 1280)
    plt.show()

