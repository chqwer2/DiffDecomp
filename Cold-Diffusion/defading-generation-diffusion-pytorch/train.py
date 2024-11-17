#from comet_ml import Experiment
from defading_diffusion_pytorch import GaussianDiffusion, Trainer
from defading_diffusion_pytorch import Unet, TwoBranchModel
import torchvision
import os
import errno
import shutil
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--data_path', default='../deblurring-diffusion-pytorch/root_celebA_128_train_new/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='default', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--loss_type', default='l1', type=str)

# Defade specific arguments
parser.add_argument('--initial_mask', default=11, type=int)
parser.add_argument('--kernel_std', default=0.15, type=float)
parser.add_argument('--reverse', action="store_true")

parser.add_argument('--dataset', default='brain', type=str)
parser.add_argument('--domain', default=None, type=str)
parser.add_argument('--aux_modality', default=None, type=str)
parser.add_argument('--deviceid', default=0, type=int)
parser.add_argument('--num_channels', default=1, type=int)
parser.add_argument('--train_bs', default=24, type=int)
parser.add_argument('--diffusion_type', default='twobranch_fade', type=str)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--image_size', default=128, type=int)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.deviceid)

image_channels = 1

diffusion_type = args.diffusion_type
# diffusion_type = "twobranch_fade"           # model_degradation      # fade | kspace
model_name = diffusion_type.split("_")[0]   # unet | twobranch

if args.debug:
    args.train_steps = 100
    args.time_steps  = 5



if model_name == "unet":
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=3,
        with_time_emb=not(args.remove_time_embed),
        residual=args.residual
    ).cuda()

elif model_name == "twobranch":
    downsample = [4, 4, 4]
    disc_channels = 64
    disc_layers = 3
    discriminator_iter_start = 10000
    disc_loss_type = "hinge"
    image_gan_weight = 1.0
    video_gan_weight = 1.0
    l1_weight = 4.0
    gan_feat_weight = 4.0
    perceptual_weight = 4.0 
    i3d_feat = False
    restart_thres = 1.0
    no_random_restart = False
    norm_type = "group"
    padding_type = "replicate"
    num_groups = 32
    
    base_num_every_group = 2
    num_features = 64
    act = "PReLU"
    num_channels = 1


    model = TwoBranchModel(
        image_channels, 
        disc_channels, disc_layers, disc_loss_type, 
        gan_feat_weight, image_gan_weight,
        discriminator_iter_start,
        perceptual_weight, l1_weight, 
        num_features, act, base_num_every_group, num_channels
    ).cuda()



diffusion = GaussianDiffusion(
    diffusion_type,
    model,
    image_size = args.image_size,
    channels = image_channels,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    reverse = args.reverse,
    
    kernel_std = args.kernel_std,
    initial_mask=args.initial_mask,
    
    num_channels = args.num_channels
).cuda()



import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

print("=== train_steps:", args.train_steps)


trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = 128,
    train_batch_size = args.train_bs,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = args.dataset,
    domain = args.domain,
    aux_modality = args.aux_modality,
    debug = args.debug,
    num_channels = args.num_channels
)

trainer.train()

