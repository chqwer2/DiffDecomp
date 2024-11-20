from defading_diffusion_pytorch import GaussianDiffusion, Trainer, Model
import torchvision
import os
import errno
import shutil
import argparse
from defading_diffusion_pytorch import TwoBranchModel
import torch

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


create = 0

if create:
    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)
    root = './root_cifar10/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default=None, type=str)

parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_cifar10/', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")

# Defade specific arguments
# parser.add_argument('--initial_mask', default=11, type=int)
parser.add_argument('--kernel_std', default=0.1, type=float)

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
model_name = diffusion_type.split("_")[0]  # unet | twobranch

if args.debug:
    args.train_steps = 100
    args.time_steps = 5

if model_name == "unet":
    # model = Unet(            # Used to be Model
    #     dim=64,
    #     dim_mults=(1, 2, 4, 8),
    #     channels=3,
    #     with_time_emb=not (args.remove_time_embed),
    #     residual=args.residual
    # ).cuda()
    model = Model(resolution=32,
                  in_channels=3,
                  out_ch=3,
                  ch=128,
                  ch_mult=(1, 2, 2, 2),
                  num_res_blocks=2,
                  attn_resolutions=(16,),
                  dropout=0.1).cuda()


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
    image_size=args.image_size,    # Used to be 32
    channels=image_channels,
    device_of_kernel='cuda',
    timesteps=args.time_steps,
    loss_type='l1',
    kernel_std=args.kernel_std,
    fade_routine=args.fade_routine,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
    # num_channels=args.num_channels     # ?
).cuda()


diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

print("=== train_steps:", args.train_steps)
if args.debug:
    args.save_folder = args.save_folder + "_debug"
else:
    if os.path.exists(args.save_folder):
        number = os.listdir(args.save_folder).count
        args.save_folder = args.save_folder + f"_{number}"


trainer = Trainer(
    diffusion,
    args.data_path,
    image_size=args.image_size,   # Used to be 32
    train_batch_size=args.train_bs,
    train_lr=2e-5,
    train_num_steps=args.train_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    save_and_sample_every=1000,
    fp16=False,
    results_folder=args.save_folder,
    load_path=args.load_path,
    dataset=args.dataset,
    domain=args.domain,
    aux_modality=args.aux_modality,
    debug=args.debug,
    num_channels=args.num_channels
)

trainer.train()
