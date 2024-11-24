from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Model
from Fid import calculate_fid_given_samples
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


create = 0

if create:
    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True)
    root = './root_cifar10_test/'
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
parser.add_argument('--sample_steps', default=None, type=int)
parser.add_argument('--kernel_std', default=0.1, type=float)
parser.add_argument('--save_folder', default='progression_cifar', type=str)
parser.add_argument('--load_path', default='/cmlscratch/eborgnia/cold_diffusion/paper_defading_random_1/model.pt', type=str)
parser.add_argument('--data_path', default='./root_cifar10_test/', type=str)
parser.add_argument('--test_type', default='test_paper_showing_diffusion_images_diff', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--residual', action="store_true")

args = parser.parse_args()
print(args)

img_path=None
if 'train' in args.test_type:
    img_path = args.data_path
elif 'test' in args.test_type:
    img_path = args.data_path

print("Img Path is ", img_path)



image_channels = 1

if model_name == "unet":
    model = Model(resolution=args.image_size,
                  in_channels=1,
                  out_ch=1,
                  ch=128,
                  ch_mult=(1, 2, 2, 2),
                  num_res_blocks=2,
                  attn_resolutions=(16,),
                  dropout=0.1).cuda()

elif model_name == "twounet":

    model = TwoBranchNewModel(resolution=args.image_size,
                  in_channels=1,
                  out_ch=1,
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
    loss_type=args.loss_type,  #$'l1',
    kernel_std=args.kernel_std,
    fade_routine=args.fade_routine,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()


trainer = Trainer(
    diffusion,
    img_path,
    image_size = 32,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path
)




if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=args.sample_steps)

elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=args.sample_steps)

elif args.test_type == 'mixup_train_data':
    trainer.test_with_mixup('train')

elif args.test_type == 'mixup_test_data':
    trainer.test_with_mixup('test')

elif args.test_type == 'test_random':
    trainer.test_from_random('random')

elif args.test_type == 'test_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'test_paper_invert_section_images':
    trainer.paper_invert_section_images()

elif args.test_type == 'test_paper_showing_diffusion_images_diff':
    trainer.paper_showing_diffusion_images()
