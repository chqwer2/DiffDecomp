

mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch
# cd STEP1.AutoencoderModel2D
git stash
git pull


mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch


datapath=/home/hao/data/medical/Brain/
# /gamedrive/Datasets/medical/Brain/

dataset=Brain
domain=BraTS-GLI-T1C     # T1C
aux_modality=T2F         # T1C, T1N, T2W, T2F
num_channels=1
train_bs=24   # 4 | 32

diffusion_type=twobranch_fade    # unet | twobranch
diffusion_type=twobranch_kspace
diffusion_type=unet_kspace


diffusion_type=twounet_kspace

# diffusion_type=unet_fade    # unet | twobranch
save_folder=./results/$diffusion_type


time_step=50
image_size=64
sampling_routine=x0_step_down   # x0_step_down  | x0_step_down_fre
loss_type=l2    # l2 | l1 | l2_l1
tag=fre_loss_l2    # x0_step_down | x0_step_down_fre
deviceid=6


python  train.py --time_steps $time_step --train_steps 700000 \
            --save_folder $save_folder  --tag $tag \
            --data_path $datapath --dataset $dataset \
            --domain $domain --aux_modality $aux_modality \
            --sampling_routine $sampling_routine \
            --remove_time_embed --residual --image_size $image_size \
            --diffusion_type $diffusion_type  --train_bs $train_bs \
            --num_channels $num_channels --deviceid $deviceid \
            --kernel_std 0.15  --discrete   --loss_type $loss_type #   --debug



# --fade_routine Random_Incremental
