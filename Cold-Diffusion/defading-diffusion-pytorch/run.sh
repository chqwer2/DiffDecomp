

mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch
cd /home/cbtil3/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch
# cd STEP1.AutoencoderModel2D
#git stash
git pull


mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch
cd /home/cbtil3/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch

datapath=/home/hao/data/medical/Brain/
# /gamedrive/Datasets/medical/Brain/

dataset=Brain
domain=BraTS-GLI-T1C     # T1C
aux_modality=T2F         # T1C, T1N, T2W, T2F
num_channels=1


diffusion_type=twobranch_fade    # unet | twobranch
diffusion_type=twobranch_kspace
diffusion_type=unet_kspace


diffusion_type=twounet_kspace

# diffusion_type=unet_fade    # unet | twobranch


time_step=50
image_size=64
sampling_routine=x0_step_down # x0_step_down_fre  # default | x0_step_down  | x0_step_down_fre
loss_type=l1   #  l2 1     # l2 | l1 | l2_l1


tag=l1     # x0_step_down | x0_step_down_fre

deviceid=0
# fre_before_attn + l1
train_bs=12   # 4 | 32 | 24


save_folder=./results/$diffusion_type_$sampling_routine


datapath=/home/hao/data/medical/Brain/
datapath=/gamedrive/Datasets/medical/Brain/brats/Processed/



python  train.py --time_steps $time_step --train_steps 700000 \
            --save_folder $save_folder  --tag $tag \
            --data_path $datapath --dataset $dataset \
            --domain $domain --aux_modality $aux_modality \
            --sampling_routine $sampling_routine \
            --remove_time_embed --residual --image_size $image_size \
            --diffusion_type $diffusion_type  --train_bs $train_bs \
            --num_channels $num_channels --deviceid $deviceid \
            --kernel_std 0.15  --discrete   --loss_type $loss_type  #--debug






