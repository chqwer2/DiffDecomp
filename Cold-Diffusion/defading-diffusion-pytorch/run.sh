

mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch
# cd STEP1.AutoencoderModel2D
git stash
git pull


mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-diffusion-pytorch

deviceid=5
datapath=/home/hao/data/medical/Brain/
# /gamedrive/Datasets/medical/Brain/
dataset=Brain
domain=BraTS-GLI-T1C 
aux_modality=T2F         # T1C, T1N, T2W, T2F
num_channels=1
train_bs=24   # 4 | 32
diffusion_type=twobranch_fade    # unet | twobranch
diffusion_type=twobranch_kspace
# diffusion_type=unet_fade    # unet | twobranch
save_folder=./results/$diffusion_type



python  train.py --time_steps 50 --train_steps 700000 \
            --save_folder $save_folder \
            --data_path $datapath --dataset $dataset \
            --domain $domain --aux_modality $aux_modality \
            --sampling_routine default \
            --remove_time_embed --residual \
            --diffusion_type $diffusion_type  --train_bs $train_bs \
            --num_channels $num_channels --deviceid $deviceid \
            --kernel_std 0.15  --debug