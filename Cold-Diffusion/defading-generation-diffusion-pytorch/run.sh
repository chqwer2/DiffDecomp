

mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-generation-diffusion-pytorch
# cd STEP1.AutoencoderModel2D
git stash
git pull



datapath=/home/hao/data/medical/Brain/
dataset=Brain
domain=BraTS-GLI-T1C 
aux_modality=T2F         # T1C, T1N, T2W, T2F


# data_modality="t1c"     # t2w, t1c, t1n, t2f
# aux_modality="t2w"      # flair


python  train.py --time_steps 50 --train_steps 700000 \
            --save_folder ./results_cifar10 \
            --data_path $datapath --dataset $dataset \
            --domain $domain --aux_modality $aux_modality \
            --train_routine Final --sampling_routine default \
            --remove_time_embed --residual --loss_type l1 \
            --initial_mask 11 --kernel_std 0.15 --reverse



