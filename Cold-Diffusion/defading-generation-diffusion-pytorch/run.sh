

mamba activate diffmri
cd /home/hao/repo/DiffDecomp/Cold-Diffusion/defading-generation-diffusion-pytorch
# cd STEP1.AutoencoderModel2D
git stash
git pull



datapath=/home/hao/data/medical/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
# data_modality="t1c"     # t2w, t1c, t1n, t2f
# aux_modality="t2w"      # flair


python  train.py --time_steps 50 --train_steps 700000 \
            --save_folder ./results_cifar10 \
            --data_path $datapath \
            --train_routine Final --sampling_routine default \
            --remove_time_embed --residual --loss_type l1 \
            --initial_mask 11 --kernel_std 0.15 --reverse



