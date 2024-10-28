
# module load mamba/latest # only for Sol

# mamba create -n difftumor python=3.9
# source activate difftumor
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install -r ../requirements.txt




mamba activate diffmri
cd /home/hao/repo/DiffDecomp/DiffTumor
cd STEP1.AutoencoderModel2D
git stash
git pull

datapath=/home/hao/data/medical/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/

data_modality="t1c"     # t2w, t1c, t1n, t2f
aux_modality="t2w"      # flair



backbone=unet   # unet | two-branch
vqgan_ckpt=pretrained_models/AutoencoderModel.ckpt
fold=0


# datapath=/scratch/zzhou82/data/Task03_Liver/
# tumorlabel=/scratch/zzhou82/data/preprocessed_labels/



python train.py dataset.name=liver_tumor_train dataset.fold=$fold dataset.data_root_path=$datapath \
                dataset.label_root_path=$tumorlabel \
                dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="liver_early_tumor_fold'$fold'"  \
                model.vqgan_ckpt=$vqgan_ckpt

# sbatch --error=logs/diffusion_model.out --output=logs/diffusion_model.out hg.sh
