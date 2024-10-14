#!/bin/bash
#SBATCH --job-name=autoencoder_model

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -p general
#SBATCH -t 7-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu


mamba activate diffmri
cd /home/hao/repo/DiffDecomp/DiffTumor
git stash
git pull


cd STEP1.AutoencoderModel3D

datapath=/home/hao/data/medical/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
data_modality="t1c"     # t2w, t1c, t1n, t2f
aux_modality="t2w"  # flair


cache_rate=0.001
batch_size=8
dataset_list="AbdomenAtlas1.0Mini"
deviceid=1


# single GPU
gpu_num=1
python train.py dataset.data_root_path=$datapath  \
       model.deviceid=$deviceid \
       dataset.data_modality=$data_modality dataset.aux_modality=$aux_modality \
       dataset.dataset_list=$dataset_list \
       dataset.cache_rate=$cache_rate \
       dataset.batch_size=$batch_size \
       model.gpus=$gpu_num

# sbatch --error=logs/autoencoder_model.out --output=logs/autoencoder_model.out hg.sh
# 