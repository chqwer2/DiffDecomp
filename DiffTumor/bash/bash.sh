
# Step 01 Pretrain feature auto-enocder
### VQ-GAN-3D

cd STEP1.AutoencoderModel/
datapath=/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData        # (e.g., /data/bdomenAtlasMini1.0/)
dataset_list="brats"
modality=t1c
gpu_num=1
cache_rate=0.05
batch_size=4
HYDRA_FULL_ERROR=1 python train.py dataset.data_root_path=$datapath dataset.dataset_list=$dataset_list \
       dataset.cache_rate=$cache_rate dataset.batch_size=$batch_size \
       dataset.data_modality=$modality  model.gpus=$gpu_num



# Step 02 Train the DMs
cd STEP2.DiffusionModel/
vqgan_ckpt=<pretrained-AutoencoderModel> # (e.g., /pretrained_models/AutoencoderModel.ckpt)
fold=0
datapath=<your-datapath> # (e.g., /data/10_Decathlon/Task03_Liver/)
tumorlabel=<your-labelpath> # (e.g., /data/preprocessed_labels/)
python train.py dataset.name=liver_tumor_train dataset.fold=$fold \
       dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel \
       dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False \
       model.results_folder_postfix="liver_early_tumor_fold'$fold'"  model.vqgan_ckpt=$vqgan_ckpt


# Step 03 Train the Tumor Segmentation Model
cd STEP3.SegmentationModel\

healthy_datapath=<your-datapath> (e.g., /data/HealthyCT/)
datapath=<your-datapath> (e.g., /data/10_Decathlon/Task03_Liver/)
cache_rate=1.0
batch_size=12
val_every=50
workers=12
organ=liver
fold=0

# U-Net
backbone=unet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

# nnU-Net
backbone=nnunet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

# Swin-UNETR
backbone=swinunetr
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir

