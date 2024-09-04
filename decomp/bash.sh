hao; mamba activate MRI
cd medical/MRI-Recon/decomp; git pull

# Install
pip install -e .

# "mask"
# Train
DESC="--comment slice_single_mi_newattention"
MODEL_FLAGS="--emb_dim 64 --enc_channels 128 --loss_mode mse --components_fusion mask "   # "
TRAIN_FLAGS="--batch_size 16 --image_size 64 --dataset brats --data_dir ../"

# mse_mi

### Single train
python scripts/image_train.py $DESC  $MODEL_FLAGS $TRAIN_FLAGS --use_dist False


### Distributed train
# DEVICE=$CUDA_VISIBLE_DEVICES
# NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# python -m torch.distributed.run --nproc_per_node=$NUM_DEVICES scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS


# mse -> ☑️  blur





