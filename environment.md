## Dependency
The code is tested on `python 3.8, Pytorch 1.13`.

##### Setup environment

```bash
mamba create -n MRI python=3.8
mamba activate MRI 
pip install torch==2.4+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 
pip install einops h5py matplotlib scikit_image tensorboardX yacs pandas opencv-python timm ml_collections  SimpleITK   pyyaml  # omegaconf
pip install omegaconf
pip install albumentations==1 threadpoolctl  scikit-video 
pip install pytorch_lightning==1.8.4 lightning_utilities  
pip install hydra-core --upgrade

# Monai
mamba install -c conda-forge monai[nibabel]
mamba install -c conda-forge monai==0.9.0

#   --no-dependencies 
mamba install -c conda-forge xorg-libx11 -y
mamba install -c conda-forge xorg-libxext -y
mamba install -c conda-forge libgomp -y
mamba install -c anaconda libxcb -y
```
