mamba create -n diffmri python=3.10

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113


pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124       \
 --extra-index-url https://download.pytorch.org/whl/cu124


pip install -r ./requirements.txt
