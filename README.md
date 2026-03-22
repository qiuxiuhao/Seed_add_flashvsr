# Deployment Guide for Seed_add_flashvsr

**Note!!!** The current model requires at least 4 A800 80GB GPUs to run. The following steps are based on H-series or A-series GPUs and require internet access.


## 0. Clone the Project
```bash
git clone https://github.com/qiuxiuhao/Seed_add_flashvsr.git
cd Seed_add_flashvsr
```

## 1. Deploy SeedVR Model

### 1.1 Create Environment

#### 1.1.1 Create Basic `seedvr` Environment

```Bash

cd SeedVR/SeedVR
conda create -n seedvr python=3.10 -y
conda activate seedvr
pip install -r requirements.txt
```
#### If you encounter issues creating a Conda virtual environment, you can find answers at Seed_add_flashvsr/SeedVR/SeedVR/README.md
#### 1.1.2 Install `flash_attn`

```Bash

pip install flash_attn==2.5.9.post1 --no-build-isolation
# If an error occurs, download the corresponding version of flash_attn first, then install the local package
```

#### 1.1.3 Install `apex`

```Bash

# (1) First return to the parent directory of SeedVR
cd ..

# (2) Clone NVIDIA's official apex repository (already included in the compressed package)
git clone https://github.com/NVIDIA/apex

# (3) Enter the apex directory
cd apex

# (4) Force specify the compilation target architecture
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"

# (5) Start compiling and installing (Note: This step usually takes 5-10 minutes, please wait patiently)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

#### 1.1.4 Supplementary Environments

```Bash

cd ../SeedVR
pip install av
pip install -r lora_requirements.txt
pip install -r lora_finetune_requirements.txt
```

### 1.2 Download Basic Weights: SeedVR2-7B

```Bash

# Currently in the SeedVR/SeedVR directory
huggingface-cli download ByteDance-Seed/SeedVR2-7B \
  --local-dir ckpts/ \
  --cache-dir ckpts/cache \
  --local-dir-use-symlinks False \
  --include "*.json" "*.safetensors" "*.pth" "*.bin" "*.py" "*.md" "*.txt" \
  --exclude "seedvr2_ema_7b_sharp.pth"
```

### 1.3 Download LoRA Fine-tuning Weights

Our weight URL is:  https://drive.google.com/drive/folders/1Vsie9V3f08r0z6Pp3mreVr2EFsJ5yM8z?usp=sharing

After downloading, please place it in `SeedVR/SeedVR/lora_weights/lora.safetensors`.

## 2. Deploy FlashVSR Model

### 2.0 Return to the `Seed_add_flashvsr` Directory First

### 2.1 Create Basic `flashvsr` Environment
### If you encounter issues creating a Conda virtual environment, you can find answers at Seed_add_flashvsr/FlashVSR/FLASHVSR/README.md
```Bash

cd FlashVSR/FlashVSR
conda create -n flashvsr python=3.11.13 -y
conda activate flashvsr
pip install setuptools==69.5.1 wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-build-isolation
pip install -r requirements.txt
pip install modelscope
cd ..
```

### 2.2 Install Block-Sparse-Attention

```Bash

git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install # Ensure dependencies from GitHub can be cloned normally during installation
cd ..
```

### 2.3 Download Weights

```Bash

cd FlashVSR/examples/WanVSR

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='JunhaoZhuang/FlashVSR-v1.1',
    local_dir='./FlashVSR-v1.1',
    max_workers=8,
    endpoint='https://hf-mirror.com',
    ignore_patterns=['*.git*', 'README.md']
)"
```

## 3. Inference (Execute in the `Seed_add_flashvsr` Directory)

### 3.0 Prepare Test Data

Place test data in `SeedVR_add_flashvsr/test-input-final/test_data`.

### 3.1 Install Tools

```Bash

apt update
apt install -y ffmpeg
```

### 3.2 Run Inference

```Bash

LOCAL=~/SeedVR_add_flashvsr
REMOTE=root@<SERVER_HOST>:~/autodl-tmp/SeedVR_add_flashvsr
python run_dual_vsr.py --mode blend --seed-profile lora --flash-profile full_test --alpha 0.7 --input-root test-input-final/test_data --result-root result
```
