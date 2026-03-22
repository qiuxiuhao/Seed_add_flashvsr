# 注意！！！当前模型至少需在4卡A800 80GB上运行，以下步骤基于H系列 or A系列GPU，要求可联网
# 1.创建环境
## 1.1创建基础seedvr环境
```bash
conda create -n seedvr python=3.10 -y
conda activate seedvr
# autodl上可能会遇到需conda init
# conda init
# ource ~/.bashrc
# conda activate seedvr
pip install -r requirements.txt
```
## 1.2安装flash_attn
```bash
pip install flash_attn==2.5.9.post1 --no-build-isolation

# 如果报错，则需提前下载对应的版本flash_attn(已提前下载至本地) ,再执行安装本地包
# pip install flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
## 1.3安装apx(A800 or H200)
```bash
# (1)进入apex中
cd ../apex
# (2) 强制指定编译目标架构为 A800 (sm_80)，H200的话，应该是9.0
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
# (3) 开始编译并安装 (注意：这一步通常需要 5 到 10 分钟，请耐心等待它跑完)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./
```

## 1.4一些补充环境
```bash
cd ../SeedVR
pip install av
pip install -r lora_requirements.txt
pip install -r lora_finetune_requirements.txt
```
# 2.下载权重
```bash
#进入SeedVR/SeedVR目录
cd SeedVR
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download ByteDance-Seed/SeedVR2-7B \
  --local-dir ckpts/ \
  --cache-dir ckpts/cache \
  --local-dir-use-symlinks False \
  --include "*.json" "*.safetensors" "*.pth" "*.bin" "*.py" "*.md" "*.txt" \
  --exclude "seedvr2_ema_7b_sharp.pth"
```