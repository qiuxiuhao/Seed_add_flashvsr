# 1.创建环境
## 1.1创建基础的flashvsr环境
```bash
cd FlashVSR
conda create -n flashvsr python=3.11.13 -y
conda activate flashvsr
pip install setuptools==69.5.1 wheel
sed -i 's/torch==2.6.0+cu124/torch/g' setup.py
sed -i 's/torch==2.6.0+cu124/torch/g' requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
sed -i -E 's/torchaudio==[0-9\.\+a-zA-Z]+/torchaudio/g' setup.py requirements.txt
sed -i -E 's/torchvision==[0-9\.\+a-zA-Z]+/torchvision/g' setup.py requirements.txt
pip install -e . --no-build-isolation
pip install -r requirements.txt
cd ..
```
## 1.2安装Block-Sparse-Attention
```bash
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
cd ..
```
# 2.下载权重
```bash
cd examples/WanVSR
# 设置环境变量，确保 Python 脚本也走镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行下载脚本
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='JunhaoZhuang/FlashVSR-v1.1',
    local_dir='./FlashVSR-v1.1',
    max_workers=8,
    endpoint='https://hf-mirror.com',
    ignore_patterns=['*.git*', 'README.md']
)"

pip install modelscope
```
# 3.推理
## 3.1推理未微调版
```bash
#cd FlashVSR/examples/WanVSR
python infer_flashvsr_v1.1_full_test.py
```
## 3.1推理未微调版
```bash
#cd FlashVSR/examples/WanVSR
#只加载 LoRA和lq_proj_in
python infer_flashvsr_v1.1_full_lora.py \
  --lora_ckpt rola_weights/best.pt \
  --lora_config ../../train_rola/configs/stage1.yaml \
  --load_lq_proj true

#只加载 LoRA，不加载 lq_proj_in
python infer_flashvsr_v1.1_full_lora.py \
  --lora_ckpt rola_weights/best.pt \
  --load_lq_proj false
```