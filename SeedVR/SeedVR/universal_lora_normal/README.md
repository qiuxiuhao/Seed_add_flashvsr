# Universal LoRA Normal (UGC) 训练指南

本目录是基于 `universal_lora` 的独立分支，只用于“正常训练策略”（非 latent-only 默认）。

- 训练入口：`universal_lora_normal/train_seedvr2_7b_lora_normal.py`
- 支持分辨率：`--video_size 1024x640` 与 `--video_size 512x320`
- 默认模式：`--train_mode normal`

## 1. UGC 默认损失（normal）

默认值（可被命令行 `--w_*` 覆盖）：

- `w_latent=1.0`
- `w_pix=0.10`
- `w_ssim=0.05`
- `w_temp=0.05`
- `w_lpips=0.0`
- `pixel_loss=charbonnier`

总损失：

```text
total = w_latent * latent_mse
      + w_pix    * pixel_loss
      + w_ssim   * ssim_loss
      + w_temp   * temporal_diff_loss
      + w_lpips  * lpips_loss
```

说明：

- `pixel_loss` 与 `temporal_diff_loss` 默认均使用 Charbonnier，更适合 UGC 压缩噪声与离群像素。
- LPIPS 默认只在评估按需开启（`--eval_lpips`），训练默认不启用（`w_lpips=0`）。

## 2. 回退模式（latent-only）

可随时回退：

```bash
torchrun --nproc_per_node=8 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train/seedvr2_lora_1024_full \
  --video_size 1024x640 \
  --train_mode latent_only \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 20 \
  --eval_every 0 \
  --save_every 20 \
  --output_dir outputs/seedvr2_7b_lora_normal_1024_latent_smoke
```

## 3. 1024x640（8xH200）命令矩阵

### 3.1 Smoke（20 steps）

```bash
torchrun --nproc_per_node=8 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train/seedvr2_lora_1024_full \
  --video_size 1024x640 \
  --train_mode normal \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 20 \
  --warmup_steps 20 \
  --eval_every 0 \
  --save_every 20 \
  --output_dir outputs/seedvr2_7b_lora_normal_1024_smoke
```

### 3.2 Pilot（200 steps）

```bash
torchrun --nproc_per_node=8 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train/seedvr2_lora_1024_full \
  --video_size 1024x640 \
  --train_mode normal \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 200 \
  --warmup_steps 100 \
  --lr 8e-5 \
  --eval_every 50 \
  --save_every 50 \
  --no-eval_lpips \
  --output_dir outputs/seedvr2_7b_lora_normal_1024_pilot
```

### 3.3 Full（900 steps）

```bash
torchrun --nproc_per_node=8 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train/seedvr2_lora_1024_full \
  --video_size 1024x640 \
  --train_mode normal \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 900 \
  --warmup_steps 200 \
  --lr 8e-5 \
  --eval_every 100 \
  --save_every 100 \
  --no-eval_lpips \
  --output_dir outputs/seedvr2_7b_lora_normal_1024_full
```

## 4. 512x320（同一份代码）

### 4.1 Smoke（20 steps）

```bash
torchrun --nproc_per_node=4 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train_cropped_512_320/synthetic_cropped_512_320 \
  --video_size 512x320 \
  --train_mode normal \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 20 \
  --warmup_steps 20 \
  --eval_every 0 \
  --save_every 20 \
  --output_dir outputs/seedvr2_7b_lora_normal_512_smoke

torchrun --nproc_per_node=4 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train_cropped_512_320/synthetic_cropped_512_320 \
  --video_size 512x320 \
  --train_mode latent_only \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 20 \
  --warmup_steps 20 \
  --eval_every 0 \
  --save_every 20 \
  --output_dir outputs/seedvr2_7b_lora_normal_512_smoke
```


### 4.2 Full（示例：3600 steps）

```bash
torchrun --nproc_per_node=8 -m universal_lora_normal.train_seedvr2_7b_lora_normal \
  --launcher ddp \
  --config configs_7b/main.yaml \
  --base_ckpt ckpts/seedvr2_ema_7b.pth \
  --dataset_root data/train/seedvr2_lora_512_full_3200 \
  --video_size 512x320 \
  --train_mode normal \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 3600 \
  --warmup_steps 200 \
  --lr 8e-5 \
  --eval_every 200 \
  --save_every 200 \
  --no-eval_lpips \
  --output_dir outputs/seedvr2_7b_lora_normal_512_full
```

## 5. 常用调参

- 如果细节偏软：`--w_pix 0.15 --w_ssim 0.05 --w_temp 0.03`
- 如果时序闪烁：`--w_temp 0.08`（注意可能过平滑）
- 如果要训练期启用 LPIPS：`--w_lpips 0.02 --eval_lpips`
- 如果显存紧张：优先降低评估频率（`--eval_every`）和关闭评估 LPIPS（`--no-eval_lpips`）

## 6. 产物

- `best_lora.safetensors`
- `last_lora.safetensors`
- `last_training_state.pt`
- `checkpoints/step_xxxxxxxx/`
