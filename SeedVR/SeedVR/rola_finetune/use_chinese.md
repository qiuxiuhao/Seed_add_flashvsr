# 把rola_finetune里的metrics这个文件夹移到SeedVR/ckpts目录下

# 补充环境
```bash
pip install -r lora_finetune_requirements.txt
```

# 执行指令
```bash
#H200 大块
torchrun --nproc_per_node=8 -m rola_finetune.train.main \
  --config rola_finetune/configs/h200_full.yaml \
  --model_config configs_7b/main.yaml \
  --base_ckpt ./ckpts/seedvr2_ema_7b.pth \
  --pos_emb ./pos_emb.pt \
  --data_root ./data/train \
  --output_dir ./outputs/rola_h200_full \
  --resume_adapter ./outputs/rola_h200_full/checkpoints/stage1/step_00003200 \
  --resume_state ./outputs/rola_h200_full/checkpoints/stage1/step_00003200/train_state.pt
#H200 小块
torchrun --nproc_per_node=8 -m rola_finetune.train.main \
  --config rola_finetune/configs/A800_full.yaml \
  --model_config configs_7b/main.yaml \
  --base_ckpt ./ckpts/seedvr2_ema_7b.pth \
  --pos_emb ./pos_emb.pt \
  --data_root ./data/train \
  --output_dir ./outputs/rola_A800_full \
  --resume_adapter ./outputs/rola_A800_full/checkpoints/stage1/step_00007200 \
  --resume_state ./outputs/rola_A800_full/checkpoints/stage1/step_00007200/train_state.pt
```
