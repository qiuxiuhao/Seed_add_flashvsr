# 环境部署
#
#
apt update
apt install -y ffmpeg
# 仅使用flashvsr
```bash
LOCAL=~/SeedVR_add_flashvsr
REMOTE=root@<SERVER_HOST>:~/autodl-tmp/SeedVR_add_flashvsr
#仅使用flashvsr
python run_dual_vsr.py --mode flashvsr --flash-profile full_test   --input-root test-input-final/test_data   --result-root result
```
# 仅使用SeedVR finetune
## 更改微调权重
### 更新adapter_path
conda install -n seedvr -c conda-forge ffmpeg -y
cd ~/autodl-tmp/SeedVR_add_flashvsr
python - <<'PY'
import json
p="dual_vsr_profiles.json"
d=json.load(open(p,"r",encoding="utf-8"))
d["seed_profiles"]["finetuned"]["args"]["adapter_path"]="lora_weights/step_00007200"
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("updated finetuned adapter_path")
PY

python run_dual_vsr.py --mode seed --seed-profile finetuned \
  --input-root test-input-final/test_data \
  --result-root result/SeedVR_adapter_2400

# 修改改lora_path
```bash
cd ~/autodl-tmp/SeedVR_add_flashvsr
python - <<'PY'
import json
p="dual_vsr_profiles.json"
d=json.load(open(p,"r",encoding="utf-8"))
d["seed_profiles"]["lora"]["args"]["lora_path"]="lora_weights/lora.safetensors"
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("updated lora_path")
PY

```
