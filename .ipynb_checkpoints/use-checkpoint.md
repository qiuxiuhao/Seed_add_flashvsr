# 环境部署
#
#
# 仅使用flashvsr
```bash
LOCAL=~/SeedVR_add_flashvsr
REMOTE=root@<SERVER_HOST>:~/autodl-tmp/SeedVR_add_flashvsr
#仅使用flashvsr
python run_dual_vsr.py --mode flashvsr --flash-profile full_test   --input-root test-input-final/test_data   --result-root result
```
# 仅使用SeedVR finetune
# 更新adapter_path
cd ~/autodl-tmp/SeedVR_add_flashvsr
python - <<'PY'
import json
p="dual_vsr_profiles.json"
d=json.load(open(p,"r",encoding="utf-8"))
d["seed_profiles"]["finetuned"]["args"]["adapter_path"]="lora/step_00007200"
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("updated finetuned adapter_path")
PY

```