from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .rotated_lora import RotatedLoRAConfig, RotatedLoRALinear

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

    _HAS_SAFETENSORS = True
except Exception:
    safe_load_file = None
    safe_save_file = None
    _HAS_SAFETENSORS = False


DEFAULT_TARGET_KEYWORDS: Tuple[str, ...] = (
    "proj_qkv",
    "proj_out",
    "proj_in",
)
DEFAULT_TARGET_SCOPES: Tuple[str, ...] = ("attn", "mlp")


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _matches_target(
    module_name: str,
    target_keywords: Sequence[str],
    target_scopes: Sequence[str],
) -> bool:
    segments = module_name.split(".")
    has_keyword = any(k in segments for k in target_keywords)
    has_scope = any(s in segments for s in target_scopes)
    return has_keyword and has_scope


def inject_rola_layers(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_keywords: Sequence[str] = DEFAULT_TARGET_KEYWORDS,
    target_scopes: Sequence[str] = DEFAULT_TARGET_SCOPES,
) -> List[str]:
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, RotatedLoRALinear):
            continue
        if not _matches_target(name, target_keywords, target_scopes):
            continue

        wrapper = RotatedLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_module_by_name(model, name, wrapper)
        replaced.append(name)
    return replaced


def freeze_non_rola_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, RotatedLoRALinear):
            module.rola_A.requires_grad_(True)
            module.rola_B.requires_grad_(True)
            module.rola_R.requires_grad_(True)


def iter_rola_modules(model: nn.Module) -> Iterable[RotatedLoRALinear]:
    for module in model.modules():
        if isinstance(module, RotatedLoRALinear):
            yield module


def merge_rola_weights(model: nn.Module) -> None:
    for module in iter_rola_modules(model):
        module.merge()


def unmerge_rola_weights(model: nn.Module) -> None:
    for module in iter_rola_modules(model):
        module.unmerge()


def collect_rola_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("rola_A") or name.endswith("rola_B") or name.endswith("rola_R"):
            state[name] = tensor.detach().cpu()
    return state


def load_rola_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
    model_state = model.state_dict()
    missing: List[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            if strict:
                raise KeyError(f"ROLA key not found in model: {key}")
            continue
        model_state[key].copy_(value.to(device=model_state[key].device, dtype=model_state[key].dtype))

    if strict:
        for key in model_state.keys():
            if (key.endswith("rola_A") or key.endswith("rola_B") or key.endswith("rola_R")) and key not in state_dict:
                missing.append(key)
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise KeyError(f"Missing ROLA keys: {preview}{suffix}")


def _save_state_file(path: Path, state: Dict[str, torch.Tensor]) -> Path:
    if _HAS_SAFETENSORS:
        path = path.with_suffix(".safetensors")
        safe_save_file(state, str(path))
        return path
    path = path.with_suffix(".pt")
    torch.save(state, str(path))
    return path


def _load_state_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors is not installed, cannot read .safetensors adapter")
        return safe_load_file(str(path))
    data = torch.load(str(path), map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected adapter format: {type(data)}")
    return data


def save_adapter(
    output_dir: str,
    model: nn.Module,
    *,
    rola_config: RotatedLoRAConfig,
    target_keywords: Sequence[str],
    target_scopes: Sequence[str],
    step: Optional[int] = None,
    extra: Optional[Dict[str, object]] = None,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = collect_rola_state_dict(model)
    model_path = _save_state_file(out_dir / "adapter_model", state)

    cfg: Dict[str, object] = {
        "format": "rotated_lora",
        "rank": rola_config.rank,
        "alpha": rola_config.alpha,
        "dropout": rola_config.dropout,
        "target_keywords": list(target_keywords),
        "target_scopes": list(target_scopes),
        "model_file": model_path.name,
    }
    if step is not None:
        cfg["step"] = int(step)
    if extra:
        cfg.update(extra)

    with (out_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return str(model_path)


def load_adapter(
    adapter_path: str,
    model: nn.Module,
    strict: bool = False,
) -> Dict[str, object]:
    path = Path(adapter_path)
    cfg: Dict[str, object] = {}

    if path.is_dir():
        cfg_path = path / "adapter_config.json"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        model_file = cfg.get("model_file", "") if cfg else ""
        if model_file:
            state_path = path / str(model_file)
        elif (path / "adapter_model.safetensors").exists():
            state_path = path / "adapter_model.safetensors"
        elif (path / "adapter_model.pt").exists():
            state_path = path / "adapter_model.pt"
        else:
            raise FileNotFoundError(f"Cannot find adapter model file under {path}")
    else:
        state_path = path

    state = _load_state_file(state_path)
    load_rola_state_dict(model, state, strict=strict)
    return cfg


def get_trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
