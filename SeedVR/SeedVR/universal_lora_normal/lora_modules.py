from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    """A lightweight LoRA wrapper for nn.Linear."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank, device=device, dtype=dtype)
        )
        self.reset_parameters()

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def lora_weight(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A

    def merge(self) -> None:
        if self.merged:
            return
        self.base_layer.weight.data += self.lora_weight().to(self.base_layer.weight.dtype) * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base_layer.weight.data -= self.lora_weight().to(self.base_layer.weight.dtype) * self.scaling
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.merged:
            return base

        lora_x = self.dropout(x)
        lora_x = F.linear(lora_x, self.lora_A)
        lora_x = F.linear(lora_x, self.lora_B)
        return base + lora_x * self.scaling

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )


TARGET_LAYER_DEFAULT = ("proj_qkv", "proj_out", "proj_in", "proj_in_gate")
TARGET_SCOPE_DEFAULT = ("attn", "mlp")


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
    has_keyword = any(keyword in segments for keyword in target_keywords)
    has_scope = any(scope in segments for scope in target_scopes)
    return has_keyword and has_scope


def inject_lora_layers(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_keywords: Sequence[str] = TARGET_LAYER_DEFAULT,
    target_scopes: Sequence[str] = TARGET_SCOPE_DEFAULT,
) -> List[str]:
    """Inject LoRALinear into selected nn.Linear layers.

    Matching is path-segment based and restricted by scopes (`attn`/`mlp`) so
    patch/embed/output projection layers are not injected by mistake.
    """
    replaced: List[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, target_keywords, target_scopes):
            continue
        if isinstance(module, LoRALinear):
            continue

        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_module_by_name(model, name, wrapper)
        replaced.append(name)

    return replaced


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def collect_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            state[name] = tensor.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
    model_state = model.state_dict()
    missing: List[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            if strict:
                raise KeyError(f"LoRA key not found in model: {key}")
            continue
        model_state[key].copy_(value.to(device=model_state[key].device, dtype=model_state[key].dtype))

    if strict:
        for name in model_state.keys():
            if (name.endswith("lora_A") or name.endswith("lora_B")) and name not in state_dict:
                missing.append(name)
        if missing:
            raise KeyError(f"Missing LoRA keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")


def save_lora_safetensors(
    model: nn.Module,
    output_path: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = collect_lora_state_dict(model)
    save_file(state, str(path), metadata=metadata or {})


def load_lora_safetensors(model: nn.Module, input_path: str, strict: bool = False) -> None:
    state = load_file(str(input_path))
    load_lora_state_dict(model, state, strict=strict)


def get_trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def merge_lora_weights(model: nn.Module) -> None:
    for module in iter_lora_modules(model):
        module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    for module in iter_lora_modules(model):
        module.unmerge()
