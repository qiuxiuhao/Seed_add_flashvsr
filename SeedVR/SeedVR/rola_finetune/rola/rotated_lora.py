from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RotatedLoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05


class RotatedLoRALinear(nn.Module):
    """Rotated LoRA wrapper for nn.Linear.

    DeltaW = B @ R @ A, where R is re-orthogonalized by QR each forward pass.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.rola_A = nn.Parameter(torch.empty(self.rank, in_features, device=device, dtype=dtype))
        self.rola_B = nn.Parameter(torch.empty(out_features, self.rank, device=device, dtype=dtype))
        self.rola_R = nn.Parameter(torch.empty(self.rank, self.rank, device=device, dtype=dtype))

        self.reset_parameters()

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.rola_A, a=math.sqrt(5))
        nn.init.zeros_(self.rola_B)
        nn.init.eye_(self.rola_R)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_layer.bias

    def orthogonal_rotation(self) -> torch.Tensor:
        # QR gives an orthonormal basis; rank is small so overhead is negligible.
        q, _ = torch.linalg.qr(self.rola_R.float(), mode="reduced")
        return q.to(dtype=self.rola_R.dtype)

    def rola_weight(self) -> torch.Tensor:
        r_orth = self.orthogonal_rotation()
        return self.rola_B @ r_orth @ self.rola_A

    def merge(self) -> None:
        if self.merged:
            return
        self.base_layer.weight.data.add_(self.rola_weight().to(self.base_layer.weight.dtype), alpha=self.scaling)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base_layer.weight.data.add_(self.rola_weight().to(self.base_layer.weight.dtype), alpha=-self.scaling)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.merged:
            return out

        z = self.dropout(x)
        z = F.linear(z, self.rola_A)
        z = F.linear(z, self.orthogonal_rotation())
        z = F.linear(z, self.rola_B)
        return out + z * self.scaling

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )
