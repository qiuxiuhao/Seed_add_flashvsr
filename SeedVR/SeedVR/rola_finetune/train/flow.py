from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class RaftEstimator:
    def __init__(self, device: torch.device, ckpt_path: Optional[str] = None):
        self.device = device
        self.available = False
        self.model = None

        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        except Exception:
            return

        try:
            if ckpt_path:
                self.model = raft_small(weights=None, progress=False).to(device).eval()
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
            else:
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights, progress=False).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    @torch.no_grad()
    def estimate(self, frame_1: torch.Tensor, frame_2: torch.Tensor) -> torch.Tensor:
        # frame: N C H W in [-1,1]
        if (not self.available) or (self.model is None):
            n, _, h, w = frame_1.shape
            return torch.zeros((n, 2, h, w), device=frame_1.device, dtype=frame_1.dtype)

        x1 = frame_1.add(1.0).mul(0.5).clamp(0.0, 1.0)
        x2 = frame_2.add(1.0).mul(0.5).clamp(0.0, 1.0)

        # RAFT expects reasonably sized feature maps; keep original size for fidelity.
        flows = self.model(x1, x2)
        flow = flows[-1]
        return flow


def _make_base_grid(n: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # H W 2
    return grid.unsqueeze(0).repeat(n, 1, 1, 1)


def warp_with_flow(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # frame: N C H W ; flow: N 2 H W, in pixel units.
    n, _, h, w = frame.shape
    grid = _make_base_grid(n, h, w, frame.device, frame.dtype)

    fx = flow[:, 0] / max(1.0, (w - 1) / 2.0)
    fy = flow[:, 1] / max(1.0, (h - 1) / 2.0)
    flow_norm = torch.stack([fx, fy], dim=-1)

    sample_grid = grid + flow_norm
    return F.grid_sample(frame, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def raft_temporal_consistency_loss(
    pred_frames: torch.Tensor,
    guide_frames: torch.Tensor,
    estimator: RaftEstimator,
) -> torch.Tensor:
    # pred/guide: T C H W
    if pred_frames.shape[0] < 2:
        return torch.zeros((), device=pred_frames.device, dtype=torch.float32)

    losses = []
    for i in range(pred_frames.shape[0] - 1):
        guide_t = guide_frames[i : i + 1]
        guide_n = guide_frames[i + 1 : i + 2]
        flow = estimator.estimate(guide_t, guide_n)

        pred_n = pred_frames[i + 1 : i + 2]
        pred_t = pred_frames[i : i + 1]
        pred_n_warp = warp_with_flow(pred_n, flow)
        losses.append((pred_n_warp - pred_t).abs().mean())

    return torch.stack(losses).mean()
