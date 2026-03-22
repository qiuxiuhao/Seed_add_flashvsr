from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as ssim_fn
except Exception:
    ssim_fn = None


@dataclass
class SupervisedLossOutput:
    loss_l1: torch.Tensor
    loss_ssim: torch.Tensor
    loss_lpips: torch.Tensor
    metric_ssim: torch.Tensor
    metric_lpips: torch.Tensor


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def pointwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l1",
    eps: float = 1e-3,
    delta: float = 0.05,
) -> torch.Tensor:
    if loss_type == "charbonnier":
        return charbonnier_loss(pred, target, eps=eps)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=delta)
    return F.l1_loss(pred, target)


def select_decode_frames(
    pred: torch.Tensor,
    target: torch.Tensor,
    frame_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # pred/target: T C H W
    if frame_cap <= 0:
        return pred, target
    t = pred.shape[0]
    if t <= frame_cap:
        return pred, target
    idx = torch.linspace(0, t - 1, steps=frame_cap, device=pred.device)
    idx = idx.round().long().unique(sorted=True)
    return pred[idx], target[idx]


def compute_supervised_recon_losses(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    lpips_model: Optional[torch.nn.Module],
    frame_cap: int,
    pixel_loss_type: str,
    pixel_eps: float,
    pixel_delta: float,
) -> SupervisedLossOutput:
    pred_frames, gt_frames = select_decode_frames(pred_frames, gt_frames, frame_cap)

    loss_l1 = pointwise_loss(
        pred_frames,
        gt_frames,
        loss_type=pixel_loss_type,
        eps=pixel_eps,
        delta=pixel_delta,
    )

    if ssim_fn is None:
        metric_ssim = torch.zeros((), device=pred_frames.device, dtype=torch.float32)
        loss_ssim = torch.zeros((), device=pred_frames.device, dtype=torch.float32)
    else:
        metric_ssim = ssim_fn(pred_frames.float(), gt_frames.float(), data_range=2.0, size_average=True)
        loss_ssim = 1.0 - metric_ssim

    if lpips_model is None:
        metric_lpips = torch.zeros((), device=pred_frames.device, dtype=torch.float32)
        loss_lpips = torch.zeros((), device=pred_frames.device, dtype=torch.float32)
    else:
        metric_lpips = lpips_model(pred_frames.float(), gt_frames.float()).mean()
        loss_lpips = metric_lpips

    return SupervisedLossOutput(
        loss_l1=loss_l1,
        loss_ssim=loss_ssim,
        loss_lpips=loss_lpips,
        metric_ssim=metric_ssim,
        metric_lpips=metric_lpips,
    )


def temporal_smooth_loss(frames: torch.Tensor, loss_type: str = "charbonnier") -> torch.Tensor:
    # frames: T C H W
    if frames.shape[0] < 2:
        return torch.zeros((), device=frames.device, dtype=torch.float32)
    diff = frames[1:] - frames[:-1]
    if loss_type == "l1":
        return diff.abs().mean()
    return charbonnier_loss(diff, torch.zeros_like(diff), eps=1e-3)


def simple_degradation(frames: torch.Tensor, downscale: int = 2) -> torch.Tensor:
    # frames: T C H W in [-1,1]
    if downscale <= 1:
        return frames

    t, c, h, w = frames.shape
    x = frames.add(1.0).mul(0.5).clamp(0.0, 1.0)

    # lightweight blur
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    d_h = max(1, h // downscale)
    d_w = max(1, w // downscale)
    x = F.interpolate(x, size=(d_h, d_w), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    x = x.mul(2.0).sub(1.0)
    return x


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10((data_range**2) / (mse + 1e-8))


def maybe_build_lpips(device: torch.device) -> Optional[torch.nn.Module]:
    try:
        import lpips as lpips_lib
    except Exception:
        return None

    model = lpips_lib.LPIPS(net="vgg").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
