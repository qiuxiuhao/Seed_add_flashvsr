from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision.io import read_video

try:
    from pytorch_msssim import ssim as ssim_fn
except Exception:
    ssim_fn = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic holdout metrics")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--frame_cap", type=int, default=0, help="Use at most N frames per video (0 = all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_lpips", action="store_true")
    return parser.parse_args()


def normalize_tchw(video: torch.Tensor) -> torch.Tensor:
    # T C H W uint8 -> T C H W float in [-1,1]
    if video.dtype == torch.uint8:
        video = video.float().div(255.0)
    else:
        video = video.float()
        if video.max() > 1.5:
            video = video / 255.0
    return video.mul(2.0).sub(1.0)


def align(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = min(pred.shape[0], gt.shape[0])
    h = min(pred.shape[2], gt.shape[2])
    w = min(pred.shape[3], gt.shape[3])
    pred = pred[:t, :, :h, :w]
    gt = gt[:t, :, :h, :w]
    return pred, gt


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(pred, gt)
    psnr = 10.0 * torch.log10((data_range**2) / (mse + 1e-8))
    return float(psnr.item())


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError("pred_dir or gt_dir does not exist")

    pred_map = {p.name: p for p in sorted(pred_dir.glob("*.mp4"))}
    gt_map = {p.name: p for p in sorted(gt_dir.glob("*.mp4"))}
    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common:
        raise RuntimeError("No overlapping .mp4 filenames found between pred_dir and gt_dir")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips as lpips_lib

            lpips_model = lpips_lib.LPIPS(net="vgg").to(device).eval()
            for p in lpips_model.parameters():
                p.requires_grad_(False)
        except Exception:
            lpips_model = None
            print("[WARN] LPIPS unavailable, skip LPIPS metric")

    sums: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    n = 0

    for name in common:
        pred, _, _ = read_video(str(pred_map[name]), output_format="TCHW", pts_unit="sec")
        gt, _, _ = read_video(str(gt_map[name]), output_format="TCHW", pts_unit="sec")

        pred = normalize_tchw(pred).to(device)
        gt = normalize_tchw(gt).to(device)
        pred, gt = align(pred, gt)

        if args.frame_cap > 0 and pred.shape[0] > args.frame_cap:
            idx = torch.linspace(0, pred.shape[0] - 1, args.frame_cap, device=device).round().long()
            pred = pred[idx]
            gt = gt[idx]

        sums["psnr"] += compute_psnr(pred, gt)

        if ssim_fn is None:
            ssim_v = 0.0
        else:
            ssim_v = float(ssim_fn(pred.float(), gt.float(), data_range=2.0, size_average=True).item())
        sums["ssim"] += ssim_v

        if lpips_model is not None:
            lpips_v = float(lpips_model(pred.float(), gt.float()).mean().item())
        else:
            lpips_v = float("nan")
        sums["lpips"] += 0.0 if (lpips_v != lpips_v) else lpips_v

        n += 1

    out = {
        "count": n,
        "PSNR": sums["psnr"] / max(1, n),
        "SSIM": sums["ssim"] / max(1, n),
        "LPIPS": (sums["lpips"] / max(1, n)) if lpips_model is not None else float("nan"),
    }

    print(json_dumps(out))


def json_dumps(obj: Dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
