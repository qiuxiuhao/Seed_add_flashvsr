from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torchvision.io import read_video

from rola_finetune.train.flow import RaftEstimator, raft_temporal_consistency_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind eval for wild set: MUSIQ + warp proxy")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--raft_ckpt", type=str, default="")
    parser.add_argument("--musiq_ckpt", type=str, default="")
    parser.add_argument("--frame_cap", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def normalize_tchw(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        video = video.float().div(255.0)
    else:
        video = video.float()
        if video.max() > 1.5:
            video = video / 255.0
    return video.mul(2.0).sub(1.0)


def align(pred: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = min(pred.shape[0], src.shape[0])
    h = min(pred.shape[2], src.shape[2])
    w = min(pred.shape[3], src.shape[3])
    return pred[:t, :, :h, :w], src[:t, :, :h, :w]


def build_musiq(device: torch.device, ckpt_path: str = ""):
    try:
        import pyiqa
    except Exception:
        return None

    if ckpt_path and Path(ckpt_path).exists():
        # Best-effort: API compatibility differs by pyiqa versions.
        for kwargs in (
            {"pretrained_model_path": ckpt_path},
            {"model_path": ckpt_path},
            {},
        ):
            try:
                return pyiqa.create_metric("musiq", device=device, **kwargs)
            except Exception:
                continue
        return None

    try:
        return pyiqa.create_metric("musiq", device=device)
    except Exception:
        return None


def compute_musiq_score(musiq_model, frames: torch.Tensor) -> float:
    # frames: T C H W in [-1,1]
    if musiq_model is None:
        return float("nan")

    x = frames.add(1.0).mul(0.5).clamp(0.0, 1.0)
    vals = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            score = musiq_model(x[i : i + 1])
            if torch.is_tensor(score):
                vals.append(float(score.mean().item()))
            else:
                vals.append(float(score))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def main() -> None:
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    input_dir = Path(args.input_dir) if args.input_dir else None

    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    pred_map = {p.name: p for p in sorted(pred_dir.glob("*.mp4"))}
    if not pred_map:
        raise RuntimeError(f"No mp4 files found in {pred_dir}")

    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        inp_map = {p.name: p for p in sorted(input_dir.glob("*.mp4"))}
        names = sorted(set(pred_map.keys()) & set(inp_map.keys()))
        if not names:
            raise RuntimeError("No overlapping videos between pred_dir and input_dir")
    else:
        inp_map = {}
        names = sorted(pred_map.keys())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raft = RaftEstimator(device, ckpt_path=args.raft_ckpt if args.raft_ckpt else None)
    if not raft.available:
        print("[WARN] RAFT unavailable, warp error will be NaN")

    musiq = build_musiq(device, ckpt_path=args.musiq_ckpt)
    if musiq is None:
        print("[WARN] MUSIQ unavailable, MUSIQ score will be NaN")

    warp_sum = 0.0
    musiq_sum = 0.0
    warp_count = 0
    musiq_count = 0

    for name in names:
        pred, _, _ = read_video(str(pred_map[name]), output_format="TCHW", pts_unit="sec")
        pred = normalize_tchw(pred).to(device)

        if input_dir is not None:
            src, _, _ = read_video(str(inp_map[name]), output_format="TCHW", pts_unit="sec")
            src = normalize_tchw(src).to(device)
        else:
            src = pred.clone()

        pred, src = align(pred, src)

        if args.frame_cap > 0 and pred.shape[0] > args.frame_cap:
            idx = torch.linspace(0, pred.shape[0] - 1, args.frame_cap, device=device).round().long()
            pred = pred[idx]
            src = src[idx]

        musiq_v = compute_musiq_score(musiq, pred)
        if musiq_v == musiq_v:
            musiq_sum += musiq_v
            musiq_count += 1

        if raft.available:
            warp = raft_temporal_consistency_loss(pred, src, raft)
            warp_sum += float(warp.item())
            warp_count += 1

    result: Dict[str, float] = {
        "count": float(len(names)),
        "MUSIQ": (musiq_sum / max(1, musiq_count)) if musiq_count > 0 else float("nan"),
        "warp_error_proxy": (warp_sum / max(1, warp_count)) if warp_count > 0 else float("nan"),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
