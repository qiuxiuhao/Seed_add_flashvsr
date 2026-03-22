from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import re
import time
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_msssim import ssim as ssim_fn

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_video
from tqdm import tqdm

# Ensure repo root is importable when launched as a script via torchrun.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_config
from common.diffusion.types import PredictionType
from projects.video_diffusion_sr.infer import VideoDiffusionInfer

from universal_lora_normal.lora_modules import (
    TARGET_LAYER_DEFAULT,

    freeze_non_lora_parameters,
    get_total_parameter_count,
    get_trainable_parameter_count,
    inject_lora_layers,
    load_lora_safetensors,
    save_lora_safetensors,
)


@dataclasses.dataclass
class LossPack:
    total: torch.Tensor
    latent_mse: torch.Tensor
    pixel_loss: torch.Tensor
    ssim_loss: torch.Tensor
    temp_loss: torch.Tensor
    lpips_loss: torch.Tensor
    psnr: torch.Tensor
    ssim_metric: torch.Tensor
    lpips_metric: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedVR2-7B LoRA finetuning for UGC short videos (normal + latent-only)"
    )

    parser.add_argument("--launcher", type=str, default="ddp", choices=["ddp", "accelerate"])

    parser.add_argument("--config", type=str, default="configs_7b/main.yaml")
    parser.add_argument("--base_ckpt", type=str, default="ckpts/seedvr2_ema_7b.pth")
    parser.add_argument("--pos_emb", type=str, default="pos_emb.pt")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root containing input/*.mp4 and target/*.mp4",
    )
    parser.add_argument(
        "--video_size",
        type=str,
        required=True,
        choices=["512x320", "1024x640"],
        help="Target video size in WxH. Orientation swap is allowed.",
    )

    parser.add_argument(
        "--manifest_train",
        type=str,
        default="",
        help="Optional prebuilt train manifest. If provided, manifest_val must also be set.",
    )
    parser.add_argument(
        "--manifest_val",
        type=str,
        default="",
        help="Optional prebuilt val manifest. If provided, manifest_train must also be set.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio by source group.")

    parser.add_argument("--output_dir", type=str, default="outputs/seedvr2_7b_lora_normal")

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target",
        type=str,
        nargs="+",
        default=["proj_qkv", "proj_out"],
        help="Path-segment match for target Linear layers",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=600)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["normal", "latent_only"],
        default="normal",
        help="Loss preset: normal for UGC short video, latent_only for fallback.",
    )
    parser.add_argument(
        "--pixel_loss",
        type=str,
        choices=["charbonnier", "l1", "huber"],
        default="charbonnier",
        help="Point-wise reconstruction loss type for pixel and temporal diff terms.",
    )
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--w_latent", type=float, default=None)
    parser.add_argument("--w_pix", type=float, default=None)
    parser.add_argument("--w_ssim", type=float, default=None)
    parser.add_argument("--w_temp", type=float, default=None)
    parser.add_argument("--w_lpips", type=float, default=None)
    parser.add_argument(
        "--eval_lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LPIPS during eval (training still controlled by --w_lpips).",
    )

    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)

    parser.add_argument("--cond_noise_scale", type=float, default=-1.0)
    parser.add_argument("--time_loc", type=float, default=0.0)
    parser.add_argument("--time_scale", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_lora", type=str, default="")

    parser.add_argument("--disable_runtime_checkpointing", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")

    args = parser.parse_args()
    preset = (
        {
            "w_latent": 1.0,
            "w_pix": 0.10,
            "w_ssim": 0.05,
            "w_temp": 0.05,
            "w_lpips": 0.0,
        }
        if args.train_mode == "normal"
        else {
            "w_latent": 1.0,
            "w_pix": 0.0,
            "w_ssim": 0.0,
            "w_temp": 0.0,
            "w_lpips": 0.0,
        }
    )
    for key, default_value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, default_value)

    for key in ("w_latent", "w_pix", "w_ssim", "w_temp", "w_lpips"):
        value = float(getattr(args, key))
        if value < 0.0:
            raise ValueError(f"--{key} must be >= 0, got {value}")
        setattr(args, key, value)

    if args.charbonnier_eps <= 0.0:
        raise ValueError(f"--charbonnier_eps must be > 0, got {args.charbonnier_eps}")
    if args.huber_delta <= 0.0:
        raise ValueError(f"--huber_delta must be > 0, got {args.huber_delta}")
    return args

class ManifestVideoPairDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.items: List[Dict[str, object]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        if not self.items:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.items[index]


CROP_SUFFIX_PATTERN = re.compile(r"(?:__crop\d+|_crop\d+|-crop\d+)$", re.IGNORECASE)


def parse_video_size(video_size: str) -> Tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", video_size.strip().lower())
    if match is None:
        raise ValueError(f"Invalid --video_size format: {video_size}")
    width = int(match.group(1))
    height = int(match.group(2))
    if (width, height) not in {(512, 320), (1024, 640)}:
        raise ValueError(f"Unsupported --video_size={video_size}. Allowed: 512x320, 1024x640")
    return width, height


def infer_group_id(sample_id: str) -> str:
    base = CROP_SUFFIX_PATTERN.sub("", sample_id)
    return base if base else sample_id


def read_video_meta(video_path: Path) -> Tuple[int, int, int]:
    try:
        import av
    except Exception as exc:
        raise RuntimeError("Missing dependency 'av'. Install via pip install av") from exc

    with av.open(str(video_path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise RuntimeError(f"No video stream found: {video_path}")
        width = int(video_stream.width or 0)
        height = int(video_stream.height or 0)
        frames = int(video_stream.frames or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video spatial shape for {video_path}: {width}x{height}")
    return width, height, max(0, frames)


def ensure_manifest_field(item: Dict[str, object], key: str, manifest_path: str, line_no: int) -> object:
    if key not in item:
        raise KeyError(f"{manifest_path}:{line_no} missing field '{key}'")
    return item[key]


def write_manifest(manifest_path: Path, items: List[Dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def collect_dataset_pairs(dataset_root: Path) -> List[Tuple[str, Path, Path]]:
    input_dir = dataset_root / "input"
    target_dir = dataset_root / "target"
    if not input_dir.exists() or not target_dir.exists():
        raise FileNotFoundError(
            f"dataset_root must contain input/ and target/ directories: {dataset_root}"
        )

    input_map = {path.name: path for path in sorted(input_dir.glob("*.mp4"))}
    target_map = {path.name: path for path in sorted(target_dir.glob("*.mp4"))}
    if not input_map:
        raise RuntimeError(f"No input videos found: {input_dir}")
    if not target_map:
        raise RuntimeError(f"No target videos found: {target_dir}")

    input_keys = set(input_map.keys())
    target_keys = set(target_map.keys())
    if input_keys != target_keys:
        only_input = sorted(input_keys - target_keys)
        only_target = sorted(target_keys - input_keys)
        details = [
            f"Pair mismatch under {dataset_root}",
            f"only_input_count={len(only_input)} examples={only_input[:8]}",
            f"only_target_count={len(only_target)} examples={only_target[:8]}",
        ]
        raise RuntimeError("\n".join(details))

    pairs: List[Tuple[str, Path, Path]] = []
    for name in sorted(input_map.keys()):
        sample_id = Path(name).stem
        pairs.append((sample_id, input_map[name].resolve(), target_map[name].resolve()))
    return pairs


def build_manifests_from_dataset(
    dataset_root: Path,
    video_size: str,
    allowed_sizes: Set[Tuple[int, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[str, str]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"--val_ratio must be in (0, 1), got {val_ratio}")

    pairs = collect_dataset_pairs(dataset_root)
    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    for sample_id, input_path, target_path in pairs:
        in_w, in_h, in_frames = read_video_meta(input_path)
        tgt_w, tgt_h, _ = read_video_meta(target_path)
        if (in_w, in_h) != (tgt_w, tgt_h):
            errors.append(
                f"{sample_id}: input/target size mismatch input={in_w}x{in_h} target={tgt_w}x{tgt_h}"
            )
            continue
        if (in_w, in_h) not in allowed_sizes:
            allowed = ", ".join([f"{w}x{h}" for (w, h) in sorted(allowed_sizes)])
            errors.append(f"{sample_id}: invalid size {in_w}x{in_h}, expected one of [{allowed}]")
            continue

        rows.append(
            {
                "id": sample_id,
                "group_id": infer_group_id(sample_id),
                "input_path": str(input_path),
                "target_path": str(target_path),
                "width": in_w,
                "height": in_h,
                "frames": in_frames,
                "split": "",
            }
        )

    if errors:
        preview = "\n".join(errors[:20])
        suffix = "\n..." if len(errors) > 20 else ""
        raise RuntimeError(
            f"Dataset validation failed with {len(errors)} invalid pairs for --video_size {video_size}:\n{preview}{suffix}"
        )
    if not rows:
        raise RuntimeError(f"No valid pairs found under {dataset_root}")

    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)

    group_ids = sorted(groups.keys())
    if len(group_ids) < 2:
        raise RuntimeError(
            "Group split requires at least 2 source groups. "
            f"Found {len(group_ids)} group(s) in dataset_root={dataset_root}"
        )

    rng = random.Random(seed)
    rng.shuffle(group_ids)
    val_group_count = int(round(len(group_ids) * val_ratio))
    val_group_count = max(1, min(len(group_ids) - 1, val_group_count))

    val_group_set = set(group_ids[:val_group_count])
    train_items: List[Dict[str, object]] = []
    val_items: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        if str(item["group_id"]) in val_group_set:
            item["split"] = "val"
            val_items.append(item)
        else:
            item["split"] = "train"
            train_items.append(item)

    train_groups = {str(item["group_id"]) for item in train_items}
    val_groups = {str(item["group_id"]) for item in val_items}
    overlap = train_groups.intersection(val_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected during split: overlap groups={sorted(overlap)[:8]}")
    if not train_items or not val_items:
        raise RuntimeError(
            f"Split resulted in empty subset: train={len(train_items)} val={len(val_items)}"
        )

    train_manifest = (dataset_root / f"manifest_train_{video_size}.jsonl").resolve()
    val_manifest = (dataset_root / f"manifest_val_{video_size}.jsonl").resolve()
    write_manifest(train_manifest, train_items)
    write_manifest(val_manifest, val_items)
    return str(train_manifest), str(val_manifest)


def validate_manifest(
    manifest_path: str,
    split_name: str,
    allowed_sizes: Set[Tuple[int, int]],
) -> Set[str]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    groups: Set[str] = set()
    errors: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no} bad json: {exc}")
                continue

            sample_id = str(ensure_manifest_field(item, "id", str(path), line_no))
            input_path = Path(str(ensure_manifest_field(item, "input_path", str(path), line_no)))
            target_path = Path(str(ensure_manifest_field(item, "target_path", str(path), line_no)))
            group_id = str(item.get("group_id", infer_group_id(sample_id)))

            if not input_path.exists() or not target_path.exists():
                errors.append(
                    f"{path}:{line_no} missing files input_exists={input_path.exists()} target_exists={target_path.exists()}"
                )
                continue

            in_w, in_h, _ = read_video_meta(input_path)
            tgt_w, tgt_h, _ = read_video_meta(target_path)
            if (in_w, in_h) != (tgt_w, tgt_h):
                errors.append(
                    f"{path}:{line_no} {sample_id} input/target mismatch input={in_w}x{in_h} target={tgt_w}x{tgt_h}"
                )
                continue
            if (in_w, in_h) not in allowed_sizes:
                allowed = ", ".join([f"{w}x{h}" for (w, h) in sorted(allowed_sizes)])
                errors.append(
                    f"{path}:{line_no} {sample_id} invalid size {in_w}x{in_h}, expected one of [{allowed}]"
                )
                continue
            groups.add(group_id)

    if errors:
        preview = "\n".join(errors[:20])
        suffix = "\n..." if len(errors) > 20 else ""
        raise RuntimeError(
            f"Manifest validation failed for {split_name} ({path}) with {len(errors)} issue(s):\n{preview}{suffix}"
        )
    if not groups:
        raise RuntimeError(f"Manifest has no valid rows: {path}")
    return groups


def resolve_manifest_paths(ctx: "DistContext", args: argparse.Namespace) -> Tuple[str, str]:
    target_w, target_h = parse_video_size(args.video_size)
    allowed_sizes = {(target_w, target_h), (target_h, target_w)}
    dataset_root = Path(args.dataset_root).resolve()

    has_train = bool(args.manifest_train)
    has_val = bool(args.manifest_val)
    if has_train != has_val:
        raise ValueError("--manifest_train and --manifest_val must be provided together.")

    if has_train and has_val:
        train_manifest = str(Path(args.manifest_train).resolve())
        val_manifest = str(Path(args.manifest_val).resolve())
        if ctx.is_main:
            train_groups = validate_manifest(train_manifest, "train", allowed_sizes)
            val_groups = validate_manifest(val_manifest, "val", allowed_sizes)
            overlap = train_groups.intersection(val_groups)
            if overlap:
                raise RuntimeError(
                    f"Group leakage between provided manifests: overlap groups={sorted(overlap)[:8]}"
                )
            rank0_print(
                ctx,
                f"[Data] Using provided manifests train={train_manifest} val={val_manifest}",
            )
        ctx.barrier()
        return train_manifest, val_manifest

    train_manifest = str((dataset_root / f"manifest_train_{args.video_size}.jsonl").resolve())
    val_manifest = str((dataset_root / f"manifest_val_{args.video_size}.jsonl").resolve())

    if ctx.is_main:
        train_manifest, val_manifest = build_manifests_from_dataset(
            dataset_root=dataset_root,
            video_size=args.video_size,
            allowed_sizes=allowed_sizes,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_groups = validate_manifest(train_manifest, "train", allowed_sizes)
        val_groups = validate_manifest(val_manifest, "val", allowed_sizes)
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(
                f"Group leakage between generated manifests: overlap groups={sorted(overlap)[:8]}"
            )
        rank0_print(
            ctx,
            f"[Data] Generated manifests train={train_manifest} val={val_manifest}",
        )
    ctx.barrier()

    return train_manifest, val_manifest

class DistContext:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.accelerator = None
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        self.rank = 0
        self.world_size = 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main = True

        if args.launcher == "accelerate":
            from accelerate import Accelerator

            self.accelerator = Accelerator(
                mixed_precision="bf16",
                gradient_accumulation_steps=args.grad_accum,
            )
            self.device = self.accelerator.device
            self.rank = self.accelerator.process_index
            self.world_size = self.accelerator.num_processes
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.is_main = self.accelerator.is_main_process
            if self.device.type == "cuda":
                torch.cuda.set_device(self.local_rank)
        else:
            if "RANK" in os.environ and not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                self.is_main = self.rank == 0
            if self.device.type == "cuda":
                torch.cuda.set_device(self.local_rank)

    def barrier(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        elif dist.is_initialized():
            dist.barrier()

    def unwrap(self, model: nn.Module) -> nn.Module:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        if isinstance(model, DDP):
            return model.module
        return model

    def backward(self, loss: torch.Tensor) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def clip_grad_norm(self, parameters: Iterable[torch.Tensor], max_norm: float) -> None:
        if max_norm <= 0:
            return
        if self.accelerator is not None:
            self.accelerator.clip_grad_norm_(parameters, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def sync_gradients(self) -> bool:
        if self.accelerator is not None:
            return self.accelerator.sync_gradients
        return True

    def should_no_sync(self, micro_step: int, model: nn.Module, grad_accum: int) -> bool:
        if self.accelerator is not None:
            return False
        if not isinstance(model, DDP):
            return False
        return ((micro_step + 1) % grad_accum) != 0


def rank0_print(ctx: DistContext, msg: str) -> None:
    if ctx.is_main:
        print(msg, flush=True)


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def read_pair_to_device(item: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    in_path = str(item["input_path"])
    tgt_path = str(item["target_path"])

    lq, _, _ = read_video(in_path, output_format="TCHW", pts_unit="sec")
    gt, _, _ = read_video(tgt_path, output_format="TCHW", pts_unit="sec")

    if lq.shape[0] != gt.shape[0]:
        t = min(lq.shape[0], gt.shape[0])
        lq = lq[:t]
        gt = gt[:t]

    lq = lq.to(device=device, dtype=torch.bfloat16).div_(255.0).mul_(2.0).sub_(1.0)
    gt = gt.to(device=device, dtype=torch.bfloat16).div_(255.0).mul_(2.0).sub_(1.0)

    lq = rearrange(lq, "t c h w -> c t h w").contiguous()
    gt = rearrange(gt, "t c h w -> c t h w").contiguous()

    return lq, gt


def infer_na_module(config) -> object:
    path = str(config.dit.model.__object__.path)
    if "dit_v2" in path:
        from models.dit_v2 import na as na_ops
    else:
        from models.dit import na as na_ops
    return na_ops


def enable_runtime_gradient_checkpointing() -> List[str]:
    """Patch NaDiT runtime checkpoint wrapper used in original repo.

    The upstream helper is a no-op for inference; here we swap it with
    torch checkpoint to lower training activation memory without editing
    original model files.
    """
    from torch.utils.checkpoint import checkpoint

    def _gc(module, *args, enabled: bool, **kwargs):
        if enabled and torch.is_grad_enabled():
            def _forward(*inner_args):
                return module(*inner_args, **kwargs)

            return checkpoint(_forward, *args, use_reentrant=False)
        return module(*args, **kwargs)

    patched: List[str] = []
    try:
        from models.dit import nadit as nadit_v1

        nadit_v1.gradient_checkpointing = _gc
        patched.append("models.dit.nadit")
    except Exception:
        pass

    try:
        from models.dit_v2 import nadit as nadit_v2

        nadit_v2.gradient_checkpointing = _gc
        patched.append("models.dit_v2.nadit")
    except Exception:
        pass

    return patched


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10((data_range**2) / (mse + 1e-8))



def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def pointwise_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    charbonnier_eps: float,
    huber_delta: float,
) -> torch.Tensor:
    if loss_type == "charbonnier":
        return charbonnier_loss(pred, target, eps=charbonnier_eps)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=huber_delta)
    return F.l1_loss(pred, target)


def temporal_diff_loss(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    loss_type: str,
    charbonnier_eps: float,
    huber_delta: float,
) -> torch.Tensor:
    # pred/gt shape: [T, C, H, W]
    if pred_frames.shape[0] < 2:
        return torch.zeros((), device=pred_frames.device, dtype=torch.float32)

    pred_dt = pred_frames[1:] - pred_frames[:-1]
    gt_dt = gt_frames[1:] - gt_frames[:-1]
    return pointwise_recon_loss(
        pred_dt,
        gt_dt,
        loss_type=loss_type,
        charbonnier_eps=charbonnier_eps,
        huber_delta=huber_delta,
    )

def sync_sum_and_count(value_sum: float, count: float, device: torch.device) -> Tuple[float, float]:
    t = torch.tensor([value_sum, count], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item())


def build_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sample_training_timestep(
    runner: VideoDiffusionInfer,
    latent_shape: torch.Tensor,
    device: torch.device,
    loc: float,
    scale: float,
) -> torch.Tensor:
    z = torch.randn((1,), device=device) * scale + loc
    t = torch.sigmoid(z) * float(runner.schedule.T)
    return runner.timestep_transform(t, latent_shape)


def decode_latents_with_grad(runner: VideoDiffusionInfer, latents: List[torch.Tensor]) -> List[torch.Tensor]:
    # runner.vae_decode is decorated with @torch.no_grad().
    # This version keeps graph for prediction branch while VAE parameters remain frozen.
    outputs: List[torch.Tensor] = []
    device = next(runner.vae.parameters()).device
    dtype = getattr(torch, runner.config.vae.dtype)

    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)

    if isinstance(scale, (list, tuple)):
        scale = torch.tensor(scale, device=device, dtype=dtype)
    if isinstance(shift, (list, tuple)):
        shift = torch.tensor(shift, device=device, dtype=dtype)

    for latent in latents:
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / scale + shift
        latent = rearrange(latent.unsqueeze(0), "b ... c -> b c ...")
        latent = latent.squeeze(2)
        sample = runner.vae.decode(latent).sample
        if hasattr(runner.vae, "postprocess"):
            sample = runner.vae.postprocess(sample)
        outputs.append(sample.squeeze(0))
    return outputs


def prepare_text_embeddings(na_ops, pos_emb_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_emb = torch.load(pos_emb_path, map_location="cpu")
    if not torch.is_tensor(pos_emb):
        raise TypeError(f"pos_emb must be a tensor, got {type(pos_emb)}")
    pos_emb = pos_emb.to(device=device, dtype=torch.float32)
    txt, txt_shape = na_ops.flatten([pos_emb])
    return txt, txt_shape


def build_condition_latent(
    runner: VideoDiffusionInfer,
    lq_latent: torch.Tensor,
    latent_shape: torch.Tensor,
    cond_noise_scale: float,
) -> torch.Tensor:
    if cond_noise_scale > 0:
        aug_noise = torch.randn_like(lq_latent)
        t_cond = torch.tensor([runner.schedule.T * cond_noise_scale], device=lq_latent.device)
        t_cond = runner.timestep_transform(t_cond, latent_shape)
        lq_blur = runner.schedule.forward(lq_latent, aug_noise, t_cond)
    else:
        lq_blur = lq_latent

    dummy = torch.zeros_like(lq_blur)
    return runner.get_condition(dummy, latent_blur=lq_blur, task="sr")


def compute_losses_for_sample(
    *,
    runner: VideoDiffusionInfer,
    dit_model: nn.Module,
    na_ops,
    lq_video: torch.Tensor,
    gt_video: torch.Tensor,
    txt_embed: torch.Tensor,
    txt_shape: torch.Tensor,
    lpips_model: Optional[nn.Module],
    args: argparse.Namespace,
    eval_mode: bool,
) -> LossPack:
    device = lq_video.device

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lq_latent, gt_latent = runner.vae_encode([lq_video, gt_video])

    latent_shape = torch.tensor(gt_latent.shape[:-1], device=device, dtype=torch.long).unsqueeze(0)

    if eval_mode:
        t = torch.tensor([float(runner.schedule.T)], device=device)
        t = runner.timestep_transform(t, latent_shape)
    else:
        t = sample_training_timestep(
            runner=runner,
            latent_shape=latent_shape,
            device=device,
            loc=args.time_loc,
            scale=args.time_scale,
        )

    noise = torch.randn_like(gt_latent)
    x_t = runner.schedule.forward(gt_latent, noise, t)
    target_v = noise - gt_latent

    cond_scale = args.cond_noise_scale
    if cond_scale < 0:
        cond_scale = float(runner.config.condition.get("noise_scale", 0.0))
    condition = build_condition_latent(
        runner=runner,
        lq_latent=lq_latent,
        latent_shape=latent_shape,
        cond_noise_scale=cond_scale,
    )

    x_t_flat, vid_shape = na_ops.flatten([x_t])
    cond_flat, _ = na_ops.flatten([condition])
    target_v_flat, _ = na_ops.flatten([target_v])

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_v = dit_model(
            vid=torch.cat([x_t_flat, cond_flat], dim=-1),
            txt=txt_embed,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            timestep=t,
        ).vid_sample

        latent_mse = F.mse_loss(pred_v.float(), target_v_flat.float())

    need_recon_branch = eval_mode or (args.w_pix > 0.0) or (args.w_ssim > 0.0) or (args.w_temp > 0.0) or (args.w_lpips > 0.0)

    if need_recon_branch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_x0_flat, _ = runner.schedule.convert_from_pred(
                pred=pred_v,
                pred_type=PredictionType.v_lerp,
                x_t=x_t_flat,
                t=t,
            )

            pred_x0 = na_ops.unflatten(pred_x0_flat, vid_shape)[0]
            pred_dec = decode_latents_with_grad(runner, [pred_x0])[0]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gt_dec = runner.vae_decode([gt_latent])[0]

        pred_frames = rearrange(pred_dec, "c t h w -> t c h w").clamp(-1, 1)
        gt_frames = rearrange(gt_dec, "c t h w -> t c h w").clamp(-1, 1)

        pixel_loss = pointwise_recon_loss(
            pred_frames,
            gt_frames,
            loss_type=args.pixel_loss,
            charbonnier_eps=args.charbonnier_eps,
            huber_delta=args.huber_delta,
        )
        ssim_metric = ssim_fn(pred_frames.float(), gt_frames.float(), data_range=2.0, size_average=True)
        ssim_loss = 1.0 - ssim_metric
        temp_loss = temporal_diff_loss(
            pred_frames,
            gt_frames,
            loss_type=args.pixel_loss,
            charbonnier_eps=args.charbonnier_eps,
            huber_delta=args.huber_delta,
        )

        need_lpips = (args.w_lpips > 0.0) or (eval_mode and (lpips_model is not None))
        if need_lpips:
            if lpips_model is None:
                raise RuntimeError("LPIPS model is required when --w_lpips > 0.")
            lpips_metric = lpips_model(pred_frames.float(), gt_frames.float()).mean()
            lpips_loss = lpips_metric
        else:
            lpips_metric = torch.zeros((), device=device, dtype=torch.float32)
            lpips_loss = torch.zeros((), device=device, dtype=torch.float32)

        psnr = compute_psnr(pred_frames.detach(), gt_frames.detach(), data_range=2.0)
    else:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        pixel_loss = zero
        ssim_metric = zero
        ssim_loss = zero
        temp_loss = zero
        lpips_metric = zero
        lpips_loss = zero
        psnr = zero

    total = (
        args.w_latent * latent_mse
        + args.w_pix * pixel_loss
        + args.w_ssim * ssim_loss
        + args.w_temp * temp_loss
        + args.w_lpips * lpips_loss
    )

    return LossPack(
        total=total,
        latent_mse=latent_mse.detach(),
        pixel_loss=pixel_loss.detach(),
        ssim_loss=ssim_loss.detach(),
        temp_loss=temp_loss.detach(),
        lpips_loss=lpips_loss.detach(),
        psnr=psnr.detach(),
        ssim_metric=ssim_metric.detach(),
        lpips_metric=lpips_metric.detach(),
    )

@torch.no_grad()
def evaluate(
    *,
    ctx: DistContext,
    runner: VideoDiffusionInfer,
    dit_model: nn.Module,
    na_ops,
    val_loader: DataLoader,
    txt_embed: torch.Tensor,
    txt_shape: torch.Tensor,
    lpips_model: Optional[nn.Module],
    compute_lpips: bool,
    args: argparse.Namespace,
) -> Dict[str, float]:
    was_training = dit_model.training
    dit_model.eval()

    sums = {"psnr": 0.0, "ssim": 0.0}
    if compute_lpips:
        sums["lpips"] = 0.0
    count = 0.0

    for batch in val_loader:
        for item in batch:
            lq, gt = read_pair_to_device(item, ctx.device)
            loss_pack = compute_losses_for_sample(
                runner=runner,
                dit_model=dit_model,
                na_ops=na_ops,
                lq_video=lq,
                gt_video=gt,
                txt_embed=txt_embed,
                txt_shape=txt_shape,
                lpips_model=lpips_model if compute_lpips else None,
                args=args,
                eval_mode=True,
            )
            sums["psnr"] += float(loss_pack.psnr.item())
            sums["ssim"] += float(loss_pack.ssim_metric.item())
            if compute_lpips:
                sums["lpips"] += float(loss_pack.lpips_metric.item())
            count += 1.0

    totals = {}
    for key, value_sum in sums.items():
        total_sum, total_count = sync_sum_and_count(value_sum, count, ctx.device)
        totals[key] = total_sum / max(1.0, total_count)
    if not compute_lpips:
        totals["lpips"] = float("nan")

    if was_training:
        dit_model.train()

    return totals

def maybe_get_lpips_model(ctx: DistContext, cache: Dict[str, object], required: bool) -> Optional[nn.Module]:
    if not required:
        return None
    model = cache.get("model")
    if model is not None:
        return model  # type: ignore[return-value]
    if bool(cache.get("unavailable", False)):
        return None

    try:
        import lpips as lpips_lib
    except Exception as exc:  # pragma: no cover
        cache["unavailable"] = True
        rank0_print(ctx, f"[WARN] LPIPS unavailable ({exc}); LPIPS metric/loss will be skipped.")
        return None

    if ctx.world_size > 1:
        if ctx.is_main:
            _lpips_bootstrap = lpips_lib.LPIPS(net="vgg")
            del _lpips_bootstrap
        ctx.barrier()

    model = lpips_lib.LPIPS(net="vgg").to(ctx.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cache["model"] = model
    return model


def format_lpips_metric(lpips_value: float) -> str:
    if math.isnan(lpips_value):
        return "skip"
    return f"{lpips_value:.6f}"

def save_checkpoint(
    *,
    ctx: DistContext,
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    best_psnr: float,
    current_metrics: Optional[Dict[str, float]],
    is_best: bool,
) -> None:
    if not ctx.is_main:
        return

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints" / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_model = ctx.unwrap(model)
    metadata = {
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lora_dropout": str(args.lora_dropout),
        "step": str(step),
    }

    save_lora_safetensors(raw_model, str(ckpt_dir / "lora.safetensors"), metadata=metadata)

    state = {
        "step": step,
        "best_psnr": best_psnr,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "metrics": current_metrics or {},
    }
    torch.save(state, ckpt_dir / "training_state.pt")

    save_lora_safetensors(raw_model, str(output_dir / "last_lora.safetensors"), metadata=metadata)
    torch.save(state, output_dir / "last_training_state.pt")

    if is_best:
        save_lora_safetensors(raw_model, str(output_dir / "best_lora.safetensors"), metadata=metadata)
        with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(current_metrics or {}, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    os.makedirs(args.output_dir, exist_ok=True)
    ctx = DistContext(args)
    set_seed(args.seed, ctx.rank)

    writer = SummaryWriter(args.output_dir) if ctx.is_main else None

    rank0_print(ctx, f"[Init] launcher={args.launcher} world_size={ctx.world_size} device={ctx.device}")
    rank0_print(
        ctx,
        "[Loss] "
        + f"mode={args.train_mode} pixel_loss={args.pixel_loss} "
        + f"w_latent={args.w_latent:.3f} w_pix={args.w_pix:.3f} "
        + f"w_ssim={args.w_ssim:.3f} w_temp={args.w_temp:.3f} "
        + f"w_lpips={args.w_lpips:.3f} eval_lpips={args.eval_lpips}",
    )

    resolved_manifest_train, resolved_manifest_val = resolve_manifest_paths(ctx, args)
    args.manifest_train = resolved_manifest_train
    args.manifest_val = resolved_manifest_val

    config = load_config(args.config)
    patched_modules = []
    if not args.disable_runtime_checkpointing:
        patched_modules = enable_runtime_gradient_checkpointing()
    if (not args.disable_runtime_checkpointing) and patched_modules:
        rank0_print(ctx, f"[Memory] Runtime checkpoint patched modules: {patched_modules}")
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)

    runner.configure_dit_model(device="cuda", checkpoint=args.base_ckpt)
    if (not args.disable_runtime_checkpointing) and hasattr(runner.dit, "set_gradient_checkpointing"):
        runner.dit.set_gradient_checkpointing(True)
    elif (not args.disable_runtime_checkpointing) and hasattr(runner.dit, "gradient_checkpointing"):
        runner.dit.gradient_checkpointing = True
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit") and hasattr(runner.config.vae, "memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.configure_diffusion()

    runner.vae.requires_grad_(False).eval()
    na_ops = infer_na_module(config)

    replaced = inject_lora_layers(
        runner.dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_keywords=args.lora_target,
    )
    freeze_non_lora_parameters(runner.dit)

    if args.resume_lora:
        load_lora_safetensors(runner.dit, args.resume_lora, strict=False)
        rank0_print(ctx, f"[Resume] Loaded LoRA from {args.resume_lora}")

    # LoRA modules are injected at runtime; ensure all parameters stay on the rank device.
    runner.dit.to(ctx.device)

    trainable_params = [p for p in runner.dit.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable LoRA parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
    )
    scheduler = build_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    train_dataset = ManifestVideoPairDataset(args.manifest_train)
    val_dataset = ManifestVideoPairDataset(args.manifest_val)

    train_sampler = None
    val_sampler = None
    if args.launcher == "ddp" and ctx.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    collate_fn = lambda batch: batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    lpips_cache: Dict[str, object] = {"model": None, "unavailable": False}
    lpips_model_train = maybe_get_lpips_model(ctx, lpips_cache, required=(args.w_lpips > 0.0))
    if (args.w_lpips > 0.0) and (lpips_model_train is None):
        raise RuntimeError(
            "w_lpips > 0 requires LPIPS dependency/model. Install lpips and retry, or set --w_lpips 0."
        )

    runner.dit.train()

    if args.launcher == "accelerate":
        runner.dit, optimizer, train_loader, val_loader = ctx.accelerator.prepare(
            runner.dit, optimizer, train_loader, val_loader
        )
    elif args.launcher == "ddp" and ctx.world_size > 1:
        auto_find_unused = (args.w_pix <= 0.0) and (args.w_ssim <= 0.0) and (args.w_temp <= 0.0) and (args.w_lpips <= 0.0)
        find_unused = args.ddp_find_unused_parameters or auto_find_unused
        if find_unused:
            rank0_print(
                ctx,
                f"[DDP] find_unused_parameters=True (manual={args.ddp_find_unused_parameters}, auto={auto_find_unused})",
            )
        runner.dit = DDP(
            runner.dit,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=find_unused,
        )

    txt_embed, txt_shape = prepare_text_embeddings(na_ops, args.pos_emb, ctx.device)

    rank0_print(ctx, f"[LoRA] Replaced linear layers: {len(replaced)}")
    rank0_print(
        ctx,
        f"[Params] total={get_total_parameter_count(ctx.unwrap(runner.dit)):,} "
        f"trainable={get_trainable_parameter_count(ctx.unwrap(runner.dit)):,}",
    )

    global_step = 0
    best_psnr = float("-inf")
    micro_step = 0
    epoch = 0
    train_start = time.time()
    pbar = tqdm(
        total=args.max_steps,
        initial=global_step,
        desc="LoRA Train",
        dynamic_ncols=True,
        disable=not ctx.is_main,
    )

    while global_step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            no_sync = ctx.should_no_sync(micro_step, runner.dit, args.grad_accum)
            sync_context = runner.dit.no_sync() if no_sync else contextlib.nullcontext()

            with sync_context:
                if ctx.accelerator is not None:
                    accumulate_ctx = ctx.accelerator.accumulate(runner.dit)
                else:
                    accumulate_ctx = contextlib.nullcontext()

                with accumulate_ctx:
                    per_item_losses: List[LossPack] = []
                    for item in batch:
                        lq, gt = read_pair_to_device(item, ctx.device)
                        losses = compute_losses_for_sample(
                            runner=runner,
                            dit_model=runner.dit,
                            na_ops=na_ops,
                            lq_video=lq,
                            gt_video=gt,
                            txt_embed=txt_embed,
                            txt_shape=txt_shape,
                            lpips_model=lpips_model_train,
                            args=args,
                            eval_mode=False,
                        )
                        per_item_losses.append(losses)

                    total_loss = torch.stack([l.total for l in per_item_losses]).mean()
                    scaled_loss = total_loss / float(args.grad_accum)
                    ctx.backward(scaled_loss)

                    should_step = False
                    if ctx.accelerator is not None:
                        should_step = ctx.sync_gradients()
                    else:
                        should_step = ((micro_step + 1) % args.grad_accum) == 0

                    if should_step:
                        ctx.clip_grad_norm(trainable_params, args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1

                        mean_latent = torch.stack([l.latent_mse for l in per_item_losses]).mean()
                        mean_pix = torch.stack([l.pixel_loss for l in per_item_losses]).mean()
                        mean_ssim_loss = torch.stack([l.ssim_loss for l in per_item_losses]).mean()
                        mean_temp_loss = torch.stack([l.temp_loss for l in per_item_losses]).mean()
                        mean_lpips_loss = torch.stack([l.lpips_loss for l in per_item_losses]).mean()
                        mean_psnr = torch.stack([l.psnr for l in per_item_losses]).mean()
                        mean_ssim_metric = torch.stack([l.ssim_metric for l in per_item_losses]).mean()
                        mean_lpips_metric = torch.stack([l.lpips_metric for l in per_item_losses]).mean()

                        if ctx.is_main:
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "step": f"{global_step}/{args.max_steps}",
                                    "loss": f"{total_loss.item():.4f}",
                                    "psnr": f"{mean_psnr.item():.3f}",
                                    "ssim": f"{mean_ssim_metric.item():.4f}",
                                    "temp": f"{mean_temp_loss.item():.4f}",
                                }
                            )

                        if global_step % args.log_every == 0:
                            rank0_print(
                                ctx,
                                "[Train] "
                                f"step={global_step}/{args.max_steps} "
                                f"loss={total_loss.item():.6f} "
                                f"latent={mean_latent.item():.6f} "
                                f"pix={mean_pix.item():.6f} "
                                f"ssim_loss={mean_ssim_loss.item():.6f} "
                                f"temp_loss={mean_temp_loss.item():.6f} "
                                f"lpips_loss={mean_lpips_loss.item():.6f} "
                                f"psnr={mean_psnr.item():.4f} "
                                f"ssim={mean_ssim_metric.item():.6f} "
                                f"lpips={mean_lpips_metric.item():.6f} "
                                f"lr={scheduler.get_last_lr()[0]:.6e}"
                            )

                        if writer is not None:
                            writer.add_scalar("train/loss_total", total_loss.item(), global_step)
                            writer.add_scalar("train/loss_latent", mean_latent.item(), global_step)
                            writer.add_scalar("train/loss_pix", mean_pix.item(), global_step)
                            writer.add_scalar("train/loss_ssim", mean_ssim_loss.item(), global_step)
                            writer.add_scalar("train/loss_temp", mean_temp_loss.item(), global_step)
                            writer.add_scalar("train/loss_lpips", mean_lpips_loss.item(), global_step)
                            writer.add_scalar("train/psnr", mean_psnr.item(), global_step)
                            writer.add_scalar("train/ssim", mean_ssim_metric.item(), global_step)
                            writer.add_scalar("train/lpips_metric", mean_lpips_metric.item(), global_step)
                            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                        do_eval = (args.eval_every > 0) and (global_step % args.eval_every == 0)
                        metrics = None
                        if do_eval:
                            ctx.barrier()
                            lpips_model_eval = maybe_get_lpips_model(
                                ctx,
                                lpips_cache,
                                required=bool(args.eval_lpips),
                            )
                            compute_lpips_eval = bool(args.eval_lpips) and (lpips_model_eval is not None)
                            metrics = evaluate(
                                ctx=ctx,
                                runner=runner,
                                dit_model=runner.dit,
                                na_ops=na_ops,
                                val_loader=val_loader,
                                txt_embed=txt_embed,
                                txt_shape=txt_shape,
                                lpips_model=lpips_model_eval,
                                compute_lpips=compute_lpips_eval,
                                args=args,
                            )
                            rank0_print(
                                ctx,
                                "[Eval] "
                                f"step={global_step} "
                                f"PSNR={metrics['psnr']:.4f} "
                                f"SSIM={metrics['ssim']:.6f} "
                                f"LPIPS={format_lpips_metric(metrics['lpips'])}",
                            )
                            if writer is not None:
                                writer.add_scalar("eval/psnr", metrics["psnr"], global_step)
                                writer.add_scalar("eval/ssim", metrics["ssim"], global_step)
                                if not math.isnan(metrics["lpips"]):
                                    writer.add_scalar("eval/lpips", metrics["lpips"], global_step)

                            if metrics["psnr"] > best_psnr:
                                best_psnr = metrics["psnr"]
                                save_checkpoint(
                                    ctx=ctx,
                                    args=args,
                                    model=runner.dit,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    step=global_step,
                                    best_psnr=best_psnr,
                                    current_metrics=metrics,
                                    is_best=True,
                                )

                        do_save = (args.save_every > 0) and (global_step % args.save_every == 0)
                        if do_save:
                            save_checkpoint(
                                ctx=ctx,
                                args=args,
                                model=runner.dit,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=global_step,
                                best_psnr=best_psnr,
                                current_metrics=metrics,
                                is_best=False,
                            )

            micro_step += 1

    lpips_model_final = maybe_get_lpips_model(
        ctx,
        lpips_cache,
        required=bool(args.eval_lpips),
    )
    compute_lpips_final = bool(args.eval_lpips) and (lpips_model_final is not None)
    final_metrics = evaluate(
        ctx=ctx,
        runner=runner,
        dit_model=runner.dit,
        na_ops=na_ops,
        val_loader=val_loader,
        txt_embed=txt_embed,
        txt_shape=txt_shape,
        lpips_model=lpips_model_final,
        compute_lpips=compute_lpips_final,
        args=args,
    )

    if final_metrics["psnr"] > best_psnr:
        best_psnr = final_metrics["psnr"]
        save_checkpoint(
            ctx=ctx,
            args=args,
            model=runner.dit,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            best_psnr=best_psnr,
            current_metrics=final_metrics,
            is_best=True,
        )

    save_checkpoint(
        ctx=ctx,
        args=args,
        model=runner.dit,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        best_psnr=best_psnr,
        current_metrics=final_metrics,
        is_best=False,
    )
    if ctx.is_main:
        pbar.close()

    if writer is not None:
        writer.close()

    elapsed = time.time() - train_start
    rank0_print(
        ctx,
        "[Done] "
        f"steps={global_step} best_psnr={best_psnr:.4f} "
        f"elapsed={elapsed / 60.0:.2f} min",
    )

    ctx.barrier()


if __name__ == "__main__":
    main()















































