#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import math
import glob
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

from diffsynth import ModelManager, FlashVSRFullPipeline
from utils.utils import Causal_LQ4x_Proj

# Avoid OpenMP over-subscription in container inference.
os.environ["OMP_NUM_THREADS"] = "1"

LORA_DEFAULTS = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": [
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2",
    ],
}


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def smallest_8n1_geq(n: int) -> int:
    return ((n - 2) // 8 + 1) * 8 + 1


def is_video(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    arr = np.array(img, dtype=np.uint8).copy()
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # CHW in [-1, 1]
    return t.to(dtype)


def save_video(frames, save_path, fps=30, source_meta=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    codec = "libx264"
    pix_fmt = "yuv420p"

    if source_meta:
        extracted_codec = source_meta.get("codec") or source_meta.get("video_codec", "")
        if extracted_codec:
            low = extracted_codec.lower()
            if "hevc" in low or "h265" in low:
                codec = "libx265"
            elif "h264" in low:
                codec = "libx264"

        extracted_pix_fmt = source_meta.get("pix_fmt", "")
        if extracted_pix_fmt:
            pix_fmt = extracted_pix_fmt.split("(")[0].split()[0].strip()

    ffmpeg_params = ["-crf", "17", "-preset", "slow"]
    if codec == "libx265":
        ffmpeg_params.extend(["-tag:v", "hvc1"])

    writer = imageio.get_writer(
        save_path,
        fps=fps,
        codec=codec,
        pixelformat=pix_fmt,
        macro_block_size=None,
        ffmpeg_params=ffmpeg_params,
    )

    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        writer.append_data(np.array(f))

    writer.close()


def prepare_input_tensor_for_1x(path: str, dtype=torch.bfloat16, device="cuda"):
    if not is_video(path):
        raise ValueError(f"Input must be a video file, got: {path}")

    rdr = imageio.get_reader(path)
    first = Image.fromarray(rdr.get_data(0)).convert("RGB")
    w0, h0 = first.size
    print(f"[{os.path.basename(path)}] Original Resolution: {w0}x{h0}")

    meta = {}
    try:
        meta = rdr.get_meta_data()
    except Exception:
        pass

    fps_val = meta.get("fps", 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    def count_frames(reader):
        try:
            nf = meta.get("nframes", None)
            if isinstance(nf, int) and nf > 0:
                return nf
        except Exception:
            pass
        try:
            return reader.count_frames()
        except Exception:
            n = 0
            try:
                while True:
                    reader.get_data(n)
                    n += 1
            except Exception:
                return n

    total_frames = count_frames(rdr)
    if total_frames <= 0:
        rdr.close()
        raise RuntimeError(f"Cannot read frames from {path}")

    tW = max(128, int(math.ceil(w0 / 128.0)) * 128)
    tH = max(128, int(math.ceil(h0 / 128.0)) * 128)

    pad_x = (tW - w0) // 2
    pad_y = (tH - h0) // 2

    print(f"[{os.path.basename(path)}] Model Canvas: {tW}x{tH} | Padding: x={pad_x}, y={pad_y}")

    required_total = smallest_8n1_geq(total_frames + 4)
    idx = list(range(total_frames))
    padding_needed = required_total - total_frames
    for _ in range(padding_needed):
        idx.append(total_frames - 1)

    print(f"[{os.path.basename(path)}] Original Frames: {total_frames} -> Padded to {required_total} | FPS: {fps}")

    frames = []
    try:
        for i in idx:
            img = Image.fromarray(rdr.get_data(i)).convert("RGB")
            padded_img = Image.new("RGB", (tW, tH), (0, 0, 0))
            padded_img.paste(img, (pad_x, pad_y))

            lq_img = padded_img.resize((tW // 4, tH // 4), Image.Resampling.BICUBIC)
            hr_lq_img = lq_img.resize((tW, tH), Image.Resampling.BICUBIC)

            frames.append(pil_to_tensor_neg1_1(hr_lq_img, dtype, device))
    finally:
        try:
            rdr.close()
        except Exception:
            pass

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid, tH, tW, required_total, fps, w0, h0, pad_x, pad_y, meta, total_frames


def _load_lora_hparams(config_path: str | None) -> Dict[str, Any]:
    cfg = dict(LORA_DEFAULTS)
    cfg["target_modules"] = list(LORA_DEFAULTS["target_modules"])

    if not config_path:
        return cfg

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"--lora_config not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    raw: Dict[str, Any]
    if ext in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to read --lora_config yaml files.") from exc
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError("--lora_config only supports .yaml/.yml/.json")

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid lora config format: {type(raw)}")

    lora_cfg = raw.get("lora", raw)
    if not isinstance(lora_cfg, dict):
        raise ValueError("Invalid lora section in --lora_config")

    cfg["r"] = int(lora_cfg.get("r", cfg["r"]))
    cfg["alpha"] = int(lora_cfg.get("alpha", cfg["alpha"]))
    cfg["dropout"] = float(lora_cfg.get("dropout", cfg["dropout"]))

    target_modules = lora_cfg.get("target_modules", cfg["target_modules"])
    if not isinstance(target_modules, list) or len(target_modules) == 0:
        raise ValueError("lora target_modules must be a non-empty list")
    cfg["target_modules"] = target_modules

    return cfg


def _apply_lora_from_checkpoint(pipe, lora_ckpt: str, lora_config: str | None, load_lq_proj: bool):
    if not lora_ckpt:
        print("[LoRA] --lora_ckpt not provided. Running base model only.")
        return

    if os.path.isdir(lora_ckpt):
        raise ValueError(
            "--lora_ckpt points to a directory. This script only accepts train_rola checkpoint files "
            "(best.pt / step_xxx.pt), not merged export directories."
        )

    if lora_ckpt.lower().endswith(".safetensors"):
        raise ValueError(
            "--lora_ckpt should be a train_rola checkpoint (.pt), not merged diffusion safetensors."
        )

    if not os.path.isfile(lora_ckpt):
        raise FileNotFoundError(f"--lora_ckpt not found: {lora_ckpt}")

    ckpt = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format: {type(ckpt)}")

    if "lora_state" not in ckpt:
        raise KeyError("Checkpoint missing 'lora_state'. Please pass train_rola best.pt/step_xxx.pt")

    hparams = _load_lora_hparams(lora_config)

    try:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # type: ignore
    except Exception as exc:
        raise RuntimeError("peft is required for LoRA inference loading. Please install peft.") from exc

    peft_cfg = LoraConfig(
        r=hparams["r"],
        lora_alpha=hparams["alpha"],
        lora_dropout=hparams["dropout"],
        bias="none",
        target_modules=hparams["target_modules"],
    )

    pipe.dit = get_peft_model(pipe.dit, peft_cfg)
    load_ret = set_peft_model_state_dict(pipe.dit, ckpt["lora_state"])

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    if hasattr(load_ret, "missing_keys"):
        missing_keys = list(getattr(load_ret, "missing_keys") or [])
    if hasattr(load_ret, "unexpected_keys"):
        unexpected_keys = list(getattr(load_ret, "unexpected_keys") or [])

    hit = max(0, len(ckpt["lora_state"]) - len(unexpected_keys))
    print(
        f"[LoRA] loaded checkpoint={lora_ckpt}\n"
        f"       target_modules={len(hparams['target_modules'])}, r={hparams['r']}, alpha={hparams['alpha']}, dropout={hparams['dropout']}\n"
        f"       state_keys={len(ckpt['lora_state'])}, hit~={hit}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
    )

    if load_lq_proj:
        if "lq_proj_in" not in ckpt:
            raise KeyError("Checkpoint missing 'lq_proj_in' but --load_lq_proj=true")
        pipe.denoising_model().LQ_proj_in.load_state_dict(ckpt["lq_proj_in"], strict=True)
        print("[LoRA] loaded lq_proj_in from checkpoint.")
    else:
        print("[LoRA] skipped lq_proj_in loading (--load_lq_proj=false).")


def init_pipeline(lora_ckpt: str, lora_config: str | None, load_lq_proj: bool):
    print(f"Device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    diffusion_path = "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"
    vae_path = "./FlashVSR-v1.1/Wan2.1_VAE.pth"
    lq_proj_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"

    print(f"[Init] base diffusion: {diffusion_path}")
    print(f"[Init] base vae:       {vae_path}")
    print(f"[Init] lora ckpt:      {lora_ckpt if lora_ckpt else '(none)'}")
    print(f"[Init] load_lq_proj:   {load_lq_proj}")

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([diffusion_path, vae_path])

    pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        "cuda", dtype=torch.bfloat16
    )

    if os.path.exists(lq_proj_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_proj_path, map_location="cpu"), strict=True)

    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    # New: inject LoRA + (optional) LQ_proj_in checkpoint weights.
    _apply_lora_from_checkpoint(pipe, lora_ckpt=lora_ckpt, lora_config=lora_config, load_lq_proj=load_lq_proj)

    # Keep original order unchanged.
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 full inference with optional LoRA checkpoint loading")
    parser.add_argument(
        "--input_root",
        type=str,
        default="./test_data/test-input-codabench",
        help="Input dataset root. Script scans all .mp4 recursively under this directory.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="./test_results",
        help="Output root. Relative paths under input_root are preserved here.",
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        default="",
        help="Path to train_rola checkpoint (.pt), e.g., best.pt or step_xxx.pt",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default=None,
        help="Optional training config (.yaml/.json) to read LoRA hparams/target_modules",
    )
    parser.add_argument(
        "--load_lq_proj",
        type=str2bool,
        default=True,
        help="Whether to load lq_proj_in from checkpoint (default: true)",
    )
    return parser.parse_args()


def main(args):
    input_root = args.input_root
    result_root = args.result_root

    inputs = glob.glob(os.path.join(input_root, "**/*.mp4"), recursive=True)
    inputs.sort()

    if not inputs:
        print(f"No .mp4 files found in '{input_root}'")
        return

    print(f"Found {len(inputs)} videos. Starting batch processing...")

    seed, dtype, device = 0, torch.bfloat16, "cuda"
    sparse_ratio = 2.0
    pipe = init_pipeline(args.lora_ckpt, args.lora_config, args.load_lq_proj)

    for p in inputs:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        rel_path = os.path.relpath(p, input_root)
        name = os.path.basename(rel_path)
        if name.startswith("."):
            continue

        save_dir = os.path.dirname(os.path.join(result_root, rel_path))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, name)

        print(f"\n--- Processing: {rel_path} ---")
        try:
            (
                LQ,
                th,
                tw,
                F,
                fps,
                orig_w,
                orig_h,
                pad_x,
                pad_y,
                source_meta,
                orig_total_frames,
            ) = prepare_input_tensor_for_1x(p, dtype=dtype, device=device)
        except Exception as e:
            print(f"[Error] {name}: {e}")
            continue

        print(f"[{name}] Starting FlashVSR Inference...")
        video_tensor = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
            tiled=False,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )

        restored_video_frames = tensor2video(video_tensor)
        restored_video_frames = restored_video_frames[:orig_total_frames]

        final_frames = []
        for frame in restored_video_frames:
            cropped = frame.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h))
            final_frames.append(cropped)

        save_video(final_frames, save_path, fps=fps, source_meta=source_meta)
        print(f"[{name}] Completed -> {save_path}")

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main(parse_args())
