#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import math
import os

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from diffsynth import FlashVSRFullPipeline, ModelManager
from utils.utils import Causal_LQ4x_Proj

# Avoid OpenMP over-subscription in container inference.
os.environ["OMP_NUM_THREADS"] = "1"


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


def init_pipeline():
    print(f"Device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models(
        [
            "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
            "./FlashVSR-v1.1/Wan2.1_VAE.pth",
        ]
    )
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        "cuda", dtype=torch.bfloat16
    )
    lq_proj_in_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(lq_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_proj_in_path, map_location="cpu"), strict=True)

    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR v1.1 full inference")
    parser.add_argument("--input_root", type=str, default="./test_data/test-input-codabench")
    parser.add_argument("--result_root", type=str, default="./test_results")
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
    pipe = init_pipeline()

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
                lq,
                th,
                tw,
                num_frames,
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
            LQ_video=lq,
            num_frames=num_frames,
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
