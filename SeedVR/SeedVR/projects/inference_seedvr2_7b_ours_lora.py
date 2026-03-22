# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
import torch
import mediapy
from einops import rearrange
from omegaconf import OmegaConf
print(os.getcwd())
import datetime
from tqdm import tqdm
from models.dit import na
import gc

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not avaliable!')
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
from torchvision.io import read_image


from common.distributed import (
    get_device,
    init_torch,
)

from common.distributed.advanced import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
)

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.distributed.ops import sync_data
from common.seed import set_seed
from common.partition import partition_by_groups, partition_by_size
import argparse
# LoRA 相关工具：只负责注入/加载微调权重，不改动原有推理流程。
from universal_lora_normal.lora_modules import (
    inject_lora_layers,
    load_lora_safetensors,
    merge_lora_weights,
)

def configure_sequence_parallel(sp_size):
    if sp_size > 1:
        init_sequence_parallel(sp_size)

def is_image_file(filename):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_exts

def configure_runner(sp_size):
    config_path = os.path.join('./configs_7b', 'main.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    configure_sequence_parallel(sp_size)
    runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr2_ema_7b.pth')
    runner.configure_vae_model()
    # Set memory limit.
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner

def generation_step(runner, text_embeds_dict, cond_latents):
    def _move_to_cuda(x):
        return [i.to(get_device()) for i in x]

    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
    noises, aug_noises, cond_latents = list(
        map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents))
    )
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        t = (
            torch.tensor([1000.0], device=get_device())
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        print(
            f"Timestep shifting from"
            f" {1000.0 * cond_noise_scale} to {t}."
        )
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors

    return samples

# ======================================================================
# [NEW CODE START] 辅助高斯遮罩函数
# ======================================================================
def get_gaussian_mask(tile_h, tile_w, sigma=0.5):
    """生成2D高斯权重遮罩，实现平滑边缘融合"""
    y = torch.linspace(-1, 1, tile_h)
    x = torch.linspace(-1, 1, tile_w)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    mask = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
# ======================================================================
# [NEW CODE END]
# ======================================================================

def generation_loop(runner, video_path='./test_videos', output_dir='./results', batch_size=1, cfg_scale=1.0, cfg_rescale=0.0, sample_steps=50, seed=666, res_h=1280, res_w=720, sp_size=1, out_fps=None):

    def _build_pos_and_neg_prompt():
        positive_text = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, \
        hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, \
        skin pore detailing, hyper sharpness, perfect without deformations."
        negative_text = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, \
        CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, \
        signature, jpeg artifacts, deformed, lowres, over-smooth"
        return positive_text, negative_text

    def _build_test_prompts(video_path):
        positive_text, negative_text = _build_pos_and_neg_prompt()
        original_videos = []
        prompts = {}
        video_list = os.listdir(video_path)
        for f in video_list:
            original_videos.append(f)
            prompts[f] = positive_text
        print(f"Total prompts to be generated: {len(original_videos)}")
        return original_videos, prompts, negative_text

    def _extract_text_embeds():
        positive_prompts_embeds = []
        for texts_pos in tqdm(original_videos_local):
            text_pos_embeds = torch.load('pos_emb.pt')
            text_neg_embeds = torch.load('neg_emb.pt')

            positive_prompts_embeds.append(
                {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
            )
        gc.collect()
        torch.cuda.empty_cache()
        return positive_prompts_embeds

    def cut_videos(videos, sp_size):
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            print(f"Cut input video size: {videos.size()}")
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        else:
            padding = [videos[:, -1].unsqueeze(1)] * (
                4 * sp_size - ((t - 1) % (4 * sp_size))
            )
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            assert (videos.size(1) - 1) % (4 * sp_size) == 0
            return videos

    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()

    set_seed(seed, same_across_ranks=True)
    os.makedirs(output_dir, exist_ok=True)
    tgt_path = output_dir

    original_videos, _, _ = _build_test_prompts(video_path)

    original_videos_group = partition_by_groups(
        original_videos,
        get_data_parallel_world_size() // get_sequence_parallel_world_size(),
    )
    original_videos_local = original_videos_group[
        get_data_parallel_rank() // get_sequence_parallel_world_size()
    ]
    original_videos_local = partition_by_size(original_videos_local, batch_size)

    positive_prompts_embeds = _extract_text_embeds()

    # ======================================================================
    # [ORIGINAL CODE COMMENTED OUT] 废弃基于暴力缩放裁剪的Transform
    # ======================================================================
    # video_transform = Compose(
    #     [
    #         NaResize(
    #             resolution=(
    #                 res_h * res_w
    #             )
    #             ** 0.5,
    #             mode="area",
    #             downsample_only=False,
    #         ),
    #         Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
    #         DivisibleCrop((16, 16)),
    #         Normalize(0.5, 0.5),
    #         Rearrange("t c h w -> c t h w"),
    #     ]
    # )
    # ======================================================================

    # ======================================================================
    # [NEW CODE START] 替换为纯净版Transform，只负责维度重排和归一化
    # ======================================================================
    video_transform_base = Compose([
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    # ======================================================================
    # [NEW CODE END]
    # ======================================================================

    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        # ======================================================================
        # [ORIGINAL CODE COMMENTED OUT] 废弃原有的“全图一口气推理”循环逻辑
        # ======================================================================
        # cond_latents = []
        # fps_lists = []
        # for video in videos:
        #     if is_image_file(video):
        #         video = read_image(
        #             os.path.join(video_path, video)
        #         ).unsqueeze(0) / 255.0
        #         if sp_size > 1:
        #             raise ValueError("Sp size should be set to 1 for image inputs!")
        #     else:
        #         video, _, info = read_video(
        #             os.path.join(video_path, video), output_format="TCHW"
        #             )
        #         video = video / 255.0
        #         fps_lists.append(info["video_fps"] if out_fps is None else out_fps)
        #     print(f"Read video size: {video.size()}")
        #     cond_latents.append(video_transform(video.to(get_device())))
        #
        # ori_lengths = [video.size(1) for video in cond_latents]
        # input_videos = cond_latents
        # cond_latents = [cut_videos(video, sp_size) for video in cond_latents]
        #
        # runner.dit.to("cpu")
        # print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
        # runner.vae.to(get_device())
        # cond_latents = runner.vae_encode(cond_latents)
        # runner.vae.to("cpu")
        # runner.dit.to(get_device())
        #
        # for i, emb in enumerate(text_embeds["texts_pos"]):
        #     text_embeds["texts_pos"][i] = emb.to(get_device())
        # for i, emb in enumerate(text_embeds["texts_neg"]):
        #     text_embeds["texts_neg"][i] = emb.to(get_device())
        #
        # samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
        # runner.dit.to("cpu")
        # del cond_latents
        # ======================================================================
        
        # ======================================================================
        # [NEW CODE START] 基于高斯矩阵的时空滑窗融合推理 (Tile-based Blending)
        # ======================================================================
        fps_lists = []
        ori_lengths = []
        input_videos = []
        samples = []

        # 文本向量仅需一次 to(device)
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(get_device())
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(get_device())

        for video_name in videos:
            video_full_path = os.path.join(video_path, video_name)
            if is_image_file(video_name):
                raise ValueError("Spatial tiling script tailored for videos, got image.")
            
            video, _, info = read_video(video_full_path, output_format="TCHW")
            video = video / 255.0
            fps_lists.append(info["video_fps"] if out_fps is None else out_fps)

            print(f"\n🚀 开始处理视频: {video_name}")
            ori_lengths.append(video.size(0)) # 记录真实帧数 T
            
            video_base = video_transform_base(video.to("cpu"))
            input_videos.append(video_base) 
            C, T, H, W = video_base.size()

            # ================= [修复核心] 智能识别横竖屏，动态翻转滑窗 =================
            if H > W:  # 竖屏视频 (如 1080x1920)
                tile_h, tile_w = 1024, 640
                stride_h, stride_w = 896, 448
                print("📐 识别为竖屏，滑窗方向调整为 1024x640")
            else:      # 横屏视频 (如 1920x1080)
                tile_h, tile_w = 640, 1024
                stride_h, stride_w = 448, 896
                print("📐 识别为横屏，滑窗方向调整为 640x1024")

            # 针对当前视频尺寸生成高斯遮罩
            gaussian_mask = get_gaussian_mask(tile_h, tile_w, sigma=0.5).to("cpu")
            # =====================================================================

            # 动态 Padding：保证长宽能被网格覆盖（如 1080 -> 1088）
            pad_h = (32 - (H % 32)) % 32
            pad_w = (32 - (W % 32)) % 32
            video_padded = torch.nn.functional.pad(video_base, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, pH, pW = video_padded.size()

            # 建立全局累加画布 (全部在 CPU 上操作，节约显存)
            final_pixel_video = torch.zeros((T, C, pH, pW), dtype=torch.float32, device="cpu")
            weight_video = torch.zeros((T, C, pH, pW), dtype=torch.float32, device="cpu")

            h_starts = list(range(0, pH - tile_h + 1, stride_h))
            if h_starts[-1] + tile_h < pH: h_starts.append(pH - tile_h)
            w_starts = list(range(0, pW - tile_w + 1, stride_w))
            if w_starts[-1] + tile_w < pW: w_starts.append(pW - tile_w)

            print(f"📦 空间网格规划完成: 将被切割为 {len(h_starts)}x{len(w_starts)} 块滑窗进行高斯融合")

            # 启动滑窗推理
            for hs in h_starts:
                for ws in w_starts:
                    print(f"   => 正在解码区域: H[{hs}:{hs+tile_h}], W[{ws}:{ws+tile_w}]")
                    tile = video_padded[:, :, hs:hs+tile_h, ws:ws+tile_w]
                    
                    # 触发底层自动补帧 (到 sp_size*4 的倍数)
                    tile_cut = cut_videos(tile.to(get_device()), sp_size)

                    # 安全显存挪移：编码
                    runner.dit.to("cpu")
                    runner.vae.to(get_device())
                    cond_latents = runner.vae_encode([tile_cut])
                    runner.vae.to("cpu")
                    runner.dit.to(get_device())

                    # 核心 7B DiT 推理与解码
                    sample_tiles = generation_step(runner, text_embeds, cond_latents=cond_latents)
                    runner.dit.to("cpu")
                    del cond_latents

                    out_tile = sample_tiles[0].to("cpu") # [T_padded, C, tile_h, tile_w]
                    out_tile = out_tile[:T] # 裁掉官方底层悄悄补上的冗余帧，还原真实长度

                    # 与高斯遮罩相乘，并累加回全局画布
                    final_pixel_video[:, :, hs:hs+tile_h, ws:ws+tile_w] += out_tile * gaussian_mask
                    weight_video[:, :, hs:hs+tile_h, ws:ws+tile_w] += gaussian_mask

                    torch.cuda.empty_cache()
                    gc.collect()

            # 全局归一化，消除重叠区高亮
            final_pixel_video = final_pixel_video / torch.clamp(weight_video, min=1e-8)
            # 裁掉之前打底用的 Pad 边缘，还原原始分辨率 (1080x1920)
            final_pixel_video = final_pixel_video[:, :, :H, :W]

            samples.append(final_pixel_video)
        # ======================================================================
        # [NEW CODE END]
        # ======================================================================

        # dump samples to the output directory
        if get_sequence_parallel_rank() == 0:
            for path, input, sample, ori_length, save_fps in zip(
                videos, input_videos, samples, ori_lengths, fps_lists
            ):
                if ori_length < sample.shape[0]:
                    sample = sample[:ori_length]
                filename = os.path.join(tgt_path, os.path.basename(path))
                # color fix
                input = (
                    rearrange(input[:, None], "c t h w -> t c h w")
                    if input.ndim == 3
                    else rearrange(input, "c t h w -> t c h w")
                )
                if use_colorfix:
                    sample = wavelet_reconstruction(
                        sample.to("cpu"), input[: sample.size(0)].to("cpu")
                    )
                else:
                    sample = sample.to("cpu")
                sample = (
                    rearrange(sample[:, None], "t c h w -> t h w c")
                    if sample.ndim == 3
                    else rearrange(sample, "t c h w -> t h w c")
                )
                sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
                sample = sample.to(torch.uint8).numpy()

                if sample.shape[0] == 1:
                    mediapy.write_image(filename, sample.squeeze(0))
                else:
                    mediapy.write_video(
                        filename, sample, fps=save_fps
                    )
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # 原有推理参数（保持与 base 脚本一致）
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)

    # 新增 LoRA 参数：仅用于加载微调权重
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", type=str, nargs="+", default=["proj_qkv", "proj_out"])
    parser.add_argument("--merge_lora", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    runner = configure_runner(args.sp_size)
    # 在 DiT 上注入 LoRA 结构，并加载微调权重
    replaced = inject_lora_layers(
        runner.dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_keywords=args.lora_target,
    )
    load_lora_safetensors(runner.dit, args.lora_path, strict=False)
    # 推理默认合并 LoRA，可减少额外分支开销
    if args.merge_lora:
        merge_lora_weights(runner.dit)
    print(f"[LoRA] loaded={args.lora_path} replaced_layers={len(replaced)} merged={args.merge_lora}")

    # 去掉 LoRA 专用参数，避免传给原始 generation_loop 造成参数不匹配
    infer_args = dict(vars(args))
    for key in ("lora_path", "lora_rank", "lora_alpha", "lora_dropout", "lora_target", "merge_lora"):
        infer_args.pop(key, None)
    generation_loop(runner, **infer_args)

