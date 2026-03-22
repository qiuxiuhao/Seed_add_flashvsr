from __future__ import annotations

import datetime
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange

from projects.video_diffusion_sr.infer import VideoDiffusionInfer


@dataclass
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def init_distributed(timeout_sec: int = 1800) -> DistContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training pipeline")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1 and (not dist.is_initialized()):
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_sec),
        )

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=(rank == 0),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    seed_v = int(seed) + int(rank)
    random.seed(seed_v)
    np.random.seed(seed_v)
    torch.manual_seed(seed_v)
    torch.cuda.manual_seed_all(seed_v)


def rank0_print(ctx: DistContext, msg: str) -> None:
    if ctx.is_main:
        print(msg, flush=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module  # type: ignore[return-value]
    return model


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def reduce_scalar_dict(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return metrics

    names = sorted(metrics.keys())
    vals = torch.tensor([float(metrics[k]) for k in names], device=device, dtype=torch.float64)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(names, vals)}


def cycle_dataloader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


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


def enable_runtime_gradient_checkpointing() -> List[str]:
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


def infer_na_module(config) -> object:
    path = str(config.dit.model.__object__.path)
    if "dit_v2" in path:
        from models.dit_v2 import na as na_ops
    else:
        from models.dit import na as na_ops
    return na_ops


def prepare_text_embeddings(na_ops, pos_emb_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_emb = torch.load(pos_emb_path, map_location="cpu")
    if not torch.is_tensor(pos_emb):
        raise TypeError(f"pos_emb must be a tensor, got {type(pos_emb)}")
    pos_emb = pos_emb.to(device=device, dtype=torch.float32)
    txt, txt_shape = na_ops.flatten([pos_emb])
    return txt, txt_shape


def sample_training_timestep(
    runner: VideoDiffusionInfer,
    latent_shape: torch.Tensor,
    device: torch.device,
    loc: float,
    scale: float,
) -> torch.Tensor:
    z = torch.randn((1,), device=device) * float(scale) + float(loc)
    t = torch.sigmoid(z) * float(runner.schedule.T)
    return runner.timestep_transform(t, latent_shape)


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


def decode_latents_with_grad(runner: VideoDiffusionInfer, latents: List[torch.Tensor]) -> List[torch.Tensor]:
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
