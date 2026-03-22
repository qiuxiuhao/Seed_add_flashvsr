from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SeedVR2 ours-base inference with external ROLA finetune adapter"
    )
    parser.add_argument("--adapter_path", type=str, required=True, help="ROLA adapter directory or adapter file path")

    # Keep the same interface as projects/inference_seedvr2_7b_ours_base.py
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    return parser.parse_args()


def _to_str_tuple(values: Sequence[str] | None, fallback: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(str(v) for v in fallback)
    return tuple(str(v) for v in values)


def _load_adapter_cfg(adapter_path: Path) -> dict:
    if not adapter_path.is_dir():
        return {}

    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Invalid adapter config format: {type(data)}")
    return data


def main() -> None:
    args = parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_path does not exist: {adapter_path}")

    # Keep base flow untouched; inject adapter right after runner is created.
    from rola_finetune.rola import (
        DEFAULT_TARGET_KEYWORDS,
        DEFAULT_TARGET_SCOPES,
        inject_rola_layers,
        load_adapter,
    )
    import projects.inference_seedvr2_7b_ours_base as base_infer

    runner = base_infer.configure_runner(args.sp_size)

    cfg = _load_adapter_cfg(adapter_path)
    rank = int(cfg.get("rank", 16))
    alpha = float(cfg.get("alpha", 32.0))
    dropout = float(cfg.get("dropout", 0.05))
    target_keywords = _to_str_tuple(cfg.get("target_keywords"), DEFAULT_TARGET_KEYWORDS)
    target_scopes = _to_str_tuple(cfg.get("target_scopes"), DEFAULT_TARGET_SCOPES)

    replaced = inject_rola_layers(
        runner.dit,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_keywords=target_keywords,
        target_scopes=target_scopes,
    )
    if len(replaced) == 0:
        raise RuntimeError("No ROLA target layers found during adapter injection")

    load_adapter(str(adapter_path), runner.dit, strict=False)
    print(
        f"[ROLA] injected={len(replaced)} adapter={adapter_path} "
        f"rank={rank} alpha={alpha} dropout={dropout}"
    )

    base_infer.generation_loop(
        runner,
        video_path=args.video_path,
        output_dir=args.output_dir,
        seed=args.seed,
        res_h=args.res_h,
        res_w=args.res_w,
        sp_size=args.sp_size,
        out_fps=args.out_fps,
    )


if __name__ == "__main__":
    main()
