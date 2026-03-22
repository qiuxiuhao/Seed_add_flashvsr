from __future__ import annotations

import argparse
from pathlib import Path

from rola_finetune.rola import inject_rola_layers, load_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeedVR2 inference with external ROLA adapter")
    parser.add_argument("--adapter_path", type=str, required=True, help="Adapter directory or file path")

    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Keep the original inference implementation unchanged.
    import projects.inference_seedvr2_7b_ours as base_infer

    runner = base_infer.configure_runner(args.sp_size)

    adapter_path = Path(args.adapter_path)
    cfg = {}
    cfg_path = adapter_path / "adapter_config.json" if adapter_path.is_dir() else None
    if cfg_path and cfg_path.exists():
        import json

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    rank = int(cfg.get("rank", 16))
    alpha = float(cfg.get("alpha", 32.0))
    dropout = float(cfg.get("dropout", 0.05))
    target_keywords = tuple(cfg.get("target_keywords", ["proj_qkv", "proj_out", "proj_in"]))
    target_scopes = tuple(cfg.get("target_scopes", ["attn", "mlp"]))

    replaced = inject_rola_layers(
        runner.dit,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_keywords=target_keywords,
        target_scopes=target_scopes,
    )
    if len(replaced) == 0:
        raise RuntimeError("No ROLA target layers found during inference adapter injection")

    load_adapter(str(adapter_path), runner.dit, strict=False)
    print(f"[ROLA] injected={len(replaced)} layers, adapter={adapter_path}")

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
