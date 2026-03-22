#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified runner for SeedVR + FlashVSR + blend workflow.")
    parser.add_argument("--mode", choices=["seed", "flashvsr", "blend"], required=True)
    parser.add_argument("--config", type=str, default="dual_vsr_profiles.json")
    parser.add_argument("--input-root", type=str, default="test-input-final/test_data")
    parser.add_argument("--result-root", type=str, default="result")

    parser.add_argument("--seed-profile", type=str, default="base")
    parser.add_argument("--flash-profile", type=str, default="full_test")
    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--seed-env", type=str, default="seedvr")
    parser.add_argument("--flash-env", type=str, default="flashvsr")
    parser.add_argument("--seed-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--flash-gpus", type=str, default="0")

    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_gpu_list(raw: str) -> list[str]:
    devices = [x.strip() for x in raw.split(",") if x.strip()]
    if not devices:
        raise ValueError("GPU list cannot be empty.")
    return devices


def load_profiles(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Profile config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Profile config must be a JSON object.")
    if "seed_profiles" not in data or "flash_profiles" not in data:
        raise ValueError("Profile config must include 'seed_profiles' and 'flash_profiles'.")
    return data


def list_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def build_cli_args(arg_map: dict[str, Any]) -> list[str]:
    cmd: list[str] = []
    for key, value in arg_map.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            cmd.extend([flag, "true" if value else "false"])
        elif isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([flag, str(item)])
        else:
            cmd.extend([flag, str(value)])
    return cmd


def format_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print(f"\n[cmd] {format_cmd(cmd)}")
    print(f"[cwd] {cwd}")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def stage_input_subset(input_root: Path, selected_files: list[Path], stage_base: Path) -> Path:
    stage_base.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix="subset_", dir=str(stage_base)))
    for src in selected_files:
        rel = src.relative_to(input_root)
        dst = stage_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return stage_root


def select_inputs_for_run(
    input_root: Path, output_root: Path, force: bool, dry_run: bool, stage_base: Path
) -> tuple[Path | None, dict[str, int], Path | None, bool]:
    all_inputs = list_videos(input_root)
    total = len(all_inputs)
    if total == 0:
        raise RuntimeError(f"No videos found under input root: {input_root}")

    if force:
        stats = {"total": total, "selected": total, "skipped_existing": 0}
        return input_root, stats, None, False

    selected = []
    for video in all_inputs:
        rel = video.relative_to(input_root)
        if not (output_root / rel).exists():
            selected.append(video)

    skipped = total - len(selected)
    stats = {"total": total, "selected": len(selected), "skipped_existing": skipped}
    if not selected:
        return None, stats, None, False

    if len(selected) == total:
        return input_root, stats, None, False

    if dry_run:
        # Dry-run should avoid file staging side effects.
        return input_root, stats, None, True

    staged = stage_input_subset(input_root, selected, stage_base)
    return staged, stats, staged, False


def run_seed_mode(
    repo_root: Path,
    args: argparse.Namespace,
    seed_profile: dict[str, Any],
    input_root: Path,
    output_root: Path,
) -> None:
    stage_base = repo_root / ".tmp_dual_vsr" / "seed"
    run_input_root, stats, staged_dir, subset_note = select_inputs_for_run(
        input_root=input_root,
        output_root=output_root,
        force=args.force,
        dry_run=args.dry_run,
        stage_base=stage_base,
    )
    print(
        f"[seed] total={stats['total']} selected={stats['selected']} "
        f"skipped_existing={stats['skipped_existing']}"
    )
    if run_input_root is None:
        print("[seed] Nothing to run.")
        return
    if subset_note:
        print("[seed] dry-run note: command preview uses full input root; real run stages only missing files.")

    seed_cwd = repo_root / "SeedVR" / "SeedVR"
    if not seed_cwd.exists():
        raise FileNotFoundError(f"SeedVR working directory not found: {seed_cwd}")

    videos = list_videos(run_input_root)
    groups: dict[Path, list[Path]] = defaultdict(list)
    for video in videos:
        rel_parent = video.parent.relative_to(run_input_root)
        groups[rel_parent].append(video)

    nproc = int(seed_profile.get("nproc_per_node", len(parse_gpu_list(args.seed_gpus))))
    entry_type = seed_profile.get("entry_type", "module")
    entry = seed_profile.get("entry")
    profile_args = dict(seed_profile.get("args", {}))
    if not entry:
        raise ValueError("Seed profile missing 'entry'.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(parse_gpu_list(args.seed_gpus))

    try:
        for rel_parent in sorted(groups.keys()):
            video_dir = run_input_root / rel_parent
            out_dir = output_root / rel_parent
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = ["conda", "run", "--no-capture-output", "-n", args.seed_env, "torchrun", f"--nproc_per_node={nproc}"]
            if entry_type == "module":
                cmd.extend(["-m", str(entry)])
            elif entry_type == "script":
                cmd.append(str(entry))
            else:
                raise ValueError(f"Unsupported seed entry_type: {entry_type}")

            cmd.extend(["--video_path", str(video_dir.resolve())])
            cmd.extend(["--output_dir", str(out_dir.resolve())])
            cmd.extend(build_cli_args(profile_args))
            run_command(cmd, cwd=seed_cwd, env=env, dry_run=args.dry_run)
    finally:
        if staged_dir is not None:
            shutil.rmtree(staged_dir, ignore_errors=True)


def run_flash_mode(
    repo_root: Path,
    args: argparse.Namespace,
    flash_profile: dict[str, Any],
    input_root: Path,
    output_root: Path,
) -> None:
    stage_base = repo_root / ".tmp_dual_vsr" / "flash"
    run_input_root, stats, staged_dir, subset_note = select_inputs_for_run(
        input_root=input_root,
        output_root=output_root,
        force=args.force,
        dry_run=args.dry_run,
        stage_base=stage_base,
    )
    print(
        f"[flash] total={stats['total']} selected={stats['selected']} "
        f"skipped_existing={stats['skipped_existing']}"
    )
    if run_input_root is None:
        print("[flash] Nothing to run.")
        return
    if subset_note:
        print("[flash] dry-run note: command preview uses full input root; real run stages only missing files.")

    flash_cwd = repo_root / "FlashVSR" / "FlashVSR" / "examples" / "WanVSR"
    if not flash_cwd.exists():
        raise FileNotFoundError(f"FlashVSR working directory not found: {flash_cwd}")

    entry = flash_profile.get("entry")
    if not entry:
        raise ValueError("Flash profile missing 'entry'.")
    profile_args = dict(flash_profile.get("args", {}))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(parse_gpu_list(args.flash_gpus))

    try:
        cmd = ["conda", "run", "--no-capture-output", "-n", args.flash_env, "python", str(entry)]
        cmd.extend(["--input_root", str(run_input_root.resolve())])
        cmd.extend(["--result_root", str(output_root.resolve())])
        cmd.extend(build_cli_args(profile_args))
        run_command(cmd, cwd=flash_cwd, env=env, dry_run=args.dry_run)
    finally:
        if staged_dir is not None:
            shutil.rmtree(staged_dir, ignore_errors=True)


def run_blend_mode(
    repo_root: Path,
    args: argparse.Namespace,
    seed_output: Path,
    flash_output: Path,
    blend_output: Path,
) -> None:
    if args.alpha is None:
        raise ValueError("--alpha is required when --mode blend.")
    cmd = [
        sys.executable,
        str((repo_root / "blend.py").resolve()),
        "--alpha",
        str(args.alpha),
        "--seed-dir",
        str(seed_output.resolve()),
        "--flash-dir",
        str(flash_output.resolve()),
        "--output-dir",
        str(blend_output.resolve()),
    ]
    if args.force:
        cmd.append("--force")
    env = os.environ.copy()
    run_command(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    config_path = resolve_path(repo_root, args.config)
    profiles = load_profiles(config_path)
    seed_profiles = profiles.get("seed_profiles", {})
    flash_profiles = profiles.get("flash_profiles", {})

    if args.seed_profile not in seed_profiles:
        raise KeyError(f"Unknown seed profile: {args.seed_profile}")
    if args.flash_profile not in flash_profiles:
        raise KeyError(f"Unknown flash profile: {args.flash_profile}")

    input_root = resolve_path(repo_root, args.input_root)
    result_root = resolve_path(repo_root, args.result_root)
    seed_output = result_root / "SeedVR"
    flash_output = result_root / "FlashVSR"
    blend_output = result_root / "SeedVR_blend_FlashVSR"

    seed_output.mkdir(parents=True, exist_ok=True)
    flash_output.mkdir(parents=True, exist_ok=True)
    blend_output.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Input root: {input_root}")
    print(f"Result root: {result_root}")
    print(f"Seed profile: {args.seed_profile}")
    print(f"Flash profile: {args.flash_profile}")
    print(f"Dry-run: {args.dry_run}, Force: {args.force}")

    if args.mode in {"seed", "blend"}:
        run_seed_mode(
            repo_root=repo_root,
            args=args,
            seed_profile=seed_profiles[args.seed_profile],
            input_root=input_root,
            output_root=seed_output,
        )

    if args.mode in {"flashvsr", "blend"}:
        run_flash_mode(
            repo_root=repo_root,
            args=args,
            flash_profile=flash_profiles[args.flash_profile],
            input_root=input_root,
            output_root=flash_output,
        )

    if args.mode == "blend":
        run_blend_mode(
            repo_root=repo_root,
            args=args,
            seed_output=seed_output,
            flash_output=flash_output,
            blend_output=blend_output,
        )


if __name__ == "__main__":
    main()
