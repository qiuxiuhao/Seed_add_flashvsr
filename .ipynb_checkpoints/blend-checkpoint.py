import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def list_videos(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required binary '{name}' was not found in PATH.")


def get_video_info(video_path: Path) -> tuple[int, int, float, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr.strip()}")

    info = json.loads(result.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    pix_fmt = str(stream["pix_fmt"])

    rate = str(stream.get("r_frame_rate", "30/1"))
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        fps = float(num) / den_f if den_f != 0 else 30.0
    else:
        fps = float(rate)
    return width, height, fps, pix_fmt


def blend_videos_ffmpeg(seed_path: Path, flash_path: Path, output_path: Path, alpha: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height, fps, pix_fmt = get_video_info(seed_path)
    blend_expr = f"A*{alpha} + B*{1.0 - alpha}"
    filter_complex = (
        f"[1:v]scale={width}:{height},fps={fps},format={pix_fmt}[flash];"
        f"[0:v]fps={fps},format={pix_fmt}[seed];"
        f"[seed][flash]blend=all_expr='{blend_expr}'[outv]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(seed_path),
        "-i",
        str(flash_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "0:a?",
        "-c:v",
        "libx265",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        pix_fmt,
        "-crf",
        "16",
        "-preset",
        "slow",
        "-c:a",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {seed_path.name}:\n{result.stderr.strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend SeedVR and FlashVSR outputs by alpha.")
    parser.add_argument("--alpha", type=float, required=True, help="Blend weight for SeedVR. Must be in [0, 1].")
    parser.add_argument("--seed-dir", type=str, default="result/SeedVR")
    parser.add_argument("--flash-dir", type=str, default="result/FlashVSR")
    parser.add_argument("--output-dir", type=str, default="result/SeedVR_blend_FlashVSR")
    parser.add_argument("--force", action="store_true", help="Overwrite existing blended outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

    require_binary("ffprobe")
    require_binary("ffmpeg")

    seed_dir = Path(args.seed_dir).resolve()
    flash_dir = Path(args.flash_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory does not exist: {seed_dir}")
    if not flash_dir.exists():
        raise FileNotFoundError(f"FlashVSR directory does not exist: {flash_dir}")

    seed_files = list_videos(seed_dir)
    if not seed_files:
        print(f"No videos found under {seed_dir}")
        return

    tasks: list[tuple[Path, Path, Path]] = []
    missing_flash: list[Path] = []
    skipped_existing = 0

    for seed_file in seed_files:
        rel = seed_file.relative_to(seed_dir)
        flash_file = flash_dir / rel
        out_file = output_dir / rel
        if not flash_file.exists():
            missing_flash.append(rel)
            continue
        if out_file.exists() and not args.force:
            skipped_existing += 1
            continue
        tasks.append((seed_file, flash_file, out_file))

    print(f"Alpha: {args.alpha}")
    print(f"Seed files: {len(seed_files)}")
    print(f"Matched tasks: {len(tasks)}")
    print(f"Missing flash files: {len(missing_flash)}")
    print(f"Skipped existing blended files: {skipped_existing}")

    errors: list[str] = []
    for seed_file, flash_file, out_file in tqdm(tasks, desc="Blending", unit="video"):
        try:
            blend_videos_ffmpeg(seed_file, flash_file, out_file, args.alpha)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{seed_file.relative_to(seed_dir)} -> {exc}")

    if missing_flash:
        print("\nMissing flash files (first 20):")
        for rel in missing_flash[:20]:
            print(f"- {rel}")
        if len(missing_flash) > 20:
            print(f"... and {len(missing_flash) - 20} more")

    if errors:
        print("\nBlend failures:")
        for msg in errors[:20]:
            print(f"- {msg}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        raise RuntimeError(f"Blending finished with {len(errors)} failures.")

    print(f"\nBlend completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
