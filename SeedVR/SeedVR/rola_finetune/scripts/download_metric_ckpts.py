from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


DEFAULT_FILES = {
    "raft_small_C_T_V2-01064c6d.pth": "https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth",
    "musiq_koniq_ckpt-e95806b9.pth": "https://download.pytorch.org/models/musiq_koniq_ckpt-e95806b9.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download metric checkpoints for wild evaluation")
    parser.add_argument("--out_dir", type=str, default="./ckpts/metrics")
    return parser.parse_args()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        done = min(total_size, count * block_size)
        ratio = done / total_size
        bar = int(ratio * 30)
        print(f"\r[{('=' * bar).ljust(30)}] {ratio * 100:6.2f}% {dst.name}", end="")

    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, str(dst), reporthook=_progress)
    print()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DEFAULT_FILES.items():
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"Skip existing: {dst}")
            continue
        try:
            download(url, dst)
        except Exception as exc:
            if dst.exists():
                try:
                    os.remove(dst)
                except OSError:
                    pass
            print(f"Failed to download {name}: {exc}")


if __name__ == "__main__":
    main()
