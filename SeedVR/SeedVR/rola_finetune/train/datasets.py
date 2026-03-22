from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


@dataclass
class VideoSample:
    sample_id: str
    group_id: str
    lq_path: Path
    hq_path: Path | None = None


def _extract_group_id(name: str) -> str:
    stem = Path(name).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _adaptive_hw(height: int, width: int, long_side: int, short_side: int) -> Tuple[int, int]:
    if width >= height:
        return short_side, long_side
    return long_side, short_side


def _resize_to_fit(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    # video: T C H W, uint8/float on CPU
    t, c, h, w = video.shape
    if h >= target_h and w >= target_w:
        return video

    scale = max(float(target_h) / max(1, h), float(target_w) / max(1, w))
    new_h = max(target_h, int(round(h * scale)))
    new_w = max(target_w, int(round(w * scale)))

    v = video.float()
    v = F.interpolate(v, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return v


def _crop_video(video: torch.Tensor, target_h: int, target_w: int, train: bool) -> torch.Tensor:
    video = _resize_to_fit(video, target_h, target_w)
    _, _, h, w = video.shape

    if h == target_h:
        top = 0
    elif train:
        top = random.randint(0, h - target_h)
    else:
        top = (h - target_h) // 2

    if w == target_w:
        left = 0
    elif train:
        left = random.randint(0, w - target_w)
    else:
        left = (w - target_w) // 2

    return video[:, :, top : top + target_h, left : left + target_w]


def _sample_clip(video: torch.Tensor, clip_len: int, train: bool) -> torch.Tensor:
    # video: T C H W
    t = int(video.shape[0])
    if t <= 0:
        raise RuntimeError("Video has zero frame")

    if t >= clip_len:
        if train:
            start = random.randint(0, t - clip_len)
        else:
            start = max(0, (t - clip_len) // 2)
        return video[start : start + clip_len]

    pad_count = clip_len - t
    pad = video[-1:].repeat(pad_count, 1, 1, 1)
    return torch.cat([video, pad], dim=0)


def _normalize(video: torch.Tensor) -> torch.Tensor:
    # input [0,255] or [0,1] -> output [-1,1]
    if video.dtype == torch.uint8:
        video = video.float().div_(255.0)
    elif video.dtype.is_floating_point:
        if video.max() > 1.5:
            video = video.div(255.0)
    else:
        video = video.float().div_(255.0)
    return video.mul(2.0).sub(1.0)


def _read_video_tchw(path: Path) -> torch.Tensor:
    video, _, _ = read_video(str(path), output_format="TCHW", pts_unit="sec")
    # to T C H W
    if video.ndim != 4:
        raise RuntimeError(f"Unexpected video tensor shape for {path}: {tuple(video.shape)}")
    return video


def collect_synthetic_pairs(data_root: str) -> List[VideoSample]:
    root = Path(data_root)
    syn_root = root / "synthetic"

    hq_dirs = [syn_root / "HQ-synthetic1", syn_root / "HQ-synthetic2"]
    lq_dirs = [syn_root / "LQ-synthetic1", syn_root / "LQ-synthetic2"]

    hq_map: Dict[str, Path] = {}
    lq_map: Dict[str, Path] = {}

    for d in hq_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            hq_map[p.name] = p.resolve()

    for d in lq_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.mp4")):
            lq_map[p.name] = p.resolve()

    hq_keys = set(hq_map.keys())
    lq_keys = set(lq_map.keys())
    if hq_keys != lq_keys:
        only_hq = sorted(hq_keys - lq_keys)
        only_lq = sorted(lq_keys - hq_keys)
        raise RuntimeError(
            "Synthetic HQ/LQ pair mismatch: "
            f"only_hq={len(only_hq)} only_lq={len(only_lq)} "
            f"examples_hq={only_hq[:5]} examples_lq={only_lq[:5]}"
        )

    samples: List[VideoSample] = []
    for name in sorted(hq_keys):
        sample_id = Path(name).stem
        samples.append(
            VideoSample(
                sample_id=sample_id,
                group_id=_extract_group_id(name),
                lq_path=lq_map[name],
                hq_path=hq_map[name],
            )
        )
    return samples


def split_train_holdout(
    samples: Sequence[VideoSample],
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[VideoSample], List[VideoSample]]:
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError(f"holdout_ratio must be in (0,1), got {holdout_ratio}")

    group_to_samples: Dict[str, List[VideoSample]] = {}
    for s in samples:
        group_to_samples.setdefault(s.group_id, []).append(s)

    groups = sorted(group_to_samples.keys())
    if len(groups) < 2:
        raise RuntimeError(f"Need at least 2 groups for holdout split, got {len(groups)}")

    rng = random.Random(seed)
    rng.shuffle(groups)
    holdout_n = max(1, min(len(groups) - 1, int(round(len(groups) * holdout_ratio))))
    holdout_group = set(groups[:holdout_n])

    train_samples: List[VideoSample] = []
    holdout_samples: List[VideoSample] = []
    for g, g_samples in group_to_samples.items():
        if g in holdout_group:
            holdout_samples.extend(g_samples)
        else:
            train_samples.extend(g_samples)

    if not train_samples or not holdout_samples:
        raise RuntimeError("Invalid split, got empty train or holdout set")

    return sorted(train_samples, key=lambda x: x.sample_id), sorted(holdout_samples, key=lambda x: x.sample_id)


class SyntheticPairDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VideoSample],
        *,
        clip_len: int,
        long_side: int,
        short_side: int,
        train: bool,
    ):
        self.samples = list(samples)
        if len(self.samples) == 0:
            raise RuntimeError("SyntheticPairDataset is empty")
        self.clip_len = int(clip_len)
        self.long_side = int(long_side)
        self.short_side = int(short_side)
        self.train = bool(train)

    def set_clip_len(self, clip_len: int) -> None:
        self.clip_len = int(max(1, clip_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        lq = _read_video_tchw(sample.lq_path)
        gt = _read_video_tchw(sample.hq_path)  # type: ignore[arg-type]

        t = min(lq.shape[0], gt.shape[0])
        lq = lq[:t]
        gt = gt[:t]

        lq = _sample_clip(lq, clip_len=self.clip_len, train=self.train)
        gt = _sample_clip(gt, clip_len=self.clip_len, train=self.train)

        h, w = int(lq.shape[-2]), int(lq.shape[-1])
        target_h, target_w = _adaptive_hw(h, w, self.long_side, self.short_side)

        lq = _crop_video(lq, target_h, target_w, train=self.train)
        gt = _crop_video(gt, target_h, target_w, train=self.train)

        lq = _normalize(lq)
        gt = _normalize(gt)

        # T C H W -> C T H W
        lq = lq.permute(1, 0, 2, 3).contiguous()
        gt = gt.permute(1, 0, 2, 3).contiguous()

        return {
            "id": sample.sample_id,
            "group_id": sample.group_id,
            "lq": lq,
            "gt": gt,
        }


def collect_wild_samples(data_root: str) -> List[VideoSample]:
    root = Path(data_root)
    wild_root = root / "wild"
    if not wild_root.exists():
        raise FileNotFoundError(f"Wild directory not found: {wild_root}")

    samples: List[VideoSample] = []
    for p in sorted(wild_root.glob("*.mp4")):
        name = p.name
        samples.append(
            VideoSample(
                sample_id=Path(name).stem,
                group_id=_extract_group_id(name),
                lq_path=p.resolve(),
                hq_path=None,
            )
        )
    if not samples:
        raise RuntimeError(f"No wild videos found under {wild_root}")
    return samples


class WildDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VideoSample],
        *,
        clip_len: int,
        long_side: int,
        short_side: int,
        train: bool,
    ):
        self.samples = list(samples)
        if len(self.samples) == 0:
            raise RuntimeError("WildDataset is empty")
        self.clip_len = int(clip_len)
        self.long_side = int(long_side)
        self.short_side = int(short_side)
        self.train = bool(train)

    def set_clip_len(self, clip_len: int) -> None:
        self.clip_len = int(max(1, clip_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        lq = _read_video_tchw(sample.lq_path)

        lq = _sample_clip(lq, clip_len=self.clip_len, train=self.train)

        h, w = int(lq.shape[-2]), int(lq.shape[-1])
        target_h, target_w = _adaptive_hw(h, w, self.long_side, self.short_side)
        lq = _crop_video(lq, target_h, target_w, train=self.train)
        lq = _normalize(lq)

        lq = lq.permute(1, 0, 2, 3).contiguous()  # C T H W

        return {
            "id": sample.sample_id,
            "group_id": sample.group_id,
            "lq": lq,
        }
