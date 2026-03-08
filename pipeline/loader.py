import json
import os
from glob import glob
from pathlib import Path
from typing import Literal


Phase = Literal["train", "val", "test"]
Subset = Literal["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
_FF_FAKES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _load_split_ids(root: str, phase: Phase) -> set[str]:
    with open(Path(root) / f"{phase}.json") as f:
        return {vid for pair in json.load(f) for vid in pair}


def _glob_videos(path: str) -> list[str]:
    return sorted(glob(os.path.join(path, "*.mp4")))


def _filter_by_ids(videos: list[str], ids: set[str]) -> list[str]:
    return [v for v in videos if Path(v).name[:3] in ids]


def load_raw(input_dir: str) -> list[str]:
    return sorted(glob(os.path.join(input_dir, "*.mp4")))


def is_raw_mode(input_dir: str) -> bool:
    return not (Path(input_dir) / "test.json").exists()


def load_ff(root: str, phase: Phase = "test", subset: Subset = "all") -> tuple[list[str], list[int]]:
    ids = _load_split_ids(root, phase)
    fakes = _FF_FAKES if subset == "all" else [subset]

    real = _filter_by_ids(
        _glob_videos(os.path.join(root, "original_sequences", "youtube", "raw", "videos")), ids
    )
    fake = [
        v
        for fake_type in fakes
        for v in _filter_by_ids(
            _glob_videos(os.path.join(root, "manipulated_sequences", fake_type, "raw", "videos")), ids
        )
    ]

    return real + fake, [0] * len(real) + [1] * len(fake)
