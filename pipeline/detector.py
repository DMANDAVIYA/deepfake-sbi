import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SelfBlendedImages" / "src"))

from model import Detector
from inference.preprocess import extract_frames
from retinaface.pre_trained_models import get_model


def _load_model(weights: str, device: torch.device) -> Detector:
    model = Detector().to(device)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.eval()
    return model


def _load_face_detector(device: torch.device):
    fd = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    fd.eval()
    return fd


def _score_faces(faces: list, idx_list: list, model: Detector, device: torch.device) -> float:
    if not faces:
        return 0.5
    with torch.no_grad():
        preds = model(torch.from_numpy(np.stack(faces)).to(device).float() / 255).softmax(1)[:, 1]

    buckets: dict[int, list[float]] = {}
    for idx, p in zip(idx_list, preds.tolist()):
        buckets.setdefault(idx, []).append(p)

    return float(np.mean([max(v) for v in buckets.values()]))


class SBIDetector:
    def __init__(self, weights: str, device: str = "cuda", n_frames: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_frames = n_frames
        self.model = _load_model(weights, self.device)
        self.face_detector = _load_face_detector(self.device)

    def predict(self, video_path: str) -> float:
        faces, idx_list = extract_frames(video_path, self.n_frames, self.face_detector)
        return _score_faces(faces, idx_list, self.model, self.device)

    def predict_batch(self, video_paths: list[str]) -> list[float]:
        return [self.predict(p) for p in video_paths]


class SBIEnsemble:
    def __init__(self, weights_paths: list[str], device: str = "cuda", n_frames: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_frames = n_frames
        self.models = [_load_model(w, self.device) for w in weights_paths]
        self.face_detector = _load_face_detector(self.device)

    def predict(self, video_path: str) -> float:
        faces, idx_list = extract_frames(video_path, self.n_frames, self.face_detector)
        return max(_score_faces(faces, idx_list, m, self.device) for m in self.models)

    def predict_batch(self, video_paths: list[str]) -> list[float]:
        return [self.predict(p) for p in video_paths]
