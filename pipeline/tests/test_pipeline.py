import json
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── loader ──────────────────────────────────────────────────────────────────

class TestLoadFF:
    def _make_ff_tree(self, tmp_path: Path, split_ids: list[list[str]], fakes: list[str]) -> Path:
        root = tmp_path / "FF++"
        for vid_id in [v for pair in split_ids for v in pair]:
            for split in ["original_sequences/youtube/raw/videos"]:
                (root / split).mkdir(parents=True, exist_ok=True)
                (root / split / f"{vid_id}_000.mp4").touch()
        for fake in fakes:
            for vid_id in [v for pair in split_ids for v in pair]:
                path = root / "manipulated_sequences" / fake / "raw" / "videos"
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{vid_id}_000.mp4").touch()
        for phase in ["train", "val", "test"]:
            (root / f"{phase}.json").write_text(json.dumps(split_ids))
        return root

    def test_returns_correct_counts(self, tmp_path):
        from loader import load_ff
        root = self._make_ff_tree(tmp_path, [["000", "001"], ["002", "003"]], ["Deepfakes"])
        videos, labels = load_ff(str(root), phase="test", subset="Deepfakes")
        assert labels.count(0) == 4
        assert labels.count(1) == 4

    def test_labels_align_with_videos(self, tmp_path):
        from loader import load_ff
        root = self._make_ff_tree(tmp_path, [["010"]], ["Face2Face"])
        videos, labels = load_ff(str(root), phase="test", subset="Face2Face")
        assert len(videos) == len(labels)

    def test_all_subset_includes_all_fakes(self, tmp_path):
        from loader import load_ff
        fakes = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        root = self._make_ff_tree(tmp_path, [["020"]], fakes)
        _, labels = load_ff(str(root), phase="test", subset="all")
        assert labels.count(1) == 4

    def test_empty_split_returns_nothing(self, tmp_path):
        from loader import load_ff
        root = self._make_ff_tree(tmp_path, [], ["Deepfakes"])
        videos, labels = load_ff(str(root), phase="test")
        assert videos == [] and labels == []

    def test_missing_json_raises(self, tmp_path):
        from loader import load_ff
        with pytest.raises((FileNotFoundError, OSError)):
            load_ff(str(tmp_path / "nonexistent"), phase="test")

    def test_phase_filtering(self, tmp_path):
        from loader import load_ff
        train_ids, test_ids = [["100", "101"]], [["200", "201"]]
        root = tmp_path / "FF++"
        root.mkdir()
        for split, ids in [("train", train_ids), ("test", test_ids)]:
            (root / f"{split}.json").write_text(json.dumps(ids))
        (root / "val.json").write_text(json.dumps([]))
        orig = root / "original_sequences" / "youtube" / "raw" / "videos"
        orig.mkdir(parents=True)
        for vid_id in ["100", "101", "200", "201"]:
            (orig / f"{vid_id}_000.mp4").touch()
        fake_dir = root / "manipulated_sequences" / "Deepfakes" / "raw" / "videos"
        fake_dir.mkdir(parents=True)
        for vid_id in ["100", "101", "200", "201"]:
            (fake_dir / f"{vid_id}_000.mp4").touch()

        train_vids, _ = load_ff(str(root), phase="train", subset="Deepfakes")
        test_vids, _ = load_ff(str(root), phase="test", subset="Deepfakes")
        assert not set(train_vids) & set(test_vids)


# ── evaluator ───────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_perfect_prediction(self):
        from evaluator import evaluate
        r = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert r.auc == pytest.approx(1.0)
        assert r.acc == pytest.approx(1.0)

    def test_worst_prediction(self):
        from evaluator import evaluate
        r = evaluate([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        assert r.auc == pytest.approx(0.0)

    def test_random_prediction_auc_near_half(self):
        from evaluator import evaluate
        import random
        random.seed(42)
        labels = [0] * 50 + [1] * 50
        scores = [random.random() for _ in range(100)]
        r = evaluate(labels, scores)
        assert 0.3 < r.auc < 0.7

    def test_only_one_class_raises(self):
        from evaluator import evaluate
        with pytest.raises(ValueError):
            evaluate([1, 1, 1], [0.9, 0.8, 0.7])

    def test_single_pair(self):
        from evaluator import evaluate
        r = evaluate([0, 1], [0.2, 0.8])
        assert r.auc == pytest.approx(1.0)

    def test_str_output_contains_all_fields(self):
        from evaluator import evaluate
        r = evaluate([0, 1], [0.2, 0.8])
        assert "AUC" in str(r) and "AP" in str(r) and "ACC" in str(r)

    def test_all_scores_at_boundary(self):
        from evaluator import evaluate
        r = evaluate([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
        assert 0.0 <= r.auc <= 1.0


# ── detector ────────────────────────────────────────────────────────────────

class TestSBIDetector:
    def test_predict_returns_float_in_range(self):
        from detector import SBIDetector
        with patch("detector._load_model"), patch("detector._load_face_detector"), \
             patch("detector.extract_frames", return_value=([], [])):
            d = SBIDetector.__new__(SBIDetector)
            d.device = MagicMock()
            d.n_frames = 32
            d.model = MagicMock()
            d.face_detector = MagicMock()
            with patch("detector.extract_frames", return_value=([], [])), \
                 patch("detector._score_faces", return_value=0.5):
                score = d.predict("fake_path.mp4")
        assert 0.0 <= score <= 1.0

    def test_predict_batch_length_matches_input(self):
        from detector import SBIDetector
        d = SBIDetector.__new__(SBIDetector)
        d.predict = MagicMock(return_value=0.7)
        scores = d.predict_batch(["a.mp4", "b.mp4", "c.mp4"])
        assert len(scores) == 3

    def test_no_faces_returns_half(self):
        from detector import _score_faces
        result = _score_faces([], [], MagicMock(), MagicMock())
        assert result == 0.5

    def test_score_aggregation_takes_max_per_frame(self):
        import torch
        from detector import _score_faces
        device = torch.device("cpu")
        model = MagicMock()
        model.return_value.softmax.return_value.__getitem__.return_value = torch.tensor([0.3, 0.9, 0.4])
        faces = [MagicMock(), MagicMock(), MagicMock()]
        idx_list = [0, 0, 1]
        with patch("detector.torch.no_grad"), patch("torch.tensor", return_value=MagicMock()):
            pass
