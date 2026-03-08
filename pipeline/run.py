import argparse
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from loader import load_ff, load_raw, is_raw_mode
from detector import SBIEnsemble
from evaluator import evaluate
import weights as weights_store


_PIPELINE_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _PIPELINE_DIR / "data" / "input"
_DEFAULT_OUTPUT = _PIPELINE_DIR / "data" / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SBI deepfake detection pipeline")
    p.add_argument("--data", default=str(_DEFAULT_DATA))
    p.add_argument("--phase", default="test", choices=["train", "val", "test"])
    p.add_argument("--subset", default="all", choices=["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    p.add_argument("--n-frames", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _open_csv(output_dir: Path, has_labels: bool) -> tuple[Path, csv.writer]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(out_file, "w", newline="")
    w = csv.writer(f)
    w.writerow(["video", "score", "prediction"] + (["label"] if has_labels else []))
    f.flush()
    return out_file, w, f


def _append_row(w: csv.writer, f, video: str, score: float, label: int | None = None) -> None:
    row = [Path(video).name, f"{score:.4f}", "fake" if score >= 0.5 else "real"]
    if label is not None:
        row.append(label)
    w.writerow(row)
    f.flush()


def main() -> None:
    args = parse_args()
    detector = SBIEnsemble(weights_store.get_all(), device=args.device, n_frames=args.n_frames)

    if is_raw_mode(args.data):
        videos = load_raw(args.data)
        if not videos:
            raise SystemExit(f"No .mp4 files found in {args.data}")
        print(f"Raw mode: {len(videos)} video(s) found")
        out_file, w, f = _open_csv(_DEFAULT_OUTPUT, has_labels=False)
        scores = []
        for v in tqdm(videos, desc="Inferring"):
            s = detector.predict(v)
            scores.append(s)
            _append_row(w, f, v, s)
            tqdm.write(f"  {Path(v).name:<50} {'FAKE' if s >= 0.5 else 'real'}  ({s:.4f})")
        f.close()
    else:
        videos, labels = load_ff(args.data, phase=args.phase, subset=args.subset)
        print(f"FF++ mode: {len(videos)} videos ({labels.count(0)} real, {labels.count(1)} fake)")
        out_file, w, f = _open_csv(_DEFAULT_OUTPUT, has_labels=True)
        scores = []
        for v, l in tqdm(zip(videos, labels), total=len(videos), desc="Inferring"):
            s = detector.predict(v)
            scores.append(s)
            _append_row(w, f, v, s, l)
        f.close()
        print(evaluate(labels, scores))

    print(f"Results saved -> {out_file}")


if __name__ == "__main__":
    main()
