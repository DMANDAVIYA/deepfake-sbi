# deepfake-sbi

Deepfake detection pipeline built on top of [Self-Blended Images (SBI)](https://github.com/mapooon/SelfBlendedImages).  
On first run the pipeline automatically:
- Clones the SBI source repository into `SelfBlendedImages/`
- Downloads pre-trained weights (`FFraw.tar`, `FFc23.tar`) from Google Drive via `gdown`

No manual setup beyond installing Python dependencies is required.

---

## Requirements

- Python 3.10+
- Git (must be on `PATH` for the auto-clone to work)
- CUDA-capable GPU recommended (falls back to CPU automatically)

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Install dependencies
pip install -r pipeline/requirements.txt
```

---

## Usage

All commands are run from the `pipeline/` directory.

```bash
cd pipeline
```

### Predict on your own videos (raw mode)

Drop `.mp4` files into `pipeline/data/input/` then run:

```bash
python run.py
```

Results are written to `pipeline/data/output/results_<timestamp>.csv` with columns:

| video | score | prediction |
|-------|-------|------------|
| clip.mp4 | 0.9231 | fake |

A score ≥ 0.5 is classified as **fake**.

### Evaluate on FaceForensics++ (FF++ mode)

Point `--data` at a directory that contains a `test.json` split file (standard FF++ layout):

```bash
python run.py --data /path/to/FaceForensics++
```

Optional flags for FF++ mode:

| Flag | Default | Choices |
|------|---------|---------|
| `--phase` | `test` | `train`, `val`, `test` |
| `--subset` | `all` | `all`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` |

### All CLI options

```
python run.py --help

  --data      PATH    Input directory (raw .mp4s or FF++ root). Default: pipeline/data/input
  --phase     STR     FF++ split to use.                        Default: test
  --subset    STR     FF++ manipulation type to evaluate.       Default: all
  --n-frames  INT     Frames sampled per video.                 Default: 32
  --device    STR     Torch device string.                      Default: cuda
```

---

## Project structure

```
pipeline/
  run.py          # Entry point
  detector.py     # SBIDetector / SBIEnsemble — auto-clones SBI repo on first import
  weights.py      # Auto-downloads pre-trained weights via gdown
  loader.py       # Video list helpers for raw mode and FF++ mode
  evaluator.py    # AUC / AP / ACC metrics
  requirements.txt

SelfBlendedImages/  # Auto-cloned from github.com/mapooon/SelfBlendedImages (git-ignored)
```

---

## Output metrics (FF++ mode)

After evaluation the terminal prints:

```
AUC=0.9800  AP=0.9750  ACC=0.9300  threshold=0.50
```

| Metric | Description |
|--------|-------------|
| AUC | Area under the ROC curve |
| AP | Average precision |
| ACC | Accuracy at the best threshold |

---

## Running tests

```bash
cd pipeline
pytest tests/
```
