# Project 3: Synthetic-to-Real Image Translation for Chessboard Rendering

This project implements a Pix2Pix-based image translation model that converts synthetic Blender-rendered chessboard images into photorealistic images resembling real chess footage.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Formats (TA Requirement)](#dataset-formats-ta-requirement)
- [Training Data Structure (Used by Our Code)](#training-data-structure-used-by-our-code)
- [Preprocessing / Dataset Packaging](#preprocessing--dataset-packaging)
- [Training](#training)
- [Inference / Evaluation](#inference--evaluation)
- [Smoke Tests](#smoke-tests)
- [GPU Usage Notes](#gpu-usage-notes)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites

- Python 3.9+
- Blender 3.x or 4.x (for synthetic rendering)
- CUDA-capable GPU (recommended, but CPU works)

### Installation

```bash
# 1) Clone
git clone https://github.com/pokerjoker11/Synthetic-to-Real-Image-Translation-for-Chessboard-Rendering.git chess-s2r
cd chess-s2r

# 2) Create + activate venv
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt
```

### Blender Setup

The synthetic renderer requires Blender. Either:

1. Add Blender to your system PATH, or
2. Set the `BLENDER_PATH` environment variable:

```bash
# Windows (PowerShell)
$env:BLENDER_PATH = "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe"

# Linux/macOS
export BLENDER_PATH=/usr/bin/blender
```

---

## Dataset Formats (TA Requirement)

The TA clarified there are multiple dataset formats:

1. **Original format** (as provided in the course materials).
2. **Drive format (required upload)** — the format the TA requested you upload to the shared drive.
3. **Training format (optional, but required if your training uses a different layout)** — if you trained using a custom structure, you must upload that too and clearly state which one is used for training.

### Format (2): Drive format (required)

This format matches the course-requested structure:

- `real_trainval/images/` + `real_trainval/gt.csv` (`image_name, fen, viewpoint`)
- `synth_trainval/images/` + `synth_trainval/gt.csv` (`image_name, fen, viewpoint`)

**Download link (shared drive):** `https://drive.google.com/file/d/1YAnLLUbW9wPSmkZbRDwc2xDHAFe8Foe2/view?usp=drive_link`

### Format (3): Training format (used by this repository)

Our training uses:

- **Paired** (synth, real) samples (Pix2Pix)
- **Manual masks** for a piece-weighted loss term

So we provide a separate ZIP that unpacks directly into the repository structure below.

**Download link (shared drive):** `https://drive.google.com/file/d/1TxRd3qaeASq_fAHo2fznuxCglATPkeDW/view?usp=drive_link`

### Curation note

The train/val subset is curated. Frames with severe blur or occlusions (e.g., hands) were excluded.

---

## Training Data Structure (Used by Our Code)

Unzip the **training-format ZIP** into the **repo root**, so it creates:

```
data/
├── real/
│   └── images/
│       └── game*_frame_*.jpg
├── synth/
│   └── images/
│       └── *.png (or jpg)
├── masks_manual/
│   └── game*_frame_*.png     # binary masks, filename matches REAL image stem
└── pairs/
    ├── train.csv             # columns: real, synth, fen, viewpoint
    └── val.csv
```

### CSV Format

`data/pairs/train.csv` and `data/pairs/val.csv` contain:

| Column | Description | Example |
|--------|-------------|---------|
| `real` | Path to real image (relative to repo root) | `data/real/images/game6_frame_029716.jpg` |
| `synth` | Path to synthetic image | `data/synth/images/row000000_white_*.png` |
| `fen` | FEN string (board part) | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` |
| `viewpoint` | Camera perspective | `white` or `black` |

### Checkpoint Placement

Trained model checkpoints are written to `checkpoints/`:

```
checkpoints/
├── best.pt       # Best validation checkpoint
└── latest.pt     # Most recent checkpoint
```

---

## Preprocessing / Dataset Packaging

Most users (including graders) should **not** rebuild the dataset from raw sources. Instead, use the two shared-drive ZIPs:

- **Drive format (2)** ZIP (course requirement)
- **Training format (3)** ZIP (required to train this repo, because of masks + pairing)

### Optional: regenerate the ZIPs from an existing local dataset

If you already have the images + pairs CSVs locally and want to recreate the ZIPs:

```bash
# Build training-format ZIP (includes masks + pairs)
python scripts/build_cloud_dataset_package.py --require_masks --mask_dir data/masks_manual --out_dir cloud_dataset_trainval --zip_name cloud_dataset_trainval_with_masks.zip

# Build Drive-format ZIP (course-required structure)
python scripts/build_drive_format_zip.py --out_dir drive_format_trainval --zip_name drive_format_trainval.zip
```

### Optional: verify masks coverage

Masks are required for training; this must be 0 missing:

```bash
python scripts/check_masks_coverage.py --fail
```

---

## Training

`train.py` is the top-level entrypoint (course requirement). It forwards CLI arguments to the underlying trainer.

### Basic training command (recommended defaults)

```bash
python train.py --ckpt_dir checkpoints --max_steps 5000 --device cuda --mask_dir data/masks_manual --lambda_piece 5 --sample_nocrop
```

### CPU example (quick sanity)

```bash
python train.py --device cpu --num_workers 0 --max_steps 50 --log_every 10 --val_every 50 --sample_every 25 --ckpt_dir checkpoints --mask_dir data/masks_manual --lambda_piece 5 --sample_nocrop
```

### Outputs

- `checkpoints/latest.pt` — most recent checkpoint
- `checkpoints/best.pt` — best validation checkpoint
- `results/train_samples/step_XXXXXX.png` — training sample grid (input | generated | target)

> Note on samples: `--sample_nocrop` creates report-friendly samples (no extra crop). The model itself is trained on standard Pix2Pix crops/augs.

---

## Inference / Evaluation

The required evaluation function is in `eval_api.py`:

```python
from eval_api import generate_chessboard_image

generate_chessboard_image(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    viewpoint="white"  # or "black"
)
```

This creates (inside `./results/` only):

- `./results/synthetic.png` — Blender-rendered synthetic image
- `./results/realistic.png` — generator-translated realistic image
- `./results/side_by_side.png` — comparison

### CLI usage

```bash
# Default checkpoint resolution (uses checkpoints/best.pt then latest.pt)
python eval_api.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --viewpoint white

# Explicit checkpoint
python eval_api.py --ckpt checkpoints/best.pt
```

### Report sample generation (recommended)

Run a few FENs and copy `results/side_by_side.png` into a `results/report_samples/` folder:

```bash
python eval_api.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --viewpoint white
python eval_api.py --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R" --viewpoint white
python eval_api.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR" --viewpoint black
```

---

## Smoke Tests

```bash
# Verify help works
python train.py --help
python eval_api.py --help
```

Quick training test:

```bash
python train.py --device cpu --num_workers 0 --max_steps 3 --log_every 1 --val_every 3 --sample_every 3 --ckpt_dir checkpoints --mask_dir data/masks_manual --lambda_piece 5 --sample_nocrop
```

---

## GPU Usage Notes

### Local Machine

- Training works on CPU but is slow
- Recommended: NVIDIA GPU with 4GB+ VRAM
- AMP (Automatic Mixed Precision) is enabled by default on CUDA (if supported)
- To disable AMP: `--no_amp`

### Cluster / Remote

- Code auto-detects CUDA availability
- Set `--device cpu` to force CPU

### Blender Rendering

- Blender rendering uses Cycles with GPU if available
- Falls back to CPU if GPU is not available

---

## Troubleshooting

### "Missing pairs csv" error

Ensure the training-format ZIP was unzipped into the repo root and you have:

- `data/pairs/train.csv`
- `data/pairs/val.csv`

### "Missing mask" error

Masks are required for training. Ensure:

- `data/masks_manual/<real_image_stem>.png` exists for every real image referenced by the pairs CSVs

You can verify coverage with:

```bash
python scripts/check_masks_coverage.py --fail
```

### "No checkpoint found" error (eval_api)

Train a model first or specify a checkpoint:

```bash
python train.py --max_steps 100 --device cpu --mask_dir data/masks_manual --lambda_piece 5 --sample_nocrop
# or
python eval_api.py --ckpt /path/to/checkpoint.pt
```
