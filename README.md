# Project 3: Synthetic-to-Real Image Translation for Chessboard Rendering

This project implements a Pix2Pix-based image translation model that converts synthetic
Blender-rendered chessboard images to photorealistic images resembling real chess footage.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Structure](#data-structure)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference / Evaluation](#inference--evaluation)
- [Smoke Tests](#smoke-tests)
- [GPU Usage Notes](#gpu-usage-notes)

---

## Environment Setup

### Prerequisites

- Python 3.9+
- [Blender](https://www.blender.org/download/) 3.x or 4.x (for synthetic rendering)
- CUDA-capable GPU (recommended, but CPU works)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd chess-s2r

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Blender Setup

The synthetic renderer requires Blender. Either:

1. Add Blender to your system PATH, or
2. Set the `BLENDER_PATH` environment variable:

```bash
# Windows (PowerShell)
$env:BLENDER_PATH = "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"

# Linux/macOS
export BLENDER_PATH=/usr/bin/blender
```

---

## Data Structure

The training pipeline expects the following folder structure:

```
data/
├── real/                      # Real chessboard images
│   ├── gt.csv                 # Ground truth: image, fen, viewpoint, game, frame
│   └── images/
│       └── game*_frame_*.jpg
│
├── synth_rect_rerender/      # Rendered synthetic images (aligned with real)
│   └── images/
│       └── row*_*.png
│
├── splits_rect/              # Train/val splits (CSV files)
│   ├── train.csv             # Columns: real, synth, fen, viewpoint
│   └── val.csv
│
└── filters/                  # (Optional) Manual drop lists
    └── manual_drop.txt
```

### CSV Format

Training CSVs (`train.csv`, `val.csv`) must have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `real` | Path to real image (relative to repo root) | `data/real/images/game6_frame_029716.jpg` |
| `synth` | Path to synthetic image | `data/synth_rect_rerender/images/row000000_white_*.png` |
| `fen` | FEN string for the position | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` |
| `viewpoint` | Camera perspective | `white` or `black` |

**Note:** Data files are not included in this repository. See your course materials for
data download instructions.

### Checkpoint Placement

Trained model checkpoints should be placed in the `checkpoints/` directory:

```
checkpoints/
├── best.pt       # Best validation checkpoint (preferred)
└── latest.pt     # Most recent training checkpoint
```

If you have a pre-trained checkpoint, place it in this directory before running inference.
The evaluation API will automatically find it (see Checkpoint Resolution below).

---

## Preprocessing

If starting from raw data, run these preprocessing steps in order:

### Step 1: Prepare Real Images (if using raw zip files)

```bash
# Extracts tagged images from game*_per_frame.zip files
# Input: data/raw_real/game*_per_frame.zip
# Output: data/real/images/*.jpg + data/real/gt.csv
python scripts/prepare_real.py
```

### Step 2: Render Synthetic Dataset

```bash
# Renders synthetic images for each entry in gt.csv using Blender
# Input: data/real/gt.csv
# Output: data/synth/images/*.png + data/pairs/pairs.csv
python scripts/render_synth_dataset.py
```

**Note:** This step requires Blender and can take several hours depending on dataset size.

### Step 3: Create Train/Val Splits

```bash
# Splits pairs.csv into train/val by game
# Input: data/pairs/pairs.csv
# Output: data/splits/train.csv, data/splits/val.csv
python scripts/make_splits.py
```

### Step 4: (Optional) Clean Splits

```bash
# Remove blurry/bad samples based on manual list and blur detection
# Input: data/splits/train.csv, data/splits/val.csv
# Output: data/splits_clean/train_clean.csv, val_clean.csv
python scripts/clean_splits.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv
```

---

## Training

### Basic Training Command

```bash
python train.py \
    --train_csv data/splits_rect/train.csv \
    --val_csv data/splits_rect/val.csv \
    --ckpt_dir checkpoints \
    --max_steps 5000 \
    --device cuda
```

### Full Training Command (with all options)

```bash
python train.py \
    --train_csv data/splits_rect/train.csv \
    --val_csv data/splits_rect/val.csv \
    --ckpt_dir checkpoints \
    --samples_dir results/train_samples \
    --device cuda \
    --batch_size 1 \
    --image_size 256 \
    --load_size 286 \
    --max_steps 5000 \
    --lr 2e-4 \
    --lambda_l1 100.0 \
    --lambda_grad 0.0 \
    --gan_loss bce \
    --log_every 50 \
    --sample_every 500 \
    --val_every 1000 \
    --seed 123
```

### Resume Training

```bash
# Auto-resumes from checkpoints/latest.pt if it exists
python train.py --ckpt_dir checkpoints --max_steps 10000

# Or specify explicit checkpoint
python train.py --resume checkpoints/best.pt --max_steps 10000
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_csv` | `data/splits_rect/train.csv` | Training data CSV |
| `--val_csv` | `data/splits_rect/val.csv` | Validation data CSV |
| `--ckpt_dir` | `checkpoints` | Checkpoint save directory |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--max_steps` | `5000` | Total training steps |
| `--batch_size` | `1` | Batch size (1 recommended for Pix2Pix) |
| `--lambda_l1` | `100.0` | L1 reconstruction loss weight |
| `--resume` | None | Explicit checkpoint path to resume from |

### Training Outputs

- `checkpoints/latest.pt` - Most recent checkpoint
- `checkpoints/best.pt` - Best validation L1 checkpoint
- `results/train_samples/step_XXXXXX.png` - Training samples (input | generated | target)

---

## Inference / Evaluation

### Project 3 Evaluation API

The required evaluation function is in `eval_api.py`:

```python
from eval_api import generate_chessboard_image

# Generate images for a given FEN and viewpoint
generate_chessboard_image(
    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    viewpoint="white"  # or "black"
)
```

This creates:
- `./results/synthetic.png` - Blender-rendered synthetic image
- `./results/realistic.png` - Generator-translated realistic image
- `./results/side_by_side.png` - Side-by-side comparison

### CLI Usage

```bash
# Default FEN (starting position)
python eval_api.py

# Custom FEN and viewpoint
python eval_api.py --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R" --viewpoint white

# Specify checkpoint explicitly
python eval_api.py --ckpt checkpoints/best.pt
```

### Checkpoint Resolution

The evaluation API finds checkpoints in this order:

1. `CKPT_PATH` environment variable (if set)
2. `checkpoints/best.pt`
3. `checkpoints/latest.pt`
4. Most recently modified `*.pt` in `checkpoints/`

To use a specific checkpoint:

```bash
# Via environment variable
export CKPT_PATH=/path/to/checkpoint.pt
python eval_api.py

# Via CLI argument
python eval_api.py --ckpt /path/to/checkpoint.pt
```

### Sample Output Generation

To generate sample outputs demonstrating the model's capabilities:

```bash
# Generate outputs for the starting position
python eval_api.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --viewpoint white

# Generate outputs for a mid-game position
python eval_api.py --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R" --viewpoint white

# Generate outputs from black's viewpoint
python eval_api.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR" --viewpoint black
```

Each command produces three files in `./results/`:
- `synthetic.png`: The Blender-rendered synthetic chessboard
- `realistic.png`: The model's translation to a realistic-looking image  
- `side_by_side.png`: A concatenated comparison of both images

---

## Smoke Tests

### Verify Installation

```bash
# Check all imports work and show help
python train.py --help
python eval_api.py --help
```

### Test Model Forward Pass

```bash
# Requires data/splits_rect/train.csv to exist
python scripts/smoke_test_model_forward.py
```

### Test Dataset Loading

```bash
# Requires data/splits_rect/train.csv and val.csv
python scripts/smoke_test_dataset.py
```

### Quick Training Test (10 steps)

```bash
python train.py \
    --train_csv data/splits_rect/train.csv \
    --val_csv data/splits_rect/val.csv \
    --max_steps 10 \
    --device cpu \
    --log_every 1 \
    --sample_every 5 \
    --val_every 10
```

### End-to-End Pipeline Test

A comprehensive test script verifies the entire pipeline:

```bash
# Run all tests (imports, data, model, dataset, training, inference)
python test_pipeline.py

# Skip training test (faster)
python test_pipeline.py --no-train

# Skip inference test (if no checkpoint/Blender)
python test_pipeline.py --no-infer

# Custom number of training steps
python test_pipeline.py --train-steps 10
```

---

## GPU Usage Notes

### Local Machine

- Training works on CPU but is slow (~0.5 it/s vs ~10 it/s on GPU)
- Recommended: NVIDIA GPU with 4GB+ VRAM
- AMP (Automatic Mixed Precision) is enabled by default on CUDA
- To disable AMP: `--no_amp`

### Cluster / Remote

- The code auto-detects CUDA availability
- No hardcoded device assumptions
- Set `--device cpu` to force CPU even when GPU is available

### Blender Rendering

- Blender rendering uses Cycles with GPU if available
- Set in Blender script: `scene.cycles.device = 'GPU'`
- Fallback to CPU if GPU not available

---

## Determinism Notes

The codebase includes best-effort determinism:

- Python `random.seed()`
- NumPy `np.random.seed()`
- PyTorch `torch.manual_seed()` and `torch.cuda.manual_seed_all()`

**Limitations:**
- Some CUDA operations are inherently non-deterministic
- Blender rendering may have minor variations
- Full determinism would require `torch.use_deterministic_algorithms(True)` which may fail on some operations

For reproducibility, use the same seed and hardware configuration.

---

## Project Structure

```
chess-s2r/
├── train.py                 # Top-level training entrypoint
├── eval_api.py              # Evaluation API (generate_chessboard_image)
├── test_pipeline.py         # End-to-end pipeline test
├── smoke_test.py            # Quick installation smoke test
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── assets/
│   ├── chess-set.blend      # Blender scene file
│   └── chess_position_api_v2.py  # Blender rendering script
│
├── scripts/
│   ├── train_pix2pix.py     # Core training implementation
│   ├── prepare_real.py      # Preprocessing: extract real images
│   ├── render_synth_dataset.py  # Preprocessing: render synthetic
│   ├── make_splits.py       # Preprocessing: create train/val splits
│   ├── clean_splits.py      # Preprocessing: filter bad samples
│   ├── smoke_test_dataset.py    # Smoke test for data loading
│   └── smoke_test_model_forward.py  # Smoke test for model
│
├── src/
│   ├── datasets/
│   │   └── pairs_dataset.py # PyTorch Dataset for paired images
│   └── models/
│       └── pix2pix_nets.py  # UNet Generator + PatchGAN Discriminator
│
├── checkpoints/             # Trained model checkpoints
│   ├── best.pt
│   └── latest.pt
│
├── results/                 # Training samples and inference outputs
│   └── train_samples/
│
├── legacy/                  # Old/unused files from previous iterations
│
└── data/                    # (gitignored) Training data
    └── ...
```

---

## Troubleshooting

### "Missing pairs csv" error

Ensure your CSV files exist and contain the correct columns. Check:
```bash
head -n 5 data/splits_rect/train.csv
```

### "No checkpoint found" error

Train a model first or specify a checkpoint path:
```bash
python train.py --max_steps 100  # Quick train
# or
export CKPT_PATH=/path/to/checkpoint.pt
```

### Blender not found

Set the BLENDER_PATH environment variable:
```bash
export BLENDER_PATH=/path/to/blender
```

### CUDA out of memory

- Reduce batch size (default is 1, which should work on 4GB VRAM)
- Use `--no_amp` to disable mixed precision
- Use `--device cpu` for CPU-only training
