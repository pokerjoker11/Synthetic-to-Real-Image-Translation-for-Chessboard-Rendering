"""
Entry-point training script for the project.

Course requirement: provide a single command to train the model (reproducible).

This file is intentionally a thin wrapper that forwards *all* CLI arguments to
`scripts/train_pix2pix.py`, so your exact experiment flags are preserved.

Examples:
    python train.py --train_csv data/splits_rect/train_clean.csv --val_csv data/splits_rect/val_final.csv --max_steps 5000

    # Piece-focused training (sharpness-oriented)
    python train.py --train_csv data/splits_rect/train_clean.csv --val_csv data/splits_rect/val_final.csv \
        --gan_loss hinge --spectral_norm --d_train_freq 2 --r1_gamma 10 \
        --use_piece_mask --piece_mask_dir data/masks --piece_weight 6 \
        --use_piece_D --piece_crop_size 96 --piece_patches_per_image 2 \
        --lambda_l1 12 --lambda_grad 60 --lambda_perceptual 0.15 \
        --lambda_piece_l1 4 --lambda_piece_grad 12
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "scripts" / "train_pix2pix.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing training script: {script}")

    # Forward all args to the real trainer.
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
