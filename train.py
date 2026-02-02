"""
Entry-point training script for the project.

Course requirement: provide a single command to train the model (reproducible).

This file is intentionally a thin wrapper that forwards *all* CLI arguments to
`scripts/train_pix2pix.py`, so your exact experiment flags are preserved.
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
