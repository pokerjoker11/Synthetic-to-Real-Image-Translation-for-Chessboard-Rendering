"""
Project 3 Evaluation API: Synthetic-to-Real Chess Image Translation

Required function:
    generate_chessboard_image(fen: str, viewpoint: str) -> None

Outputs:
    ./results/synthetic.png   - Blender-rendered synthetic chessboard
    ./results/realistic.png   - Generator-translated realistic image
    ./results/side_by_side.png - Left: synthetic, Right: realistic

Checkpoint resolution (in order):
    1. CKPT_PATH environment variable
    2. checkpoints/best.pt
    3. checkpoints/latest.pt
    4. Most recent *.pt in checkpoints/
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

# Lazy imports for torch/numpy to allow --help without ML dependencies
torch = None
np = None


def _ensure_ml_imports():
    """Import torch and numpy on first use."""
    global torch, np
    if torch is None:
        import torch as _torch
        import numpy as _np
        torch = _torch
        np = _np

# ----------------------------
# Config
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
BLEND_FILE = ASSETS_DIR / "chess-set.blend"
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v2.py"

RESULTS_DIR = REPO_ROOT / "results"
TMP_RENDERS_DIR = REPO_ROOT / "renders"
DEFAULT_RES = 1024
DEFAULT_SAMPLES = 64
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints"

# Blender output view to use as synthetic
SYNTH_VIEW_NAME = "1_overhead.png"


def _set_deterministic(seed: int = 42) -> None:
    """
    Best-effort determinism for reproducibility.
    Note: Full determinism requires CUBLAS_WORKSPACE_CONFIG and may not be
    achievable with all operations. This provides reasonable reproducibility.
    """
    _ensure_ml_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: torch.use_deterministic_algorithms(True) may fail on some ops
    # We skip it to avoid runtime errors on the grader's machine


def _py_single_quote(s: str) -> str:
    """Safely embed Windows paths in a Python single-quoted string literal."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _find_blender() -> Optional[str]:
    """
    Finds Blender executable.
    Priority:
      1) BLENDER_PATH env var
      2) 'blender' in PATH
    """
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    return "blender"


def _find_checkpoint(ckpt_dir: Path = DEFAULT_CKPT_DIR) -> Path:
    """
    Find checkpoint in priority order:
      1. CKPT_PATH environment variable
      2. ckpt_dir/best.pt
      3. ckpt_dir/latest.pt
      4. Most recently modified *.pt in ckpt_dir
    
    Raises FileNotFoundError with helpful message if none found.
    """
    # 1. Environment variable
    env_ckpt = os.environ.get("CKPT_PATH")
    if env_ckpt:
        p = Path(env_ckpt)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"CKPT_PATH environment variable set to '{env_ckpt}' but file does not exist."
        )

    # 2. best.pt
    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        return best_pt

    # 3. latest.pt
    latest_pt = ckpt_dir / "latest.pt"
    if latest_pt.exists():
        return latest_pt

    # 4. Most recent *.pt
    if ckpt_dir.exists():
        pt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pt_files:
            return pt_files[0]

    # None found - provide helpful error
    raise FileNotFoundError(
        f"No checkpoint found.\n"
        f"Searched:\n"
        f"  1. CKPT_PATH env var (not set)\n"
        f"  2. {best_pt}\n"
        f"  3. {latest_pt}\n"
        f"  4. Any *.pt in {ckpt_dir}/\n\n"
        f"To fix:\n"
        f"  - Train a model: python train.py --max_steps 5000\n"
        f"  - Or set CKPT_PATH=/path/to/checkpoint.pt"
    )


def _load_generator(ckpt_path: Path, device) -> "torch.nn.Module":
    """Load the trained generator from checkpoint."""
    _ensure_ml_imports()
    # Import here to avoid circular imports and keep eval_api standalone-ish
    from src.models.pix2pix_nets import UNetGenerator

    G = UNetGenerator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G


def _render_synthetic_with_blender(fen: str, viewpoint: str) -> Path:
    """
    Runs Blender in background mode to render the synthetic chessboard.
    Returns path to the rendered PNG.
    """
    blender_bin = _find_blender()

    if not BLEND_FILE.exists():
        raise FileNotFoundError(f"Missing blend file: {BLEND_FILE}")
    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing blender script: {BLENDER_SCRIPT}")

    # Clean old renders
    if TMP_RENDERS_DIR.exists():
        shutil.rmtree(TMP_RENDERS_DIR, ignore_errors=True)

    chdir_expr = f"import os; os.chdir('{_py_single_quote(str(REPO_ROOT))}')"

    cmd = [
        blender_bin,
        str(BLEND_FILE),
        "--background",
        "--python-expr",
        chdir_expr,
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--fen",
        fen,
        "--view",
        viewpoint,
        "--resolution",
        str(DEFAULT_RES),
        "--samples",
        str(DEFAULT_SAMPLES),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Blender render failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{proc.stdout}"
        )

    # Primary expected location
    synth_path = TMP_RENDERS_DIR / SYNTH_VIEW_NAME
    if synth_path.exists():
        return synth_path

    # Fallback: parse Blender stdout for actual path
    import re
    m = re.search(r"Saved:\s*'([^']*1_overhead\.png)'", proc.stdout)
    if m:
        p = Path(m.group(1))
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Expected render not found: {synth_path}\n"
        f"Blender output:\n{proc.stdout}"
    )


def _translate_image(G, img_path: Path, device) -> Image.Image:
    """
    Run generator on a single image.
    Returns PIL Image in RGB.
    """
    _ensure_ml_imports()
    import torchvision.transforms.functional as TF

    # Load and preprocess
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (W, H)

    # Resize to 256x256 for model (standard Pix2Pix input)
    img_resized = img.resize((256, 256), Image.BICUBIC)

    # To tensor and normalize to [-1, 1]
    x = TF.to_tensor(img_resized)  # [0, 1]
    x = x * 2.0 - 1.0  # [-1, 1]
    x = x.unsqueeze(0).to(device)  # (1, 3, 256, 256)

    # Generate
    with torch.no_grad():
        y = G(x)

    # Post-process
    y = y.squeeze(0).cpu()
    y = (y + 1.0) * 0.5  # [0, 1]
    y = y.clamp(0, 1)
    y = (y * 255).byte()
    y = y.permute(1, 2, 0).numpy()

    out_img = Image.fromarray(y, mode="RGB")

    # Resize back to original resolution
    out_img = out_img.resize(original_size, Image.BICUBIC)

    return out_img


def _make_side_by_side(left_png: Path, right_png: Path, out_png: Path) -> None:
    """Create side-by-side comparison image."""
    left = Image.open(left_png).convert("RGB")
    right = Image.open(right_png).convert("RGB")

    # Match heights
    if left.height != right.height:
        right = right.resize(
            (int(right.width * (left.height / right.height)), left.height),
            Image.BICUBIC
        )

    canvas = Image.new("RGB", (left.width + right.width, left.height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(out_png, format="PNG")


# -------------------------------------------------------
# REQUIRED BY PROJECT 3 EVALUATION
# -------------------------------------------------------
def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """
    Generate synthetic and realistic chessboard images from a given FEN.

    Args:
        fen: FEN string describing the chess position (board part only, no move info needed)
        viewpoint: "white" or "black" - which side to view the board from

    Saves:
        ./results/synthetic.png    - Blender-rendered synthetic image
        ./results/realistic.png    - Generator-translated realistic image
        ./results/side_by_side.png - Left: synthetic, Right: realistic

    Raises:
        ValueError: if viewpoint not in {"white", "black"}
        FileNotFoundError: if Blender files or checkpoint not found
    """
    if viewpoint not in ("white", "black"):
        raise ValueError('viewpoint must be "white" or "black"')

    # Best-effort determinism
    _set_deterministic(seed=42)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_out = RESULTS_DIR / "synthetic.png"
    realistic_out = RESULTS_DIR / "realistic.png"
    side_by_side_out = RESULTS_DIR / "side_by_side.png"

    # Step 1: Render synthetic with Blender
    synth_png = _render_synthetic_with_blender(fen, viewpoint)
    shutil.copyfile(synth_png, synthetic_out)

    # Step 2: Load generator and translate to realistic
    _ensure_ml_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _find_checkpoint()
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")

    G = _load_generator(ckpt_path, device)
    realistic_img = _translate_image(G, synthetic_out, device)
    realistic_img.save(realistic_out, format="PNG")

    # Step 3: Create side-by-side
    _make_side_by_side(synthetic_out, realistic_out, side_by_side_out)

    print(f"[OK] Saved: {synthetic_out}")
    print(f"[OK] Saved: {realistic_out}")
    print(f"[OK] Saved: {side_by_side_out}")


# CLI for standalone testing
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate chessboard images from FEN")
    ap.add_argument("--fen", type=str,
                    default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                    help="FEN string (board part)")
    ap.add_argument("--viewpoint", type=str, default="white",
                    choices=["white", "black"])
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Override checkpoint path (can also use CKPT_PATH env var)")

    args = ap.parse_args()

    # Allow CLI override of checkpoint
    if args.ckpt:
        os.environ["CKPT_PATH"] = args.ckpt

    generate_chessboard_image(args.fen, args.viewpoint)
