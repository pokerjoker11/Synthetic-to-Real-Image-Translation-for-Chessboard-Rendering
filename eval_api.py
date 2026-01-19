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
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v3.py"

RESULTS_DIR = REPO_ROOT / "results"
TMP_RENDERS_DIR = REPO_ROOT / "renders"
DEFAULT_RES = 512  # Match training data render resolution
DEFAULT_SAMPLES = 64
DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints_clean"

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
      2. Checkpoints from plateau period (step 35k-50k) - best quality
      3. ckpt_dir/best.pt
      4. ckpt_dir/latest.pt
      5. Most recently modified *.pt in ckpt_dir
    
    Raises FileNotFoundError with helpful message if none found.
    """
    # Import torch directly for checkpoint loading
    import torch
    
    # 1. Environment variable
    env_ckpt = os.environ.get("CKPT_PATH")
    if env_ckpt:
        p = Path(env_ckpt)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"CKPT_PATH environment variable set to '{env_ckpt}' but file does not exist."
        )

    # 2. Prefer checkpoints from plateau period (35k-50k steps) - best quality
    # Training analysis showed quality improved until ~40k-43k, then plateaued
    # Explicitly check for known best checkpoint first
    best_plateau_ckpt = ckpt_dir / "best_step40000_backup.pt"
    if best_plateau_ckpt.exists():
        try:
            ckpt = torch.load(best_plateau_ckpt, map_location="cpu", weights_only=False)
            step = ckpt.get("step", 0)
            if 35000 <= step <= 50000:
                return best_plateau_ckpt
        except Exception:
            pass
    
    # Fallback: search all checkpoints for plateau period
    if ckpt_dir.exists():
        plateau_checkpoints = []
        for pt_file in ckpt_dir.glob("*.pt"):
            try:
                ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
                step = ckpt.get("step", 0)
                # Prefer checkpoints around 35k-50k (plateau period)
                if 35000 <= step <= 50000:
                    plateau_checkpoints.append((pt_file, step))
            except Exception:
                # Silently skip corrupted checkpoints
                continue
        
        if plateau_checkpoints:
            # Sort by step number, prefer those closer to 43k (plateau point)
            plateau_checkpoints.sort(key=lambda x: abs(x[1] - 43000))
            return plateau_checkpoints[0][0]

    # 3. best.pt
    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        return best_pt

    # 4. latest.pt
    latest_pt = ckpt_dir / "latest.pt"
    if latest_pt.exists():
        return latest_pt

    # 5. Most recent *.pt
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


def _match_training_style(img: Image.Image) -> Image.Image:
    """
    Post-process the Blender render to match the training data style.
    
    Training synth images have:
    - Pure black dark squares (very dark, ~50-70)
    - Higher contrast
    - More matte appearance (no bright white highlights)
    
    Current Blender output has:
    - Medium gray dark squares (~100-120)
    - Bright white highlights on pieces
    - Lower contrast overall
    
    This function adjusts levels and contrast to match.
    """
    import numpy as np_local
    from PIL import ImageEnhance
    
    # Increase contrast to match training style
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.4)  # Boost contrast
    
    # Adjust brightness slightly (training images are darker overall)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.85)
    
    # Apply levels adjustment to deepen the blacks
    arr = np_local.array(img).astype(np_local.float32)
    
    # Stretch the dark values to be darker (like training data)
    # Input range adjustment: darken darks more aggressively
    arr = np_local.clip(arr, 0, 255)
    
    # Apply gamma correction to deepen midtones and darks
    arr = 255.0 * np_local.power(arr / 255.0, 1.15)
    
    arr = np_local.clip(arr, 0, 255).astype(np_local.uint8)
    
    return Image.fromarray(arr)


def _apply_sharpening(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """
    Apply subtle unsharp masking to enhance details.
    
    Args:
        img: Input PIL Image
        strength: Sharpening strength (0.0 = no sharpening, 1.0 = maximum)
    
    Returns:
        Sharpened PIL Image
    """
    from PIL import ImageFilter
    
    # Unsharp mask: enhances edges while preserving smooth areas
    # Strength controls the intensity
    if strength <= 0.0:
        return img
    
    # Apply unsharp mask filter
    # Parameters: radius (blur), percent (strength), threshold (edge detection)
    sharpened = img.filter(ImageFilter.UnsharpMask(
        radius=1,  # Small radius for fine details
        percent=int(100 * strength),  # Convert to percentage
        threshold=3  # Only sharpen edges, not smooth areas
    ))
    
    # Blend with original to control strength
    if strength < 1.0:
        from PIL import Image
        sharpened = Image.blend(img, sharpened, strength)
    
    return sharpened


def _apply_default_perspective_transform(img: Image.Image) -> Image.Image:
    """
    Apply a default perspective transform to synthetic images to match training data.
    
    During training, synthetic images get perspective transforms to match real images.
    For test positions, we apply a typical perspective (forward tilt + slight side offset)
    that represents common camera angles in real chess images.
    """
    import torchvision.transforms.functional as TF
    from numpy.linalg import solve
    
    w, h = img.size
    perspective_max_tilt = 0.05  # Same as training
    
    # Enlarge by ~25% to accommodate tilt (same as training)
    scale_factor = 1.25
    enlarged_w = int(w * scale_factor)
    enlarged_h = int(h * scale_factor)
    
    # Resize to larger size
    enlarged = TF.resize(img, [enlarged_h, enlarged_w], interpolation=Image.BICUBIC)
    
    # Default perspective parameters (typical camera angle)
    max_shift = int(min(w, h) * perspective_max_tilt)
    pitch_shift = max_shift * 0.3  # Forward tilt (typical camera position)
    roll_shift = 0  # Minimal roll
    side_offset = 0.02 * 0.4 * min(w, h)  # Subtle horizontal offset
    
    # Original corners
    orig_corners = [
        [0, 0],  # top-left
        [enlarged_w, 0],  # top-right
        [enlarged_w, enlarged_h],  # bottom-right
        [0, enlarged_h],  # bottom-left
    ]
    
    # New corners with perspective distortion
    new_corners = [
        [orig_corners[0][0] - roll_shift - side_offset, orig_corners[0][1] - pitch_shift],
        [orig_corners[1][0] + roll_shift - side_offset, orig_corners[1][1] - pitch_shift],
        [orig_corners[2][0] + roll_shift + side_offset, orig_corners[2][1] + pitch_shift],
        [orig_corners[3][0] - roll_shift + side_offset, orig_corners[3][1] + pitch_shift],
    ]
    
    # Compute perspective transform coefficients
    A = np.zeros((8, 8))
    b = np.zeros(8)
    
    for i, ((x, y), (xp, yp)) in enumerate(zip(orig_corners, new_corners)):
        A[i*2] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp]
        b[i*2] = xp
        A[i*2+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp]
        b[i*2+1] = yp
    
    try:
        coeffs = solve(A, b)
        perspective_coeffs = tuple(coeffs)
        
        # Apply perspective transform with brown fill (matches training)
        brown_color = (101, 67, 33)
        tilted = enlarged.transform(
            enlarged.size, Image.PERSPECTIVE, perspective_coeffs,
            Image.BICUBIC, fillcolor=brown_color
        )
        
        # Center crop back to original size (can crop into brown border)
        tilted_w, tilted_h = tilted.size
        target_crop_size = max(w, h)
        crop_w = min(tilted_w, target_crop_size + int(w * 0.1))
        crop_h = min(tilted_h, target_crop_size + int(h * 0.1))
        crop_x = (tilted_w - crop_w) // 2
        crop_y = (tilted_h - crop_h) // 2
        crop_x = max(0, min(crop_x, tilted_w - crop_w))
        crop_y = max(0, min(crop_y, tilted_h - crop_h))
        
        cropped = tilted.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
        # Resize to original size
        return cropped.resize((w, h), Image.BICUBIC)
    except:
        # If transform fails, return original
        return img


def _crop_to_board(img: Image.Image) -> Image.Image:
    """
    Crop the Blender render using the same 1% border crop as training data.
    
    Training data uses synth_v3_cropped which was created by cropping 1% from
    all edges of 512x512 Blender renders, resulting in ~506x506 images.
    This function replicates that exact same crop to match training preprocessing.
    """
    w, h = img.size
    
    # Same 1% crop as training data (crop_synth_images.py with --border-pct 0.01)
    border_pct = 0.01
    border_x = int(w * border_pct)
    border_y = int(h * border_pct)
    
    left = border_x
    top = border_y
    right = w - border_x
    bottom = h - border_y
    
    cropped = img.crop((left, top, right, bottom))
    return cropped


def _translate_image(G, img_path: Path, device, already_processed: bool = False) -> Image.Image:
    """
    Run generator on a single image.
    Returns PIL Image in RGB.
    
    Args:
        already_processed: If True, skip crop and style matching (image already done)
    """
    _ensure_ml_imports()
    import torchvision.transforms.functional as TF

    # Load and preprocess
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (W, H)
    
    # Crop and style match if not already done
    if not already_processed:
        img = _crop_to_board(img)
        img = _match_training_style(img)

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
    
    # Post-processing: Apply sharpening to enhance piece details
    # Increased strength for sharper pieces (can increase artifacts if model output is already noisy)
    out_img = _apply_sharpening(out_img, strength=0.35)

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
    
    # Load and preprocess to match training data
    synth_img = Image.open(synth_png).convert("RGB")
    synth_img_cropped = _crop_to_board(synth_img)
    
    # Apply perspective transform to match training data
    # During training, synthetic images get perspective transforms to match real images
    # For test positions, apply a default perspective (typical camera angle)
    synth_img_tilted = _apply_default_perspective_transform(synth_img_cropped)
    
    synth_img_tilted.save(synthetic_out, format="PNG")
    print(f"[INFO] Applied perspective transform and cropped synthetic image to match training data")

    # Step 2: Load generator and translate to realistic
    _ensure_ml_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _find_checkpoint()
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")

    G = _load_generator(ckpt_path, device)
    realistic_img = _translate_image(G, synthetic_out, device, already_processed=True)
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
