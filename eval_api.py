import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

# ----------------------------
# Config (edit to match your repo)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
BLEND_FILE = ASSETS_DIR / "chess-set.blend"
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v2.py"

RESULTS_DIR = REPO_ROOT / "results"
TMP_RENDERS_DIR = REPO_ROOT / "renders"  # Blender script writes here by default
DEFAULT_RES = 1024
DEFAULT_SAMPLES = 64

# Choose which Blender render to treat as "the" synthetic image.
# The script outputs:
#   renders/1_overhead.png, renders/2_west.png, renders/3_east.png
SYNTH_VIEW_NAME = "1_overhead.png"

def _py_single_quote(s: str) -> str:
    # Safely embed Windows paths in a Python single-quoted string literal
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _find_blender() -> Optional[str]:
    """
    Finds Blender executable.
    Priority:
      1) env BLENDER_PATH
      2) 'blender' in PATH
    """
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # fallback: rely on PATH
    return "blender"


def _render_synthetic_with_blender(fen: str, viewpoint: str) -> Path:
    """
    Runs Blender in background mode to render the synthetic chessboard for given FEN+viewpoint.
    Returns the path to the chosen synthetic PNG.
    """
    blender_bin = _find_blender()

    if not BLEND_FILE.exists():
        raise FileNotFoundError(f"Missing blend file: {BLEND_FILE}")
    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing blender script: {BLENDER_SCRIPT}")

    # Clean old renders to avoid accidentally using stale files
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

    # Primary expected location (repo-local)
    synth_path = TMP_RENDERS_DIR / SYNTH_VIEW_NAME
    if synth_path.exists():
        return synth_path

    # Fallback: parse Blender stdout for the actual saved path
    import re
    m = re.search(r"Saved:\s*'([^']*1_overhead\.png)'", proc.stdout)
    if m:
        p = Path(m.group(1))
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Expected render not found: {synth_path}\n"
        f"Also failed to find fallback path from Blender output.\n"
        f"Blender output was:\n{proc.stdout}"
    )

def _make_side_by_side(left_png: Path, right_png: Path, out_png: Path) -> None:
    left = Image.open(left_png).convert("RGB")
    right = Image.open(right_png).convert("RGB")

    # Make heights match (simple + deterministic)
    if left.height != right.height:
        right = right.resize((int(right.width * (left.height / right.height)), left.height))

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
    Saves:
      ./results/synthetic.png
      ./results/realistic.png
      ./results/side_by_side.png
    """
    if viewpoint not in ("white", "black"):
        raise ValueError('viewpoint must be "white" or "black"')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_out = RESULTS_DIR / "synthetic.png"
    realistic_out = RESULTS_DIR / "realistic.png"
    side_by_side_out = RESULTS_DIR / "side_by_side.png"

    # Step 1 baseline:
    # - render synthetic with Blender
    # - "realistic" is just a copy for now (we’ll replace with the trained translator later)
    synth_png = _render_synthetic_with_blender(fen, viewpoint)
    shutil.copyfile(synth_png, synthetic_out)

    # Baseline "translator" (identity)
    shutil.copyfile(synthetic_out, realistic_out)

    _make_side_by_side(synthetic_out, realistic_out, side_by_side_out)


# Simple local test runner (not used by the evaluator)
if __name__ == "__main__":
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    generate_chessboard_image(test_fen, "white")
    print("Wrote:")
    print(" - results/synthetic.png")
    print(" - results/realistic.png")
    print(" - results/side_by_side.png")
