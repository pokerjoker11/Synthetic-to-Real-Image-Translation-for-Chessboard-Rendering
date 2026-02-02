# scripts/render_synth_dataset.py
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
BLEND_FILE = ASSETS_DIR / "chess-set.blend"
BLENDER_SCRIPT = ASSETS_DIR / "chess_position_api_v2.py"

REAL_GT = REPO_ROOT / "data" / "real" / "gt.csv"
SYN_DIR = REPO_ROOT / "data" / "synth" / "images"
PAIRS_CSV = REPO_ROOT / "data" / "pairs" / "pairs.csv"

TMP_RENDERS_DIR = REPO_ROOT / "renders"
SYN_VIEW_NAME = "1_overhead.png"

# Training-time render settings (faster than eval)
RESOLUTION = 512
SAMPLES = 16

# Progress reporting
REPORT_EVERY = 5  # update progress every N items


def _find_blender() -> str:
    p = os.environ.get("BLENDER_PATH")
    if p and Path(p).exists():
        return p
    return "blender"


def _py_single_quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _render_one(fen: str, viewpoint: str) -> Path:
    blender_bin = _find_blender()
    if not BLEND_FILE.exists():
        raise FileNotFoundError(f"Missing: {BLEND_FILE}")
    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {BLENDER_SCRIPT}")

    # ensure Blender treats repo as CWD (so ./renders is repo-local)
    chdir_expr = f"import os; os.chdir('{_py_single_quote(str(REPO_ROOT))}')"

    # clean old renders to avoid stale reads
    if TMP_RENDERS_DIR.exists():
        shutil.rmtree(TMP_RENDERS_DIR, ignore_errors=True)

    cmd = [
        blender_bin,
        str(BLEND_FILE),
        "--background",
        "--python-expr",
        chdir_expr,
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--fen", fen,
        "--view", viewpoint,
        "--resolution", str(RESOLUTION),
        "--samples", str(SAMPLES),
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
        raise RuntimeError(f"Blender failed.\nCMD: {' '.join(cmd)}\n\n{proc.stdout}")

    # normal expected location
    p = TMP_RENDERS_DIR / SYN_VIEW_NAME
    if p.exists():
        return p

    # fallback: parse "Saved: '...1_overhead.png'"
    m = re.search(r"Saved:\s*'([^']*1_overhead\.png)'", proc.stdout)
    if m:
        q = Path(m.group(1))
        if q.exists():
            return q

    raise FileNotFoundError(
        f"Render missing. Expected: {p}\nBlender output:\n{proc.stdout}"
    )


def main():
    if not REAL_GT.exists():
        raise SystemExit(f"[ERR] Missing {REAL_GT}")

    SYN_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(REAL_GT, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    pairs = []

    start_t = time.perf_counter()
    done = 0
    rendered = 0
    skipped_existing = 0

    for r in rows:
        real_rel = r["image"]  # e.g. images/game2_frame_000200.jpg
        fen = r["fen"]
        viewpoint = r.get("viewpoint", "white")
        game = r.get("game", "game?")
        frame = int(r.get("frame", "0"))

        out_name = Path(real_rel).with_suffix(".png").name  # same basename, .png
        synth_out = SYN_DIR / out_name

        if synth_out.exists():
            skipped_existing += 1
        else:
            synth_tmp = _render_one(fen, viewpoint)
            shutil.copyfile(synth_tmp, synth_out)
            rendered += 1

        pairs.append({
            "real": f"data/real/{real_rel}",
            "synth": f"data/synth/images/{out_name}",
            "fen": fen,
            "viewpoint": viewpoint,
            "game": game,
            "frame": frame,
        })

        done += 1
        if done == 1 or done % REPORT_EVERY == 0 or done == n:
            elapsed = time.perf_counter() - start_t
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = n - done
            eta_sec = remaining / rate if rate > 0 else float("inf")

            if eta_sec == float("inf"):
                eta_str = "??:??"
            else:
                eta_min = int(eta_sec // 60)
                eta_s = int(eta_sec % 60)
                eta_str = f"{eta_min:02d}:{eta_s:02d}"

            pct = 100.0 * done / n
            msg = (
                f"\r[PROGRESS] {done}/{n} ({pct:5.1f}%) | "
                f"rendered={rendered} skipped={skipped_existing} | "
                f"{rate:5.2f} it/s | ETA {eta_str}"
            )
            print(msg, end="", flush=True)

    print()  # newline after carriage-return progress line

    # write pairs.csv
    with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["real", "synth", "fen", "viewpoint", "game", "frame"])
        w.writeheader()
        w.writerows(pairs)

    print("\n==== Summary ====")
    print(f"Rows in gt.csv         : {n}")
    print(f"Rendered newly         : {rendered}")
    print(f"Skipped existing       : {skipped_existing}")
    print(f"Synth images dir       : {SYN_DIR}")
    print(f"Pairs CSV              : {PAIRS_CSV}")
    print(f"Render settings        : res={RESOLUTION}, samples={SAMPLES}")


if __name__ == "__main__":
    main()
