# scripts/generate_piece_masks_from_fen_trainspace.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

PIECE_CHARS = set("PNBRQKpnbrqk")

PIECE_CHARS = set("PNBRQKpnbrqk")

def fen_to_piece_grid(fen: str) -> np.ndarray:
    placement = fen.split()[0]
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError("Bad FEN")
    grid = np.full((8, 8), ".", dtype="<U1")  # row0 = rank8 (top)
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            elif ch in PIECE_CHARS:
                grid[r, c] = ch
                c += 1
            else:
                raise ValueError(f"Bad FEN char: {ch}")
        if c != 8:
            raise ValueError("Bad FEN row width")
    return grid

def grid_to_piece_mask(grid: np.ndarray, size: int, viewpoint: str) -> Image.Image:
    sq = size // 8
    rot180 = (str(viewpoint).lower().strip() == "black")

    # Expansion per piece type (fractions of a square)
    # (up, left/right, down)
    EXP = {
        "p": (0.45, 0.16, 0.08),
        "n": (0.70, 0.22, 0.10),
        "b": (0.75, 0.22, 0.10),
        "r": (0.80, 0.22, 0.10),
        "q": (0.95, 0.26, 0.12),
        "k": (0.90, 0.26, 0.12),
    }

    mask = np.zeros((size, size), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            ch = grid[r, c]
            if ch == ".":
                continue

            rr, cc = (7 - r, 7 - c) if rot180 else (r, c)

            base_y0, base_y1 = rr * sq, (rr + 1) * sq
            base_x0, base_x1 = cc * sq, (cc + 1) * sq

            t = ch.lower()
            up_f, lr_f, dn_f = EXP.get(t, (0.75, 0.22, 0.10))

            up = int(round(up_f * sq))
            lr = int(round(lr_f * sq))
            dn = int(round(dn_f * sq))

            y0 = max(0, base_y0 - up)
            y1 = min(size, base_y1 + dn)
            x0 = max(0, base_x0 - lr)
            x1 = min(size, base_x1 + lr)

            mask[y0:y1, x0:x1] = 255

    # Slight dilation to catch edges / slight misalignment
    m = Image.fromarray(mask).convert("L").filter(ImageFilter.MaxFilter(size=7))
    return m



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--fen_col", default="fen")
    ap.add_argument("--view_col", default="viewpoint")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("synth", args.fen_col, args.view_col):
        if col not in df.columns:
            raise SystemExit(f"Missing column {col}. Found: {df.columns.tolist()}")

    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = 0
    for _, r in df.iterrows():
        synth = Path(str(r["synth"]))
        outp = out_dir / (synth.stem + ".png")
        if outp.exists():
            skipped += 1
            continue
        grid = fen_to_piece_grid(str(r[args.fen_col]))
        m = grid_to_piece_mask(grid, size=args.size, viewpoint=str(r[args.view_col]))

        m.save(outp, format="PNG")
        made += 1

    print(f"[OK] made={made} skipped={skipped} out={out_dir.resolve()}")

if __name__ == "__main__":
    main()
