import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

PIECE_SCALE = {
    "p": (0.30, 0.45),
    "n": (0.35, 0.55),
    "b": (0.35, 0.55),
    "r": (0.38, 0.60),
    "q": (0.42, 0.70),
    "k": (0.42, 0.70),
}

def parse_fen_board(fen: str):
    board = fen.split()[0]
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Bad FEN board: {board}")
    grid = []
    for r in ranks:
        row = []
        for ch in r:
            if ch.isdigit():
                row += [""] * int(ch)
            else:
                row.append(ch)
        if len(row) != 8:
            raise ValueError(f"Bad rank in FEN: {r}")
        grid.append(row)
    return grid  # [row=0..7 for rank8..rank1][col=0..7 for a..h]

def make_mask_from_fen(fen: str, viewpoint: str, size: int, src_size: int, bbox):
    x0, y0, x1, y1 = bbox
    s = size / float(src_size)
    x0 *= s; x1 *= s; y0 *= s; y1 *= s

    sw = (x1 - x0) / 8.0
    sh = (y1 - y0) / 8.0

    grid = parse_fen_board(fen)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    vp = (viewpoint or "").lower().strip()
    is_black = vp.startswith("black")

    for r in range(8):        # r=0 is rank8 (top)
        for c in range(8):    # c=0 is file a (left)
            piece = grid[r][c]
            if not piece:
                continue

            rr, cc = r, c
            if is_black:
                rr = 7 - r
                cc = 7 - c

            cx = x0 + (cc + 0.5) * sw
            cy = y0 + (rr + 0.5) * sh

            key = piece.lower()
            sx, sy = PIECE_SCALE.get(key, (0.35, 0.55))
            rx = sw * sx
            ry = sh * sy

            left = max(0, cx - rx)
            right = min(size - 1, cx + rx)
            top = max(0, cy - ry)
            bottom = min(size - 1, cy + ry)

            draw.ellipse([left, top, right, bottom], fill=255)

    return mask

def process_csv(csv_path: Path, mask_dir: Path, real_col: str, fen_col: str, view_col: str,
                size: int, src_size: int, bbox, overwrite: bool):
    df = pd.read_csv(csv_path)
    made = skipped = missing = 0
    for _, row in df.iterrows():
        real = str(row[real_col])
        fen = row.get(fen_col, "")
        view = row.get(view_col, "white")

        if not isinstance(fen, str) or not fen.strip():
            missing += 1
            continue

        out_name = Path(real).stem + ".png"  # KEYED TO REAL FRAME
        out_path = mask_dir / out_name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        m = make_mask_from_fen(fen, view, size=size, src_size=src_size, bbox=bbox)
        m.save(out_path)
        made += 1

    return made, skipped, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, default="data/masks")
    ap.add_argument("--real_col", type=str, default="real")
    ap.add_argument("--fen_col", type=str, default="fen")
    ap.add_argument("--view_col", type=str, default="viewpoint")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--src_size", type=int, default=502)
    ap.add_argument("--bbox", type=str, required=True, help="x0,y0,x1,y1 in src_size coords (e.g. 26,23,475,476)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    bbox = tuple(int(x) for x in args.bbox.split(","))
    if len(bbox) != 4:
        raise ValueError("bbox must be x0,y0,x1,y1")

    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    tr = process_csv(Path(args.train_csv), mask_dir, args.real_col, args.fen_col, args.view_col,
                     args.size, args.src_size, bbox, args.overwrite)
    va = process_csv(Path(args.val_csv), mask_dir, args.real_col, args.fen_col, args.view_col,
                     args.size, args.src_size, bbox, args.overwrite)

    print(f"[TRAIN] made={tr[0]} skipped={tr[1]} missing_fen={tr[2]}")
    print(f"[VAL]   made={va[0]} skipped={va[1]} missing_fen={va[2]}")
    print(f"[OK] masks in: {mask_dir}")

if __name__ == "__main__":
    main()
