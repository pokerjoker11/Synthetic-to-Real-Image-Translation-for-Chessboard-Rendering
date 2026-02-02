# scripts/calibrate_board_bbox.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def find_grid_bbox(img: Image.Image, pad: int = 4):
    g = np.array(img.convert("L"), dtype=np.float32)
    # high-frequency emphasis
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    e = gx + gy

    # sum energy per row/col
    col = e.sum(axis=0)
    row = e.sum(axis=1)

    # smooth
    k = 9
    col_s = np.convolve(col, np.ones(k)/k, mode="same")
    row_s = np.convolve(row, np.ones(k)/k, mode="same")

    # threshold to find "active" region (grid area)
    ct = col_s.mean() + 0.8 * col_s.std()
    rt = row_s.mean() + 0.8 * row_s.std()

    xs = np.where(col_s > ct)[0]
    ys = np.where(row_s > rt)[0]
    if len(xs) < 10 or len(ys) < 10:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # expand slightly
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(g.shape[1]-1, x1 + pad); y1 = min(g.shape[0]-1, y1 + pad)
    return x0, y0, x1, y1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--pad", type=int, default=4)
    args = ap.parse_args()

    img = Image.open(args.img).convert("RGB")
    bbox = find_grid_bbox(img, pad=args.pad)
    if bbox is None:
        print("[ERROR] Could not estimate bbox")
        return
    x0,y0,x1,y1 = bbox
    print(f"[OK] bbox x0={x0} y0={y0} x1={x1} y1={y1} (w={x1-x0+1} h={y1-y0+1})")

    # save visualization
    arr = np.array(img).copy()
    arr[y0:y0+2, x0:x1+1] = [255,0,0]
    arr[y1-1:y1+1, x0:x1+1] = [255,0,0]
    arr[y0:y1+1, x0:x0+2] = [255,0,0]
    arr[y0:y1+1, x1-1:x1+1] = [255,0,0]
    out = Path("results") / "bbox_debug.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(arr).save(out)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
