import argparse
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------
def resolve_path(repo_root: Path, rel_or_abs: str) -> Path:
    rel_or_abs = (rel_or_abs or "").replace("\\", "/")
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p).resolve()


def safe_stem_from_relpath(relpath: str, maxlen: int = 180) -> str:
    """
    Safer than just Path(stem) if you ever get collisions.
    Keeps filename stem + short hash of full relpath.
    """
    rp = relpath.replace("\\", "/")
    base = Path(rp).stem
    h = hashlib.md5(rp.encode("utf-8")).hexdigest()[:8]
    s = f"{base}_{h}"
    return s[:maxlen]


def load_coarse_mask(mask_dir: Path, real_rel: str, synth_rel: str, out_size: int) -> np.ndarray:
    """
    Try real stem then synth stem; returns binary 0/255, resized to out_size.
    """
    real_stem = Path(real_rel).stem
    synth_stem = Path(synth_rel).stem

    p = mask_dir / f"{real_stem}.png"
    if not p.exists():
        p = mask_dir / f"{synth_stem}.png"

    if not p.exists():
        return np.zeros((out_size, out_size), dtype=np.uint8)

    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    return m


def dilate_mask(mask01: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask01
    k = 2 * px + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask01, kernel, iterations=1)


def overlay_fg(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Overlay mask in GREEN (to match FG scribble color).
    mask01 is 0/255 or 0..255 uint8.
    """
    out = img_bgr.copy().astype(np.float32)
    m = (mask01.astype(np.float32) / 255.0)[..., None]
    # green channel in BGR is index 1
    out[..., 1] = np.clip(out[..., 1] * (1.0 - alpha * m[..., 0]) + 255.0 * alpha * m[..., 0], 0, 255)
    return out.astype(np.uint8)

def make_help_panel(annot, idx: int, total: int) -> np.ndarray:
    """
    Returns a BGR image with a readable list of controls.
    """
    W, H = 640, 520
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    lines = [
        "Manual Mask Refine - Controls",
        "",
        f"Image: {idx+1}/{total}",
        f"Mode: {annot.mode.upper()}   Brush: {annot.brush}   Allowed_px: {annot.allowed_dilate_px}",
        "",
        "Mouse:",
        "  Left-drag   : Foreground scribble (piece)",
        "  Right-drag  : Background scribble (board / NOT piece)",
        "",
        "Keys:",
        "  g           : Run GrabCut (refine mask from scribbles)",
        "  s           : Save mask",
        "  n / Space   : Next image",
        "  p           : Previous image",
        "  r           : Reset (clear scribbles + result)",
        "  + / -       : Brush size",
        "  ] / [       : Increase / Decrease allowed region (spill) by 2 px",
        "  a           : Toggle allowed-region visualization (green)",
        "  o           : Toggle mask overlay (red)",
        "  q / Esc     : Quit",
        "",
        "Tip: Use BOTH FG (inside piece) and BG (around piece), then press 'g'.",
    ]

    y = 28
    for k, t in enumerate(lines):
        if k == 0:
            cv2.putText(img, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 2, cv2.LINE_AA)
            y += 32
            continue
        cv2.putText(img, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24

    return img

# ----------------------------
# Interactive annotator state
# ----------------------------
class Annotator:
    def __init__(self, img_bgr: np.ndarray, coarse_mask: np.ndarray, allowed_dilate_px: int):
        self.img = img_bgr
        self.H, self.W = img_bgr.shape[:2]

        # Coarse occupancy area -> allowed region (dilated)
        self.coarse = coarse_mask.copy()
        self.allowed = dilate_mask(self.coarse, allowed_dilate_px)

        # GrabCut mask labels: 0=bg,1=fg,2=prob bg,3=prob fg
        self.gc_mask = np.full((self.H, self.W), 2, dtype=np.uint8)  # probable BG
        self.gc_mask[self.allowed > 0] = 3  # probable FG inside allowed

        # User scribbles
        self.brush = 8
        self.mode = "fg"  # 'fg' or 'bg'
        self.show_allowed = True
        self.show_overlay = True

        # Current result mask (binary 0/255)
        self.result = self.coarse.copy()

        self._drawing = False
        self.allowed_dilate_px = int(allowed_dilate_px)

    def reset(self):
        self.gc_mask[:] = 2
        self.gc_mask[self.allowed > 0] = 3
        self.result = self.coarse.copy()

    def set_mode_fg(self):
        self.mode = "fg"

    def set_mode_bg(self):
        self.mode = "bg"

    def inc_brush(self):
        self.brush = min(80, self.brush + 2)

    def dec_brush(self):
        self.brush = max(1, self.brush - 2)

    def paint(self, x, y):
        # fg scribble -> definite fg (1), bg scribble -> definite bg (0)
        lbl = 1 if self.mode == "fg" else 0
        cv2.circle(self.gc_mask, (x, y), self.brush, lbl, -1)

        # preview: show strokes immediately in the displayed mask
        if self.mode == "fg":
            cv2.circle(self.result, (x, y), self.brush, 255, -1)
        else:
            cv2.circle(self.result, (x, y), self.brush, 0, -1)


    def run_grabcut(self, iters=2):
        # Force outside allowed region to bg (helps a lot)
        self.gc_mask[self.allowed == 0] = 0

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(self.img, self.gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)

        # extract fg/probfg
        fg = np.where((self.gc_mask == 1) | (self.gc_mask == 3), 255, 0).astype(np.uint8)

        # safety: keep only within allowed region (but allow spill inside allowed dilation)
        fg = cv2.bitwise_and(fg, fg, mask=(self.allowed > 0).astype(np.uint8) * 255)

        self.result = fg
        print("gc_mask counts:",
            int((self.gc_mask==0).sum()), "BG",
            int((self.gc_mask==1).sum()), "FG",
            int((self.gc_mask==2).sum()), "PBG",
            int((self.gc_mask==3).sum()), "PFG")
        print("result fg pixels:", int((self.result>0).sum()))

    def render(self):
        vis = self.img.copy()

        if self.show_allowed:
            # show allowed region faint green
            g = vis.copy().astype(np.float32)
            a = (self.allowed.astype(np.float32) / 255.0)[..., None]
            g[..., 1] = np.clip(g[..., 1] * (1.0 - 0.35 * a[..., 0]) + 255.0 * 0.35 * a[..., 0], 0, 255)
            vis = g.astype(np.uint8)

        if self.show_overlay:
            vis = overlay_fg(vis, self.result, alpha=0.55)

        # draw UI text
        txt = f"mode={self.mode.upper()}  brush={self.brush}  allowed_px={self.allowed_dilate_px}   (see 'Controls' window)"
        cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Show GrabCut scribbles: FG=green, BG=blue
        scrib = np.zeros_like(vis, dtype=np.uint8)
        scrib[self.gc_mask == 1] = (0, 255, 0)   # definite FG (BGR)
        scrib[self.gc_mask == 0] = (255, 0, 0)   # definite BG (BGR)
        vis = cv2.addWeighted(vis, 1.0, scrib, 0.7, 0)
        
        return vis
    
    def set_allowed_dilate(self, new_px: int):
        self.allowed_dilate_px = int(max(0, new_px))
        self.allowed = dilate_mask(self.coarse, self.allowed_dilate_px)

        # enforce constraints immediately
        self.gc_mask[self.allowed == 0] = 0
        self.result = cv2.bitwise_and(self.result, self.result, mask=(self.allowed > 0).astype(np.uint8) * 255)



# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pairs CSV (must contain columns: real, synth)")
    ap.add_argument("--mask_dir", required=True, help="Directory containing coarse masks (your current data/masks)")
    ap.add_argument("--out_dir", required=True, help="Where to save refined masks")
    ap.add_argument("--size", type=int, default=256, help="Resize images/masks to this size (default 256)")
    ap.add_argument("--allowed_dilate_px", type=int, default=10, help="Expand allowed region to catch projected spillover (default 10)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip frames that already have a saved refined mask")
    ap.add_argument("--use_safe_names", action="store_true", help="Save masks as <stem>_<hash>.png to avoid any future collisions")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    csv_path = Path(args.csv)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "real" not in df.columns or "synth" not in df.columns:
        raise ValueError("CSV must contain columns named 'real' and 'synth'")

    rows = df.to_dict("records")
    i = 0

    win = "Manual Mask Refine (GrabCut)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    win2 = "Controls"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win2, 640, 520)


    annot = None
    cur_paths = None

    def load_index(idx: int):
        nonlocal annot, cur_paths
        r = rows[idx]
        real_rel = r["real"]
        synth_rel = r["synth"]

        real_path = resolve_path(repo_root, real_rel)
        synth_path = resolve_path(repo_root, synth_rel)

        if not real_path.exists():
            print(f"[SKIP] missing real: {real_path}")
            return False

        img = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] unreadable real: {real_path}")
            return False

        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

        coarse = load_coarse_mask(mask_dir, real_rel, synth_rel, args.size)

        annot = Annotator(img, coarse, args.allowed_dilate_px)
        cur_paths = (real_rel, synth_rel)
        return True

    # initial load (skip missing)
    while i < len(rows) and not load_index(i):
        i += 1
    if i >= len(rows):
        print("[ERR] No valid images found in CSV.")
        return

    # mouse callback
    def on_mouse(event, x, y, flags, param):
        nonlocal annot
        if annot is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            annot._drawing = True
            annot.set_mode_fg()
            annot.paint(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            annot._drawing = True
            annot.set_mode_bg()
            annot.paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and annot._drawing:
            annot.paint(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            annot._drawing = False

    cv2.setMouseCallback(win, on_mouse)

    def out_name(real_rel: str) -> str:
        if args.use_safe_names:
            return safe_stem_from_relpath(real_rel) + ".png"
        return Path(real_rel).stem + ".png"

    while True:
        vis = annot.render()
        cv2.imshow(win, vis)
        cv2.imshow(win2, make_help_panel(annot, i, len(rows)))

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

        # brush
        if key == ord("+") or key == ord("="):
            annot.inc_brush()
        if key == ord("-") or key == ord("_"):
            annot.dec_brush()

        # toggles
        if key == ord("a"):
            annot.show_allowed = not annot.show_allowed
        if key == ord("o"):
            annot.show_overlay = not annot.show_overlay

        # actions
        if key == ord("r"):
            annot.reset()
        if key == ord("g"):
            annot.run_grabcut(iters=5)

        # save
        if key == ord("s"):
            real_rel, synth_rel = cur_paths
            fn = out_name(real_rel)
            out_path = out_dir / fn
            cv2.imwrite(str(out_path), annot.result)
            print(f"[OK] saved {out_path}")

        # next/prev
        if key == ord("n") or key == ord(" "):
            # optionally skip existing
            j = i + 1
            while j < len(rows):
                real_rel = rows[j]["real"]
                fn = out_name(real_rel)
                if args.skip_existing and (out_dir / fn).exists():
                    j += 1
                    continue
                if load_index(j):
                    i = j
                    break
                j += 1

        if key == ord("p"):
            j = i - 1
            while j >= 0:
                real_rel = rows[j]["real"]
                fn = out_name(real_rel)
                if args.skip_existing and (out_dir / fn).exists():
                    j -= 1
                    continue
                if load_index(j):
                    i = j
                    break
                j -= 1
        
        if key == ord("]"):
            annot.set_allowed_dilate(annot.allowed_dilate_px + 2)
        if key == ord("["):
            annot.set_allowed_dilate(annot.allowed_dilate_px - 2)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
