import argparse
from pathlib import Path
import hashlib
import shutil
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


def resolve_path(repo_root: Path, rel_or_abs: str) -> Path:
    rel_or_abs = (rel_or_abs or "").replace("\\", "/")
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p).resolve()


def safe_name(relpath: str) -> str:
    rp = relpath.replace("\\", "/")
    base = Path(rp).stem
    h = hashlib.md5(rp.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}.png"


def overlay_green(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    out = img_bgr.astype(np.float32).copy()
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    out[..., 1] = np.clip(out[..., 1] * (1.0 - alpha * m[..., 0]) + 255.0 * alpha * m[..., 0], 0, 255)
    return out.astype(np.uint8)


class Painter:
    def __init__(self):
        self.brush = 10
        self.drawing = False
        self.mode = "paint"  # or "erase"
        self.mask = None

    def set_mask(self, mask: np.ndarray):
        self.mask = mask

    def paint_at(self, x, y):
        if self.mask is None:
            return
        val = 255 if self.mode == "paint" else 0
        cv2.circle(self.mask, (x, y), self.brush, int(val), -1)

    def inc_brush(self):
        self.brush = min(100, self.brush + 2)

    def dec_brush(self):
        self.brush = max(1, self.brush - 2)


def pick_columns(df: pd.DataFrame):
    cols = set(df.columns)
    if "real" in cols and "synth" in cols:
        return "real", "synth"
    # fallbacks (in case your CSV columns changed)
    real_candidates = ["real_path", "B", "target", "real_img"]
    synth_candidates = ["synth_path", "A", "input", "synth_img"]
    real_col = next((c for c in real_candidates if c in cols), None)
    synth_col = next((c for c in synth_candidates if c in cols), None)
    if real_col and synth_col:
        return real_col, synth_col
    raise ValueError(f"CSV must contain (real,synth) or known alternatives. Columns found: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pairs CSV with columns real,synth (or A,B etc.)")
    ap.add_argument("--out_dir", required=True, help="Where to save your manual masks")
    ap.add_argument("--size", type=int, default=256)

    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip already-saved masks when moving forward (Prev still works)")
    ap.add_argument("--use_safe_names", action="store_true",
                    help="Save masks as stem+hash to avoid any future collisions")

    ap.add_argument("--shuffle", action="store_true", help="Shuffle annotation order")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit)")

    ap.add_argument("--bad_dir", type=str, default="bad_pairs",
                    help="Where to move dropped pairs (relative to repo root)")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    csv_path = Path(args.csv)

    df = pd.read_csv(csv_path)
    real_col, synth_col = pick_columns(df)
    rows = df.to_dict("records")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_root = (repo_root / args.bad_dir).resolve()
    bad_root.mkdir(parents=True, exist_ok=True)

    # order list
    order = list(range(len(rows)))
    if args.shuffle:
        rng = np.random.RandomState(args.seed)
        rng.shuffle(order)
    if args.limit and args.limit > 0:
        order = order[:min(args.limit, len(order))]

    if len(order) == 0:
        print("[ERR] Nothing to label: order list is empty.")
        return

    def out_path_for(real_rel: str) -> Path:
        return out_dir / (safe_name(real_rel) if args.use_safe_names else (Path(real_rel).stem + ".png"))

    painter = Painter()
    win = "FG Mask Painter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # current state
    pos = 0
    cur_img = None
    cur_real_rel = None
    cur_synth_rel = None

    # dropped tracking (so we skip them in-session)
    dropped_pairs = set()

    # one-time backup on first drop
    backup_made = False

    def ensure_backup():
        nonlocal backup_made
        if backup_made:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = csv_path.with_suffix(csv_path.suffix + f".bak_{ts}")
        shutil.copy2(csv_path, bak)
        backup_made = True
        print(f"[OK] backup csv -> {bak}")

    def rewrite_csv_in_place():
        # remove dropped pairs from df and write back to same csv path
        nonlocal df
        if not dropped_pairs:
            return
        ensure_backup()

        mi = pd.MultiIndex.from_frame(df[[real_col, synth_col]].astype(str))
        drop_list = list(dropped_pairs)
        keep = ~mi.isin(drop_list)
        df2 = df.loc[keep].copy()

        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        df2.to_csv(tmp, index=False)
        tmp.replace(csv_path)
        df = df2
        print(f"[OK] updated csv in-place -> {csv_path} (rows now {len(df)})")

    def load_mask_for_frame(real_rel: str) -> np.ndarray:
        # Only load YOUR saved mask if it exists; otherwise blank
        saved_p = out_path_for(real_rel)
        if saved_p.exists():
            m = cv2.imread(str(saved_p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return np.zeros((args.size, args.size), dtype=np.uint8)
            m = cv2.resize(m, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
            return (m > 127).astype(np.uint8) * 255
        return np.zeros((args.size, args.size), dtype=np.uint8)

    def load_index(p: int, ignore_skip: bool = False) -> bool:
        nonlocal cur_img, cur_real_rel, cur_synth_rel

        if p < 0 or p >= len(order):
            return False

        i = order[p]
        r = rows[i]
        real_rel = str(r[real_col])
        synth_rel = str(r[synth_col])

        # skip dropped in-session
        if (real_rel, synth_rel) in dropped_pairs:
            return False

        # skip saved masks when moving forward (but allow Prev to load them)
        saved_p = out_path_for(real_rel)
        if args.skip_existing and (not ignore_skip) and saved_p.exists():
            return False

        real_path = resolve_path(repo_root, real_rel)
        if not real_path.exists():
            print(f"[SKIP] missing real: {real_path}")
            return False

        img = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] unreadable real: {real_path}")
            return False

        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        mask = load_mask_for_frame(real_rel)

        cur_img = img
        cur_real_rel = real_rel
        cur_synth_rel = synth_rel
        painter.set_mask(mask)
        return True

    def save_current():
        if painter.mask is None or cur_real_rel is None:
            return
        p = out_path_for(cur_real_rel)
        cv2.imwrite(str(p), painter.mask)
        print(f"[OK] saved mask -> {p}")

    def move_to_bad(abs_src: Path, rel_for_layout: str, bucket: str) -> bool:
        """
        Move file to bad_pairs/<bucket>/<original_relpath>
        Returns True on success.
        """
        if not abs_src.exists():
            return False
        relp = Path(rel_for_layout.replace("\\", "/"))
        dst = bad_root / bucket / relp
        dst.parent.mkdir(parents=True, exist_ok=True)

        # avoid overwrite
        if dst.exists():
            h = hashlib.md5(str(abs_src).encode("utf-8")).hexdigest()[:6]
            dst = dst.with_name(dst.stem + f"_{h}" + dst.suffix)

        try:
            shutil.move(str(abs_src), str(dst))
            return True
        except Exception as e:
            print(f"[WARN] failed moving {abs_src} -> {dst}: {e}")
            return False

    def drop_current_and_next():
        """
        Move real+synth (+manual mask if exists) to bad_pairs and remove row from CSV (in-place).
        """
        if cur_real_rel is None or cur_synth_rel is None:
            return

        real_rel = cur_real_rel
        synth_rel = cur_synth_rel
        dropped_pairs.add((real_rel, synth_rel))

        # move files (real & synth)
        real_abs = resolve_path(repo_root, real_rel)
        synth_abs = resolve_path(repo_root, synth_rel)

        moved_real = move_to_bad(real_abs, real_rel, "real")
        moved_synth = move_to_bad(synth_abs, synth_rel, "synth")

        # move manual mask if exists
        mp = out_path_for(real_rel)
        if mp.exists():
            move_to_bad(mp, str(Path("masks_manual") / mp.name), "masks")

        # update CSV immediately so training pool is updated now
        rewrite_csv_in_place()

        print(f"[DROP] removed from CSV + moved files | real_moved={moved_real} synth_moved={moved_synth}")
        go_next(save_first=False)

    def go_next(save_first: bool):
        nonlocal pos
        if save_first:
            save_current()
        j = pos + 1
        while j < len(order) and not load_index(j, ignore_skip=False):
            j += 1
        if j < len(order):
            pos = j
        else:
            print("[INFO] Reached end of list.")

    def go_prev():
        nonlocal pos
        j = pos - 1
        # allow prev even if saved masks exist
        while j >= 0 and not load_index(j, ignore_skip=True):
            j -= 1
        if j >= 0:
            pos = j
        else:
            print("[INFO] At start of list.")

    # initial load
    while pos < len(order) and not load_index(pos, ignore_skip=False):
        pos += 1
    if pos >= len(order):
        print("[ERR] No valid images to label (missing files or all skipped).")
        return

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            painter.drawing = True
            painter.mode = "paint"
            painter.paint_at(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            painter.drawing = True
            painter.mode = "erase"
            painter.paint_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and painter.drawing:
            painter.paint_at(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            painter.drawing = False

    cv2.setMouseCallback(win, on_mouse)

    while True:
        if cur_img is None or painter.mask is None:
            print("[ERR] Current image/mask not loaded. Exiting.")
            break

        # Left: painted overlay, Right: raw mask (B/W)
        paint_view = overlay_green(cur_img, painter.mask, alpha=0.65)
        mask_view = cv2.cvtColor(painter.mask, cv2.COLOR_GRAY2BGR)
        panel = np.concatenate([paint_view, mask_view], axis=1)

        status = (
            f"                              {pos+1}/{len(order)}"
        )
        cv2.putText(panel, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, panel)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            painter.inc_brush()
        if key in (ord("-"), ord("_")):
            painter.dec_brush()

        if key == ord("s"):
            go_next(save_first=True)
        if key == ord("n") or key == ord(" "):
            go_next(save_first=False)
        if key == ord("p"):
            go_prev()
        if key == ord("m"):
            drop_current_and_next()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
