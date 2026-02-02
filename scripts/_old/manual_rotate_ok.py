#!/usr/bin/env python3
"""
Manual rotation tool for rectified OK chessboards.

UI:
- Counter/filename ABOVE the image
- Image in the middle
- Intended POV BELOW the image (just above buttons)
- Buttons:
    Back | Rotate 90° CCW | Rotate 180° | Pass | Rotate 90° CW

Keybinds:
A = CCW, D = CW, R = 180, Backspace = back, Space/Enter = pass, Q/Esc = quit

Logs to CSV and saves resume state.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, List

import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_pov_map(pairs_csv: Optional[Path]) -> Dict[str, str]:
    if not pairs_csv or not pairs_csv.exists():
        return {}

    pov_map: Dict[str, str] = {}
    with pairs_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_path = (row.get("real_path") or "").strip()
            rid = (row.get("id") or "").strip()

            key = ""
            if real_path:
                key = Path(real_path).stem
            elif rid:
                key = rid
            if not key:
                continue

            pov = (row.get("pov") or row.get("pov_desired") or "").strip().lower()
            if pov in {"white", "black"}:
                pov_map[key] = pov

    return pov_map


def infer_pov_from_name(stem: str) -> str:
    s = stem.lower()
    if "_bpov" in s:
        return "black"
    if "_wpov" in s:
        return "white"
    return "unknown"


def rotate_image_inplace(path: Path, degrees: int) -> None:
    """
    degrees: +90 => CCW 90
             -90 => CW 90
             180 => 180
    """
    img = Image.open(path).convert("RGB")
    img = img.rotate(degrees, expand=False)  # keep same size (square expected)
    ext = path.suffix.lower()
    fmt = "PNG" if ext == ".png" else "JPEG" if ext in {".jpg", ".jpeg"} else "PNG"
    img.save(path, format=fmt)


def list_images(ok_dir: Path) -> List[Path]:
    files = []
    for p in ok_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort(key=lambda x: x.as_posix())
    return files


def load_state(state_json: Path) -> int:
    if not state_json.exists():
        return 0
    try:
        data = json.loads(state_json.read_text(encoding="utf-8"))
        return int(data.get("index", 0))
    except Exception:
        return 0


def save_state(state_json: Path, index: int) -> None:
    state_json.parent.mkdir(parents=True, exist_ok=True)
    state_json.write_text(json.dumps({"index": index}, indent=2), encoding="utf-8")


def append_log(log_csv: Path, row: Dict[str, str]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "stem", "action", "degrees", "intended_pov"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


class ReviewerApp:
    def __init__(self, root: tk.Tk, images: List[Path], pov_map: Dict[str, str],
                 state_json: Path, log_csv: Path):
        self.root = root
        self.images = images
        self.pov_map = pov_map
        self.state_json = state_json
        self.log_csv = log_csv

        self.idx = load_state(state_json)
        self.idx = max(0, min(self.idx, max(0, len(self.images) - 1)))

        self.root.title("Manual Rotate OK Boards")
        self.root.geometry("980x740")

        # Counter ABOVE the image
        self.counter_label = tk.Label(root, text="", font=("Arial", 16))
        self.counter_label.pack(pady=10)

        # Image
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=6)

        # Intended POV BELOW the image (just above buttons)
        self.pov_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.pov_label.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)

        self.btn_back = tk.Button(btn_frame, text="Back", width=10, command=self.prev_image)
        self.btn_back.grid(row=0, column=0, padx=8)

        self.btn_left = tk.Button(btn_frame, text="Rotate ⟲ (90°)", width=16, command=self.rotate_ccw)
        self.btn_left.grid(row=0, column=1, padx=8)

        self.btn_180 = tk.Button(btn_frame, text="Rotate 180°", width=14, command=self.rotate_180)
        self.btn_180.grid(row=0, column=2, padx=8)

        self.btn_pass = tk.Button(btn_frame, text="Pass", width=10, command=self.pass_image)
        self.btn_pass.grid(row=0, column=3, padx=8)

        self.btn_right = tk.Button(btn_frame, text="Rotate ⟳ (90°)", width=16, command=self.rotate_cw)
        self.btn_right.grid(row=0, column=4, padx=8)

        # Keybinds
        root.bind("<a>", lambda e: self.rotate_ccw())
        root.bind("<A>", lambda e: self.rotate_ccw())
        root.bind("<d>", lambda e: self.rotate_cw())
        root.bind("<D>", lambda e: self.rotate_cw())
        root.bind("<r>", lambda e: self.rotate_180())
        root.bind("<R>", lambda e: self.rotate_180())
        root.bind("<BackSpace>", lambda e: self.prev_image())
        root.bind("<space>", lambda e: self.pass_image())
        root.bind("<Return>", lambda e: self.pass_image())
        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<q>", lambda e: self.quit())
        root.bind("<Q>", lambda e: self.quit())

        self.photo = None  # keep reference
        self.show_current()

    def current(self) -> Path:
        return self.images[self.idx]

    def intended_pov(self, path: Path) -> str:
        key = path.stem
        pov = self.pov_map.get(key)
        if pov in {"white", "black"}:
            return pov
        return infer_pov_from_name(key)

    def show_current(self) -> None:
        if not self.images:
            self.counter_label.config(text="No images found.")
            self.pov_label.config(text="")
            return

        p = self.current()
        pov = self.intended_pov(p)

        self.counter_label.config(text=f"{self.idx+1}/{len(self.images)}   {p.name}")
        self.pov_label.config(text=f"Intended POV: {pov.upper()}")

        img = Image.open(p).convert("RGB")

        # Scale for viewing
        max_w, max_h = 860, 560
        w, h = img.size
        scale = min(max_w / w, max_h / h, 4.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_disp = img.resize((new_w, new_h), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(img_disp)
        self.img_label.config(image=self.photo)

        save_state(self.state_json, self.idx)

    def rotate_ccw(self) -> None:
        self._rotate(+90, "rotate_ccw")

    def rotate_cw(self) -> None:
        self._rotate(-90, "rotate_cw")

    def rotate_180(self) -> None:
        self._rotate(180, "rotate_180")

    def _rotate(self, degrees: int, action: str) -> None:
        p = self.current()
        pov = self.intended_pov(p)

        try:
            rotate_image_inplace(p, degrees)
            append_log(self.log_csv, {
                "path": str(p),
                "stem": p.stem,
                "action": action,
                "degrees": str(degrees),
                "intended_pov": pov,
            })
        except Exception as e:
            messagebox.showerror("Rotate failed", f"Could not rotate:\n{p}\n\n{e}")
            return

        # stay on same image so you can rotate multiple times
        self.show_current()

    def pass_image(self) -> None:
        if self.idx + 1 >= len(self.images):
            messagebox.showinfo("Done", "Reached the end of the image list.")
            return
        self.idx += 1
        self.show_current()

    def prev_image(self) -> None:
        if self.idx <= 0:
            return
        self.idx -= 1
        self.show_current()

    def quit(self) -> None:
        save_state(self.state_json, self.idx)
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok_dir", type=str, required=True,
                    help="Folder with OK rectified boards (e.g., data/chessred2k_rect/real_ok)")
    ap.add_argument("--pairs_csv", type=str, default="",
                    help="Optional: pairs_ok.csv to read intended POV (column 'pov' or 'pov_desired')")
    ap.add_argument("--log_csv", type=str, default="manual_rotations.csv",
                    help="Where to append rotation actions")
    ap.add_argument("--state_json", type=str, default="manual_rotate_state.json",
                    help="Where to store resume state (current index)")
    ap.add_argument("--reset", action="store_true",
                    help="Ignore existing state and start from the first image")
    args = ap.parse_args()

    ok_dir = Path(args.ok_dir)
    if not ok_dir.exists():
        raise SystemExit(f"[ERROR] ok_dir does not exist: {ok_dir}")

    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else None
    log_csv = Path(args.log_csv)
    state_json = Path(args.state_json)

    if args.reset and state_json.exists():
        try:
            state_json.unlink()
        except Exception:
            pass

    images = list_images(ok_dir)
    if not images:
        raise SystemExit(f"[ERROR] No images found in: {ok_dir}")

    pov_map = load_pov_map(pairs_csv)

    root = tk.Tk()
    ReviewerApp(root, images, pov_map, state_json, log_csv)
    root.mainloop()


if __name__ == "__main__":
    main()
