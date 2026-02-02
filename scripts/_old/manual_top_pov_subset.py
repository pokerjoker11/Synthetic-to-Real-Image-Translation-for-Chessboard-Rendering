#!/usr/bin/env python3
"""
Manual "Top-POV subset" selector for rectified OK boards.

Goal:
- Iterate through all images in --ok_dir (e.g., .../real_ok)
- For each image, you decide if it's "extra aligned with top POV"
- If YES: move it into --top_dir (your smaller subset bucket)
- If NO: keep it in ok_dir

UI:
- Counter/filename ABOVE the image
- Image in the middle
- Two buttons below:
    Left  = "Top POV (move)"  -> moves file to top_dir
    Right = "Keep"           -> keeps file where it is

Keybinds:
- Left Arrow / J  = Top POV (move)
- Right Arrow / K = Keep
- Backspace       = Back (does NOT undo moves)
- Q / Esc         = Quit

Logs decisions to CSV and saves resume state.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict

import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(root_dir: Path) -> List[Path]:
    files = []
    for p in root_dir.rglob("*"):
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
        fieldnames = ["path", "action", "moved_to"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(dst)  # atomic on same filesystem
    except Exception:
        shutil.copy2(src, dst)
        src.unlink()


class TopPOVApp:
    def __init__(self, root: tk.Tk, ok_dir: Path, top_dir: Path,
                 images: List[Path], state_json: Path, log_csv: Path):
        self.root = root
        self.ok_dir = ok_dir
        self.top_dir = top_dir
        self.images = images
        self.state_json = state_json
        self.log_csv = log_csv

        self.idx = load_state(state_json)
        self.idx = max(0, min(self.idx, max(0, len(self.images) - 1)))

        self.root.title("Pick Top-POV Subset (Move / Keep)")
        self.root.geometry("980x760")

        # Counter above image
        self.counter_label = tk.Label(root, text="", font=("Arial", 16))
        self.counter_label.pack(pady=10)

        # Image
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=6)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=14)

        self.btn_move = tk.Button(
            btn_frame,
            text="Top POV (move)",
            width=22,
            command=self.move_top
        )
        self.btn_move.grid(row=0, column=0, padx=14)

        self.btn_keep = tk.Button(
            btn_frame,
            text="Keep",
            width=18,
            command=self.keep
        )
        self.btn_keep.grid(row=0, column=1, padx=14)

        # Keybinds
        root.bind("<Left>", lambda e: self.move_top())
        root.bind("<j>", lambda e: self.move_top())
        root.bind("<J>", lambda e: self.move_top())

        root.bind("<Right>", lambda e: self.keep())
        root.bind("<k>", lambda e: self.keep())
        root.bind("<K>", lambda e: self.keep())

        root.bind("<BackSpace>", lambda e: self.back())
        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<q>", lambda e: self.quit())
        root.bind("<Q>", lambda e: self.quit())

        self.photo = None
        self.show_current()

    def current(self) -> Path:
        return self.images[self.idx]

    def show_current(self) -> None:
        if not self.images:
            self.counter_label.config(text="No images found.")
            return

        # Skip missing (moved earlier / deleted)
        while self.idx < len(self.images) and not self.images[self.idx].exists():
            self.idx += 1

        if self.idx >= len(self.images):
            messagebox.showinfo("Done", "Reached the end of the list.")
            self.quit()
            return

        p = self.current()
        rel = p.relative_to(self.ok_dir) if p.is_relative_to(self.ok_dir) else p.name
        self.counter_label.config(text=f"{self.idx+1}/{len(self.images)}   {rel}")

        img = Image.open(p).convert("RGB")

        # Scale for viewing
        max_w, max_h = 880, 600
        w, h = img.size
        scale = min(max_w / w, max_h / h, 4.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_disp = img.resize((new_w, new_h), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(img_disp)
        self.img_label.config(image=self.photo)

        save_state(self.state_json, self.idx)

    def keep(self) -> None:
        p = self.current()
        append_log(self.log_csv, {"path": str(p), "action": "keep", "moved_to": ""})
        self._next()

    def move_top(self) -> None:
        p = self.current()
        try:
            rel = p.relative_to(self.ok_dir)
        except Exception:
            rel = Path(p.name)

        dst = self.top_dir / rel
        try:
            safe_move(p, dst)
            append_log(self.log_csv, {"path": str(p), "action": "move_top", "moved_to": str(dst)})
        except Exception as e:
            messagebox.showerror("Move failed", f"Could not move:\n{p}\n-> {dst}\n\n{e}")
            return

        self._next()

    def _next(self) -> None:
        if self.idx + 1 >= len(self.images):
            messagebox.showinfo("Done", "Reached the end of the list.")
            self.quit()
            return
        self.idx += 1
        self.show_current()

    def back(self) -> None:
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
    ap.add_argument("--top_dir", type=str, required=True,
                    help="Destination folder for selected Top-POV boards (e.g., data/chessred2k_rect/real_top_pov)")
    ap.add_argument("--log_csv", type=str, default="manual_top_pov_select.csv",
                    help="Decision log CSV")
    ap.add_argument("--state_json", type=str, default="manual_top_pov_select_state.json",
                    help="Resume state JSON")
    ap.add_argument("--reset", action="store_true", help="Start from the first image")
    args = ap.parse_args()

    ok_dir = Path(args.ok_dir)
    top_dir = Path(args.top_dir)
    log_csv = Path(args.log_csv)
    state_json = Path(args.state_json)

    if not ok_dir.exists():
        raise SystemExit(f"[ERROR] ok_dir does not exist: {ok_dir}")
    top_dir.mkdir(parents=True, exist_ok=True)

    if args.reset and state_json.exists():
        try:
            state_json.unlink()
        except Exception:
            pass

    images = list_images(ok_dir)
    if not images:
        raise SystemExit(f"[ERROR] No images found in: {ok_dir}")

    root = tk.Tk()
    TopPOVApp(root, ok_dir=ok_dir, top_dir=top_dir,
              images=images, state_json=state_json, log_csv=log_csv)
    root.mainloop()


if __name__ == "__main__":
    main()
