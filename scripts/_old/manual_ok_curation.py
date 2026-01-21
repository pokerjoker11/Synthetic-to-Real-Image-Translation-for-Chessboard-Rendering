#!/usr/bin/env python3
"""
Manual OK-board curation tool.

- Iterates images in --ok_dir (default: .../real_ok)
- Displays each image
- Buttons:
    Reject  -> moves the file from ok_dir to extreme_dir
    Approve -> keeps file and advances
- Also supports keys:
    Left Arrow / J = Reject
    Right Arrow / K = Approve
    Backspace = Back
    Q / Esc = Quit
- Saves resume state to --state_json and logs decisions to --log_csv

Notes:
- Uses atomic move when possible; falls back to copy+delete across filesystems.
- Preserves relative path under ok_dir when moving to extreme_dir.
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


class CurateApp:
    def __init__(self, root: tk.Tk, ok_dir: Path, extreme_dir: Path,
                 images: List[Path], state_json: Path, log_csv: Path):
        self.root = root
        self.ok_dir = ok_dir
        self.extreme_dir = extreme_dir
        self.images = images
        self.state_json = state_json
        self.log_csv = log_csv

        self.idx = load_state(state_json)
        self.idx = max(0, min(self.idx, max(0, len(self.images) - 1)))

        self.root.title("Curate OK Boards (Approve / Reject)")
        self.root.geometry("980x760")

        # Counter above image
        self.counter_label = tk.Label(root, text="", font=("Arial", 16))
        self.counter_label.pack(pady=10)

        # Image display
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=6)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=14)

        self.btn_reject = tk.Button(btn_frame, text="Reject (move to extreme)", width=24, command=self.reject)
        self.btn_reject.grid(row=0, column=0, padx=12)

        self.btn_approve = tk.Button(btn_frame, text="Approve", width=18, command=self.approve)
        self.btn_approve.grid(row=0, column=1, padx=12)

        self.btn_back = tk.Button(btn_frame, text="Back", width=12, command=self.back)
        self.btn_back.grid(row=0, column=2, padx=12)

        # Keybinds
        root.bind("<Left>", lambda e: self.reject())
        root.bind("<Right>", lambda e: self.approve())
        root.bind("<j>", lambda e: self.reject())
        root.bind("<J>", lambda e: self.reject())
        root.bind("<k>", lambda e: self.approve())
        root.bind("<K>", lambda e: self.approve())
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

        # Skip missing (e.g. moved previously)
        while self.idx < len(self.images) and not self.images[self.idx].exists():
            self.idx += 1
        if self.idx >= len(self.images):
            messagebox.showinfo("Done", "Reached the end of the list.")
            self.quit()
            return

        p = self.current()
        self.counter_label.config(text=f"{self.idx+1}/{len(self.images)}   {p.relative_to(self.ok_dir)}")

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

    def approve(self) -> None:
        p = self.current()
        append_log(self.log_csv, {"path": str(p), "action": "approve", "moved_to": ""})
        if self.idx + 1 >= len(self.images):
            messagebox.showinfo("Done", "Reached the end of the list.")
            self.quit()
            return
        self.idx += 1
        self.show_current()

    def reject(self) -> None:
        p = self.current()
        try:
            rel = p.relative_to(self.ok_dir)
        except Exception:
            rel = p.name

        dst = self.extreme_dir / rel
        try:
            safe_move(p, dst)
            append_log(self.log_csv, {"path": str(p), "action": "reject", "moved_to": str(dst)})
        except Exception as e:
            messagebox.showerror("Move failed", f"Could not move:\n{p}\n-> {dst}\n\n{e}")
            return

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
    ap.add_argument("--extreme_dir", type=str, required=True,
                    help="Folder to move rejected boards into (e.g., data/chessred2k_rect/real_extreme)")
    ap.add_argument("--log_csv", type=str, default="manual_ok_curation.csv",
                    help="Decision log CSV")
    ap.add_argument("--state_json", type=str, default="manual_ok_curation_state.json",
                    help="Resume state JSON")
    ap.add_argument("--reset", action="store_true", help="Start from the first image")
    args = ap.parse_args()

    ok_dir = Path(args.ok_dir)
    extreme_dir = Path(args.extreme_dir)
    log_csv = Path(args.log_csv)
    state_json = Path(args.state_json)

    if not ok_dir.exists():
        raise SystemExit(f"[ERROR] ok_dir does not exist: {ok_dir}")

    extreme_dir.mkdir(parents=True, exist_ok=True)

    if args.reset and state_json.exists():
        try:
            state_json.unlink()
        except Exception:
            pass

    images = list_images(ok_dir)
    if not images:
        raise SystemExit(f"[ERROR] No images found in: {ok_dir}")

    root = tk.Tk()
    CurateApp(root, ok_dir=ok_dir, extreme_dir=extreme_dir,
              images=images, state_json=state_json, log_csv=log_csv)
    root.mainloop()


if __name__ == "__main__":
    main()
