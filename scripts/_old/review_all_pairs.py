#!/usr/bin/env python3
"""
Review ALL training pairs (including games 1-7).

This is a specialized version of manual_review_pairs.py that:
- Works on train_final.csv (all games)
- Uses a separate progress file (review_all_progress.csv)
- Outputs train_clean.csv when done

Usage:
    python scripts/review_all_pairs.py
    python scripts/review_all_pairs.py --start 100  # Start from row 100

Controls:
    A or Left Arrow  = Accept (keep this pair)
    R or Right Arrow = Reject (discard this pair - has hands or other issues)
    S                = Skip (decide later)
    Q                = Quit and save progress
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: tkinter not available")
    sys.exit(1)

from PIL import Image, ImageTk, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

PIECE_SYMBOLS = {
    'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
    'k': 'K', 'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N', 'p': 'P',
}


def fen_to_board_array(fen: str) -> list:
    """Convert FEN to 8x8 array"""
    board_part = fen.split()[0]
    ranks = board_part.split('/')
    
    board = []
    for rank_str in ranks:
        row = []
        for char in rank_str:
            if char.isdigit():
                row.extend(['.'] * int(char))
            else:
                row.append(char)
        board.append(row)
    return board


def create_board_overlay(fen: str, viewpoint: str = 'white', size: int = 400) -> Image.Image:
    """Create a semi-transparent board overlay from FEN."""
    board = fen_to_board_array(fen)
    
    if viewpoint == 'black':
        board = [row[::-1] for row in board[::-1]]
    
    square_size = size // 8
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font_size = int(square_size * 0.75)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            
            piece = board[row][col]
            if piece != '.':
                symbol = PIECE_SYMBOLS.get(piece, piece)
                is_white = piece.isupper()
                
                if is_white:
                    piece_color = (255, 255, 0, 230)
                    outline_color = (0, 0, 0, 255)
                else:
                    piece_color = (255, 0, 0, 230)
                    outline_color = (255, 255, 255, 255)
                
                try:
                    bbox = font.getbbox(symbol)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except:
                    tw, th = square_size // 2, square_size // 2
                
                tx = x1 + (square_size - tw) // 2
                ty = y1 + (square_size - th) // 2 - 3
                
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        if dx or dy:
                            draw.text((tx+dx, ty+dy), symbol, fill=outline_color, font=font)
                draw.text((tx, ty), symbol, fill=piece_color, font=font)
    
    grid_color = (0, 255, 0, 100)
    for i in range(9):
        pos = i * square_size
        draw.line([(pos, 0), (pos, size)], fill=grid_color, width=2)
        draw.line([(0, pos), (size, pos)], fill=grid_color, width=2)
    
    return img


class ReviewApp:
    def __init__(self, csv_path: Path, start_idx: int = 0):
        self.csv_path = csv_path
        self.start_idx = start_idx
        
        self.rows = []
        self.load_csv()
        
        self.accepted = []
        self.rejected = []
        self.current_idx = start_idx
        
        # Use separate progress file for full review
        self.progress_file = REPO_ROOT / "data" / "review_all_progress.csv"
        self.load_progress()
        
        self.root = tk.Tk()
        self.root.title("Full Dataset Review - Reject images with hands!")
        self.root.geometry("700x800")
        self.root.configure(bg='#2b2b2b')
        
        self.setup_ui()
        self.bind_keys()
        self.show_current()
    
    def load_csv(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        print(f"Loaded {len(self.rows)} rows from {self.csv_path}")
    
    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row['idx'])
                    decision = row['decision']
                    if decision == 'accept':
                        self.accepted.append(idx)
                    elif decision == 'reject':
                        self.rejected.append(idx)
            
            reviewed = set(self.accepted + self.rejected)
            for i in range(len(self.rows)):
                if i not in reviewed:
                    self.current_idx = max(self.start_idx, i)
                    break
            
            print(f"Loaded progress: {len(self.accepted)} accepted, {len(self.rejected)} rejected")
            print(f"Resuming from index {self.current_idx}")
    
    def save_progress(self):
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.progress_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['idx', 'decision', 'timestamp'])
            writer.writeheader()
            
            for idx in self.accepted:
                writer.writerow({'idx': idx, 'decision': 'accept', 'timestamp': datetime.now().isoformat()})
            for idx in self.rejected:
                writer.writerow({'idx': idx, 'decision': 'reject', 'timestamp': datetime.now().isoformat()})
        
        print(f"Progress saved: {len(self.accepted)} accepted, {len(self.rejected)} rejected")
    
    def save_clean_csv(self):
        """Save accepted rows to train_clean.csv"""
        output_path = REPO_ROOT / "data" / "splits_rect" / "train_clean.csv"
        
        accepted_rows = [self.rows[i] for i in sorted(self.accepted) if i < len(self.rows)]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if accepted_rows:
                writer = csv.DictWriter(f, fieldnames=accepted_rows[0].keys())
                writer.writeheader()
                writer.writerows(accepted_rows)
        
        print(f"Saved {len(accepted_rows)} clean rows to {output_path}")
        return output_path
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Warning banner
        warning_frame = tk.Frame(main_frame, bg='#ff6b6b', padx=10, pady=5)
        warning_frame.pack(fill=tk.X, pady=5)
        tk.Label(warning_frame, text="REJECT images with HANDS or other obstructions!", 
                 bg='#ff6b6b', fg='white', font=('Arial', 12, 'bold')).pack()
        
        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var, font=('Arial', 12))
        progress_label.pack(pady=5)
        
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        center_frame = ttk.LabelFrame(images_frame, text="Real Game + FEN Overlay (Yellow=White, Red=Black)", padding=5)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.overlay_label = ttk.Label(center_frame)
        self.overlay_label.pack()
        
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_var = tk.StringVar()
        info_label = ttk.Label(info_frame, textvariable=self.info_var, font=('Consolas', 9), wraplength=900)
        info_label.pack()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        style = ttk.Style()
        style.configure('Accept.TButton', font=('Arial', 12, 'bold'))
        style.configure('Reject.TButton', font=('Arial', 12, 'bold'))
        
        ttk.Button(button_frame, text="ACCEPT (A) - Clean image", command=self.accept, 
                   style='Accept.TButton', width=25).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(button_frame, text="SKIP (S)", command=self.skip, width=15).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(button_frame, text="REJECT (R) - Has hands/issues", command=self.reject,
                   style='Reject.TButton', width=25).pack(side=tk.RIGHT, padx=20)
        
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_var = tk.StringVar()
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var, font=('Arial', 10))
        stats_label.pack()
        
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Prev", command=self.prev_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Quit (Q)", command=self.quit_app, width=15).pack(side=tk.RIGHT, padx=5)
    
    def bind_keys(self):
        self.root.bind('a', lambda e: self.accept())
        self.root.bind('A', lambda e: self.accept())
        self.root.bind('<Left>', lambda e: self.accept())
        
        self.root.bind('r', lambda e: self.reject())
        self.root.bind('R', lambda e: self.reject())
        self.root.bind('<Right>', lambda e: self.reject())
        
        self.root.bind('s', lambda e: self.skip())
        self.root.bind('S', lambda e: self.skip())
        
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('Q', lambda e: self.quit_app())
        self.root.bind('<Escape>', lambda e: self.quit_app())
    
    def show_current(self):
        if self.current_idx >= len(self.rows):
            messagebox.showinfo("Complete", f"All {len(self.rows)} rows reviewed!\n\n"
                               f"Accepted: {len(self.accepted)}\nRejected: {len(self.rejected)}")
            self.quit_app()
            return
        
        row = self.rows[self.current_idx]
        fen = row.get('fen', '')
        viewpoint = row.get('viewpoint', 'white')
        real_rel = row.get('real', '').replace('\\', '/')
        real_path = REPO_ROOT / real_rel
        
        reviewed = len(self.accepted) + len(self.rejected)
        self.progress_var.set(f"Row {self.current_idx + 1} / {len(self.rows)}  |  Reviewed: {reviewed}")
        
        self.stats_var.set(f"Accepted: {len(self.accepted)}  |  Rejected: {len(self.rejected)}  |  Remaining: {len(self.rows) - reviewed}")
        
        self.info_var.set(f"FEN: {fen}\nViewpoint: {viewpoint}  |  File: {real_rel}")
        
        img_size = 512
        if real_path.exists():
            real_img = Image.open(real_path).convert('RGBA')
            real_img = real_img.resize((img_size, img_size), Image.BICUBIC)
        else:
            real_img = Image.new('RGBA', (img_size, img_size), (200, 200, 200, 255))
            draw = ImageDraw.Draw(real_img)
            draw.text((img_size//4, img_size//2), "Image not found", fill='red')
        
        fen_overlay = create_board_overlay(fen, viewpoint, img_size)
        
        combined = real_img.copy()
        combined.paste(fen_overlay, (0, 0), fen_overlay)
        combined_rgb = combined.convert('RGB')
        
        self.overlay_photo = ImageTk.PhotoImage(combined_rgb)
        self.overlay_label.configure(image=self.overlay_photo)
        
        if self.current_idx in self.accepted:
            self.progress_var.set(self.progress_var.get() + "  [ACCEPTED]")
        elif self.current_idx in self.rejected:
            self.progress_var.set(self.progress_var.get() + "  [REJECTED]")
    
    def accept(self):
        if self.current_idx in self.rejected:
            self.rejected.remove(self.current_idx)
        if self.current_idx not in self.accepted:
            self.accepted.append(self.current_idx)
        self.next_unreviewed()
    
    def reject(self):
        if self.current_idx in self.accepted:
            self.accepted.remove(self.current_idx)
        if self.current_idx not in self.rejected:
            self.rejected.append(self.current_idx)
        self.next_unreviewed()
    
    def skip(self):
        self.current_idx += 1
        self.show_current()
    
    def next_unreviewed(self):
        reviewed = set(self.accepted + self.rejected)
        for i in range(self.current_idx + 1, len(self.rows)):
            if i not in reviewed:
                self.current_idx = i
                self.show_current()
                return
        self.current_idx += 1
        self.show_current()
    
    def prev_item(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()
    
    def next_item(self):
        if self.current_idx < len(self.rows) - 1:
            self.current_idx += 1
            self.show_current()
    
    def quit_app(self):
        self.save_progress()
        output_path = self.save_clean_csv()
        
        messagebox.showinfo("Saved", f"Progress saved!\n\n"
                           f"Accepted: {len(self.accepted)}\n"
                           f"Rejected: {len(self.rejected)}\n\n"
                           f"Clean CSV: {output_path}")
        
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Review all training pairs")
    parser.add_argument('--start', type=int, default=0, help="Start from this row index")
    
    args = parser.parse_args()
    
    # Use train_final.csv
    csv_path = REPO_ROOT / "data" / "splits_rect" / "train_final.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1
    
    print("="*60)
    print("FULL DATASET REVIEW")
    print("Reject any images that have hands or other obstructions!")
    print("="*60)
    
    app = ReviewApp(csv_path, args.start)
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
