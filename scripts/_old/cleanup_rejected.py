#!/usr/bin/env python3
"""
Review and delete rejected frames and their synthetic counterparts.

Loads rejected entries from review_progress.csv, displays each one,
and allows you to confirm deletion or keep the files.

Usage:
    python scripts/cleanup_rejected.py
    python scripts/cleanup_rejected.py --csv data/review_progress.csv
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Error: tkinter not available")
    sys.exit(1)

from PIL import Image, ImageTk, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH_V3_DIR = REPO_ROOT / "data" / "synth_v3" / "images"


def generate_synth_filename(row_idx: int, viewpoint: str, fen: str) -> str:
    """Generate the expected synth filename for a row."""
    fen_hash = hashlib.md5(fen.encode()).hexdigest()[:8]
    return f"row{row_idx:06d}_{viewpoint}_{fen_hash}.png"


class CleanupApp:
    def __init__(self, review_csv: Path, source_csv: Path):
        self.review_csv = review_csv
        self.source_csv = source_csv
        
        # Load rejected entries
        self.rejected = []
        self.load_rejected()
        
        if not self.rejected:
            print("No rejected entries found!")
            sys.exit(0)
        
        self.current_idx = 0
        self.to_delete = []
        self.to_keep = []
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Cleanup Rejected Frames")
        self.root.geometry("700x700")
        
        self.setup_ui()
        self.bind_keys()
        self.show_current()
    
    def load_rejected(self):
        """Load rejected entries from review progress CSV."""
        # Load review decisions
        rejected_indices = set()
        if self.review_csv.exists():
            with open(self.review_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('decision') == 'reject':
                        rejected_indices.add(int(row['idx']))
        
        print(f"Found {len(rejected_indices)} rejected entries in review CSV")
        
        # Load source CSV to get full info
        if self.source_csv.exists():
            with open(self.source_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
            
            for idx in sorted(rejected_indices):
                if idx < len(all_rows):
                    row = all_rows[idx]
                    self.rejected.append({
                        'idx': idx,
                        'real': row.get('real', ''),
                        'fen': row.get('fen', ''),
                        'viewpoint': row.get('viewpoint', 'white'),
                    })
        
        print(f"Loaded {len(self.rejected)} rejected entries with details")
    
    def setup_ui(self):
        """Setup the GUI."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress
        self.progress_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.progress_var, font=('Arial', 12)).pack(pady=5)
        
        # Image display
        self.img_label = ttk.Label(main_frame)
        self.img_label.pack(pady=10)
        
        # Info
        self.info_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.info_var, font=('Consolas', 9), wraplength=650).pack(pady=5)
        
        # File paths
        self.files_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.files_var, font=('Consolas', 8), wraplength=650, foreground='gray').pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(btn_frame, text="KEEP (K)", command=self.keep, width=20).pack(side=tk.LEFT, padx=20)
        ttk.Button(btn_frame, text="DELETE (D)", command=self.delete, width=20).pack(side=tk.RIGHT, padx=20)
        
        # Stats
        self.stats_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.stats_var, font=('Arial', 10)).pack(pady=5)
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="< Prev", command=self.prev_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next >", command=self.next_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="FINISH & EXECUTE", command=self.finish, width=20).pack(side=tk.RIGHT, padx=5)
    
    def bind_keys(self):
        """Bind keyboard shortcuts."""
        self.root.bind('k', lambda e: self.keep())
        self.root.bind('K', lambda e: self.keep())
        self.root.bind('<Left>', lambda e: self.keep())
        
        self.root.bind('d', lambda e: self.delete())
        self.root.bind('D', lambda e: self.delete())
        self.root.bind('<Right>', lambda e: self.delete())
        self.root.bind('<Delete>', lambda e: self.delete())
        
        self.root.bind('<Escape>', lambda e: self.quit_no_action())
        self.root.bind('q', lambda e: self.quit_no_action())
    
    def show_current(self):
        """Display current rejected entry."""
        if self.current_idx >= len(self.rejected):
            self.finish()
            return
        
        entry = self.rejected[self.current_idx]
        idx = entry['idx']
        real_rel = entry['real'].replace('\\', '/')
        fen = entry['fen']
        viewpoint = entry['viewpoint']
        
        real_path = REPO_ROOT / real_rel
        synth_filename = generate_synth_filename(idx, viewpoint, fen)
        synth_path = SYNTH_V3_DIR / synth_filename
        
        # Update progress
        self.progress_var.set(f"Rejected {self.current_idx + 1} / {len(self.rejected)}")
        self.stats_var.set(f"To Delete: {len(self.to_delete)}  |  To Keep: {len(self.to_keep)}  |  Remaining: {len(self.rejected) - self.current_idx}")
        
        # Update info
        self.info_var.set(f"Row: {idx}  |  FEN: {fen[:50]}...  |  View: {viewpoint}")
        
        # File status
        real_exists = "EXISTS" if real_path.exists() else "NOT FOUND"
        synth_exists = "EXISTS" if synth_path.exists() else "NOT FOUND"
        self.files_var.set(f"Real: {real_rel} [{real_exists}]\nSynth: {synth_filename} [{synth_exists}]")
        
        # Load and display real image
        img_size = 400
        if real_path.exists():
            img = Image.open(real_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.BICUBIC)
        else:
            img = Image.new('RGB', (img_size, img_size), (100, 100, 100))
            draw = ImageDraw.Draw(img)
            draw.text((img_size//4, img_size//2), "Image not found", fill='red')
        
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo)
    
    def keep(self):
        """Keep this entry (don't delete)."""
        if self.current_idx < len(self.rejected):
            entry = self.rejected[self.current_idx]
            self.to_keep.append(entry)
            self.current_idx += 1
            self.show_current()
    
    def delete(self):
        """Mark this entry for deletion."""
        if self.current_idx < len(self.rejected):
            entry = self.rejected[self.current_idx]
            self.to_delete.append(entry)
            self.current_idx += 1
            self.show_current()
    
    def prev_item(self):
        """Go to previous item."""
        if self.current_idx > 0:
            # Remove from lists if it was decided
            entry = self.rejected[self.current_idx - 1]
            if entry in self.to_delete:
                self.to_delete.remove(entry)
            if entry in self.to_keep:
                self.to_keep.remove(entry)
            self.current_idx -= 1
            self.show_current()
    
    def next_item(self):
        """Go to next item (skip)."""
        if self.current_idx < len(self.rejected) - 1:
            self.current_idx += 1
            self.show_current()
    
    def finish(self):
        """Execute deletions."""
        if not self.to_delete:
            messagebox.showinfo("Done", f"No files to delete.\nKept: {len(self.to_keep)}")
            self.root.destroy()
            return
        
        # Confirm
        msg = f"Delete {len(self.to_delete)} rejected entries?\n\nThis will remove:\n"
        msg += f"- Real images (if exist)\n"
        msg += f"- Synthetic images (if exist)\n\n"
        msg += f"Keeping: {len(self.to_keep)} entries"
        
        if not messagebox.askyesno("Confirm Deletion", msg):
            return
        
        # Execute deletions
        deleted_real = 0
        deleted_synth = 0
        
        for entry in self.to_delete:
            idx = entry['idx']
            real_rel = entry['real'].replace('\\', '/')
            fen = entry['fen']
            viewpoint = entry['viewpoint']
            
            # Delete real image
            real_path = REPO_ROOT / real_rel
            if real_path.exists():
                real_path.unlink()
                deleted_real += 1
            
            # Delete synth image
            synth_filename = generate_synth_filename(idx, viewpoint, fen)
            synth_path = SYNTH_V3_DIR / synth_filename
            if synth_path.exists():
                synth_path.unlink()
                deleted_synth += 1
        
        messagebox.showinfo("Deletion Complete", 
                           f"Deleted:\n"
                           f"  Real images: {deleted_real}\n"
                           f"  Synth images: {deleted_synth}\n\n"
                           f"Kept: {len(self.to_keep)} entries")
        
        self.root.destroy()
    
    def quit_no_action(self):
        """Quit without executing deletions."""
        if messagebox.askyesno("Quit", "Quit without deleting anything?"):
            self.root.destroy()
    
    def run(self):
        """Start the app."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Cleanup rejected frames")
    parser.add_argument('--review-csv', type=str, default='data/review_progress.csv',
                        help="Review progress CSV with decisions")
    parser.add_argument('--source-csv', type=str, default='data/new_games_annotations.csv',
                        help="Source CSV with full entry details")
    
    args = parser.parse_args()
    
    review_csv = REPO_ROOT / args.review_csv
    source_csv = REPO_ROOT / args.source_csv
    
    if not review_csv.exists():
        print(f"[ERROR] Review CSV not found: {review_csv}")
        return 1
    
    if not source_csv.exists():
        print(f"[ERROR] Source CSV not found: {source_csv}")
        return 1
    
    app = CleanupApp(review_csv, source_csv)
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
