#!/usr/bin/env python3
"""
Manual FEN annotation tool for rejected images.

Displays real game image with an empty grid overlay.
Click to place pieces and build the correct FEN manually.

Usage:
    python scripts/manual_annotate_fen.py
    python scripts/manual_annotate_fen.py --csv data/splits_rect/train.csv
    python scripts/manual_annotate_fen.py --rejected-only  # Only show rejected from review

Controls:
    Left Click  = Place selected piece
    Right Click = Remove piece from square
    C           = Clear board
    S           = Save current FEN and move to next
    R           = Reject image (bad quality)
    Q           = Quit and save progress
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
    print("Error: tkinter not available. Install with: pip install tk")
    sys.exit(1)

from PIL import Image, ImageTk, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

# Piece types
PIECE_TYPES = ['K', 'Q', 'R', 'B', 'N', 'P']
PIECE_NAMES = {
    'K': 'King', 'Q': 'Queen', 'R': 'Rook',
    'B': 'Bishop', 'N': 'Knight', 'P': 'Pawn'
}


def board_to_fen(board: list) -> str:
    """Convert 8x8 board array to FEN string."""
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows) + " w - - 0 1"


def fen_to_board(fen: str) -> list:
    """Convert FEN string to 8x8 board array."""
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
        # Pad if needed
        while len(row) < 8:
            row.append('.')
        board.append(row[:8])
    
    # Ensure 8 rows
    while len(board) < 8:
        board.append(['.'] * 8)
    
    return board[:8]


class AnnotationApp:
    def __init__(self, csv_path: Path, rejected_only: bool = False):
        self.csv_path = csv_path
        self.rejected_only = rejected_only
        
        # Load data
        self.rows = []
        self.row_indices = []  # Original indices in CSV
        self.load_data()
        
        # Current state
        self.current_idx = 0
        self.board = [['.'] * 8 for _ in range(8)]  # Empty board
        self.selected_color = 'white'  # 'white' or 'black'
        self.selected_piece = 'P'  # Default to pawn
        
        # Results
        self.annotations = {}  # idx -> {'fen': str, 'status': 'annotated'|'rejected'}
        self.progress_file = REPO_ROOT / "data" / "annotation_progress.csv"
        self.load_progress()
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Manual FEN Annotation Tool")
        self.root.geometry("900x800")
        self.root.configure(bg='#2b2b2b')
        
        self.setup_ui()
        self.bind_keys()
        self.show_current()
    
    def load_data(self):
        """Load CSV and optionally filter to rejected items."""
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        if self.rejected_only:
            # Load review progress to find rejected items
            review_file = REPO_ROOT / "data" / "review_progress.csv"
            rejected_indices = set()
            
            if review_file.exists():
                with open(review_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['decision'] == 'reject':
                            rejected_indices.add(int(row['idx']))
            
            # Filter to rejected only
            for idx, row in enumerate(all_rows):
                if idx in rejected_indices:
                    self.rows.append(row)
                    self.row_indices.append(idx)
            
            print(f"Loaded {len(self.rows)} rejected items from {len(all_rows)} total")
        else:
            self.rows = all_rows
            self.row_indices = list(range(len(all_rows)))
            print(f"Loaded {len(self.rows)} rows from {self.csv_path}")
    
    def load_progress(self):
        """Load previous annotation progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row['original_idx'])
                    self.annotations[idx] = {
                        'fen': row.get('fen', ''),
                        'status': row.get('status', 'annotated')
                    }
            
            # Find next unannotated
            for i, orig_idx in enumerate(self.row_indices):
                if orig_idx not in self.annotations:
                    self.current_idx = i
                    break
            
            print(f"Loaded {len(self.annotations)} previous annotations")
    
    def save_progress(self):
        """Save annotation progress."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.progress_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['original_idx', 'fen', 'status', 'timestamp'])
            writer.writeheader()
            
            for idx, data in self.annotations.items():
                writer.writerow({
                    'original_idx': idx,
                    'fen': data['fen'],
                    'status': data['status'],
                    'timestamp': datetime.now().isoformat()
                })
        
        print(f"Progress saved: {len(self.annotations)} annotations")
    
    def save_corrected_csv(self):
        """Save CSV with corrected FENs."""
        output_path = self.csv_path.parent / f"{self.csv_path.stem}_corrected.csv"
        
        # Read original CSV
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Apply corrections
        corrected_rows = []
        for idx, row in enumerate(all_rows):
            if idx in self.annotations:
                ann = self.annotations[idx]
                if ann['status'] == 'annotated' and ann['fen']:
                    row = dict(row)
                    row['fen'] = ann['fen']
                    corrected_rows.append(row)
                # Skip rejected
            else:
                corrected_rows.append(row)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corrected_rows)
        
        print(f"Saved {len(corrected_rows)} rows to {output_path}")
        return output_path
    
    def setup_ui(self):
        """Setup the GUI."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress label
        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var, font=('Arial', 12))
        progress_label.pack(pady=5)
        
        # Image canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas for image and grid
        self.img_size = 512
        self.canvas = tk.Canvas(canvas_frame, width=self.img_size, height=self.img_size, 
                                bg='gray', highlightthickness=2, highlightbackground='black')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)
        
        # FEN display
        self.fen_var = tk.StringVar(value="8/8/8/8/8/8/8/8 w - - 0 1")
        fen_frame = ttk.Frame(main_frame)
        fen_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fen_frame, text="Current FEN:").pack(side=tk.LEFT)
        fen_entry = ttk.Entry(fen_frame, textvariable=self.fen_var, width=60, font=('Consolas', 10))
        fen_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(fen_frame, text="Load FEN", command=self.load_fen_from_entry).pack(side=tk.LEFT, padx=5)
        
        # Piece selection frame
        piece_frame = ttk.LabelFrame(main_frame, text="Piece Selection", padding=10)
        piece_frame.pack(fill=tk.X, pady=10)
        
        # Color selection
        color_frame = ttk.Frame(piece_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(color_frame, text="Color:", font=('Arial', 11)).pack(side=tk.LEFT, padx=5)
        
        self.color_var = tk.StringVar(value='white')
        self.white_btn = ttk.Radiobutton(color_frame, text="White (1)", variable=self.color_var, 
                                         value='white', command=self.update_selection)
        self.white_btn.pack(side=tk.LEFT, padx=10)
        
        self.black_btn = ttk.Radiobutton(color_frame, text="Black (2)", variable=self.color_var,
                                         value='black', command=self.update_selection)
        self.black_btn.pack(side=tk.LEFT, padx=10)
        
        # Piece type selection
        type_frame = ttk.Frame(piece_frame)
        type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(type_frame, text="Piece:", font=('Arial', 11)).pack(side=tk.LEFT, padx=5)
        
        self.piece_var = tk.StringVar(value='P')
        self.piece_buttons = {}
        for i, piece in enumerate(PIECE_TYPES):
            btn = ttk.Radiobutton(type_frame, text=f"{PIECE_NAMES[piece]} ({piece})", 
                                  variable=self.piece_var, value=piece, command=self.update_selection)
            btn.pack(side=tk.LEFT, padx=5)
            self.piece_buttons[piece] = btn
        
        # Current selection display
        self.selection_var = tk.StringVar(value="Selected: White Pawn (P)")
        selection_label = ttk.Label(piece_frame, textvariable=self.selection_var, 
                                    font=('Arial', 12, 'bold'))
        selection_label.pack(pady=5)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Clear Board (C)", command=self.clear_board, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Load Original FEN", command=self.load_original_fen, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="SAVE & NEXT (S)", command=self.save_and_next, width=15).pack(side=tk.LEFT, padx=20)
        ttk.Button(action_frame, text="REJECT IMAGE (R)", command=self.reject_image, width=18).pack(side=tk.RIGHT, padx=5)
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_item, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save & Quit (Q)", command=self.quit_app, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Stats
        self.stats_var = tk.StringVar()
        stats_label = ttk.Label(main_frame, textvariable=self.stats_var, font=('Arial', 10))
        stats_label.pack(pady=5)
    
    def bind_keys(self):
        """Bind keyboard shortcuts."""
        self.root.bind('c', lambda e: self.clear_board())
        self.root.bind('C', lambda e: self.clear_board())
        
        self.root.bind('s', lambda e: self.save_and_next())
        self.root.bind('S', lambda e: self.save_and_next())
        
        self.root.bind('r', lambda e: self.reject_image())
        self.root.bind('R', lambda e: self.reject_image())
        
        self.root.bind('q', lambda e: self.quit_app())
        self.root.bind('Q', lambda e: self.quit_app())
        
        self.root.bind('<Escape>', lambda e: self.quit_app())
        
        # Color shortcuts
        self.root.bind('1', lambda e: self.set_color('white'))
        self.root.bind('2', lambda e: self.set_color('black'))
        
        # Piece shortcuts
        self.root.bind('k', lambda e: self.set_piece('K'))
        self.root.bind('K', lambda e: self.set_piece('K'))
        self.root.bind('q', lambda e: self.set_piece('Q'))  # Note: conflicts with quit, but quit needs Escape
        self.root.bind('b', lambda e: self.set_piece('B'))
        self.root.bind('B', lambda e: self.set_piece('B'))
        self.root.bind('n', lambda e: self.set_piece('N'))
        self.root.bind('N', lambda e: self.set_piece('N'))
        self.root.bind('p', lambda e: self.set_piece('P'))
        self.root.bind('P', lambda e: self.set_piece('P'))
        
        # Arrow keys for navigation
        self.root.bind('<Left>', lambda e: self.prev_item())
        self.root.bind('<Right>', lambda e: self.next_item())
    
    def set_color(self, color):
        self.color_var.set(color)
        self.update_selection()
    
    def set_piece(self, piece):
        self.piece_var.set(piece)
        self.update_selection()
    
    def update_selection(self):
        color = self.color_var.get()
        piece = self.piece_var.get()
        piece_name = PIECE_NAMES.get(piece, piece)
        self.selection_var.set(f"Selected: {color.title()} {piece_name} ({piece if color == 'white' else piece.lower()})")
        self.selected_color = color
        self.selected_piece = piece
    
    def show_current(self):
        """Display the current image."""
        if self.current_idx >= len(self.rows):
            messagebox.showinfo("Complete", f"All {len(self.rows)} items processed!")
            self.quit_app()
            return
        
        row = self.rows[self.current_idx]
        orig_idx = self.row_indices[self.current_idx]
        
        fen = row.get('fen', '')
        viewpoint = row.get('viewpoint', 'white')
        real_rel = row.get('real', '').replace('\\', '/')
        real_path = REPO_ROOT / real_rel
        
        # Store viewpoint for coordinate translation
        self.current_viewpoint = viewpoint
        
        # Update progress
        annotated = sum(1 for a in self.annotations.values() if a['status'] == 'annotated')
        rejected = sum(1 for a in self.annotations.values() if a['status'] == 'rejected')
        self.progress_var.set(f"Item {self.current_idx + 1} / {len(self.rows)}  |  Original row: {orig_idx}")
        self.stats_var.set(f"✓ Annotated: {annotated}  |  ✗ Rejected: {rejected}  |  Remaining: {len(self.rows) - len(self.annotations)}")
        
        # Load or initialize board
        if orig_idx in self.annotations and self.annotations[orig_idx]['fen']:
            self.board = fen_to_board(self.annotations[orig_idx]['fen'])
        else:
            # Start with empty board
            self.board = [['.'] * 8 for _ in range(8)]
        
        # Load real image
        if real_path.exists():
            self.real_img = Image.open(real_path).convert('RGBA')
            self.real_img = self.real_img.resize((self.img_size, self.img_size), Image.BICUBIC)
        else:
            self.real_img = Image.new('RGBA', (self.img_size, self.img_size), (200, 200, 200, 255))
            draw = ImageDraw.Draw(self.real_img)
            draw.text((self.img_size//4, self.img_size//2), "Image not found", fill='red')
        
        self.update_canvas()
        self.update_fen_display()
    
    def update_canvas(self):
        """Redraw the canvas with image and grid overlay."""
        # Create composite image
        composite = self.real_img.copy()
        draw = ImageDraw.Draw(composite)
        
        square_size = self.img_size // 8
        
        # Draw grid
        grid_color = (0, 255, 0, 150)
        for i in range(9):
            pos = i * square_size
            draw.line([(pos, 0), (pos, self.img_size)], fill=grid_color, width=2)
            draw.line([(0, pos), (self.img_size, pos)], fill=grid_color, width=2)
        
        # Draw pieces
        try:
            font_size = int(square_size * 0.7)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':
                    # Adjust for viewpoint
                    if self.current_viewpoint == 'black':
                        display_col = 7 - col
                        display_row = 7 - row
                    else:
                        display_col = col
                        display_row = row
                    
                    x1 = display_col * square_size
                    y1 = display_row * square_size
                    
                    is_white = piece.isupper()
                    symbol = piece.upper()
                    
                    if is_white:
                        piece_color = (255, 255, 0, 255)  # Yellow
                        outline_color = (0, 0, 0, 255)
                    else:
                        piece_color = (255, 0, 0, 255)  # Red
                        outline_color = (255, 255, 255, 255)
                    
                    try:
                        bbox = font.getbbox(symbol)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except:
                        tw, th = square_size // 2, square_size // 2
                    
                    tx = x1 + (square_size - tw) // 2
                    ty = y1 + (square_size - th) // 2 - 3
                    
                    # Draw outline
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if dx or dy:
                                draw.text((tx+dx, ty+dy), symbol, fill=outline_color, font=font)
                    draw.text((tx, ty), symbol, fill=piece_color, font=font)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(composite.convert('RGB'))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def update_fen_display(self):
        """Update the FEN display."""
        fen = board_to_fen(self.board)
        self.fen_var.set(fen)
    
    def on_canvas_click(self, event):
        """Handle left click - place piece."""
        col, row = self.canvas_to_board(event.x, event.y)
        if 0 <= row < 8 and 0 <= col < 8:
            piece = self.selected_piece if self.selected_color == 'white' else self.selected_piece.lower()
            self.board[row][col] = piece
            self.update_canvas()
            self.update_fen_display()
    
    def on_canvas_right_click(self, event):
        """Handle right click - remove piece."""
        col, row = self.canvas_to_board(event.x, event.y)
        if 0 <= row < 8 and 0 <= col < 8:
            self.board[row][col] = '.'
            self.update_canvas()
            self.update_fen_display()
    
    def canvas_to_board(self, x, y):
        """Convert canvas coordinates to board coordinates."""
        square_size = self.img_size // 8
        display_col = x // square_size
        display_row = y // square_size
        
        # Adjust for viewpoint
        if self.current_viewpoint == 'black':
            col = 7 - display_col
            row = 7 - display_row
        else:
            col = display_col
            row = display_row
        
        return col, row
    
    def clear_board(self):
        """Clear all pieces from board."""
        self.board = [['.'] * 8 for _ in range(8)]
        self.update_canvas()
        self.update_fen_display()
    
    def load_original_fen(self):
        """Load the original FEN from CSV."""
        if self.current_idx < len(self.rows):
            row = self.rows[self.current_idx]
            fen = row.get('fen', '')
            if fen:
                self.board = fen_to_board(fen)
                self.update_canvas()
                self.update_fen_display()
    
    def load_fen_from_entry(self):
        """Load FEN from the entry field."""
        fen = self.fen_var.get()
        if fen:
            self.board = fen_to_board(fen)
            self.update_canvas()
            self.update_fen_display()
    
    def save_and_next(self):
        """Save current annotation and move to next."""
        orig_idx = self.row_indices[self.current_idx]
        fen = board_to_fen(self.board)
        
        self.annotations[orig_idx] = {
            'fen': fen,
            'status': 'annotated'
        }
        
        self.next_unannotated()
    
    def reject_image(self):
        """Reject current image."""
        orig_idx = self.row_indices[self.current_idx]
        
        self.annotations[orig_idx] = {
            'fen': '',
            'status': 'rejected'
        }
        
        self.next_unannotated()
    
    def next_unannotated(self):
        """Go to next unannotated item."""
        for i in range(self.current_idx + 1, len(self.rows)):
            if self.row_indices[i] not in self.annotations:
                self.current_idx = i
                self.show_current()
                return
        
        # All done or go to next
        self.current_idx += 1
        self.show_current()
    
    def prev_item(self):
        """Go to previous item."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()
    
    def next_item(self):
        """Go to next item."""
        if self.current_idx < len(self.rows) - 1:
            self.current_idx += 1
            self.show_current()
    
    def quit_app(self):
        """Save and quit."""
        self.save_progress()
        output_path = self.save_corrected_csv()
        
        annotated = sum(1 for a in self.annotations.values() if a['status'] == 'annotated')
        rejected = sum(1 for a in self.annotations.values() if a['status'] == 'rejected')
        
        messagebox.showinfo("Saved", f"Progress saved!\n\n"
                           f"Annotated: {annotated}\n"
                           f"Rejected: {rejected}\n\n"
                           f"Corrected CSV: {output_path}")
        
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Manual FEN annotation tool")
    parser.add_argument('--csv', type=str, default='data/splits_rect/train.csv',
                        help="CSV file to annotate")
    parser.add_argument('--rejected-only', action='store_true',
                        help="Only show items rejected in the review tool")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1
    
    app = AnnotationApp(csv_path, args.rejected_only)
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
