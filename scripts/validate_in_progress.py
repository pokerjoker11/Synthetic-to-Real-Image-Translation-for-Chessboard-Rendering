#!/usr/bin/env python3
"""
Validate rendered images while re-rendering is in progress.
Reads original train.csv but looks for images in synth_v3 folder.

Usage:
    python scripts/validate_in_progress.py --random 10
    python scripts/validate_in_progress.py --all
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTH_V3_DIR = REPO_ROOT / "data" / "synth_v3" / "images"
OUTPUT_DIR = REPO_ROOT / "results" / "validation"

# Use simple letters instead of Unicode symbols (more reliable rendering)
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


def create_board_diagram(fen: str, viewpoint: str = 'white', size: int = 512) -> Image.Image:
    """Create a 2D board diagram from FEN."""
    board = fen_to_board_array(fen)
    
    if viewpoint == 'black':
        board = [row[::-1] for row in board[::-1]]
    
    square_size = size // 8
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    light_square = (240, 217, 181)
    dark_square = (181, 136, 99)
    
    try:
        font_size = int(square_size * 0.7)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            is_light = (row + col) % 2 == 0
            color = light_square if is_light else dark_square
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            piece = board[row][col]
            if piece != '.':
                symbol = PIECE_SYMBOLS.get(piece, piece)
                is_white = piece.isupper()
                piece_color = (255, 255, 255) if is_white else (0, 0, 0)
                outline = (50, 50, 50) if is_white else light_square
                
                try:
                    bbox = font.getbbox(symbol)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except:
                    tw, th = square_size // 2, square_size // 2
                
                tx = x1 + (square_size - tw) // 2
                ty = y1 + (square_size - th) // 2 - 5
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx or dy:
                            draw.text((tx+dx, ty+dy), symbol, fill=outline, font=font)
                draw.text((tx, ty), symbol, fill=piece_color, font=font)
    
    draw.rectangle([0, 0, size-1, size-1], outline='black', width=2)
    return img


def generate_filename(row_idx: int, viewpoint: str, fen: str) -> str:
    """Generate expected filename for a row."""
    fen_hash = hashlib.md5(fen.encode()).hexdigest()[:8]
    return f"row{row_idx:06d}_{viewpoint}_{fen_hash}.png"


def main():
    parser = argparse.ArgumentParser(description="Validate in-progress renders")
    parser.add_argument('--random', type=int, default=10, help="Validate N random images")
    parser.add_argument('--all', action='store_true', help="Validate all rendered images")
    args = parser.parse_args()
    
    # Read original train.csv
    train_csv = REPO_ROOT / "data" / "splits_rect" / "train.csv"
    if not train_csv.exists():
        print(f"[ERROR] train.csv not found: {train_csv}")
        return 1
    
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows from train.csv")
    
    # Find which images exist in synth_v3
    if not SYNTH_V3_DIR.exists():
        print(f"[ERROR] synth_v3 folder not found: {SYNTH_V3_DIR}")
        return 1
    
    existing_images = set(p.name for p in SYNTH_V3_DIR.glob("*.png"))
    print(f"Found {len(existing_images)} rendered images in synth_v3")
    
    # Match rows to rendered images
    valid_rows = []
    for idx, row in enumerate(rows):
        fen = row.get('fen', '')
        viewpoint = row.get('viewpoint', 'white')
        expected_filename = generate_filename(idx, viewpoint, fen)
        
        if expected_filename in existing_images:
            valid_rows.append((idx, row, expected_filename))
    
    print(f"Matched {len(valid_rows)} rows to rendered images")
    
    if not valid_rows:
        print("[ERROR] No rendered images found to validate")
        return 1
    
    # Select subset
    if not args.all and args.random < len(valid_rows):
        valid_rows = random.sample(valid_rows, args.random)
        print(f"Randomly selected {len(valid_rows)} for validation")
    
    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Validate each
    for idx, row, filename in valid_rows:
        fen = row['fen']
        viewpoint = row['viewpoint']
        real_rel = row.get('real', '')
        synth_path = SYNTH_V3_DIR / filename
        real_path = REPO_ROOT / real_rel.replace('\\', '/')
        
        print(f"\nValidating row {idx}: {filename}")
        print(f"  FEN: {fen[:40]}...")
        print(f"  Real: {real_rel}")
        
        # Load rendered synthetic image
        synth_img = Image.open(synth_path).convert('RGB')
        size = max(synth_img.size)
        
        # Create FEN diagram
        diagram = create_board_diagram(fen, viewpoint, size)
        
        # Load real image if exists
        if real_path.exists():
            real_img = Image.open(real_path).convert('RGB')
            real_img = real_img.resize((size, size), Image.BICUBIC)
        else:
            # Create placeholder if real image not found
            real_img = Image.new('RGB', (size, size), (200, 200, 200))
            draw_placeholder = ImageDraw.Draw(real_img)
            draw_placeholder.text((size//4, size//2), "Real image\nnot found", fill='red')
            print(f"  [WARN] Real image not found: {real_path}")
        
        # Resize synth if needed
        if synth_img.size != (size, size):
            synth_img = synth_img.resize((size, size), Image.BICUBIC)
        
        # Three images side by side: FEN | Synth | Real
        gap = 10
        combined_width = size * 3 + gap * 2
        combined = Image.new('RGB', (combined_width, size + 50), (255, 255, 255))
        combined.paste(diagram, (0, 0))
        combined.paste(synth_img, (size + gap, 0))
        combined.paste(real_img, (size * 2 + gap * 2, 0))
        
        # Labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        label_y = size + 5
        draw.text((size//2 - 50, label_y), "Expected (FEN)", fill='black', font=font)
        draw.text((size + gap + size//2 - 50, label_y), "Rendered Synth", fill='black', font=font)
        draw.text((size*2 + gap*2 + size//2 - 40, label_y), "Real Game", fill='black', font=font)
        
        # Add FEN string at bottom
        try:
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            small_font = ImageFont.load_default()
        draw.text((10, size + 25), f"FEN: {fen[:80]}{'...' if len(fen) > 80 else ''}", fill='gray', font=small_font)
        draw.text((10, size + 38), f"View: {viewpoint} | Row: {idx}", fill='gray', font=small_font)
        
        # Save
        out_path = OUTPUT_DIR / f"validate_{idx:04d}.png"
        combined.save(out_path)
        print(f"  Saved: {out_path}")
    
    print(f"\n{'='*60}")
    print(f"Validation complete! Check images in: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
