#!/usr/bin/env python3
"""
Validate rendered synthetic images against expected FEN positions.

Creates side-by-side comparisons showing:
- Left: Simple 2D board diagram from FEN (ground truth)
- Right: Rendered synthetic image

Usage:
    # Validate a single image
    python scripts/validate_renders.py --image data/synth_v3/images/row000001_white_xxx.png --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --viewpoint white

    # Validate all images from a CSV
    python scripts/validate_renders.py --csv data/splits_rect/train_v3.csv --output results/validation/

    # Quick spot-check: validate random N images from CSV
    python scripts/validate_renders.py --csv data/splits_rect/train_v3.csv --random 10 --output results/validation/
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

# Piece symbols for drawing
PIECE_SYMBOLS = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}

# Simple ASCII representation as fallback
PIECE_ASCII = {
    'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
    'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
}


def parse_fen(fen: str) -> dict:
    """Parse FEN string into dict of {square: piece_char}"""
    board_part = fen.split()[0]
    ranks = board_part.split('/')
    
    positions = {}
    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for char in rank_str:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                rank_number = 8 - rank_idx
                square = f"{file_letter}{rank_number}"
                positions[square] = char
                file_idx += 1
    
    return positions


def fen_to_board_array(fen: str) -> list:
    """Convert FEN to 8x8 array (row 0 = rank 8, row 7 = rank 1)"""
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
    """
    Create a simple 2D board diagram from FEN.
    Returns a PIL Image.
    """
    board = fen_to_board_array(fen)
    
    # Flip board for black viewpoint
    if viewpoint == 'black':
        board = [row[::-1] for row in board[::-1]]
    
    square_size = size // 8
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Colors
    light_square = (240, 217, 181)  # Beige
    dark_square = (181, 136, 99)    # Brown
    white_piece_color = (255, 255, 255)
    black_piece_color = (0, 0, 0)
    outline_color = (50, 50, 50)
    
    # Try to load a font for pieces
    try:
        # Try to use a larger font
        font_size = int(square_size * 0.7)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw board and pieces
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            # Square color (alternating)
            is_light = (row + col) % 2 == 0
            color = light_square if is_light else dark_square
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Piece
            piece = board[row][col]
            if piece != '.':
                # Get piece symbol
                symbol = PIECE_SYMBOLS.get(piece, piece)
                
                # Piece color
                is_white_piece = piece.isupper()
                piece_color = white_piece_color if is_white_piece else black_piece_color
                
                # Center the piece in the square
                try:
                    bbox = font.getbbox(symbol)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    text_width, text_height = square_size // 2, square_size // 2
                
                text_x = x1 + (square_size - text_width) // 2
                text_y = y1 + (square_size - text_height) // 2 - 5
                
                # Draw piece with outline for visibility
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), symbol, 
                                      fill=outline_color if is_white_piece else light_square, 
                                      font=font)
                draw.text((text_x, text_y), symbol, fill=piece_color, font=font)
    
    # Draw border
    draw.rectangle([0, 0, size-1, size-1], outline='black', width=2)
    
    # Add file/rank labels
    try:
        label_font = ImageFont.truetype("arial.ttf", 12)
    except:
        label_font = ImageFont.load_default()
    
    files = 'abcdefgh' if viewpoint == 'white' else 'hgfedcba'
    ranks = '87654321' if viewpoint == 'white' else '12345678'
    
    return img


def create_comparison_image(fen: str, synth_path: Path, viewpoint: str = 'white') -> Image.Image:
    """
    Create side-by-side comparison: FEN diagram | Rendered image
    """
    # Load rendered image
    if not synth_path.exists():
        print(f"  [WARN] Image not found: {synth_path}")
        return None
    
    synth_img = Image.open(synth_path).convert('RGB')
    
    # Create FEN diagram at same size
    size = max(synth_img.size)
    diagram = create_board_diagram(fen, viewpoint, size)
    
    # Resize synth to match if needed
    if synth_img.size != (size, size):
        synth_img = synth_img.resize((size, size), Image.BICUBIC)
    
    # Create side-by-side
    gap = 20
    combined_width = size * 2 + gap
    combined = Image.new('RGB', (combined_width, size + 40), (255, 255, 255))
    
    # Paste images
    combined.paste(diagram, (0, 0))
    combined.paste(synth_img, (size + gap, 0))
    
    # Add labels
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((size // 2 - 50, size + 10), "Expected (FEN)", fill='black', font=font)
    draw.text((size + gap + size // 2 - 50, size + 10), "Rendered", fill='black', font=font)
    
    return combined


def validate_single(fen: str, synth_path: Path, viewpoint: str, output_path: Path = None) -> bool:
    """Validate a single render and optionally save comparison image."""
    print(f"\nValidating: {synth_path.name}")
    print(f"  FEN: {fen[:50]}...")
    print(f"  Viewpoint: {viewpoint}")
    
    comparison = create_comparison_image(fen, synth_path, viewpoint)
    
    if comparison is None:
        return False
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(output_path)
        print(f"  Saved: {output_path}")
    
    return True


def validate_csv(csv_path: Path, output_dir: Path, random_n: int = None):
    """Validate renders from a CSV file."""
    print(f"\n{'='*70}")
    print(f"Validating renders from: {csv_path}")
    print(f"{'='*70}")
    
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows")
    
    # Random subset if requested
    if random_n and random_n < len(rows):
        rows = random.sample(rows, random_n)
        print(f"Randomly selected {random_n} for validation")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    
    for idx, row in enumerate(rows):
        fen = row.get('fen', '')
        viewpoint = row.get('viewpoint', 'white')
        synth_rel = row.get('synth', '')
        
        if not fen or not synth_rel:
            print(f"  [SKIP] Row {idx}: Missing fen or synth path")
            failed += 1
            continue
        
        synth_path = REPO_ROOT / synth_rel
        output_path = output_dir / f"validate_{idx:04d}.png"
        
        if validate_single(fen, synth_path, viewpoint, output_path):
            success += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Validation complete: {success} success, {failed} failed")
    print(f"Comparison images saved to: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Validate rendered chess images against FEN")
    parser.add_argument('--image', type=str, help="Single image to validate")
    parser.add_argument('--fen', type=str, help="FEN string (required with --image)")
    parser.add_argument('--viewpoint', type=str, default='white', choices=['white', 'black'])
    parser.add_argument('--csv', type=str, help="CSV file to validate")
    parser.add_argument('--output', type=str, default='results/validation/', help="Output directory")
    parser.add_argument('--random', type=int, help="Validate random N images from CSV")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.image:
        if not args.fen:
            print("Error: --fen required when using --image")
            return 1
        
        synth_path = Path(args.image)
        if not synth_path.is_absolute():
            synth_path = REPO_ROOT / synth_path
        
        output_path = output_dir / f"validate_{synth_path.stem}.png"
        validate_single(args.fen, synth_path, args.viewpoint, output_path)
        
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        
        validate_csv(csv_path, output_dir, args.random)
    else:
        print("Usage: Specify --image and --fen, or --csv")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
