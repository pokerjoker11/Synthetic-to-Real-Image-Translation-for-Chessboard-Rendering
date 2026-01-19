#!/usr/bin/env python3
"""
Process games 8, 9, 10 from the new folder structure where:
- Each folder name is a FEN (with underscores instead of slashes)
- Inside each FEN folder: white/ and black/ subfolders for viewpoint

This script:
1. Scans the new structure
2. Copies images to data/real/images/ (white POV) and data/real/images_rot180/ (black POV)
3. Creates a CSV with correct FEN and viewpoint annotations

Usage:
    python scripts/process_new_games.py
    python scripts/process_new_games.py --dry-run  # Preview without copying
    python scripts/process_new_games.py --games 8 9 10
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Source and destination directories
RAW_DIR = REPO_ROOT / "data" / "raw"
REAL_IMAGES_DIR = REPO_ROOT / "data" / "real" / "images"
REAL_IMAGES_ROT180_DIR = REPO_ROOT / "data" / "real" / "images_rot180"


def fen_folder_to_fen(folder_name: str) -> str:
    """Convert folder name (underscores) to proper FEN (slashes)."""
    # Replace underscores with slashes to get the board part
    # FEN folder format: rank8_rank7_rank6_rank5_rank4_rank3_rank2_rank1
    fen_board = folder_name.replace('_', '/')
    # Add standard FEN suffix (white to move, no castling, no en passant, etc.)
    return f"{fen_board} w - - 0 1"


def process_game(game_num: int, dry_run: bool = False, one_per_fen: bool = True) -> list:
    """
    Process a single game and return list of entries.
    
    Args:
        game_num: Game number to process
        dry_run: If True, don't copy files
        one_per_fen: If True, only select one frame per FEN (alternating white/black)
    
    Returns:
        List of dicts with: real, fen, viewpoint
    """
    game_dir = RAW_DIR / f"game{game_num}" / "images"
    
    if not game_dir.exists():
        print(f"[SKIP] Game {game_num} not found at {game_dir}")
        return []
    
    entries = []
    fen_folders = [d for d in game_dir.iterdir() if d.is_dir()]
    
    print(f"\n[Game {game_num}] Found {len(fen_folders)} FEN positions")
    
    # Alternate between white and black to balance the dataset
    use_white_next = True
    
    for fen_folder in sorted(fen_folders):
        fen = fen_folder_to_fen(fen_folder.name)
        
        white_dir = fen_folder / "white"
        black_dir = fen_folder / "black"
        
        white_files = list(white_dir.glob("*.jpg")) if white_dir.exists() else []
        black_files = list(black_dir.glob("*.jpg")) if black_dir.exists() else []
        
        if one_per_fen:
            # Select one viewpoint per FEN, alternating to balance
            if use_white_next and white_files:
                # Pick first white image
                img_file = white_files[0]
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}.jpg"
                dest_path = REAL_IMAGES_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'white',
                    'source': str(img_file)
                })
                use_white_next = False
                
            elif not use_white_next and black_files:
                # Pick first black image
                img_file = black_files[0]
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}_rot180.jpg"
                dest_path = REAL_IMAGES_ROT180_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_ROT180_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images_rot180/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'black',
                    'source': str(img_file)
                })
                use_white_next = True
                
            elif white_files:
                # Fallback to white if black not available
                img_file = white_files[0]
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}.jpg"
                dest_path = REAL_IMAGES_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'white',
                    'source': str(img_file)
                })
                
            elif black_files:
                # Fallback to black if white not available
                img_file = black_files[0]
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}_rot180.jpg"
                dest_path = REAL_IMAGES_ROT180_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_ROT180_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images_rot180/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'black',
                    'source': str(img_file)
                })
        else:
            # Original behavior: process all images
            for img_file in white_files:
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}.jpg"
                dest_path = REAL_IMAGES_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'white',
                    'source': str(img_file)
                })
            
            for img_file in black_files:
                frame_num = img_file.stem.replace("frame_", "")
                dest_name = f"game{game_num}_frame_{frame_num}_rot180.jpg"
                dest_path = REAL_IMAGES_ROT180_DIR / dest_name
                
                if not dry_run:
                    REAL_IMAGES_ROT180_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, dest_path)
                
                entries.append({
                    'real': f"data/real/images_rot180/{dest_name}",
                    'fen': fen,
                    'viewpoint': 'black',
                    'source': str(img_file)
                })
    
    white_count = len([e for e in entries if e['viewpoint'] == 'white'])
    black_count = len([e for e in entries if e['viewpoint'] == 'black'])
    print(f"  Processed {len(entries)} images ({white_count} white, {black_count} black)")
    
    return entries


def main():
    parser = argparse.ArgumentParser(description="Process new games 8, 9, 10")
    parser.add_argument('--dry-run', action='store_true', help="Preview without copying files")
    parser.add_argument('--games', nargs='+', type=int, default=[8, 9, 10],
                        help="Which games to process (default: 8 9 10)")
    parser.add_argument('--output', type=str, default='data/new_games_annotations.csv',
                        help="Output CSV file")
    parser.add_argument('--all-frames', action='store_true',
                        help="Process all frames (default: one per FEN, alternating viewpoints)")
    
    args = parser.parse_args()
    
    one_per_fen = not args.all_frames
    
    if args.dry_run:
        print("=== DRY RUN MODE - No files will be copied ===\n")
    
    if one_per_fen:
        print("Mode: One frame per FEN (alternating white/black)")
    else:
        print("Mode: All frames")
    
    all_entries = []
    
    for game_num in args.games:
        entries = process_game(game_num, args.dry_run, one_per_fen)
        all_entries.extend(entries)
    
    # Save CSV
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if all_entries:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Only write the columns needed for training (exclude 'source')
            fieldnames = ['real', 'fen', 'viewpoint']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in all_entries:
                writer.writerow({k: entry[k] for k in fieldnames})
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total images processed: {len(all_entries)}")
        print(f"  White POV: {len([e for e in all_entries if e['viewpoint'] == 'white'])}")
        print(f"  Black POV: {len([e for e in all_entries if e['viewpoint'] == 'black'])}")
        print(f"  Output CSV: {output_path}")
        
        if args.dry_run:
            print(f"\n[DRY RUN] No files were copied. Run without --dry-run to copy files.")
        else:
            print(f"\nFiles copied to:")
            print(f"  White POV: {REAL_IMAGES_DIR}")
            print(f"  Black POV: {REAL_IMAGES_ROT180_DIR}")
        
        print(f"{'='*60}")
    else:
        print("\n[WARNING] No images found to process!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
