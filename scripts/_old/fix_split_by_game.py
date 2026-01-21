#!/usr/bin/env python3
"""
Fix train/val split to be properly separated by game (no data leakage).

Reads from train_clean.csv and val_final.csv, then creates new splits
where entire games are in either train OR val, never both.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def extract_game(path_str):
    """Extract game name from path like 'data\real\images\game4_frame_001396.jpg'"""
    match = re.search(r'game(\d+)', path_str)
    return f"game{match.group(1)}" if match else None

def main():
    train_csv = REPO_ROOT / "data" / "splits_rect" / "train_clean.csv"
    val_csv = REPO_ROOT / "data" / "splits_rect" / "val_final.csv"
    
    # Read all rows
    all_rows = []
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows.extend(list(reader))
        fieldnames = reader.fieldnames
    
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows.extend(list(reader))
    
    print(f"Total rows: {len(all_rows)}")
    
    # Group by game
    games_dict = defaultdict(list)
    for row in all_rows:
        game = extract_game(row.get('real', ''))
        if game:
            games_dict[game].append(row)
        else:
            print(f"Warning: Could not extract game from: {row.get('real', '')}")
    
    print(f"\nGames found: {sorted(games_dict.keys())}")
    for game in sorted(games_dict.keys()):
        print(f"  {game}: {len(games_dict[game])} images")
    
    # Split by game: hold out 1-2 games for validation
    # Use game7 and game10 for validation (or last 2 games if they don't exist)
    all_games = sorted(games_dict.keys())
    
    # Strategy: Hold out ~10-15% of data for validation
    # With 8 games, hold out 1-2 games
    if len(all_games) >= 2:
        # Hold out last 2 games alphabetically (or game7 + one other)
        val_games = set()
        if 'game7' in all_games:
            val_games.add('game7')
        if 'game10' in all_games and len(all_games) > 2:
            val_games.add('game10')
        else:
            # If game10 doesn't exist or we need another game, use the last one
            val_games.add(all_games[-1])
        
        # If we only have 1 val game and have many games, add one more
        if len(val_games) == 1 and len(all_games) > 4:
            # Add second-to-last game
            for game in reversed(all_games):
                if game not in val_games:
                    val_games.add(game)
                    break
    else:
        # Fallback: just use last game
        val_games = {all_games[-1]}
    
    print(f"\nValidation games: {sorted(val_games)}")
    
    # Create splits
    train_rows = []
    val_rows = []
    
    for game, rows in games_dict.items():
        if game in val_games:
            val_rows.extend(rows)
            print(f"  {game} -> VAL ({len(rows)} images)")
        else:
            train_rows.extend(rows)
            print(f"  {game} -> TRAIN ({len(rows)} images)")
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_rows)} images")
    print(f"  Val: {len(val_rows)} images")
    print(f"  Train %: {100*len(train_rows)/len(all_rows):.1f}%")
    print(f"  Val %: {100*len(val_rows)/len(all_rows):.1f}%")
    
    # Verify no leakage
    train_games_set = {extract_game(r.get('real', '')) for r in train_rows}
    val_games_set = {extract_game(r.get('real', '')) for r in val_rows}
    overlap = train_games_set & val_games_set
    
    if overlap:
        print(f"\nERROR: Still have overlap: {overlap}")
        return 1
    
    print(f"\nOK: No data leakage - games properly separated")
    
    # Write new splits
    output_dir = REPO_ROOT / "data" / "splits_rect"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / "train_clean_fixed.csv"
    val_output = output_dir / "val_final_fixed.csv"
    
    with open(train_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    
    with open(val_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)
    
    print(f"\nSaved:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"\nNext steps:")
    print(f"  1. Review the new splits")
    print(f"  2. If good, replace train_clean.csv and val_final.csv")
    print(f"  3. Restart training with proper split")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
