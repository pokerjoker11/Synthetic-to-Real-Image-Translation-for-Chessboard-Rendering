#!/usr/bin/env python3
"""
Rebuild train.csv and val.csv by:
1. Removing old game 8, 9, 10 entries
2. Adding new properly annotated game 8, 9, 10 entries
3. Splitting new data 90% train / 10% val

Usage:
    python scripts/rebuild_train_csv.py
    python scripts/rebuild_train_csv.py --val-split 0.1
"""

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def filter_old_games(rows, games_to_remove):
    """Remove entries for specified games."""
    kept = []
    removed = 0
    for row in rows:
        real_path = row.get('real', '')
        is_old_game = any(g in real_path for g in games_to_remove)
        if is_old_game:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Rebuild train/val CSVs")
    parser.add_argument('--val-split', type=float, default=0.1,
                        help="Fraction of new data for validation (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    train_csv = REPO_ROOT / "data" / "splits_rect" / "train.csv"
    val_csv = REPO_ROOT / "data" / "splits_rect" / "val.csv"
    new_games_csv = REPO_ROOT / "data" / "new_games_annotations.csv"
    
    train_output = REPO_ROOT / "data" / "splits_rect" / "train_rebuilt.csv"
    val_output = REPO_ROOT / "data" / "splits_rect" / "val_rebuilt.csv"
    
    games_to_remove = ['game8', 'game9', 'game10']
    
    # Process train.csv
    print("=" * 60)
    print("Processing train.csv...")
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        train_rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"  Original: {len(train_rows)} rows")
    train_kept, train_removed = filter_old_games(train_rows, games_to_remove)
    print(f"  Removed {train_removed} old game 8/9/10 entries")
    print(f"  Kept {len(train_kept)} entries from games 2-7")
    
    # Process val.csv
    print("\nProcessing val.csv...")
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        val_rows = list(reader)
    
    print(f"  Original: {len(val_rows)} rows")
    val_kept, val_removed = filter_old_games(val_rows, games_to_remove)
    print(f"  Removed {val_removed} old game 8/9/10 entries")
    print(f"  Kept {len(val_kept)} entries from games 2-7")
    
    # Read new game 8, 9, 10 annotations
    print("\nLoading new game 8/9/10 annotations...")
    with open(new_games_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        new_rows = list(reader)
    
    print(f"  Total new entries: {len(new_rows)}")
    
    # Format new rows
    new_formatted = []
    for row in new_rows:
        new_formatted.append({
            'real': row['real'],
            'synth': '',  # Will need to be rendered
            'fen': row['fen'],
            'viewpoint': row['viewpoint']
        })
    
    # Shuffle and split new data
    random.shuffle(new_formatted)
    val_count = int(len(new_formatted) * args.val_split)
    new_val = new_formatted[:val_count]
    new_train = new_formatted[val_count:]
    
    print(f"  Split: {len(new_train)} train, {len(new_val)} val ({args.val_split*100:.0f}%)")
    
    # Combine
    final_train = train_kept + new_train
    final_val = val_kept + new_val
    
    print("\n" + "=" * 60)
    print("Final counts:")
    print(f"  train_rebuilt.csv: {len(final_train)} rows")
    print(f"    - Games 2-7: {len(train_kept)}")
    print(f"    - Games 8-10: {len(new_train)}")
    print(f"  val_rebuilt.csv: {len(final_val)} rows")
    print(f"    - Games 2-7: {len(val_kept)}")
    print(f"    - Games 8-10: {len(new_val)}")
    
    # Write outputs
    with open(train_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_train)
    
    with open(val_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_val)
    
    print(f"\nSaved:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print("\nNote: New game 8/9/10 entries have empty 'synth' paths.")
    print("Run the synthetic renderer to generate them before training.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
