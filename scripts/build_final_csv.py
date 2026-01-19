#!/usr/bin/env python3
"""
Build final train/val CSVs by:
1. Using the _v3 CSVs (with synth paths)
2. Excluding rejected entries from review

Usage:
    python scripts/build_final_csv.py
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    # Paths
    train_v3 = REPO_ROOT / "data" / "splits_rect" / "train_v3.csv"
    val_v3 = REPO_ROOT / "data" / "splits_rect" / "val_v3.csv"
    review_csv = REPO_ROOT / "data" / "review_progress.csv"
    
    train_final = REPO_ROOT / "data" / "splits_rect" / "train_final.csv"
    val_final = REPO_ROOT / "data" / "splits_rect" / "val_final.csv"
    
    # Load rejected indices
    rejected_indices = set()
    if review_csv.exists():
        with open(review_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('decision') == 'reject':
                    rejected_indices.add(int(row['idx']))
    
    print(f"Found {len(rejected_indices)} rejected entries")
    
    # Process train_v3.csv
    if train_v3.exists():
        with open(train_v3, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            train_rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Filter out rejected (match by real path containing game8/9/10 and checking against new_games indices)
        # Since rejection was done on new_games_annotations.csv, we need to match by real path
        
        # Load new_games_annotations to map indices to real paths
        new_games_csv = REPO_ROOT / "data" / "new_games_annotations.csv"
        rejected_real_paths = set()
        
        if new_games_csv.exists():
            with open(new_games_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                new_games_rows = list(reader)
            
            for idx in rejected_indices:
                if idx < len(new_games_rows):
                    real_path = new_games_rows[idx].get('real', '').replace('\\', '/')
                    rejected_real_paths.add(real_path)
        
        print(f"Rejected real paths: {len(rejected_real_paths)}")
        
        # Filter train rows
        train_filtered = []
        for row in train_rows:
            real_path = row.get('real', '').replace('\\', '/')
            if real_path not in rejected_real_paths:
                train_filtered.append(row)
        
        print(f"Train: {len(train_rows)} -> {len(train_filtered)} (removed {len(train_rows) - len(train_filtered)})")
        
        with open(train_final, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(train_filtered)
        
        print(f"Saved: {train_final}")
    else:
        print(f"[ERROR] train_v3.csv not found: {train_v3}")
    
    # Process val_v3.csv
    if val_v3.exists():
        with open(val_v3, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            val_rows = list(reader)
            fieldnames = reader.fieldnames
        
        # Filter val rows
        val_filtered = []
        for row in val_rows:
            real_path = row.get('real', '').replace('\\', '/')
            if real_path not in rejected_real_paths:
                val_filtered.append(row)
        
        print(f"Val: {len(val_rows)} -> {len(val_filtered)} (removed {len(val_rows) - len(val_filtered)})")
        
        with open(val_final, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(val_filtered)
        
        print(f"Saved: {val_final}")
    else:
        print(f"[ERROR] val_v3.csv not found: {val_v3}")
    
    print("\n" + "="*60)
    print("Final CSVs ready for training:")
    print(f"  {train_final}")
    print(f"  {val_final}")
    print("\nTo start training:")
    print(f"  python train.py --train_csv data/splits_rect/train_final.csv --val_csv data/splits_rect/val_final.csv")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
