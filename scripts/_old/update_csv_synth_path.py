#!/usr/bin/env python3
"""
Update CSV files to point to cropped synthetic images.
Replaces 'synth_v3' with 'synth_v3_cropped' in synth_path column.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits_rect"


def update_csv(csv_path: Path) -> int:
    """Update synth paths in a CSV file. Returns number of rows updated."""
    if not csv_path.exists():
        print(f"  [SKIP] Not found: {csv_path}")
        return 0
    
    lines = csv_path.read_text(encoding='utf-8').splitlines()
    if not lines:
        return 0
    
    updated_lines = []
    count = 0
    
    for line in lines:
        if 'synth_v3/images' in line and 'synth_v3_cropped' not in line:
            line = line.replace('synth_v3/images', 'synth_v3_cropped/images')
            count += 1
        updated_lines.append(line)
    
    csv_path.write_text('\n'.join(updated_lines) + '\n', encoding='utf-8')
    return count


def main():
    print("Updating CSV files to use cropped synthetic images...")
    print(f"Looking in: {SPLITS_DIR}\n")
    
    csvs = [
        SPLITS_DIR / "train_final.csv",
        SPLITS_DIR / "val_final.csv",
    ]
    
    total = 0
    for csv_path in csvs:
        count = update_csv(csv_path)
        print(f"  {csv_path.name}: {count} rows updated")
        total += count
    
    print(f"\nTotal: {total} paths updated")
    print("\nYou can now restart training with the cropped images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
