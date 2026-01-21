#!/usr/bin/env python3
"""
Remove top POV images from pairs_ok.csv.

Since top POV images have been moved to real_top_pov/ folder,
they should be removed from pairs_ok.csv to avoid duplicates.

Usage:
    python scripts/remove_top_pov_from_ok.py
    python scripts/remove_top_pov_from_ok.py --csv data/chessred2k_rect_pov/pairs_ok.csv
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_top_pov_filenames() -> set[str]:
    """Get all image filenames in real_top_pov/ folder."""
    top_pov_dir = REPO_ROOT / "data" / "chessred2k_rect_pov" / "real_top_pov"
    
    if not top_pov_dir.exists():
        print(f"[WARN] Top POV directory not found: {top_pov_dir}")
        return set()
    
    filenames = set()
    for img_path in top_pov_dir.glob("*.png"):
        filenames.add(img_path.name)
    
    return filenames


def remove_top_pov_from_csv(csv_path: Path, backup: bool = True):
    """Remove rows from CSV where real_path points to images in real_top_pov/."""
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    
    # Get top POV filenames
    top_pov_filenames = get_top_pov_filenames()
    print(f"Found {len(top_pov_filenames)} images in real_top_pov/ folder")
    
    if not top_pov_filenames:
        print("[WARN] No top POV images found, nothing to remove")
        return
    
    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Loaded {len(rows)} rows from {csv_path}")
    
    # Filter out top POV rows
    kept_rows = []
    removed_rows = []
    
    for row in rows:
        real_path = row.get("real_path", "")
        if real_path:
            filename = Path(real_path).name
            # Check if this image is in top_pov folder OR if path already points to real_top_pov
            if filename in top_pov_filenames or "real_top_pov" in real_path:
                removed_rows.append(row)
            else:
                kept_rows.append(row)
        else:
            # Keep rows without real_path (shouldn't happen, but be safe)
            kept_rows.append(row)
    
    # Print summary
    print(f"\n==== Summary ====")
    print(f"Total rows in CSV: {len(rows)}")
    print(f"  - Kept: {len(kept_rows)}")
    print(f"  - Removed (top POV): {len(removed_rows)}")
    
    if len(removed_rows) == 0:
        print("\n[OK] No top POV images found in CSV, nothing to remove")
        return
    
    # Backup original CSV
    if backup:
        backup_path = csv_path.parent / f"{csv_path.stem}_backup.csv"
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"  - Backup saved to: {backup_path}")
    
    # Write updated CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    
    print(f"\n[OK] Updated {csv_path}")
    print(f"  - Removed {len(removed_rows)} top POV entries")
    print(f"  - Kept {len(kept_rows)} entries")
    
    # Optionally save removed rows to a separate CSV for reference
    removed_csv = csv_path.parent / f"{csv_path.stem}_removed_top_pov.csv"
    with open(removed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removed_rows)
    
    print(f"  - Removed entries saved to: {removed_csv}")


def main():
    parser = argparse.ArgumentParser(description="Remove top POV images from pairs_ok.csv")
    parser.add_argument("--csv", type=str, default="data/chessred2k_rect_pov/pairs_ok.csv",
                       help="CSV file to update")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup of original CSV")
    
    args = parser.parse_args()
    
    csv_path = REPO_ROOT / args.csv
    remove_top_pov_from_csv(csv_path, backup=not args.no_backup)


if __name__ == "__main__":
    main()
