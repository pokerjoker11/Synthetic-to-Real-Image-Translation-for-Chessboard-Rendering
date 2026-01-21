#!/usr/bin/env python3
"""
Create a mixed training CSV by combining:
- ALL rows from course train.csv (527 rows)
- ALL rows from ChessReD2K TOP subset (139 rows)
- Random K rows from ChessReD2K OK-rest subset (default 75 rows)

Validates file existence and adds a 'source' column to track data origin.
"""
import argparse
import os
import sys
import pandas as pd
import random
from pathlib import Path


def normalize_path(p):
    """Normalize path to use forward slashes."""
    return str(p).replace('\\', '/')


def validate_row_files(row, base_dir='.'):
    """
    Check if real_path and synth_path files exist.
    Returns (is_valid, missing_files_list).
    """
    missing = []
    
    # Check real_path
    if 'real_path' in row:
        real_path = os.path.join(base_dir, row['real_path'])
        if not os.path.exists(real_path):
            missing.append(f"real_path: {row['real_path']}")
    else:
        missing.append("real_path column missing")
    
    # Check synth_path
    if 'synth_path' in row:
        synth_path = os.path.join(base_dir, row['synth_path'])
        if not os.path.exists(synth_path):
            missing.append(f"synth_path: {row['synth_path']}")
    else:
        missing.append("synth_path column missing")
    
    return len(missing) == 0, missing


def load_and_validate_csv(csv_path, source_name, required_count=None, base_dir='.'):
    """
    Load CSV and validate that files exist.
    If required_count is specified, ensure we can get that many valid rows.
    Returns (validated_df, num_skipped).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[{source_name}] Loaded {len(df)} rows from {csv_path}")
    
    # Validate each row
    valid_rows = []
    skipped = 0
    
    for idx, row in df.iterrows():
        is_valid, missing = validate_row_files(row, base_dir)
        if is_valid:
            valid_rows.append(row)
        else:
            skipped += 1
            if skipped <= 5:  # Show first 5 errors
                print(f"  [SKIP] Row {idx}: {', '.join(missing)}")
    
    if skipped > 5:
        print(f"  [SKIP] ... and {skipped - 5} more rows")
    
    validated_df = pd.DataFrame(valid_rows)
    
    # Check if we have enough valid rows
    if required_count is not None and len(validated_df) < required_count:
        print(f"  [WARNING] Only {len(validated_df)} valid rows available, needed {required_count}")
    
    print(f"[{source_name}] Valid rows: {len(validated_df)}, Skipped: {skipped}")
    return validated_df, skipped


def sample_rows(df, k, seed, source_name):
    """
    Randomly sample k rows from df using the given seed.
    If k >= len(df), return all rows.
    """
    if k >= len(df):
        print(f"[{source_name}] Taking all {len(df)} rows (k={k} >= available)")
        return df
    
    sampled = df.sample(n=k, random_state=seed)
    print(f"[{source_name}] Sampled {k} rows from {len(df)} available")
    return sampled


def normalize_paths_in_df(df):
    """Normalize all path columns to use forward slashes."""
    path_columns = [col for col in df.columns if 'path' in col.lower()]
    for col in path_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_path)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create mixed training CSV from course data + ChessReD2K subsets'
    )
    parser.add_argument(
        '--course_train_csv',
        default='data/splits_rect/train.csv',
        help='Path to course training CSV (default: data/splits_rect/train.csv)'
    )
    parser.add_argument(
        '--chessred_top_csv',
        required=True,
        help='Path to ChessReD2K TOP subset CSV'
    )
    parser.add_argument(
        '--chessred_okrest_csv',
        required=True,
        help='Path to ChessReD2K OK-rest subset CSV'
    )
    parser.add_argument(
        '--okrest_k',
        type=int,
        default=75,
        help='Number of rows to sample from OK-rest (default: 75)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for sampling and shuffling (default: 1337)'
    )
    parser.add_argument(
        '--out_csv',
        default='data/splits_rect/train_mixed.csv',
        help='Output CSV path (default: data/splits_rect/train_mixed.csv)'
    )
    parser.add_argument(
        '--base_dir',
        default='.',
        help='Base directory for resolving relative paths (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("Mixed Training CSV Generator")
    print("=" * 60)
    print(f"Course train CSV:    {args.course_train_csv}")
    print(f"ChessReD TOP CSV:    {args.chessred_top_csv}")
    print(f"ChessReD OK-rest CSV: {args.chessred_okrest_csv}")
    print(f"OK-rest sample size: {args.okrest_k}")
    print(f"Random seed:         {args.seed}")
    print(f"Output CSV:          {args.out_csv}")
    print(f"Base directory:      {args.base_dir}")
    print("=" * 60)
    print()
    
    # Load and validate each dataset
    try:
        # Course training data - take ALL rows
        course_df, course_skip = load_and_validate_csv(
            args.course_train_csv,
            'COURSE',
            base_dir=args.base_dir
        )
        course_df['source'] = 'course'
        
        # ChessReD TOP - take ALL rows
        top_df, top_skip = load_and_validate_csv(
            args.chessred_top_csv,
            'TOP',
            required_count=None,  # Take all available
            base_dir=args.base_dir
        )
        top_df['source'] = 'chessred_top'
        
        # ChessReD OK-rest - sample K rows
        okrest_df, okrest_skip = load_and_validate_csv(
            args.chessred_okrest_csv,
            'OK-REST',
            required_count=args.okrest_k,
            base_dir=args.base_dir
        )
        okrest_sampled = sample_rows(okrest_df, args.okrest_k, args.seed, 'OK-REST')
        okrest_sampled['source'] = 'chessred_okrest'
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load CSVs: {e}")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("Combining datasets...")
    print("-" * 60)
    
    # Combine all datasets
    combined_dfs = [course_df, top_df, okrest_sampled]
    
    # Get common columns (at minimum: synth_path, real_path, id if present)
    all_columns = set()
    for df in combined_dfs:
        all_columns.update(df.columns)
    
    # Ensure source column is present
    all_columns.add('source')
    
    # Find columns present in course_df (our reference)
    if len(course_df) > 0:
        reference_columns = list(course_df.columns)
    else:
        # Fallback to essential columns
        reference_columns = ['synth_path', 'real_path', 'source']
        if 'id' in all_columns:
            reference_columns.insert(0, 'id')
    
    # Ensure all DataFrames have the same columns
    for i, df in enumerate(combined_dfs):
        for col in reference_columns:
            if col not in df.columns:
                df[col] = None
        combined_dfs[i] = df[reference_columns]
    
    # Concatenate
    mixed_df = pd.concat(combined_dfs, ignore_index=True)
    
    print(f"Combined {len(mixed_df)} rows")
    print()
    
    # Normalize paths
    print("Normalizing paths...")
    mixed_df = normalize_paths_in_df(mixed_df)
    
    # Shuffle
    print(f"Shuffling with seed {args.seed}...")
    mixed_df = mixed_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    
    # Save
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    mixed_df.to_csv(args.out_csv, index=False)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Course rows:       {len(course_df):4d}")
    print(f"ChessReD TOP rows: {len(top_df):4d}")
    print(f"ChessReD OK rows:  {len(okrest_sampled):4d}")
    print("-" * 60)
    print(f"Total rows:        {len(mixed_df):4d}")
    print("=" * 60)
    print(f"\nOutput written to: {args.out_csv}")
    print()
    
    # Show source distribution
    source_counts = mixed_df['source'].value_counts()
    print("Source distribution:")
    for source, count in source_counts.items():
        print(f"  {source:20s}: {count:4d} rows")
    
    print()
    print("[DONE]")


if __name__ == '__main__':
    main()
