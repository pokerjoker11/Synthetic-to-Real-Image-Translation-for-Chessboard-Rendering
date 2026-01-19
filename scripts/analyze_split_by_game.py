#!/usr/bin/env python3
"""Analyze train/val split by game to check for data leakage."""

import csv
from collections import Counter
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent

def extract_game(path_str):
    """Extract game name from path like 'data\real\images\game4_frame_001396.jpg'"""
    match = re.search(r'game(\d+)', path_str)
    return f"game{match.group(1)}" if match else None

def analyze_split():
    train_csv = REPO_ROOT / "data" / "splits_rect" / "train_clean.csv"
    val_csv = REPO_ROOT / "data" / "splits_rect" / "val_final.csv"
    
    train_games = Counter()
    val_games = Counter()
    
    # Count games in train
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = extract_game(row.get('real', ''))
            if game:
                train_games[game] += 1
    
    # Count games in val
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = extract_game(row.get('real', ''))
            if game:
                val_games[game] += 1
    
    print("=" * 60)
    print("TRAIN/VAL SPLIT BY GAME")
    print("=" * 60)
    print(f"\nTRAIN GAMES ({sum(train_games.values())} total images):")
    for game in sorted(train_games.keys()):
        print(f"  {game}: {train_games[game]} images")
    
    print(f"\nVAL GAMES ({sum(val_games.values())} total images):")
    for game in sorted(val_games.keys()):
        print(f"  {game}: {val_games[game]} images")
    
    # Check for leakage
    print("\n" + "=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)
    
    train_set = set(train_games.keys())
    val_set = set(val_games.keys())
    overlap = train_set & val_set
    
    if overlap:
        print(f"WARNING: Data leakage detected!")
        print(f"   Games in BOTH train and val: {sorted(overlap)}")
        print(f"\n   This means:")
        print(f"   - Model sees some positions from these games during training")
        print(f"   - Then tests on other positions from the same games")
        print(f"   - This can make validation misleading (appears better than it is)")
        print(f"   - Model may be memorizing game-specific patterns")
    else:
        print("OK: No data leakage - games are properly separated")
        print(f"   Train games: {sorted(train_set)}")
        print(f"   Val games: {sorted(val_set)}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if overlap:
        print("Split should be by game (entire games in train or val, not mixed)")
        print("This prevents the model from seeing similar positions during training")
        print("and then testing on them.")
    else:
        print("Split is properly done by game - no action needed.")

if __name__ == "__main__":
    analyze_split()
