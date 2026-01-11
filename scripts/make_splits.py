import csv
from pathlib import Path

PAIRS_CSV = Path("data/pairs/pairs.csv")
SPLITS_DIR = Path("data/splits")


def main():
    if not PAIRS_CSV.exists():
        raise SystemExit(f"[ERR] Missing {PAIRS_CSV}. Run render_synth_dataset.py first.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    with open(PAIRS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("[ERR] pairs.csv has no rows")

    # Deterministic split by game:
    # Hold out game7 for validation (common sense: use entire game to avoid leakage).
    val_games = {"game7"}

    train_rows = [r for r in rows if r.get("game") not in val_games]
    val_rows = [r for r in rows if r.get("game") in val_games]

    # If someone has no game7, fallback to last game alphabetically
    if len(val_rows) == 0:
        games = sorted({r.get("game", "") for r in rows if r.get("game")})
        if not games:
            raise SystemExit("[ERR] No 'game' field in pairs.csv rows")
        fallback = games[-1]
        train_rows = [r for r in rows if r.get("game") != fallback]
        val_rows = [r for r in rows if r.get("game") == fallback]
        print(f"[WARN] No rows for game7; using {fallback} as validation instead.")

    fieldnames = rows[0].keys()

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(train_rows)

    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(val_rows)

    print("==== Split Summary ====")
    print(f"Total rows : {len(rows)}")
    print(f"Train      : {len(train_rows)}")
    print(f"Val        : {len(val_rows)}")
    print(f"Train CSV  : {train_csv}")
    print(f"Val CSV    : {val_csv}")


if __name__ == "__main__":
    main()
