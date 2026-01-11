import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PAIRS_CSV = Path("data/pairs/pairs.csv")
OUT_DIR = Path("results/pair_inspection")


def _open_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    if img.height == h:
        return img
    w = int(img.width * (h / img.height))
    return img.resize((w, h))


def _side_by_side(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    # match heights
    h = max(left.height, right.height)
    left2 = _resize_to_height(left, h)
    right2 = _resize_to_height(right, h)

    pad = 10
    cap_h = 60
    out = Image.new("RGB", (left2.width + right2.width + pad * 3, h + cap_h + pad * 2), (20, 20, 20))
    out.paste(left2, (pad, cap_h + pad))
    out.paste(right2, (pad * 2 + left2.width, cap_h + pad))

    draw = ImageDraw.Draw(out)
    # Font: use default to avoid OS issues
    draw.text((pad, pad), caption, fill=(240, 240, 240))
    draw.text((pad, cap_h - 22), "REAL (target)", fill=(240, 240, 240))
    draw.text((pad * 2 + left2.width, cap_h - 22), "SYNTH (input)", fill=(240, 240, 240))
    return out


def main():
    if not PAIRS_CSV.exists():
        raise SystemExit(f"[ERR] Missing {PAIRS_CSV}. Run render_synth_dataset.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PAIRS_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("[ERR] pairs.csv has no rows")

    # Sample
    k = min(12, len(rows))
    sample = random.sample(rows, k)

    missing = 0
    for i, r in enumerate(sample):
        real_p = Path(r["real"])
        synth_p = Path(r["synth"])
        fen = r.get("fen", "")
        game = r.get("game", "")
        frame = r.get("frame", "")

        if not real_p.exists() or not synth_p.exists():
            print(f"[WARN] missing files: real={real_p.exists()} synth={synth_p.exists()} | {real_p} | {synth_p}")
            missing += 1
            continue

        real_img = _open_rgb(real_p)
        synth_img = _open_rgb(synth_p)

        caption = f"{game} frame={frame}\nFEN: {fen}"
        out = _side_by_side(real_img, synth_img, caption)

        out_path = OUT_DIR / f"sample_{i:02d}.png"
        out.save(out_path)
        print(f"[OK] wrote {out_path}")

    print("\n==== Done ====")
    print(f"Samples requested : {k}")
    print(f"Samples written   : {k - missing}")
    print(f"Missing pairs     : {missing}")
    print(f"Output dir        : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
