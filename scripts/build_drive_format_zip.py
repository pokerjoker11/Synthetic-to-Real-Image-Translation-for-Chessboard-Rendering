# scripts/build_drive_format_zip.py
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(p: str) -> str:
    return str(p).strip().replace("\\", "/")


def resolve(repo: Path, p: str) -> Path:
    pp = Path(norm(p))
    return pp.resolve() if pp.is_absolute() else (repo / pp).resolve()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def safe_copy(src: Path, dst_dir: Path, key: str, name_map: Dict[str, str]) -> str:
    key = norm(key)
    if key in name_map:
        return name_map[key]
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    dst = dst_dir / base
    if dst.exists():
        base = f"{src.stem}_{short_hash(key)}{src.suffix}"
        dst = dst_dir / base
    shutil.copy2(src, dst)
    name_map[key] = base
    return base


def write_gt(gt: Dict[str, Tuple[str, str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "fen", "viewpoint"])
        for _, (img, fen, vp) in sorted(gt.items()):
            w.writerow([img, fen, vp])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/pairs/train.csv")
    ap.add_argument("--val_csv", default="data/pairs/val.csv")
    ap.add_argument("--out_dir", default="drive_format_trainval")
    ap.add_argument("--zip_name", default="drive_format_trainval.zip")
    args = ap.parse_args()

    repo = Path(".").resolve()
    out_root = (repo / args.out_dir).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    real_root = out_root / "real_trainval"
    synth_root = out_root / "synth_trainval"
    real_img_dir = real_root / "images"
    synth_img_dir = synth_root / "images"

    df = pd.concat([pd.read_csv(repo / args.train_csv), pd.read_csv(repo / args.val_csv)], ignore_index=True)
    if "fen" not in df.columns:
        df["fen"] = ""
    if "viewpoint" not in df.columns:
        df["viewpoint"] = "white"

    real_map: Dict[str, str] = {}
    synth_map: Dict[str, str] = {}
    gt_real: Dict[str, Tuple[str, str, str]] = {}
    gt_synth: Dict[str, Tuple[str, str, str]] = {}

    for _, r in df.iterrows():
        real_p = resolve(repo, r["real"])
        synth_p = resolve(repo, r["synth"])
        if not real_p.exists():
            raise FileNotFoundError(f"Missing real: {real_p}")
        if not synth_p.exists():
            raise FileNotFoundError(f"Missing synth: {synth_p}")

        fen = str(r.get("fen", ""))
        vp = str(r.get("viewpoint", "white"))

        real_key = norm(str(r["real"]))
        synth_key = norm(str(r["synth"]))

        real_fn = safe_copy(real_p, real_img_dir, real_key, real_map)
        synth_fn = safe_copy(synth_p, synth_img_dir, synth_key, synth_map)

        gt_real.setdefault(real_fn, (real_fn, fen, vp))
        gt_synth.setdefault(synth_fn, (synth_fn, fen, vp))

    write_gt(gt_real, real_root / "gt.csv")
    write_gt(gt_synth, synth_root / "gt.csv")

    manifest = out_root / "MANIFEST_SHA256.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                f.write(f"{sha256_file(p)}  {p.relative_to(out_root).as_posix()}\n")

    zip_path = (repo / args.zip_name).resolve()
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out_root).as_posix())

    print("[OK] wrote folder:", out_root)
    print("[OK] wrote zip:", zip_path)
    print("[OK] real images:", len(list(real_img_dir.glob('*'))))
    print("[OK] synth images:", len(list(synth_img_dir.glob('*'))))


if __name__ == "__main__":
    main()
