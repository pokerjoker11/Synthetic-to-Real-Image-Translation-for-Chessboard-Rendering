# scripts/build_cloud_dataset_package.py
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def uniq(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def norm_path_str(p: str) -> str:
    return str(p).strip().replace("\\", "/")


def resolve(repo_root: Path, p: str) -> Path:
    p = norm_path_str(p)
    pp = Path(p)
    return pp.resolve() if pp.is_absolute() else (repo_root / pp).resolve()


def safe_copy(src: Path, dst_dir: Path, src_key: str, name_map: Dict[str, str]) -> str:
    """
    Copy src into dst_dir with collision-safe naming.
    Returns the destination filename (basename).
    """
    src_key = norm_path_str(src_key)
    if src_key in name_map:
        return name_map[src_key]

    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    dst = dst_dir / base

    if dst.exists():
        # If it's not the same file content/path, avoid name collision
        base = f"{src.stem}_{short_hash(src_key)}{src.suffix}"
        dst = dst_dir / base

    shutil.copy2(src, dst)
    name_map[src_key] = base
    return base


def write_gt(gt_rows: Iterable[Tuple[str, str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "fen", "viewpoint"])
        for img_name, fen, viewpoint in gt_rows:
            w.writerow([img_name, fen, viewpoint])


def write_pairs(df: pd.DataFrame, out_csv: Path, real_name_map: Dict[str, str], synth_name_map: Dict[str, str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        real_src_key = norm_path_str(str(r["real"]))
        synth_src_key = norm_path_str(str(r["synth"]))

        real_fn = real_name_map[real_src_key]
        synth_fn = synth_name_map[synth_src_key]

        rows.append(
            {
                "synth": f"data/synth/images/{synth_fn}",
                "real": f"data/real/images/{real_fn}",
                "fen": str(r.get("fen", "")),
                "viewpoint": str(r.get("viewpoint", "white")),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/pairs/train.csv")
    ap.add_argument("--val_csv", default="data/pairs/val.csv")
    ap.add_argument("--mask_dir", default="data/masks_manual")
    ap.add_argument("--out_dir", default="cloud_dataset_trainval")
    ap.add_argument("--zip_name", default="cloud_dataset_trainval_with_masks.zip")
    ap.add_argument("--require_masks", action="store_true", help="Fail if any used real image is missing a mask.")
    args = ap.parse_args()

    repo = Path(".").resolve()

    train_csv = (repo / args.train_csv).resolve()
    val_csv = (repo / args.val_csv).resolve()
    mask_dir = (repo / args.mask_dir).resolve()

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train_csv: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val_csv: {val_csv}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask_dir: {mask_dir}")

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    for df, name in [(df_tr, "train"), (df_va, "val")]:
        if "real" not in df.columns or "synth" not in df.columns:
            raise ValueError(f"{name} csv must contain columns: real, synth")
        if "fen" not in df.columns:
            df["fen"] = ""
        if "viewpoint" not in df.columns:
            df["viewpoint"] = "white"

    out_root = (repo / args.out_dir).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # The packaged dataset is shaped so that unzipping into repo root recreates expected defaults:
    # data/real/images, data/synth/images, data/masks_manual, data/pairs/train.csv,val.csv
    pkg_data = out_root / "data"
    pkg_real_img = pkg_data / "real" / "images"
    pkg_synth_img = pkg_data / "synth" / "images"
    pkg_masks = pkg_data / "masks_manual"
    pkg_pairs = pkg_data / "pairs"

    real_name_map: Dict[str, str] = {}
    synth_name_map: Dict[str, str] = {}

    gt_real: Dict[str, Tuple[str, str, str]] = {}
    gt_synth: Dict[str, Tuple[str, str, str]] = {}

    def process(df: pd.DataFrame) -> None:
        for _, r in df.iterrows():
            real_p = resolve(repo, r["real"])
            synth_p = resolve(repo, r["synth"])
            if not real_p.exists():
                raise FileNotFoundError(f"Missing real image: {real_p}")
            if not synth_p.exists():
                raise FileNotFoundError(f"Missing synth image: {synth_p}")

            fen = str(r.get("fen", ""))
            viewpoint = str(r.get("viewpoint", "white"))

            real_key = norm_path_str(str(r["real"]))
            synth_key = norm_path_str(str(r["synth"]))

            real_fn = safe_copy(real_p, pkg_real_img, real_key, real_name_map)
            synth_fn = safe_copy(synth_p, pkg_synth_img, synth_key, synth_name_map)

            gt_real.setdefault(real_fn, (real_fn, fen, viewpoint))
            gt_synth.setdefault(synth_fn, (synth_fn, fen, viewpoint))

    process(df_tr)
    process(df_va)

    # Copy masks for used real images (mask name == stem of REAL image)
    pkg_masks.mkdir(parents=True, exist_ok=True)
    used_real_fns = list(gt_real.keys())
    missing_masks = []
    copied = 0
    for fn in used_real_fns:
        stem = Path(fn).stem
        mp = mask_dir / f"{stem}.png"
        if not mp.exists():
            missing_masks.append(str(mp))
            continue
        shutil.copy2(mp, pkg_masks / mp.name)
        copied += 1

    if missing_masks:
        msg = "[ERROR] Missing masks:\n" + "\n".join(missing_masks[:50])
        if args.require_masks:
            raise FileNotFoundError(msg)
        else:
            print(msg)
    print(f"[OK] masks copied: {copied} / {len(used_real_fns)}")

    # Write gt.csv (dataset_root/images + gt.csv format, for completeness)
    write_gt(gt_real.values(), pkg_data / "real" / "gt.csv")
    write_gt(gt_synth.values(), pkg_data / "synth" / "gt.csv")

    # Write pairs CSVs that match your code defaults when unzipped into repo root
    write_pairs(df_tr, pkg_pairs / "train.csv", real_name_map, synth_name_map)
    write_pairs(df_va, pkg_pairs / "val.csv", real_name_map, synth_name_map)

    # Manifest
    manifest = out_root / "MANIFEST_SHA256.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(out_root).as_posix()
                f.write(f"{sha256_file(p)}  {rel}\n")

    # Zip
    zip_path = (repo / args.zip_name).resolve()
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                z.write(p, arcname=p.relative_to(out_root).as_posix())

    print("[OK] wrote folder:", out_root)
    print("[OK] wrote zip:", zip_path)
    print("[OK] real images:", len(list(pkg_real_img.glob("*"))))
    print("[OK] synth images:", len(list(pkg_synth_img.glob("*"))))
    print("[OK] masks:", len(list(pkg_masks.glob("*.png"))))


if __name__ == "__main__":
    main()
