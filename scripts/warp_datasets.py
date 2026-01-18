# scripts/warp_datasets.py
import json
from pathlib import Path
import numpy as np
import cv2

def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path: Path, img):
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))

def warp_dir(src_dir: Path, dst_dir: Path, H: np.ndarray, out_size: int):
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files, start=1):
        img = imread_unicode(p)
        if img is None:
            print(f"[WARN] failed read: {p}")
            continue
        warped = cv2.warpPerspective(img, H, (out_size, out_size))
        imwrite_unicode(dst_dir / p.name, warped)
        if i % 200 == 0:
            print(f"[OK] warped {i}/{len(files)}")

    print(f"[DONE] {src_dir} -> {dst_dir} ({len(files)} files)")

def main():
    with open("calib/homography_real.json", "r", encoding="utf-8") as f:
        j_real = json.load(f)
    with open("calib/homography_synth.json", "r", encoding="utf-8") as f:
        j_syn = json.load(f)

    H_real = np.array(j_real["H"], dtype=np.float32)
    H_syn  = np.array(j_syn["H"], dtype=np.float32)
    out_size = int(j_real["out_size"])
    if int(j_syn["out_size"]) != out_size:
        raise RuntimeError("out_size mismatch between real and synth homographies")

    warp_dir(Path("data/real/images"),  Path("data/real_warp/images"),  H_real, out_size)
    warp_dir(Path("data/synth/images"), Path("data/synth_warp/images"), H_syn,  out_size)

    # copy gt files (filenames unchanged)
    Path("data/real_warp").mkdir(exist_ok=True)
    Path("data/synth_warp").mkdir(exist_ok=True)
    for src, dst in [
        (Path("data/real/gt.csv"), Path("data/real_warp/gt.csv")),
        (Path("data/real/gt_with_synth.csv"), Path("data/real_warp/gt_with_synth.csv")),
        (Path("data/synth/gt.csv"), Path("data/synth_warp/gt.csv")),
    ]:
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print("[OK] Copied gt.csv files into *_warp folders")

if __name__ == "__main__":
    main()
