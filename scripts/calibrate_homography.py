# scripts/calibrate_homography.py
import argparse
import json
from pathlib import Path
import numpy as np
import cv2

def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="Directory with images to calibrate on")
    ap.add_argument("--out", required=True, help="Output json path, e.g. calib/homography_real.json")
    ap.add_argument("--out_size", type=int, default=512)
    ap.add_argument("--display_size", type=int, default=1024, help="Shown size for easier clicking")
    args = ap.parse_args()

    img_dir = Path(args.image_dir)
    imgs = sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png"), *img_dir.glob("*.jpeg")])
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")

    img_path = imgs[0]
    img = imread_unicode(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h0, w0 = img.shape[:2]

    # show a resized version for easy clicking
    disp = cv2.resize(img, (args.display_size, args.display_size), interpolation=cv2.INTER_AREA)
    sx = w0 / args.display_size
    sy = h0 / args.display_size

    print(f"Using image: {img_path}")
    print("Click 4 board corners in this order: TL, TR, BR, BL")
    pts_disp = []

    win = "Click TL,TR,BR,BL (ESC to abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_disp.append([x, y])
            print(f"Point {len(pts_disp)}: ({x}, {y})")

    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = disp.copy()
        for p in pts_disp:
            cv2.circle(vis, tuple(p), 7, (0, 255, 0), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Aborted.")
        if len(pts_disp) == 4:
            break

    cv2.destroyAllWindows()

    # map clicked points back to original image coords
    pts = np.array([[p[0] * sx, p[1] * sy] for p in pts_disp], dtype=np.float32)

    out_size = args.out_size
    dst = np.array([[0, 0],
                    [out_size - 1, 0],
                    [out_size - 1, out_size - 1],
                    [0, out_size - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(pts, dst)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"out_size": out_size, "H": H.tolist(), "example_image": str(img_path)},
            f,
            indent=2
        )

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
