
# generate_dataset.py
# Python "runner" that reads fens.txt and calls Blender headless for each FEN.

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Config:
    blender_exe: Path
    blend_file: Path
    blender_script: Path

    fens_file: Path

    output_root: Path          # e.g. data/synthetic/train
    sample_prefix: str = "sample_"
    start_index: int = 0

    views: List[str] = None    # informational (meta); Blender script decides what it renders
    render_resolution: int = 512
    render_samples: int = 64
    view_side: str = "black"   # passed to blender script if supported: --view black/white

    # If True, will skip sample folder if it already exists (safe re-run).
    skip_existing: bool = True


def read_fens(fens_path: Path) -> List[str]:
    if not fens_path.exists():
        raise FileNotFoundError(f"fens.txt not found: {fens_path}")

    fens: List[str] = []
    for line in fens_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fens.append(line)
        print(f"[DEBUG] Loaded FEN: {line}")
    return fens


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_meta(sample_dir: Path, *, fen: str, cfg: Config) -> None:
    meta = {
        "fen": fen,
        "views": cfg.views or ["overhead", "west", "east"],
        "render_resolution": [cfg.render_resolution, cfg.render_resolution],
        "render_samples": cfg.render_samples,
        "view_side": cfg.view_side,
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_blender_for_fen(sample_dir: Path, *, fen: str, cfg: Config) -> None:
    """
    Runs Blender headless to render images for a single FEN.
    """
    renders_dir = sample_dir / "renders"
    # Build Blender command
    cmd = [
        str(cfg.blender_exe),
        str(cfg.blend_file),
        "--background",
        "--python", str(cfg.blender_script),
        "--",
        "--fen", fen,
        "--resolution", str(cfg.render_resolution),
        "--samples", str(cfg.render_samples),
        "--view", cfg.view_side,
        "--out_dir", str(renders_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(sample_dir),
        capture_output=True,
        text=True,
        errors="replace",
    )


    if result.returncode != 0:
        if result.stdout:
            (sample_dir / "blender_stdout.txt").write_text(result.stdout, encoding="utf-8")
        if result.stderr:
            (sample_dir / "blender_stderr.txt").write_text(result.stderr, encoding="utf-8")
        raise RuntimeError(
            f"Blender failed (code {result.returncode}) for sample {sample_dir.name}\n"
            f"See logs: {sample_dir/'blender_stdout.txt'} and {sample_dir/'blender_stderr.txt'}"
        )




def main() -> int:
    # ---- Configure paths (edit these to match your machine) ----
    current_directory = os.getcwd()
    cfg = Config(
        blender_exe=Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"),
        blend_file=Path(rf"{current_directory}\chess-set.blend"),
        blender_script=Path(rf"{current_directory}\chess_position_api_v2.py"),
        fens_file=Path(rf"{current_directory}\fens.txt"),
        output_root=Path(rf"{current_directory}\data\synthetic\train"),
        views=["overhead", "west", "east"],
        render_resolution=512,
        render_samples=64,
        view_side="black",
        skip_existing=True,
        start_index=0,
    )

    # ---- Sanity checks ----
    for p, name in [
        (cfg.blender_exe, "blender_exe"),
        (cfg.blend_file, "blend_file"),
        (cfg.blender_script, "blender_script"),
        (cfg.fens_file, "fens_file"),
    ]:
        if not p.exists():
            print(f"[ERROR] Missing {name}: {p}", file=sys.stderr)
            return 2

    ensure_dir(cfg.output_root)

    fens = read_fens(cfg.fens_file)
    if not fens:
        print("[ERROR] No FENs found in fens.txt", file=sys.stderr)
        return 3

    print(f"[INFO] Loaded {len(fens)} FENs from {cfg.fens_file}")
    print(f"[INFO] Output root: {cfg.output_root}")

    for i, fen in enumerate(fens, start=cfg.start_index):
        sample_name = f"{cfg.sample_prefix}{i:05d}"
        sample_dir = cfg.output_root / sample_name

        if sample_dir.exists() and cfg.skip_existing:
            print(f"[SKIP] {sample_name} already exists")
            continue

        ensure_dir(sample_dir)

        # Write meta first (it’s fine; it records intended settings)
        write_meta(sample_dir, fen=fen, cfg=cfg)

        print(f"[RUN ] {sample_name}  fen={fen}")
        try:
            run_blender_for_fen(sample_dir, fen=fen, cfg=cfg)
        except Exception as e:
            print(f"[FAIL] {sample_name}: {e}", file=sys.stderr)
            # Keep the sample_dir + logs for debugging; continue or stop:
            return 1

        print(f"[OK  ] {sample_name}")

    print("[DONE] Dataset generation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
