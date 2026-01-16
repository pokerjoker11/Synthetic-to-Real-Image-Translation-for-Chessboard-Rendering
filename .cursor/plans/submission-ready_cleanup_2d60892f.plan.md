---
name: Submission-Ready Cleanup
overview: Make the chess-s2r repo submission-ready by fixing the evaluation API to use the trained model, adding a top-level train.py wrapper, creating comprehensive documentation, and cleaning up dead code and missing dependencies.
todos:
  - id: train-entrypoint
    content: Create top-level train.py wrapper + add --resume flag to scripts/train_pix2pix.py
    status: pending
  - id: eval-api
    content: Update eval_api.py to load trained generator and produce realistic.png via inference
    status: pending
  - id: readme
    content: Create README.md with setup, data structure, training, inference, and smoke test docs
    status: pending
  - id: cleanup
    content: Fix requirements.txt (add torch/torchvision/pillow), fix .gitignore, remove dead files
    status: pending
---

# Submission-Ready Repo Cleanup Plan

## Deliverable 1: Training Entrypoint

The main training logic is already in [`scripts/train_pix2pix.py`](scripts/train_pix2pix.py) with full CLI support. Changes needed:

- **Create top-level `train.py`** that imports and calls `scripts/train_pix2pix.main()` (simple wrapper for discoverability)
- **Add explicit `--resume PATH`** arg to allow resuming from a specific checkpoint (currently auto-resumes only from `checkpoints/latest.pt`)
- Verify all required CLI args are present (they mostly are)

Key existing args in [`scripts/train_pix2pix.py`](scripts/train_pix2pix.py):

```python
--train_csv, --val_csv
--ckpt_dir, --samples_dir
--device, --batch_size, --image_size, --load_size
--max_steps, --lambda_l1, --lambda_grad
```

---

## Deliverable 2: Evaluation API (`generate_chessboard_image`)

Current [`eval_api.py`](eval_api.py) renders synthetic via Blender but **copies synthetic as realistic** (placeholder). Fix:

- **Load trained generator** from `checkpoints/best.pt` (or configurable path via env var `CKPT_PATH`)
- **Run inference** on the synthetic image to produce realistic.png
- **Add determinism**: `torch.manual_seed()`, `torch.backends.cudnn.deterministic`
- Handle missing checkpoint gracefully with clear error

Proposed flow:

```
FEN + viewpoint
    --> Blender renders synthetic.png
    --> Load generator checkpoint
    --> G(synthetic) --> realistic.png
    --> side_by_side.png
```

---

## Deliverable 3: README.md

Create comprehensive README covering:

- **Environment Setup**
  - Clone repo
  - Create venv/conda env
  - `pip install -r requirements.txt`
  - Blender installation note (required for synthetic rendering)

- **Data Preparation**
  - Expected folder structure:
    ```
    data/
      splits/
        train.csv   # columns: real, synth, fen, viewpoint, game, frame
        val.csv
      real/         # real chessboard images
      synth/        # rendered synthetic images
    ```

  - Note: data not included in repo; describe where to download or how to generate

- **Training**
  ```bash
  python train.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv \
    --ckpt_dir checkpoints --samples_dir results/samples --max_steps 5000 --device cuda
  ```

- **Inference/Evaluation**
  ```bash
  python eval_api.py  # uses default FEN, writes to ./results/
  ```


Or from Python:

  ```python
  from eval_api import generate_chessboard_image
  generate_chessboard_image("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "white")
  ```

- **Smoke Test** (verify setup without full training)
  ```bash
  python scripts/smoke_test_model_forward.py
  ```

- **GPU Notes**: Works on CPU but slow; recommend CUDA. Blender also benefits from GPU.

---

## Deliverable 4: Code Cleanup

- **Fix `requirements.txt`**: Add missing deps
  ```
  torch>=2.0
  torchvision>=0.15
  pillow>=9.0
  ```

- **Remove/repurpose empty files**:
  - Delete or populate `src/train.py` (avoid confusion)
  - Delete or populate `src/infer.py`
  - `src/utils/misc.py` appears untracked - check if needed

- **Fix `.gitignore`**: Currently has PowerShell wrapper garbage; replace with clean content

- **Add validation in dataset**: Clear error if CSV or images missing (already exists in `pairs_dataset.py`)

- **Ensure cross-platform**: Already using pathlib; no hardcoded separators found

---

## Verification Commands

After each deliverable, run:

- **Training smoke test** (10 steps):
  ```bash
  python train.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv --max_steps 10 --device cpu
  ```

- **Eval API test**:
  ```bash
  python eval_api.py
  # Check: results/synthetic.png, results/realistic.png, results/side_by_side.png exist
  ```

- **Model forward pass**:
  ```bash
  python scripts/smoke_test_model_forward.py
  ```