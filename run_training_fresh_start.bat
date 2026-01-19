@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo FRESH START - ANTI-OVERFITTING CONFIGURATION
echo ============================================================
echo Starting from scratch with anti-overfitting settings:
echo   - Feature matching DISABLED (0) - prevents memorization
echo   - Lower L1 (20) - less pixel-level memorization
echo   - Lower perceptual (0.25) - less texture memorization
echo   - Increased dropout (0.6) - more regularization
echo   - NO RESUME - fresh start to avoid learned bad habits
echo ============================================================
echo.

.\.venv\Scripts\python.exe -u scripts/train_pix2pix.py ^
    --train_csv data/splits_rect/train_clean.csv ^
    --val_csv data/splits_rect/val_final.csv ^
    --batch_size 8 ^
    --image_size 256 ^
    --load_size 256 ^
    --max_steps 100000 ^
    --lr 2e-4 ^
    --lambda_l1 20 ^
    --lambda_grad 25 ^
    --lambda_perceptual 0.25 ^
    --lambda_feature_match 0 ^
    --d_train_freq 2 ^
    --label_smooth 0.1 ^
    --spectral_norm ^
    --grad_gray ^
    --gan_loss bce ^
    --log_every 100 ^
    --sample_every 1000 ^
    --val_every 2000 ^
    --ckpt_dir checkpoints_fresh ^
    --samples_dir results/train_samples_fresh ^
    --no_amp

pause
