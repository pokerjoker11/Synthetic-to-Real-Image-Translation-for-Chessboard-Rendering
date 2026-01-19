@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo SHARPER PIECES - ADJUSTED LOSS WEIGHTS
echo ============================================================
echo Continuing from fresh training with adjusted weights:
echo   - Higher edge loss (30) - enforces sharp piece boundaries
echo   - Slightly higher perceptual (0.35) - better detail capture
echo   - Lower L1 (18) - less averaging at boundaries
echo   - RESUME from checkpoints_fresh/latest.pt
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
    --lambda_l1 18 ^
    --lambda_grad 30 ^
    --lambda_perceptual 0.35 ^
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
    --resume checkpoints_fresh/latest.pt ^
    --no_amp

pause
