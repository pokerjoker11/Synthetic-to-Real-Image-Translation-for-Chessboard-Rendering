@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo AGGRESSIVE ANTI-OVERFITTING - STRONGER REGULARIZATION
echo ============================================================
echo Continuing with stronger anti-overfitting measures:
echo   - Lower learning rate (1e-4) - finer tuning, less overshooting
echo   - Even lower L1 (15) - reduce averaging pressure
echo   - Higher edge loss (35) - stronger emphasis on sharp boundaries
echo   - Same perceptual (0.35) - keep detail capture
echo   - RESUME from checkpoints_fresh/latest.pt
echo ============================================================
echo.
echo NOTE: This addresses overfitting by:
echo   - Slower learning (lower LR) = more careful updates
echo   - Less averaging (lower L1) = allows sharper edges
echo   - Stronger edge enforcement (higher grad) = sharper pieces
echo ============================================================
echo.

.\.venv\Scripts\python.exe -u scripts/train_pix2pix.py ^
    --train_csv data/splits_rect/train_clean.csv ^
    --val_csv data/splits_rect/val_final.csv ^
    --batch_size 8 ^
    --image_size 256 ^
    --load_size 256 ^
    --max_steps 100000 ^
    --lr 1e-4 ^
    --lambda_l1 15 ^
    --lambda_grad 35 ^
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
