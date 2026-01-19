@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo ANTI-OVERFITTING TRAINING CONFIGURATION
echo ============================================================
echo Aggressive changes to prevent memorization:
echo   - Feature matching DISABLED (was 10) - major cause of memorization
echo   - Reduced L1 (30 to 25) - less pixel-level memorization
echo   - Increased dropout (0.5 to 0.6) - more regularization
echo   - Focus on learning generalizable features, not memorizing
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
    --lambda_l1 25 ^
    --lambda_grad 25 ^
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
    --ckpt_dir checkpoints_advanced ^
    --samples_dir results/train_samples_advanced ^
    --resume checkpoints_advanced/latest.pt ^
    --no_amp

pause
