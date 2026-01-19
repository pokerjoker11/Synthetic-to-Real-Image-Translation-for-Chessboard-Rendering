@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo ADVANCED TRAINING - Feature Matching + Spectral Normalization
echo ============================================================
echo Key improvements:
echo   - Spectral normalization: Stable D/G balance
echo   - Feature matching loss: Richer training signals
echo   - Balanced loss weights from fixed_d training
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
    --lambda_l1 30 ^
    --lambda_grad 25 ^
    --lambda_perceptual 0.35 ^
    --lambda_feature_match 10 ^
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
    --no_amp

pause
