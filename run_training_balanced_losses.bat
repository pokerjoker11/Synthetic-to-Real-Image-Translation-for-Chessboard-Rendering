@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo MAXIMUM SHARPNESS - AGGRESSIVE EDGE EMPHASIS
echo ============================================================
echo Going harder on piece sharpness:
echo   - MUCH higher edge loss (60) - maximum emphasis on sharp boundaries
echo   - Lower perceptual (0.15) - further reduce dominance
echo   - Lower L1 (12) - reduce averaging even more
echo   - RESUME from checkpoints_fresh/latest.pt
echo ============================================================
echo.
echo Current loss balance (step 45k):
echo   - Edge: ~18%% (TOO LOW - pieces still blurry)
echo   - Perceptual: ~23%% (still dominating)
echo   - GAN: ~45%%
echo   - L1: ~14%%
echo.
echo Target balance (prioritize sharpness):
echo   - Edge: ~25-30%% (SHARP PIECES - highest priority)
echo   - Perceptual: ~15%% (minimal texture, not dominating)
echo   - GAN: ~40%% (realism)
echo   - L1: ~20%% (structure)
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
    --lambda_l1 12 ^
    --lambda_grad 60 ^
    --lambda_perceptual 0.15 ^
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
