@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo GENERALIZATION-FOCUSED TRAINING
echo ============================================================
echo Key changes to improve generalization:
echo   - Reduced feature matching (was 10, now 5) - less memorization
echo   - Increased L1 slightly (30 to 35) - better structure learning
echo   - Continue from step 6000 checkpoint
echo   - More training steps to learn generalizable features
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
    --lambda_l1 35 ^
    --lambda_grad 25 ^
    --lambda_perceptual 0.35 ^
    --lambda_feature_match 5 ^
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
