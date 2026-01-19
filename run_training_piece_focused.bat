@echo off
cd /d "%~dp0"
set PYTHONPATH=.
set PYTHONUNBUFFERED=1

echo ============================================================
echo PIECE-FOCUSED SUPERVISION TRAINING
echo ============================================================
echo Piece-focused supervision for sharper chess pieces:
echo   - Piece mask weighted losses (L1, grad, perceptual)
echo   - Piece-patch discriminator (focuses on piece regions)
echo   - Hinge loss (better GAN training)
echo   - R1 regularization (stabilizes discriminator)
echo   - Piece sharpness validation metric
echo ============================================================
echo.
echo Configuration:
echo   - Piece mask dir: data/masks (if masks exist)
echo   - Piece weight: 6.0 (6x weight on piece pixels)
echo   - Piece discriminator: enabled (96x96 patches, 2 per image)
echo   - R1 gamma: 10.0 (moderate regularization)
echo   - Hinge loss: enabled (label smoothing auto-disabled)
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
    --spectral_norm ^
    --grad_gray ^
    --gan_loss hinge ^
    --use_piece_mask ^
    --piece_mask_dir data/masks ^
    --piece_weight 6.0 ^
    --use_piece_D ^
    --piece_crop_size 96 ^
    --piece_patches_per_image 2 ^
    --lambda_piece_gan 1.0 ^
    --r1_gamma 10.0 ^
    --log_every 100 ^
    --sample_every 1000 ^
    --val_every 2000 ^
    --ckpt_dir checkpoints_fresh ^
    --samples_dir results/train_samples_fresh ^
    --resume checkpoints_fresh/latest.pt ^
    --no_amp

pause
