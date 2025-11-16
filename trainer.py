#!/usr/bin/env python3
"""
Flexible trainer entrypoint for the integrated animal pipeline.

Features:
- run full training (uses pl.Trainer)
- run a single optimization step (max_steps=1) for debugging
- configurable dataset, batch size, gpus, lr, epochs

Usage examples (from repo root):
  python trainer.py --mode full --gpus 0 1 --batch_size 4
  python trainer.py --mode one_step --batch_size 1
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import argparse

# Make local modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline_integration'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AniMer'))

from train_horses import HorseTrainer, QuickHorseDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

# ---------------------------------------------------------
# DEBUG: Print what GPUs PyTorch can actually see
# ---------------------------------------------------------
print("\n===== PYTORCH GPU VISIBILITY CHECK =====")
print("CUDA available:", torch.cuda.is_available())
print("Total visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        print(f"cuda:{i} → {torch.cuda.get_device_name(i)}")
    except:
        print(f"cuda:{i} → (device name unavailable)")
print("=========================================\n")


def run_one_step(data_dir, batch_size, num_workers, lr, device):
    ds = QuickHorseDataset(data_dir, image_size=(256, 192), split='train')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = HorseTrainer(learning_rate=lr)

    use_gpu = torch.cuda.is_available() and device is not None
    accel = 'gpu' if use_gpu else 'cpu'
    precision = '16-mixed' if use_gpu else 32

    trainer = pl.Trainer(
        max_steps=1,
        limit_val_batches=0,
        accelerator=accel,
        devices=1,
        precision=precision,
        enable_progress_bar=True
    )

    trainer.fit(model, train_dataloaders=dl)


def run_full_train(data_dir, output_dir, batch_size, num_workers, max_epochs, gpus, lr, ckpt_path=None):
    ds_train = QuickHorseDataset(data_dir, image_size=(256, 192), split='train')
    ds_val = QuickHorseDataset(data_dir, image_size=(256, 192), split='val')

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = HorseTrainer(learning_rate=lr)

    # Reuse checkpoint callback and logger from train_horses for familiarity
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='horse-{epoch:02d}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=output_dir, name='logs')

    use_gpu = torch.cuda.is_available() and len(gpus) > 0
    accel = 'gpu' if use_gpu else 'cpu'

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accel,
        devices=gpus if use_gpu else None,
        strategy='ddp_find_unused_parameters_true' if use_gpu and len(gpus) > 1 else 'auto',
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision='16-mixed' if use_gpu else 32,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'one_step'], default='full')
    parser.add_argument('--data_dir', type=str, default='/mnt/zone/C/animal_datasets/Horses10')
    parser.add_argument('--output_dir', type=str, default='./outputs/horse_training')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--gpus', type=int, nargs='*', default=[1, 2])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None, help='Device string like cuda:0 (optional)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    if args.mode == 'one_step':
        run_one_step(args.data_dir, args.batch_size, args.num_workers, args.lr, args.device)
    else:
        run_full_train(args.data_dir, args.output_dir, args.batch_size, args.num_workers, args.max_epochs, args.gpus, args.lr, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
