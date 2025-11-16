# PIPELINE (Poster‑Friendly Overview)

Goal: From one animal image → predict a clean 3D mesh + keypoints and handy features for analysis.

## Big Picture (Input → Output)

```
[Image 256×192]
     │
     ├─ ViT‑H (AniMer)  →  Visual features (1280‑D)
     │        │
     │        └─ SMAL Head  →  Shape (41) + Pose (35×3×3) + Cam (3)
     │                        │
     │                        └─ SMAL Model → 3D Mesh (3889×3) + 3D KPs (26×3)
     │
     └─ Head Crop → Head Encoder → Head features (768‑D)

Concatenate [Visual 1280 | Head 768 | Params ~359] → Fusion MLP → Fused (512‑D)
```

What you get per image
- 3D Mesh: 3889 vertices (exported as .obj)
- 3D Keypoints: 26 points
- SMAL Parameters: 41 shape (betas), full pose, camera [scale, tx, ty]
- Features: visual (1280‑D), head (768‑D), fused (512‑D)
- Overlay: quick visualization (.png)

## Why this works (at a glance)
- Strong vision backbone (AniMer ViT‑H) understands animal images well.
- SMAL head + SMAL model turn features into 3D shape & pose.
- A small head encoder adds fine facial cues.
- Simple fusion MLP gives a compact 512‑D representation for retrieval/metrics.

## Minimal Workflow
1) Input image (resize to 256×192)
2) ViT‑H features → SMAL params → SMAL mesh & keypoints
3) Head crop → head features
4) Fuse (visual + head + params) → 512‑D fused features
5) Save mesh/npys/overlay

## Quick Commands

Train one step (debug)
```bash
python trainer.py --mode one_step \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --batch_size 1 --num_workers 0 --lr 1e-4
```

Full training (fresh)
```bash
python trainer.py --mode full \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --output_dir ./outputs/horse_training \
  --batch_size 4 --gpus 1 2 --max_epochs 50 --lr 1e-4
```

Resume training
```bash
python trainer.py --mode full \
  --ckpt ./outputs/horse_training/checkpoints/last.ckpt \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --output_dir ./outputs/horse_training \
  --batch_size 4 --gpus 1 2 --max_epochs 50 --lr 1e-4
```

Single‑image inference
```bash
python inference.py \
  --ckpt ./outputs/horse_training/checkpoints/last.ckpt \
  --input /mnt/zone/C/animal_datasets/Horses10/horse10/labeled-data/Sample2/0295.png \
  --out_dir ./outputs/inference/random_horse \
  --device cuda:0
```

## What’s inside (short map)
- `pipeline_integration/integrated_pipeline.py` → end‑to‑end forward
- `trainer.py` → easy CLI (full / one‑step / resume)
- `inference.py` → saves OBJ, overlays, and all .npy outputs
- `AniMer/` → backbone + SMAL head + SMAL model

## Notes 
- Input size: 256×192. ViT‑H weights live at `AniMer/data/backbone.pth`.
- SMAL files are in `AniMer/data/smal/`.
- Current setup targets horses (single SMAL topology). Multi‑species variants can be added later.

