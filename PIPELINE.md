# Project Pipeline

This document explains the full, end‑to‑end pipeline used in this project — from a raw input image to final 3D outputs, features, and saved artifacts. It also includes quick commands to train, resume, and run inference.

## Overview
- Input: RGB image of an animal (e.g., horse), optionally with a bounding box/crop.
- Backbone (visual): AniMer’s ViT‑H extracts global visual features.
- SMAL head: Predicts SMAL parameters (shape betas, joint poses) and weak‑perspective camera.
- SMAL model: Generates the 3D mesh (vertices) and 3D keypoints from the parameters.
- Head feature branch: Crops the head region, encodes to a compact feature vector.
- Feature fusion: Concatenate visual + head + parameter features, project via an MLP to a 512‑D embedding.
- Outputs: Mesh (.obj), 3D keypoints, parameters, camera, intermediate features, and an overlay visualization.

```
[Image 256x192]
   └─> ViT‑H (AniMer backbone) ──> visual_features [B, 1280]
          └─> SMAL head ──> {betas[41], pose[35], cam[3]}
                └─> SMAL model ──> vertices [B, 3889, 3], keypoints3d [B, 26, 3]

[Image] ──> head crop ──> HeadFeatureExtractor ──> head_features [B, 768]

Concat[visual 1280 | head 768 | params ~359] ──> MLP ──> fused_features [B, 512]
```

## Components & Data Flow

- Visual Backbone (AniMer ViT‑H)
  - File: `AniMer/amr/models/backbones/vit.py`
  - Config: `pipeline_integration/integrated_pipeline.py` via `load_amr_config()` (TYPE `vith`, FEATURE_DIM `1280`).
  - Input: Preprocessed image `[B, 3, 256, 192]`.
  - Output: `visual_features` `[B, 1280]` (global pooled or token-averaged).

- SMAL Head (AniMer)
  - File: `AniMer/amr/models/heads/smal_head.py`
  - Input: `visual_features` `[B, 1280]`.
  - Outputs:
    - `pred_betas` `[B, 41]` — shape coefficients.
    - `pred_pose` `[B, 35, 3, 3]` — per‑joint rotations (rotation matrices).
    - `pred_cam` `[B, 3]` — weak‑perspective camera `[scale, tx, ty]`.

- SMAL Model (Mesh Generator)
  - Lives inside AniMer AMR model, used via `self.amr_model.smal(...)`.
  - Inputs: `{betas, pose(rots), global_orient, transl(optional)}`.
  - Outputs: `pred_vertices` `[B, 3889, 3]`, `pred_keypoints_3d` `[B, 26, 3]`.

- Head Feature Branch
  - File: `pipeline_integration/integrated_pipeline.py` (class `HeadFeatureExtractor`).
  - Steps: Head detection/crop → resize to 224 → lightweight CNN encoder.
  - Output: `head_features` `[B, 768]`.

- Feature Fusion (MLP)
  - File: `pipeline_integration/integrated_pipeline.py` (class `FeatureFusionMLP`).
  - Inputs concatenated: `visual_features [1280] + head_features [768] + SMAL param vec [~359]` → `~2407‑D`.
  - Output: `fused_features` `[B, 512]` (useful for retrieval, metrics, or downstream tasks).

## Preprocessing
- Resize input image to `[256, 192]` (H×W) to match ViT‑H expectations.
- Normalize using AniMer’s defaults.
- Optional: detect/bbox crop; the AniMer backbone code may crop internally before feature extraction depending on the config.

## Outputs (per image)
- Geometry & params
  - `vertices` `[3889, 3]`
  - `keypoints3d` `[26, 3]`
  - `betas` `[41]`
  - `pose` `[35, 3, 3]`
  - `cam` `[3]` (scale, tx, ty)
- Features
  - `visual_features` `[1280]` (from ViT‑H)
  - `head_features` `[768]` (from head branch)
  - `fused_features` `[512]` (from MLP)
- Artifacts
  - Mesh `.obj`
  - Overlay image with projected keypoints/mesh
  - `.npy` dumps of all tensors above

All of these are produced by `inference.py` and saved under `outputs/inference/<subdir>/<image_id>/`.

## Training
- Script: `train_horses.py` (Lightning) and `trainer.py` (flexible CLI).
- Losses (example setup in `HorseTrainer`):
  - Shape prior: L2 on `betas`.
  - Pose smoothness: first‑order difference on per‑joint rotations.
  - Feature regularization: small L2 on `fused_features`.
- Optimizer: AdamW; Scheduler: CosineAnnealingLR; Mixed precision on GPU.
- Checkpoints: Saved to `outputs/horse_training/checkpoints/` (`last.ckpt` + top‑k by val loss).

## Checkpoints & Dependencies
- ViT‑H pretrained backbone: `AniMer/data/backbone.pth` (≈3.6 GB).
- SMAL model files: `AniMer/data/smal/` (required for mesh generation).
- AniMer codebase: copied into `LHM_animal_exp/AniMer` and imported by the pipeline.

## Code Map
- `pipeline_integration/integrated_pipeline.py` — End‑to‑end module (AMR + head + fusion + forward/output packing).
- `trainer.py` — CLI for training (full / one‑step) and resume from checkpoint.
- `train_horses.py` — Lightning module (`HorseTrainer`) and dataset wiring.
- `quick_horse_dataset.py` — Minimal Horses10 dataset loader.
- `inference.py` — Single‑image inference and artifact export.
- `INFERENCE_README.md`, `TRAINING_GUIDE.md`, `INTEGRATION_COMPLETE.md` — Additional docs.

## Quickstart Commands
Run from the project root: `LHM_animal_exp`

- One optimization step (debug)
```bash
python trainer.py --mode one_step \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --batch_size 1 --num_workers 0 --lr 1e-4
```

- Full training (fresh)
```bash
python trainer.py --mode full \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --output_dir ./outputs/horse_training \
  --batch_size 4 --gpus 1 2 --max_epochs 50 --lr 1e-4
```

- Resume from checkpoint
```bash
python trainer.py --mode full \
  --ckpt ./outputs/horse_training/checkpoints/last.ckpt \
  --data_dir /mnt/zone/C/animal_datasets/Horses10 \
  --output_dir ./outputs/horse_training \
  --batch_size 4 --gpus 1 2 --max_epochs 50 --lr 1e-4
```

- Single‑image inference
```bash
python inference.py \
  --ckpt ./outputs/horse_training/checkpoints/last.ckpt \
  --input /mnt/zone/C/animal_datasets/Horses10/horse10/labeled-data/Sample2/0295.png \
  --out_dir ./outputs/inference/random_horse \
  --device cuda:0
```

## Multi‑Species / Mesh Family (Notes)
- Current implementation uses a single SMAL topology (e.g., horse), so all batches share the same vertex count (no padding needed).
- For mixed species with different topologies, consider:
  - Padding vertices to the max per batch with a mask and masked losses, or
  - Grouped per‑species processing and fuse only fixed‑length features across species, or
  - Remeshing onto a shared template.

## Tips
- Input size: use `256x192` (H×W) for ViT‑H.
- If `amr` import errors occur, ensure `AniMer/` is present and on `sys.path` (the pipeline adds it automatically).
- Ensure `AniMer/data/backbone.pth` and `AniMer/data/smal/*` exist before training/inference.

---
For deeper details, see the code references above. The pipeline entry (`IntegratedAnimalPipeline`) is designed to be modular: you can swap the head encoder, change the fusion network (e.g., small Transformer), or enable LoRA on the ViT‑H attention layers with minimal code changes.
