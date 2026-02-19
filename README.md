# TriFuse-LWDETR: Chart Detection with HiFuse + LW-DETR

A lightweight end-to-end object detection model combining HiFuse backbone with LW-DETR for chart element detection.

## Installation

### 1. Clone and install dependencies
```bash
git clone https://github.com/Iongshiba/trifuse-lwdetr.git
cd trifuse-lwdetr
pip install -r requirements.txt
```

### 2. Build CUDA extension
```bash
cd models/ops
python setup.py build install
cd ../..
```

## Dataset

Download the ChartRec dataset from Kaggle: https://www.kaggle.com/datasets/longshiba/chart-detection-v4

The dataset should have this structure:
```
your_dataset/
├── train/           # Training images
├── val/             # Validation images
├── train.json       # COCO format annotations
└── val.json
```

## Training

```bash
python main.py \
    --coco_path /path/to/dataset \
    --output_dir ./output \
    --num_classes 1 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --lr_encoder 2e-4 \
    --encoder trifuse_tiny_caev2 \
    --dec_layers 3 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --num_queries 100 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --ia_bce_loss
```

### With pretrained weights
```bash
python main.py \
    --coco_path /path/to/dataset \
    --pretrained /path/to/pretrained.pth \
    --output_dir ./output \
    --num_classes 1 \
    --epochs 50
```

### Resume training
```bash
python main.py \
    --coco_path /path/to/dataset \
    --resume ./output/checkpoint.pth \
    --output_dir ./output
```

## Evaluation

Evaluate a trained model on the validation set:
```bash
python main.py \
    --coco_path /path/to/dataset \
    --resume /path/to/checkpoint.pth \
    --eval \
    --num_classes 1 \
    --encoder trifuse_tiny_caev2 \
    --dec_layers 3 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --num_queries 100 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --bbox_reparam \
    --lite_refpoint_refine
```

This outputs COCO metrics: AP, AP50, AP75, APs, APm, APl.

## Inference

Run inference on images:
```bash
python main.py \
    --inference \
    --input /path/to/image_or_folder \
    --resume /path/to/checkpoint.pth \
    --output_dir ./inference_output \
    --score_threshold 0.5 \
    --visualize \
    --save_json \
    --num_classes 1 \
    --encoder trifuse_tiny_caev2 \
    --dec_layers 3 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --num_queries 100 \
    --two_stage \
    --projector_scale P4 \
    --bbox_reparam \
    --lite_refpoint_refine
```

Options:
- `--visualize`: Save images with bounding boxes drawn
- `--save_json`: Save detection results as JSON
- `--score_threshold`: Confidence threshold (default: 0.5)
- `--class_names`: Class names for visualization labels

## Model Architecture

### Backbone: TriFuse
A trimmed HiFuse with three stages that fuses:
- **Global Block**: Transformer layers for long-range dependencies
- **Local Block**: CNN layers for local feature extraction
- **HFF Block**: Hierarchical feature fusion

### Head: LW-DETR
Lightweight DETR decoder with multi-scale deformable attention and learnable object queries.

## Reference

- [HiFuse](https://arxiv.org/abs/2209.10218)
- [DETR](https://arxiv.org/abs/2005.12872)
- [LW-DETR](https://arxiv.org/abs/2406.03459)
- [CAEv2](https://openreview.net/forum?id=f36LaK7M0F)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)
