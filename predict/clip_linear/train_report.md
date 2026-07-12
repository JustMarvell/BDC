# Train Report — CLIP Linear/MLP Head

Generated: 2026-07-12 13:22:40

## Backbone
- CLIP model: ViT-B/32
- Embeddings dir: /media/vellosaurus/VelloStorage/BDC/train/clip_linear/embeddings

## Architecture
- Head type: Linear probe
- Input dim: 512
- Num classes: 3

## Hyperparameters
- Epochs (max): 50
- Epochs run: 47
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 256
- Early stopping patience: 8

## Dataset
- Train samples: 23762 — {'Recyclable': 7762, 'Electronic': 8000, 'Organic': 8000}
- Val samples: 3708 — {'Recyclable': 1370, 'Electronic': 535, 'Organic': 1803}

## Result
- Best epoch: 39
- Best val macro-F1: 0.9722
- Train macro-F1 (best checkpoint): 0.9717
- Train-Val Gap: -0.0005

### Classification report (best checkpoint, val split)
              precision    recall  f1-score   support

  Recyclable     0.9586    0.9628    0.9607      1370
  Electronic     0.9708    0.9944    0.9825       535
     Organic     0.9787    0.9684    0.9735      1803

    accuracy                         0.9701      3708
   macro avg     0.9694    0.9752    0.9722      3708
weighted avg     0.9701    0.9701    0.9701      3708


## Output
- Checkpoint saved to: /media/vellosaurus/VelloStorage/BDC/train/clip_linear/head.pt
