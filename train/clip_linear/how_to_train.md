# CLIP Frozen-Embeddings Classifier

Extract frozen OpenAI CLIP image embeddings once, train a small head (linear or 1-hidden-layer MLP) on top, then predict on the test set.

## Setup
```bash
pip install -r train/clip_linear/requirements.txt
```

## 1. Extract embeddings
```bash
python3 train/clip_linear/extract_embeddings.py \
  --input-dir TrainImagesBalanced \
  --output-dir train/clip_linear/embeddings \
  --model ViT-B/32
```
Caches `train.npz` and `val.npz` (features + labels) so this only needs to run once per CLIP model.

## 2. Train the head
```bash
python3 train/clip_linear/train_head.py \
  --embeddings-dir train/clip_linear/embeddings \
  --output train/clip_linear/head.pt \
  --hidden-dim 0 \
  --epochs 50
```
- `--hidden-dim 0` → plain linear probe. Set e.g. `--hidden-dim 256` for a small MLP.
- Selects and saves the checkpoint with the best validation macro-F1 (early stopping via `--patience`).

## 3. Predict on test set
```bash
python3 train/clip_linear/predict.py \
  --test-dir TestImages \
  --head-checkpoint train/clip_linear/head.pt \
  --model ViT-B/32 \
  --output submission.csv
```
Writes `id,predicted` in the order test files sort numerically, matching `template.csv`. Codes: `0=Recyclable, 1=Electronic, 2=Organic`.

## Notes
- `--model` must match between extraction and prediction (same CLIP checkpoint).
- Swapping to a bigger backbone (e.g. `ViT-L/14`) just needs re-running step 1 with `--model ViT-L/14`.