import argparse
from pathlib import Path

import numpy as np

from utils import load_clip, collect_class_dirs, extract_features, VALID_EXT


def extract_split(split_dir: Path, model, preprocess, device, batch_size):
    dirs, classes, codes = collect_class_dirs(split_dir)
    all_files, all_labels = [], []
    for d, code in zip(dirs, codes):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in VALID_EXT)
        all_files.extend(files)
        all_labels.extend([code] * len(files))
        print(f"  {d.name}: {len(files)} images")

    feats = extract_features(all_files, model, preprocess, device, batch_size)
    labels = np.array(all_labels, dtype=np.int64)
    return feats, labels, [str(f) for f in all_files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="TrainImagesBalanced")
    ap.add_argument("--output-dir", default="train/clip_linear/embeddings")
    ap.add_argument("--model", default="ViT-B/32")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, device = load_clip(args.model)
    print(f"Loaded {args.model} on {device}")

    for split in ["train", "val"]:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Skip {split}: {split_dir} not found")
            continue
        print(f"\nExtracting {split} split from {split_dir}")
        feats, labels, files = extract_split(split_dir, model, preprocess, device, args.batch_size)
        np.savez(out_dir / f"{split}.npz", features=feats, labels=labels, files=files)
        print(f"Saved {out_dir / (split + '.npz')} ({feats.shape[0]} samples, dim={feats.shape[1]})")


if __name__ == "__main__":
    main()