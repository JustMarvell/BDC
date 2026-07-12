import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from utils import load_clip, collect_test_files, extract_features
from train_head import Head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="TestImages")
    ap.add_argument("--head-checkpoint", default="train/clip_linear/head.pt")
    ap.add_argument("--model", default="ViT-B/32")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    files = collect_test_files(test_dir)
    print(f"Found {len(files)} test images")

    clip_model, preprocess, device = load_clip(args.model)
    feats = extract_features(files, clip_model, preprocess, device, args.batch_size)

    ckpt = torch.load(args.head_checkpoint, map_location=device)
    head = Head(ckpt["in_dim"], ckpt["num_classes"], ckpt["hidden_dim"], ckpt["dropout"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    with torch.no_grad():
        x = torch.tensor(feats, dtype=torch.float32).to(device)
        preds = head(x).argmax(dim=1).cpu().numpy()

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted"])
        for file_path, pred in zip(files, preds):
            writer.writerow([file_path.stem, int(pred)])

    print(f"Saved predictions to {out_path.resolve()}")
    unique, counts = np.unique(preds, return_counts=True)
    print(dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()