import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = {0: "Recyclable", 1: "Electronic", 2: "Organic"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_clip(model_name="ViT-B/32", device=None):
    import clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device


def collect_class_dirs(split_dir: Path):
    dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    classes = [d.name for d in dirs]
    codes = [int(d.name.split("_")[0]) for d in dirs]
    return dirs, classes, codes


def numeric_sort_key(p: Path):
    m = re.search(r"\d+", p.stem)
    return int(m.group()) if m else p.stem


def collect_test_files(test_dir: Path):
    files = [p for p in test_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    return sorted(files, key=numeric_sort_key)


@torch.no_grad()
def extract_features(files, model, preprocess, device, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(files), batch_size), desc="extracting", unit="batch"):
        batch = files[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in batch]).to(device)
        f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)