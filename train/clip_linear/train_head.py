import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from utils import CLASS_NAMES
from sklearn.metrics import f1_score, classification_report


class Head(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=0, dropout=0.2):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["features"], d["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default="train/clip_linear/embeddings")
    ap.add_argument("--output", default="train/clip_linear/head.pt")
    ap.add_argument("--hidden-dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--report-dir", default="predict/clip_linear")
    args = ap.parse_args()

    emb_dir = Path(args.embeddings_dir)
    Xtr, ytr = load_npz(emb_dir / "train.npz")
    Xval, yval = load_npz(emb_dir / "val.npz")
    clip_model_name = np.load(emb_dir / "train.npz", allow_pickle=True)["model"].item()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device)
    yval_np = yval

    num_classes = int(max(ytr.max(), yval.max())) + 1
    model = Head(Xtr.shape[1], num_classes, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    best_f1, best_state, stale, best_epoch = -1, None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            preds = model(Xval_t).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(yval_np, preds, average="macro")
        print(f"epoch {epoch:3d} | loss {total_loss/len(ds):.4f} | val macro-F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_state, stale, best_epoch = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xval_t).argmax(dim=1).cpu().numpy()
    with torch.no_grad():
        train_preds = model(Xtr_t.to(device)).argmax(dim=1).cpu().numpy()
    train_f1 = f1_score(ytr, train_preds, average="macro")
    print(f"Train macro-F1 (best checkpoint): {train_f1:.4f}  |  Val macro-F1: {best_f1:.4f}  |  Gap: {train_f1 - best_f1:.4f}")
    print("\nBest checkpoint validation report:")
    print(classification_report(yval_np, preds, digits=4))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "in_dim": Xtr.shape[1],
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }, out_path)
    print(f"\nSaved best head (val macro-F1 {best_f1:.4f}) to {out_path}")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    train_counts = {CLASS_NAMES.get(int(c), int(c)): int((ytr == c).sum()) for c in sorted(set(ytr.tolist()))}
    val_counts = {CLASS_NAMES.get(int(c), int(c)): int((yval == c).sum()) for c in sorted(set(yval.tolist()))}
    report = f"""# Train Report — CLIP Linear/MLP Head

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Backbone
- CLIP model: {clip_model_name}
- Embeddings dir: {emb_dir.resolve()}

## Architecture
- Head type: {'Linear probe' if args.hidden_dim == 0 else f'MLP (hidden_dim={args.hidden_dim}, dropout={args.dropout})'}
- Input dim: {Xtr.shape[1]}
- Num classes: {num_classes}

## Hyperparameters
- Epochs (max): {args.epochs}
- Epochs run: {epoch}
- Learning rate: {args.lr}
- Weight decay: {args.weight_decay}
- Batch size: {args.batch_size}
- Early stopping patience: {args.patience}

## Dataset
- Train samples: {len(ytr)} — {train_counts}
- Val samples: {len(yval)} — {val_counts}

## Result
- Best epoch: {best_epoch}
- Best val macro-F1: {best_f1:.4f}
- Train macro-F1 (best checkpoint): {train_f1:.4f}
- Train-Val Gap: {train_f1 - best_f1:.4f}

### Classification report (best checkpoint, val split)
{classification_report(yval_np, preds, digits=4, target_names=[CLASS_NAMES.get(i, str(i)) for i in range(num_classes)])}

## Output
- Checkpoint saved to: {out_path.resolve()}
"""
    report_path = report_dir / "train_report.md"
    report_path.write_text(report)
    print(f"Saved training report to {report_path.resolve()}")


if __name__ == "__main__":
    main()