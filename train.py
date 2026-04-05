"""Training entrypoint (single-task stubs).

Example::

    python train.py --task cls --data_root path/to/oxford-iiit-pet --epochs 30 --save classifier.pth

Requires a downloaded Oxford-IIIT Pet tree (``images/``, ``annotations/``).
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _macro_f1_cls(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Macro-averaged F1 for 37-class classification (matches sklearn-style eval)."""
    from sklearn.metrics import f1_score

    model.eval()
    ys: list[np.ndarray] = []
    pr: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y, _, _ = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.append(y.detach().cpu().numpy())
            pr.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(pr)
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task: Literal["cls", "loc", "seg"],
    device: torch.device,
    ce_loss: nn.Module,
    iou_loss: IoULoss,
    mse_loss: nn.Module,
    log_every: int = 1,
) -> float:
    model.train()
    total = 0.0
    n = 0
    n_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        if task == "cls":
            x, y, _, _ = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
        elif task == "loc":
            x, _, bbox, _ = batch
            x = x.to(device)
            bbox = bbox.to(device)
            pred = model(x)
            loss = mse_loss(pred, bbox) + iou_loss(pred, bbox)
        else:
            x, _, _, mask = batch
            x = x.to(device)
            mask = mask.to(device)
            logits = model(x)
            loss = ce_loss(logits, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)

        if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == n_batches):
            print(
                f"    batch {batch_idx}/{n_batches}  loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cls", "loc", "seg"], required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", type=str, default="checkpoint.pth")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Print every N batches (1 = every batch). Set 0 to print only epoch summary.",
    )
    p.add_argument(
        "--weighted_ce",
        action="store_true",
        help="For cls: use inverse-frequency class weights (helps 37-way imbalance).",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="For cls: CrossEntropy label smoothing (0 disables).",
    )
    p.add_argument(
        "--encoder_ckpt",
        type=str,
        default="",
        help="For loc/seg: path to a trained classifier.pth — copy its encoder weights into "
        "this model so features match MultiTaskPerceptionModel (shared backbone = classifier).",
    )
    args = p.parse_args()

    device = _device()
    print(f"task={args.task}  device={device}  data_root={args.data_root}", flush=True)
    full = OxfordIIITPetDataset(root=args.data_root, split="trainval")
    n_val = max(1, int(len(full) * args.val_fraction))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.task == "cls":
        model = VGG11Classifier(num_classes=37).to(device)
    elif args.task == "loc":
        model = VGG11Localizer().to(device)
    else:
        model = VGG11UNet(num_classes=3).to(device)

    if args.task in ("loc", "seg") and args.encoder_ckpt:
        ckpt_path = args.encoder_ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--encoder_ckpt not found: {ckpt_path}")
        clf = VGG11Classifier(num_classes=37)
        clf.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print(
            f"Encoder initialized from classifier checkpoint: {ckpt_path} "
            f"(matches MultiTaskPerceptionModel backbone).",
            flush=True,
        )

    print(
        f"train samples={len(train_set)}  val samples={len(val_set)}  "
        f"batches/epoch={len(train_loader)}  batch_size={args.batch_size}",
        flush=True,
    )
    print(
        "First step can be slow on CPU (model + data). You should see batch lines shortly.",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.task == "cls":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=2
        )
    else:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2
        )

    if args.task == "cls":
        ls = args.label_smoothing if args.label_smoothing > 0 else 0.0
        if args.weighted_ce:
            # Weights from full trainval indices in train_set
            base_ds = train_set.dataset
            assert isinstance(base_ds, OxfordIIITPetDataset)
            idxs = train_set.indices
            counts = np.zeros(37, dtype=np.float64)
            for j in idxs:
                _, lab, _, _ = base_ds[j]
                counts[int(lab)] += 1.0
            w = 1.0 / (counts + 1e-6)
            w = w * (37.0 / w.sum())
            wt = torch.tensor(w, dtype=torch.float32, device=device)
            ce = nn.CrossEntropyLoss(weight=wt, label_smoothing=ls)
            print("Using weighted CrossEntropyLoss + label_smoothing", flush=True)
        else:
            ce = nn.CrossEntropyLoss(label_smoothing=ls)
    else:
        ce = nn.CrossEntropyLoss()
    iou = IoULoss()
    mse = nn.MSELoss()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    best_f1 = -1.0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        print(f"--- epoch {epoch}/{args.epochs} ---", flush=True)
        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            args.task,
            device,
            ce,
            iou,
            mse,
            log_every=args.log_every,
        )
        print(f"epoch {epoch}/{args.epochs} mean train_loss={loss:.4f}", flush=True)

        if args.task == "cls":
            vf1 = _macro_f1_cls(model, val_loader, device)
            print(f"validation macro-F1={vf1:.4f}", flush=True)
            sch.step(vf1)
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  (new best macro-F1={best_f1:.4f})", flush=True)
        else:
            sch.step(loss)

    if args.task == "cls" and best_state is not None:
        torch.save(best_state, args.save)
        print(f"saved BEST val macro-F1={best_f1:.4f} -> {args.save}", flush=True)
    else:
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
