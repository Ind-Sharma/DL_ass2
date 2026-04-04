"""Training entrypoint (single-task stubs).

Example::

    python train.py --task cls --data_root path/to/oxford-iiit-pet --epochs 5 --save checkpoints/classifier.pth

Requires a downloaded Oxford-IIIT Pet tree (``images/``, ``annotations/``).
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

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
    args = p.parse_args()

    device = _device()
    print(f"task={args.task}  device={device}  data_root={args.data_root}", flush=True)
    full = OxfordIIITPetDataset(root=args.data_root, split="trainval")
    n_val = int(len(full) * args.val_fraction)
    n_train = len(full) - n_val
    train_set, _ = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.task == "cls":
        model = VGG11Classifier(num_classes=37).to(device)
    elif args.task == "loc":
        model = VGG11Localizer().to(device)
    else:
        model = VGG11UNet(num_classes=3).to(device)

    print(
        f"train samples={len(train_set)}  batches/epoch={len(loader)}  batch_size={args.batch_size}",
        flush=True,
    )
    print(
        "First step can be slow on CPU (model + data). You should see batch lines shortly.",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    iou = IoULoss()
    mse = nn.MSELoss()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"--- epoch {epoch}/{args.epochs} ---", flush=True)
        loss = train_one_epoch(
            model,
            loader,
            opt,
            args.task,
            device,
            ce,
            iou,
            mse,
            log_every=args.log_every,
        )
        print(f"epoch {epoch}/{args.epochs} mean train_loss={loss:.4f}", flush=True)

    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
