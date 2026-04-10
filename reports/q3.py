"""
W&B Report Q3 (Assignment PDF §2.3): Transfer learning showdown for segmentation.

Strategies logged as separate W&B runs:
1) strict_feature_extractor: freeze full VGG11 encoder
2) partial_finetune: freeze early encoder blocks, unfreeze last N blocks
3) full_finetune: train whole network end-to-end
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.segmentation import VGG11UNet


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_root(data_root: str) -> str:
    root = Path(data_root)
    s = str(root).lower()
    if "kaggle" not in s:
        return str(root)

    candidates = [root, root / "new_data"]
    if root.name == "input" and root.is_dir():
        candidates.extend([d for d in root.iterdir() if d.is_dir()])

    for c in candidates:
        if (c / "annotations").is_dir() and (c / "images").is_dir():
            print(f"Kaggle mode: using data_root={c}", flush=True)
            return str(c)

    print(f"Kaggle mode: using provided data_root={root}", flush=True)
    return str(root)


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _configure_strategy(
    model: VGG11UNet, strategy: str, partial_unfreeze_blocks: int
) -> None:
    if strategy == "strict_feature_extractor":
        _set_requires_grad(model.encoder, False)
        _set_requires_grad(model.decoder, True)
        return

    if strategy == "partial_finetune":
        # Freeze all encoder, then unfreeze last N blocks (from block5 backwards).
        _set_requires_grad(model.encoder, False)
        n = max(1, min(5, int(partial_unfreeze_blocks)))
        block_names = ["block5", "block4", "block3", "block2", "block1"][:n]
        pool_names = ["pool5", "pool4", "pool3", "pool2", "pool1"][:n]
        drop_names = ["drop5", "drop4", "drop3", "drop2", "drop1"][:n]
        for name in block_names + pool_names + drop_names:
            m = getattr(model.encoder, name, None)
            if m is not None:
                _set_requires_grad(m, True)
        _set_requires_grad(model.decoder, True)
        return

    if strategy == "full_finetune":
        _set_requires_grad(model, True)
        return

    raise ValueError(
        "strategy must be one of: strict_feature_extractor, partial_finetune, full_finetune"
    )


def _load_encoder_from_classifier(model: VGG11UNet, classifier_ckpt: str) -> None:
    if not classifier_ckpt:
        return
    if not os.path.isfile(classifier_ckpt):
        raise FileNotFoundError(f"--classifier_ckpt not found: {classifier_ckpt}")
    clf = VGG11Classifier(num_classes=37)
    clf.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"), strict=True)
    model.encoder.load_state_dict(clf.encoder.state_dict(), strict=True)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, _, _, mask in loader:
        x = x.to(device)
        mask = mask.to(device)
        logits = model(x)
        loss = loss_fn(logits, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    total_px = 0
    inter = np.zeros(num_classes, dtype=np.float64)
    pred_sum = np.zeros(num_classes, dtype=np.float64)
    tgt_sum = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for x, _, _, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            logits = model(x)
            loss = loss_fn(logits, mask)
            total_loss += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)

            pred = logits.argmax(dim=1)
            correct += int((pred == mask).sum().item())
            total_px += int(mask.numel())

            for c in range(num_classes):
                p = pred == c
                t = mask == c
                inter[c] += float((p & t).sum().item())
                pred_sum[c] += float(p.sum().item())
                tgt_sum[c] += float(t.sum().item())

    val_loss = total_loss / max(n, 1)
    pixel_acc = float(correct) / max(total_px, 1)
    dice_per_class = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    macro_dice = float(np.mean(dice_per_class))
    return val_loss, pixel_acc, macro_dice


def _iter_strategies(value: str) -> Iterable[str]:
    for s in value.split(","):
        s = s.strip()
        if s:
            yield s


def _run_strategy(
    *,
    strategy: str,
    project: str,
    entity: str | None,
    data_root: str,
    classifier_ckpt: str,
    partial_unfreeze_blocks: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    num_workers: int,
) -> None:
    _seed_all(seed)
    device = _device()
    full = OxfordIIITPetDataset(root=data_root, split="trainval")
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VGG11UNet(num_classes=3).to(device)
    _load_encoder_from_classifier(model, classifier_ckpt)
    _configure_strategy(model, strategy, partial_unfreeze_blocks)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters after strategy configuration.")
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"q3_{strategy}",
        config={
            "question": "2.3",
            "task": "seg",
            "strategy": strategy,
            "partial_unfreeze_blocks": partial_unfreeze_blocks,
            "classifier_ckpt": classifier_ckpt,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "val_fraction": val_fraction,
            "device": str(device),
            "trainable_params": int(sum(p.numel() for p in params)),
        },
    )

    best_dice = -1.0
    best_val_loss = float("inf")
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        e0 = time.perf_counter()
        train_loss = _train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_pixel_acc, val_macro_dice = _eval_epoch(
            model, val_loader, loss_fn, device
        )
        epoch_s = time.perf_counter() - e0

        best_dice = max(best_dice, val_macro_dice)
        best_val_loss = min(best_val_loss, val_loss)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/pixel_acc": val_pixel_acc,
                "val/macro_dice": val_macro_dice,
                "time/epoch_s": epoch_s,
                "time/elapsed_s": time.perf_counter() - t0,
            }
        )

    wandb.summary["best_val_macro_dice"] = best_dice
    wandb.summary["best_val_loss"] = best_val_loss
    run.finish()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument(
        "--classifier_ckpt",
        type=str,
        default="classifier.pth",
        help="Optional classifier checkpoint to initialize shared encoder.",
    )
    p.add_argument(
        "--strategies",
        type=str,
        default="strict_feature_extractor,partial_finetune,full_finetune",
        help="Comma-separated strategy names.",
    )
    p.add_argument(
        "--partial_unfreeze_blocks",
        type=int,
        default=2,
        help="For partial_finetune: number of last encoder blocks to unfreeze (1..5).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    data_root = _resolve_data_root(args.data_root)
    entity = args.entity.strip() or None
    strategies = list(_iter_strategies(args.strategies))
    if not strategies:
        raise ValueError("No strategies provided.")

    for strategy in strategies:
        _run_strategy(
            strategy=strategy,
            project=args.project,
            entity=entity,
            data_root=data_root,
            classifier_ckpt=args.classifier_ckpt,
            partial_unfreeze_blocks=args.partial_unfreeze_blocks,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()

