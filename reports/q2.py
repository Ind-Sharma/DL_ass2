"""
W&B Report Q2 (Assignment PDF §2.2): Internal dynamics with dropout ablation.

Runs classification training under three conditions:
1) No dropout (p=0.0)
2) Custom dropout p=0.2
3) Custom dropout p=0.5

Logs interactive W&B curves for:
- train/loss
- val/loss
- generalization_gap/loss (val - train)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_f1_cls(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    ys: List[np.ndarray] = []
    pr: List[np.ndarray] = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            ys.append(y.detach().cpu().numpy())
            pr.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(pr)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _epoch_train_loss(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def _epoch_eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


@dataclass(frozen=True)
class RunResult:
    run_name: str
    best_val_loss: float
    best_val_macro_f1: float


def _parse_values(s: str) -> List[float]:
    vals: List[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    return vals


def _build_split(
    data_root: str,
    val_fraction: float,
    split_seed: int,
) -> tuple[Subset, Subset]:
    full = OxfordIIITPetDataset(root=data_root, split="trainval")
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed)
    )
    return train_set, val_set


def _run_single(
    *,
    project: str,
    entity: str | None,
    data_root: str,
    dropout_p: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    num_workers: int,
) -> RunResult:
    _seed_all(seed)
    device = _device()

    train_set, val_set = _build_split(
        data_root=data_root,
        val_fraction=val_fraction,
        split_seed=seed,
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

    model = VGG11Classifier(num_classes=37, dropout_p=dropout_p).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    run_name = f"q2_dropout_p{dropout_p:.1f}"
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config={
            "question": "2.2",
            "task": "cls",
            "dropout_p": dropout_p,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "val_fraction": val_fraction,
            "device": str(device),
        },
    )

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        e0 = time.perf_counter()
        train_loss = _epoch_train_loss(model, train_loader, opt, ce, device)
        val_loss = _epoch_eval_loss(model, val_loader, ce, device)
        val_f1 = _macro_f1_cls(model, val_loader, device)
        epoch_s = time.perf_counter() - e0
        best_val_loss = min(best_val_loss, val_loss)
        best_val_f1 = max(best_val_f1, val_f1)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/macro_f1": val_f1,
                "generalization_gap/loss": val_loss - train_loss,
                "time/epoch_s": epoch_s,
                "time/elapsed_s": time.perf_counter() - t0,
            }
        )

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_val_macro_f1"] = best_val_f1
    run.finish()
    return RunResult(run_name=run_name, best_val_loss=best_val_loss, best_val_macro_f1=best_val_f1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--dropout_values",
        type=str,
        default="0.0,0.2,0.5",
        help="Comma-separated dropout probabilities, e.g. 0.0,0.2,0.5",
    )
    args = p.parse_args()

    data_root = str(Path(args.data_root))
    entity = args.entity.strip() or None
    dropout_values = _parse_values(args.dropout_values)

    if not dropout_values:
        raise ValueError("No dropout values provided.")

    for p_drop in dropout_values:
        if not (0.0 <= p_drop <= 1.0):
            raise ValueError(f"Invalid dropout value {p_drop}; expected within [0, 1].")
        _run_single(
            project=args.project,
            entity=entity,
            data_root=data_root,
            dropout_p=p_drop,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()

