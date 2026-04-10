"""
W&B Report Q1 (Assignment PDF §2.1): BatchNorm effect + activation distributions.

This script is NOT part of the autograder submission; it only helps generate W&B artifacts.

What it logs (interactive in W&B):
- Train/val curves for BN vs no-BN runs (classification).
- Activation histograms for the "3rd convolutional layer" on the *same* fixed input.
- (Optional) A quick learning-rate stability sweep to estimate the max stable LR.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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


def _replace_batchnorm_with_identity(module: nn.Module) -> nn.Module:
    """In-place replace all BatchNorm2d/1d with Identity for a 'no BN' ablation."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, nn.Identity())
        else:
            _replace_batchnorm_with_identity(child)
    return module


@torch.no_grad()
def _macro_f1_cls(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    ys: List[np.ndarray] = []
    pr: List[np.ndarray] = []
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


def _train_one_epoch_cls(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ce_loss: nn.Module,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@dataclass(frozen=True)
class RunResult:
    run_name: str
    max_stable_lr: float | None


def _activation_histogram_3rd_conv(model: VGG11Classifier, x1: torch.Tensor) -> np.ndarray:
    """
    Capture activations from the '3rd conv layer' (VGG11 conv3), approximated as:
    encoder.block3 first ReLU output (Conv(128->256) + BN + ReLU).
    """
    acts: List[torch.Tensor] = []

    def hook(_m: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        acts.append(out.detach().cpu())

    # block3 layout: [conv, bn, relu, conv, bn, relu]
    h = model.encoder.block3[2].register_forward_hook(hook)
    try:
        model.eval()
        _ = model(x1)
    finally:
        h.remove()

    if not acts:
        raise RuntimeError("Activation hook did not fire; check layer indexing.")
    a = acts[0].flatten().numpy()
    return a


def _lr_sweep_max_stable(
    make_model: callable,
    device: torch.device,
    train_loader: DataLoader,
    lrs: Iterable[float],
    steps_per_lr: int,
    loss_fn: nn.Module,
    diverge_loss: float,
) -> Tuple[float | None, wandb.Table]:
    """
    Quick & dirty stability sweep: for each LR, train a fresh model for a few steps and mark stable
    if loss stays finite and below `diverge_loss`.
    """
    rows = []
    max_stable: float | None = None

    for lr in lrs:
        model = make_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        stable = True
        last_loss = None
        model.train()
        it = iter(train_loader)
        for _ in range(steps_per_lr):
            try:
                x, y, _, _ = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y, _, _ = next(it)

            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            last_loss = float(loss.detach().cpu())
            if not np.isfinite(last_loss) or last_loss > diverge_loss:
                stable = False
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        rows.append([lr, stable, last_loss])
        if stable:
            max_stable = lr

    table = wandb.Table(columns=["lr", "stable", "last_loss"], data=rows)
    return max_stable, table


def _run_experiment(
    *,
    project: str,
    entity: str | None,
    data_root: str,
    with_bn: bool,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    num_workers: int,
    fixed_input_index: int,
    do_lr_sweep: bool,
    lr_sweep_values: List[float],
    lr_sweep_steps: int,
    diverge_loss: float,
) -> RunResult:
    device = _device()
    _seed_all(seed)

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

    model = VGG11Classifier(num_classes=37)
    if not with_bn:
        _replace_batchnorm_with_identity(model)
    model = model.to(device)

    # Fixed input for activation histogram (same across runs)
    x_fixed, _, _, _ = full[fixed_input_index]
    x1 = x_fixed.unsqueeze(0).to(device)

    run_name = "q1_with_bn" if with_bn else "q1_no_bn"
    cfg = {
        "question": "2.1",
        "task": "cls",
        "with_bn": with_bn,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_fraction": val_fraction,
        "fixed_input_index": fixed_input_index,
        "device": str(device),
    }
    run = wandb.init(project=project, entity=entity, name=run_name, config=cfg)

    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        e0 = time.perf_counter()
        train_loss = _train_one_epoch_cls(model, train_loader, opt, device, ce)
        val_f1 = _macro_f1_cls(model, val_loader, device)
        epoch_s = time.perf_counter() - e0
        best_f1 = max(best_f1, val_f1)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/macro_f1": val_f1,
                "time/epoch_s": epoch_s,
                "time/elapsed_s": time.perf_counter() - t0,
            }
        )

    # Activation histogram at end of training
    act = _activation_histogram_3rd_conv(model, x1)
    wandb.log(
        {
            "activations/conv3_hist": wandb.Histogram(act),
            "activations/conv3_mean": float(act.mean()),
            "activations/conv3_std": float(act.std()),
        }
    )

    max_stable_lr = None
    if do_lr_sweep:
        # Use fresh models so sweep doesn't depend on final trained weights
        def make_model() -> VGG11Classifier:
            m = VGG11Classifier(num_classes=37)
            if not with_bn:
                _replace_batchnorm_with_identity(m)
            return m

        max_stable_lr, table = _lr_sweep_max_stable(
            make_model=make_model,
            device=device,
            train_loader=train_loader,
            lrs=lr_sweep_values,
            steps_per_lr=lr_sweep_steps,
            loss_fn=ce,
            diverge_loss=diverge_loss,
        )
        wandb.log(
            {
                "lr_sweep/table": table,
                "lr_sweep/max_stable_lr": max_stable_lr if max_stable_lr is not None else -1,
            }
        )

    wandb.summary["best_val_macro_f1"] = best_f1
    if max_stable_lr is not None:
        wandb.summary["max_stable_lr"] = max_stable_lr
    run.finish()

    return RunResult(run_name=run_name, max_stable_lr=max_stable_lr)


def _parse_lrs(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--fixed_input_index",
        type=int,
        default=0,
        help="Index into OxfordIIITPetDataset(trainval) used for activation histogram input.",
    )
    p.add_argument("--lr_sweep", action="store_true")
    p.add_argument(
        "--lr_sweep_values",
        type=str,
        default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
    )
    p.add_argument("--lr_sweep_steps", type=int, default=40)
    p.add_argument(
        "--diverge_loss",
        type=float,
        default=25.0,
        help="If loss exceeds this during LR sweep, mark as diverged/unstable.",
    )
    args = p.parse_args()

    data_root = str(Path(args.data_root))
    entity = args.entity.strip() or None
    lr_sweep_values = _parse_lrs(args.lr_sweep_values)

    # Run with BN and without BN as two separate W&B runs in the same project.
    _run_experiment(
        project=args.project,
        entity=entity,
        data_root=data_root,
        with_bn=True,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        fixed_input_index=args.fixed_input_index,
        do_lr_sweep=args.lr_sweep,
        lr_sweep_values=lr_sweep_values,
        lr_sweep_steps=args.lr_sweep_steps,
        diverge_loss=args.diverge_loss,
    )
    _run_experiment(
        project=args.project,
        entity=entity,
        data_root=data_root,
        with_bn=False,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        fixed_input_index=args.fixed_input_index,
        do_lr_sweep=args.lr_sweep,
        lr_sweep_values=lr_sweep_values,
        lr_sweep_steps=args.lr_sweep_steps,
        diverge_loss=args.diverge_loss,
    )


if __name__ == "__main__":
    main()

