"""
W&B Report Q6 (Assignment PDF §2.6): Dice vs Pixel Accuracy for segmentation.

Logs:
- Validation pixel accuracy
- Validation macro dice
- Validation loss
- 5 sample visual triplets: original image, GT trimap, predicted trimap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
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


def _denormalize(img_chw: torch.Tensor) -> np.ndarray:
    mean = np.array(OxfordIIITPetDataset.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(OxfordIIITPetDataset.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    arr = img_chw.detach().cpu().numpy()
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    return np.transpose(arr, (1, 2, 0))


def _palette(mask_hw: np.ndarray) -> np.ndarray:
    # Class ids: 0 foreground, 1 boundary, 2 background
    colors = np.array(
        [
            [255, 80, 80],    # foreground - red-ish
            [255, 220, 80],   # boundary - yellow-ish
            [70, 130, 255],   # background - blue-ish
        ],
        dtype=np.uint8,
    )
    m = np.clip(mask_hw.astype(np.int64), 0, 2)
    return colors[m]


def _eval_seg(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
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


def _collect_samples(
    model: nn.Module,
    dataset: Subset,
    device: torch.device,
    num_samples: int,
) -> wandb.Table:
    n = min(num_samples, len(dataset))
    table = wandb.Table(columns=["idx", "original", "gt_trimap", "pred_trimap"])
    model.eval()
    with torch.no_grad():
        for i in range(n):
            x, _, _, mask = dataset[i]
            xb = x.unsqueeze(0).to(device)
            pred = model(xb).argmax(dim=1).squeeze(0).cpu().numpy()
            gt = mask.cpu().numpy()
            table.add_data(
                i,
                wandb.Image(_denormalize(x), caption=f"sample_{i}"),
                wandb.Image(_palette(gt), caption="gt_trimap"),
                wandb.Image(_palette(pred), caption="pred_trimap"),
            )
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--unet_ckpt", type=str, default="unet.pth")
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=5, help="Number of visualization samples.")
    args = p.parse_args()

    _seed_all(args.seed)
    device = _device()
    data_root = _resolve_data_root(args.data_root)

    full = OxfordIIITPetDataset(root=data_root, split="trainval")
    n_val = max(1, int(len(full) * args.val_fraction))
    n_train = len(full) - n_val
    _, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VGG11UNet(num_classes=3).to(device)
    state = torch.load(args.unet_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    ce = nn.CrossEntropyLoss()
    val_loss, pixel_acc, macro_dice = _eval_seg(model, val_loader, ce, device)
    sample_table = _collect_samples(model, val_set, device, num_samples=args.num_samples)

    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name="q6_segmentation_eval",
        config={
            "question": "2.6",
            "unet_ckpt": args.unet_ckpt,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "device": str(device),
        },
    )

    wandb.log(
        {
            "val/loss": val_loss,
            "val/pixel_acc": pixel_acc,
            "val/macro_dice": macro_dice,
            "samples/segmentation_triplets": sample_table,
        }
    )
    run.finish()
    print(
        f"Q6 complete. val_loss={val_loss:.4f} val_pixel_acc={pixel_acc:.4f} val_macro_dice={macro_dice:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

