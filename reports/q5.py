"""
W&B Report Q5 (Assignment PDF §2.5): Object detection confidence + IoU table.

Builds a W&B table with at least N validation images:
- Green box: ground truth
- Red box: prediction
- Confidence score (uncertainty-based proxy from MC dropout)
- IoU score
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer


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


def _cxcywh_to_xyxy(box: torch.Tensor, img_size: int) -> Tuple[float, float, float, float]:
    xc, yc, w, h = [float(v) for v in box]
    w = max(w, 1e-3)
    h = max(h, 1e-3)
    x1 = max(0.0, min(float(img_size - 1), xc - 0.5 * w))
    y1 = max(0.0, min(float(img_size - 1), yc - 0.5 * h))
    x2 = max(0.0, min(float(img_size - 1), xc + 0.5 * w))
    y2 = max(0.0, min(float(img_size - 1), yc + 0.5 * h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou_cxcywh(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    def to_xyxy(b: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        xc, yc, w, h = b.unbind(dim=-1)
        w = w.abs().clamp_min(eps)
        h = h.abs().clamp_min(eps)
        x1 = xc - 0.5 * w
        y1 = yc - 0.5 * h
        x2 = xc + 0.5 * w
        y2 = yc + 0.5 * h
        return x1, y1, x2, y2

    px1, py1, px2, py2 = to_xyxy(pred)
    tx1, ty1, tx2, ty2 = to_xyxy(target)
    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h
    ap = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    at = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
    union = ap + at - inter + eps
    return float((inter / union).item())


def _enable_dropout_for_mc(model: nn.Module) -> None:
    model.train()
    # Keep BatchNorm deterministic for stable MC estimates on tiny batches.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


@torch.no_grad()
def _predict_with_confidence(
    model: VGG11Localizer,
    x: torch.Tensor,
    mc_samples: int,
    conf_temp: float,
) -> Tuple[torch.Tensor, float]:
    # Mean prediction + uncertainty-based confidence proxy.
    preds: List[torch.Tensor] = []
    _enable_dropout_for_mc(model)
    for _ in range(mc_samples):
        preds.append(model(x))
    stack = torch.stack(preds, dim=0)  # [T, B, 4]
    mean_pred = stack.mean(dim=0)
    std_pred = stack.std(dim=0, unbiased=False)  # [B, 4]
    sigma = float(std_pred.mean().item())
    confidence = float(np.exp(-sigma / max(conf_temp, 1e-6)))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    model.eval()
    return mean_pred, confidence


def _draw_overlay(
    img_rgb: np.ndarray,
    gt_xyxy: Tuple[float, float, float, float],
    pr_xyxy: Tuple[float, float, float, float],
    confidence: float,
    iou: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_rgb)
    ax.axis("off")

    gx1, gy1, gx2, gy2 = gt_xyxy
    px1, py1, px2, py2 = pr_xyxy
    gt_w, gt_h = gx2 - gx1, gy2 - gy1
    pr_w, pr_h = px2 - px1, py2 - py1
    ax.add_patch(
        patches.Rectangle(
            (gx1, gy1),
            gt_w,
            gt_h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="GT",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (px1, py1),
            pr_w,
            pr_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Pred",
        )
    )
    ax.set_title(f"conf={confidence:.3f} | IoU={iou:.3f}")
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--localizer_ckpt", type=str, default="localizer.pth")
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_samples", type=int, default=10, help="Rows to log (>=10 recommended).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mc_samples", type=int, default=8, help="MC dropout passes for confidence proxy.")
    p.add_argument("--conf_temp", type=float, default=12.0, help="Temperature for confidence scaling.")
    args = p.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

    _seed_all(args.seed)
    device = _device()
    data_root = _resolve_data_root(args.data_root)

    full = OxfordIIITPetDataset(root=data_root, split="trainval")
    n_val = max(1, int(len(full) * args.val_fraction))
    n_train = len(full) - n_val
    _, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    # Deterministic subset of val samples for table rows.
    idxs = list(range(min(args.num_samples, len(val_set))))
    sub = Subset(val_set, idxs)
    loader = DataLoader(
        sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = VGG11Localizer().to(device)
    state = torch.load(args.localizer_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name="q5_detection_table",
        config={
            "question": "2.5",
            "localizer_ckpt": args.localizer_ckpt,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "num_samples": args.num_samples,
            "mc_samples": args.mc_samples,
            "conf_temp": args.conf_temp,
            "device": str(device),
            "note": "Confidence is an uncertainty-based proxy from MC dropout.",
        },
    )

    table = wandb.Table(
        columns=["idx", "image_overlay", "confidence", "iou", "failure_case"]
    )

    ious: List[float] = []
    confs: List[float] = []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        x, _, bbox_gt, _ = batch
        x = x.to(device)
        bbox_gt = bbox_gt.to(device)

        bbox_pred, conf = _predict_with_confidence(
            model=model,
            x=x,
            mc_samples=args.mc_samples,
            conf_temp=args.conf_temp,
        )
        iou = _iou_cxcywh(bbox_pred[0], bbox_gt[0])

        img_rgb = _denormalize(x[0])
        gt_xyxy = _cxcywh_to_xyxy(bbox_gt[0].detach().cpu(), img_size=img_rgb.shape[0])
        pr_xyxy = _cxcywh_to_xyxy(bbox_pred[0].detach().cpu(), img_size=img_rgb.shape[0])
        fig = _draw_overlay(img_rgb, gt_xyxy, pr_xyxy, conf, iou)
        table.add_data(
            i,
            wandb.Image(fig),
            conf,
            iou,
            bool((conf > 0.7 and iou < 0.3) or iou < 0.1),
        )
        plt.close(fig)

        ious.append(iou)
        confs.append(conf)

    wandb.log(
        {
            "detection/table": table,
            "detection/mean_iou": float(np.mean(ious)) if ious else 0.0,
            "detection/mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "time/elapsed_s": time.perf_counter() - t0,
        }
    )
    run.finish()
    print("Q5 table logging complete.", flush=True)


if __name__ == "__main__":
    main()

