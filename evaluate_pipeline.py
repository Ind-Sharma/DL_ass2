"""Local evaluation mirroring Gradescope-style pipeline metrics (section 4).

Run from ``da6401_assignment_2``::

    python evaluate_pipeline.py --data_root PATH_TO_OXFORD_PET

Uses the same validation split idea as ``train.py`` (seed 42, 10%% val by default).
Loads ``MultiTaskPerceptionModel`` with ``classifier.pth``, ``localizer.pth``, ``unet.pth``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _box_iou_cxcywh(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for axis-aligned boxes [B,4] as (cx, cy, w, h)."""
    def to_xyxy(b: torch.Tensor) -> tuple[torch.Tensor, ...]:
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
    return inter / union


def _accumulate_dice(
    inter: np.ndarray, pred_sum: np.ndarray, tgt_sum: np.ndarray,
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6,
) -> None:
    """Update running per-class intersection / pred pixels / target pixels (CPU numpy)."""
    pred = logits.argmax(dim=1)
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter[c] += float((p * t).sum().item())
        pred_sum[c] += float(p.sum().item())
        tgt_sum[c] += float(t.sum().item())


def _macro_dice_from_running(
    inter: np.ndarray, pred_sum: np.ndarray, tgt_sum: np.ndarray, eps: float = 1e-6
) -> float:
    dice_per_class = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    return float(np.mean(dice_per_class))


def main() -> None:
    p = argparse.ArgumentParser(description="Local pipeline metrics (like Gradescope sec. 4)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--classifier_path", type=str, default="classifier.pth")
    p.add_argument("--localizer_path", type=str, default="localizer.pth")
    p.add_argument("--unet_path", type=str, default="unet.pth")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for path, name in [
        (args.classifier_path, "classifier"),
        (args.localizer_path, "localizer"),
        (args.unet_path, "unet"),
    ]:
        if not os.path.isfile(path):
            print(f"WARNING: missing {name} checkpoint: {path} (metrics will reflect random init for unloaded parts)")

    device = _device()
    torch.manual_seed(args.seed)

    full = OxfordIIITPetDataset(root=args.data_root, split="trainval")
    n_val = max(1, int(len(full) * args.val_fraction))
    n_train = len(full) - n_val
    _, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(device)
    model.eval()

    all_y: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_ious: list[float] = []
    dice_inter = np.zeros(3, dtype=np.float64)
    dice_pred = np.zeros(3, dtype=np.float64)
    dice_tgt = np.zeros(3, dtype=np.float64)

    print(f"device={device}  val_samples={len(val_set)}  batches={len(loader)}", flush=True)

    with torch.no_grad():
        for batch in loader:
            x, y, bbox, mask = batch
            x = x.to(device)
            y = y.cpu().numpy()
            bbox = bbox.to(device)
            mask = mask.to(device)

            out = model(x)
            logits = out["classification"]
            pred_cls = logits.argmax(dim=1).cpu().numpy()
            all_y.append(y)
            all_pred_cls.append(pred_cls)

            pb = out["localization"]
            ious = _box_iou_cxcywh(pb, bbox).cpu().numpy()
            all_ious.extend(ious.tolist())

            seg_logits = out["segmentation"]
            _accumulate_dice(dice_inter, dice_pred, dice_tgt, seg_logits, mask, num_classes=3)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred_cls)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    ious_arr = np.array(all_ious, dtype=np.float64)
    acc_50 = float(np.mean(ious_arr >= 0.5))
    acc_75 = float(np.mean(ious_arr >= 0.75))
    macro_dice = _macro_dice_from_running(dice_inter, dice_pred, dice_tgt)

    print()
    print("=== Local pipeline metrics (same spirit as Gradescope sec. 4) ===")
    print(f"Classification macro-F1:  {macro_f1:.4f}   (thresholds: >0.3 / >0.5 / >0.8)")
    print(f"Localization Acc@IoU≥0.5: {100*acc_50:.1f}%   (threshold: >60%)")
    print(f"Localization Acc@IoU≥0.75: {100*acc_75:.1f}%   (threshold: >40%)")
    print(f"Segmentation macro-Dice:  {macro_dice:.4f}   (thresholds: >0.3 / >0.5 / >0.8)")
    print()
    print("Notes:")
    print("- Uses YOUR val split + YOUR checkpoints; Gradescope uses a private test set.")
    print("- If checkpoints are missing, scores will be poor until you train & point paths.")
    print()


if __name__ == "__main__":
    main()
