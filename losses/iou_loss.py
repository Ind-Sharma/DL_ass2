"""Custom IoU loss 
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Boxes are axis-aligned in ``(x_center, y_center, width, height)`` format.
    Loss is ``1 - IoU``, clipped to ``[0, 1]`` per pair (IoU in ``[0, 1]``).
    """

    _VALID_REDUCTIONS = frozenset({"none", "mean", "sum"})

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {sorted(self._VALID_REDUCTIONS)}, got {reduction!r}"
            )

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor, eps: float) -> tuple[torch.Tensor, ...]:
        """boxes: [B, 4] (xc, yc, w, h) -> x1,y1,x2,y2 with positive sizes."""
        xc, yc, w, h = boxes.unbind(dim=-1)
        w = w.abs().clamp_min(eps)
        h = h.abs().clamp_min(eps)
        x1 = xc - 0.5 * w
        y1 = yc - 0.5 * h
        x2 = xc + 0.5 * w
        y2 = yc + 0.5 * h
        return x1, y1, x2, y2

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        if pred_boxes.shape != target_boxes.shape or pred_boxes.dim() != 2 or pred_boxes.size(-1) != 4:
            raise ValueError(
                f"Expected pred and target of shape [B, 4], got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}"
            )

        px1, py1, px2, py2 = self._to_xyxy(pred_boxes, self.eps)
        tx1, ty1, tx2, ty2 = self._to_xyxy(target_boxes, self.eps)

        inter_x1 = torch.maximum(px1, tx1)
        inter_y1 = torch.maximum(py1, ty1)
        inter_x2 = torch.minimum(px2, tx2)
        inter_y2 = torch.minimum(py2, ty2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter = inter_w * inter_h

        area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
        area_t = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
        union = area_p + area_t - inter + self.eps
        iou = inter / union
        iou = iou.clamp(0.0, 1.0)
        loss = 1.0 - iou

        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
