"""
W&B Report Q7 (Assignment PDF §2.7): Final pipeline showcase on novel internet images.

Given 3 external image URLs, this script:
- downloads each image,
- runs MultiTaskPerceptionModel (classification + localization + segmentation),
- creates output visualizations (pred bbox + seg overlay),
- logs a W&B table for report discussion.
"""

from __future__ import annotations

import argparse
import io
from typing import List, Tuple
from urllib.request import urlopen

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_rgb(url: str, timeout_s: int = 20) -> Image.Image:
    with urlopen(url, timeout=timeout_s) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _preprocess(img: Image.Image, img_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(OxfordIIITPetDataset.IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(OxfordIIITPetDataset.IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr_norm = (arr - mean) / std
    x = torch.from_numpy(arr_norm).permute(2, 0, 1).contiguous()
    return x, arr


def _cxcywh_to_xyxy(box: torch.Tensor, img_size: int = 224) -> Tuple[float, float, float, float]:
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


def _seg_palette(mask_hw: np.ndarray) -> np.ndarray:
    colors = np.array(
        [
            [255, 80, 80],    # foreground
            [255, 220, 80],   # boundary
            [70, 130, 255],   # background
        ],
        dtype=np.uint8,
    )
    m = np.clip(mask_hw.astype(np.int64), 0, 2)
    return colors[m]


def _overlay_visual(
    img_rgb: np.ndarray,
    bbox_xyxy: Tuple[float, float, float, float],
    seg_mask_hw: np.ndarray,
    class_idx: int,
    class_conf: float,
    alpha: float = 0.35,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_rgb)

    seg_rgb = _seg_palette(seg_mask_hw)
    ax.imshow(seg_rgb, alpha=alpha)

    x1, y1, x2, y2 = bbox_xyxy
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
    )
    ax.set_title(f"class_idx={class_idx} conf={class_conf:.3f}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _parse_urls(s: str) -> List[str]:
    urls = [u.strip() for u in s.split(",") if u.strip()]
    return urls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image_urls",
        type=str,
        required=True,
        help="Comma-separated 3 internet image URLs (not dataset images).",
    )
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--classifier_ckpt", type=str, default="classifier.pth")
    p.add_argument("--localizer_ckpt", type=str, default="localizer.pth")
    p.add_argument("--unet_ckpt", type=str, default="unet.pth")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _seed_all(args.seed)
    device = _device()
    urls = _parse_urls(args.image_urls)
    if len(urls) != 3:
        raise ValueError("Please provide exactly 3 image URLs in --image_urls.")

    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_ckpt,
        localizer_path=args.localizer_ckpt,
        unet_path=args.unet_ckpt,
    ).to(device)
    model.eval()

    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name="q7_pipeline_showcase",
        config={
            "question": "2.7",
            "seed": args.seed,
            "image_urls": urls,
            "classifier_ckpt": args.classifier_ckpt,
            "localizer_ckpt": args.localizer_ckpt,
            "unet_ckpt": args.unet_ckpt,
            "device": str(device),
        },
    )

    table = wandb.Table(
        columns=[
            "idx",
            "url",
            "output_overlay",
            "pred_class_idx",
            "pred_class_conf",
            "pred_bbox_cxcywh",
        ]
    )

    for i, url in enumerate(urls):
        img = _download_rgb(url)
        x, img_rgb = _preprocess(img, img_size=224)
        xb = x.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(xb)
            logits = out["classification"]
            probs = F.softmax(logits, dim=1)
            class_idx = int(probs.argmax(dim=1).item())
            class_conf = float(probs.max(dim=1).values.item())

            bbox = out["localization"][0].detach().cpu()
            seg = out["segmentation"].argmax(dim=1)[0].detach().cpu().numpy()

        bbox_xyxy = _cxcywh_to_xyxy(bbox, img_size=224)
        fig = _overlay_visual(
            img_rgb=img_rgb,
            bbox_xyxy=bbox_xyxy,
            seg_mask_hw=seg,
            class_idx=class_idx,
            class_conf=class_conf,
        )
        table.add_data(
            i,
            url,
            wandb.Image(fig),
            class_idx,
            class_conf,
            [float(v) for v in bbox.tolist()],
        )
        plt.close(fig)

    wandb.log({"q7/showcase_table": table})
    run.finish()
    print("Q7 showcase logging complete.", flush=True)


if __name__ == "__main__":
    main()

