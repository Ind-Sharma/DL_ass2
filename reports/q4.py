"""
W&B Report Q4 (Assignment PDF §2.4): Feature map visualization.

Loads a trained classifier checkpoint, runs one dog image through the model,
captures feature maps from:
1) first convolutional layer
2) last convolutional layer before final pooling
and logs visualization grids to W&B.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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


def _pick_one_dog_sample(ds: OxfordIIITPetDataset) -> Tuple[torch.Tensor, str]:
    # Oxford Pets naming convention: dog breed stems usually start lowercase, cats uppercase.
    for i in range(len(ds)):
        sample = ds[i]
        if len(sample) == 5:
            x, _, _, _, img_path = sample
        else:
            x, _, _, _ = sample
            img_path = ""
        stem = Path(img_path).stem if img_path else ""
        if stem and stem[0].islower():
            return x, img_path

    # Fallback: first sample if naming heuristic fails.
    sample = ds[0]
    if len(sample) == 5:
        x, _, _, _, img_path = sample
    else:
        x, _, _, _ = sample
        img_path = ""
    return x, img_path


def _make_feature_grid(feature_map: torch.Tensor, title: str, max_channels: int = 32) -> plt.Figure:
    fmap = feature_map.detach().cpu().squeeze(0)  # [C,H,W]
    c = min(int(fmap.size(0)), max_channels)
    cols = 8
    rows = int(np.ceil(c / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle(title)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= c:
            continue
        ch = fmap[idx].numpy()
        ch_min, ch_max = float(ch.min()), float(ch.max())
        if ch_max - ch_min > 1e-8:
            ch = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch = np.zeros_like(ch)
        ax.imshow(ch, cmap="viridis")
        ax.set_title(f"ch{idx}", fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, default="classifier.pth")
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_channels", type=int, default=32, help="Max feature channels to visualize per layer.")
    args = p.parse_args()

    _seed_all(args.seed)
    device = _device()
    data_root = _resolve_data_root(args.data_root)

    ds = OxfordIIITPetDataset(root=data_root, split="trainval", return_paths=True)
    x, img_path = _pick_one_dog_sample(ds)
    x = x.unsqueeze(0).to(device)

    model = VGG11Classifier(num_classes=37).to(device)
    state = torch.load(args.classifier_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    first_conv_out: List[torch.Tensor] = []
    last_conv_out: List[torch.Tensor] = []

    def hook_first(_m: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        first_conv_out.append(out.detach())

    def hook_last(_m: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        last_conv_out.append(out.detach())

    h1 = model.encoder.block1[0].register_forward_hook(hook_first)
    h2 = model.encoder.block5[3].register_forward_hook(hook_last)
    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())
    h1.remove()
    h2.remove()

    if not first_conv_out or not last_conv_out:
        raise RuntimeError("Feature hooks did not capture activations.")

    img_rgb = _denormalize(x[0])
    fig_first = _make_feature_grid(
        first_conv_out[0], "First Conv Layer Feature Maps", max_channels=args.max_channels
    )
    fig_last = _make_feature_grid(
        last_conv_out[0], "Last Conv Layer (Before Pool) Feature Maps", max_channels=args.max_channels
    )

    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name="q4_feature_maps",
        config={
            "question": "2.4",
            "classifier_ckpt": args.classifier_ckpt,
            "sample_image_path": img_path,
            "predicted_class_idx": pred_class,
            "max_channels": args.max_channels,
            "device": str(device),
        },
    )

    wandb.log(
        {
            "input/image": wandb.Image(img_rgb, caption=f"Sample image: {img_path}"),
            "feature_maps/first_conv": wandb.Image(fig_first),
            "feature_maps/last_conv_before_pool": wandb.Image(fig_last),
            "prediction/class_idx": pred_class,
        }
    )

    plt.close(fig_first)
    plt.close(fig_last)
    run.finish()
    print("Q4 feature map logging complete.", flush=True)


if __name__ == "__main__":
    main()

