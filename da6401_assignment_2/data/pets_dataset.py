"""Dataset loader for Oxford-IIIT Pet.

Expected layout under ``root`` (official VGG tarball):

- ``images/*.jpg`` (or nested ``images/images/*.jpg`` from some archives)
- ``annotations/list.txt``
- ``annotations/trimaps/*.png``
- ``annotations/xmls/*.xml`` (VOC-style; first ``<bndbox>`` used as head box)

Only standard-library XML parsing + Pillow + torch/numpy — no ``torchvision`` dependency.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_list_file(list_path: Path) -> List[Tuple[str, int, int, int]]:
    """Return rows: (stem, class_id, species_id, trainval_flag)."""
    rows: List[Tuple[str, int, int, int]] = []
    for line in list_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        stem, cls_id, species, tv = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        rows.append((stem, cls_id, species, tv))
    return rows


def _resolve_ann_root(root: Path) -> Path:
    """Resolve annotations root across deeply nested archive layouts."""
    base = root / "annotations"
    cur = base
    for _ in range(6):
        if (cur / "list.txt").is_file():
            return cur
        nxt = cur / "annotations"
        if not nxt.is_dir():
            break
        cur = nxt

    if base.is_dir():
        for list_path in base.rglob("list.txt"):
            return list_path.parent

    raise FileNotFoundError(
        f"Could not find list.txt under {base} (including nested annotations folders)."
    )


def _resolve_img_root(root: Path) -> Path:
    """Resolve image root across deeply nested archive layouts."""
    base = root / "images"
    cur = base
    for _ in range(6):
        if cur.is_dir() and any(cur.glob("*.jpg")):
            return cur
        nxt = cur / "images"
        if not nxt.is_dir():
            break
        cur = nxt

    if base.is_dir():
        for jpg in base.rglob("*.jpg"):
            return jpg.parent

    raise FileNotFoundError(
        f"Could not find .jpg files under {base} (including nested images folders)."
    )


def _parse_voc_bbox(xml_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(x_center, y_center, width, height)`` in **original image pixel** coords."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None
    obj = root.find(".//object")
    if obj is None:
        obj = root.find("object")
    if obj is None:
        return None
    bnd = obj.find("bndbox")
    if bnd is None:
        return None
    xmin = float(bnd.findtext("xmin"))
    ymin = float(bnd.findtext("ymin"))
    xmax = float(bnd.findtext("xmax"))
    ymax = float(bnd.findtext("ymax"))
    w = max(xmax - xmin, 1.0)
    h = max(ymax - ymin, 1.0)
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    return xc, yc, w, h


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset: breed label, head bbox, trimap mask."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        img_size: int = 224,
        return_paths: bool = False,
    ):
        """
        Args:
            root: Dataset root directory.
            split: ``"trainval"`` (list flag 1), ``"test"`` (flag 0), or ``"all"``.
            img_size: Square resize (VGG11 uses 224).
            return_paths: If True, ``__getitem__`` also returns image path strings.
        """
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.return_paths = return_paths
        self._ann_root = _resolve_ann_root(self.root)
        self._img_root = _resolve_img_root(self.root)

        list_path = self._ann_root / "list.txt"
        rows = _read_list_file(list_path)
        if split == "trainval":
            rows = [r for r in rows if r[3] == 1]
        elif split == "test":
            rows = [r for r in rows if r[3] == 0]
        elif split == "all":
            pass
        else:
            raise ValueError("split must be 'trainval', 'test', or 'all'")

        self._samples: List[Tuple[str, int]] = [(r[0], r[1]) for r in rows]
        if not self._samples:
            raise RuntimeError(f"No samples for split={split!r} in {list_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def _load_bbox(self, stem: str, w0: int, h0: int) -> torch.Tensor:
        xml_path = self._ann_root / "xmls" / f"{stem}.xml"
        box = _parse_voc_bbox(xml_path)
        if box is None:
            # Fallback: full image box
            xc, yc = 0.5 * w0, 0.5 * h0
            w, h = float(w0), float(h0)
        else:
            xc, yc, w, h = box
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        xc_n = xc * sx
        yc_n = yc * sy
        w_n = w * sx
        h_n = h * sy
        return torch.tensor([xc_n, yc_n, w_n, h_n], dtype=torch.float32)

    def _load_trimap(self, stem: str) -> torch.Tensor:
        tri_path = self._ann_root / "trimaps" / f"{stem}.png"
        mask = Image.open(tri_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        # Official trimap codes: 1 foreground, 2 boundary, 3 background -> classes 0,1,2
        mask = (mask - 1).clamp(0, 2)
        return mask

    def __getitem__(self, idx: int):
        stem, cls_id = self._samples[idx]
        img_path = self._img_root / f"{stem}.jpg"
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        arr = np.asarray(img).astype(np.float32) / 255.0
        for c in range(3):
            arr[..., c] = (arr[..., c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        bbox = self._load_bbox(stem, w0, h0)
        trimap = self._load_trimap(stem)

        # Official ``list.txt`` uses breed ids **1..37**; some mirrors use **0..36**.
        if cls_id >= 1:
            label_idx = cls_id - 1
        else:
            label_idx = cls_id
        label_idx = max(0, min(36, int(label_idx)))
        label = torch.tensor(label_idx, dtype=torch.long)
        out: Tuple = (image, label, bbox, trimap)
        if self.return_paths:
            out = out + (str(img_path),)
        return out
