"""Dry-run Gradescope-style checkpoint loading (run from project root).

1. Same working directory as ``python -m`` / autograder (files like ``classifier.pth`` here).
2. Optional: download from Drive first (same as README Step 3).

Examples::

    # Weights already in cwd (names match multitask defaults)
    python verify_checkpoint_load.py

    # Download then verify (replace IDs with yours)
    python verify_checkpoint_load.py --id-cls YOUR_CLS_ID --id-loc YOUR_LOC_ID --id-seg YOUR_SEG_ID
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Verify MultiTaskPerceptionModel loads checkpoints")
    p.add_argument("--id-cls", type=str, default="", help="Google Drive file id for classifier.pth")
    p.add_argument("--id-loc", type=str, default="", help="Google Drive file id for localizer.pth")
    p.add_argument("--id-seg", type=str, default="", help="Google Drive file id for unet.pth")
    args = p.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(root)
    print(f"[verify] cwd={root}", flush=True)

    if args.id_cls or args.id_loc or args.id_seg:
        try:
            import gdown
        except ImportError:
            print("[verify] ERROR: gdown not installed. pip install gdown", file=sys.stderr, flush=True)
            return 1
        mapping = [
            (args.id_cls, "classifier.pth"),
            (args.id_loc, "localizer.pth"),
            (args.id_seg, "unet.pth"),
        ]
        for fid, out in mapping:
            if fid:
                print(f"[verify] gdown.download id={fid[:8]}... -> {out}", flush=True)
                gdown.download(id=fid, output=out, quiet=False)

    for name in ("classifier.pth", "localizer.pth", "unet.pth"):
        exists = os.path.isfile(os.path.join(root, name))
        print(f"[verify] {name}: {'exists' if exists else 'MISSING'}", flush=True)

    import torch

    from models.multitask import MultiTaskPerceptionModel

    device = torch.device("cpu")
    model = MultiTaskPerceptionModel().to(device)
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
    assert "classification" in out and "localization" in out and "segmentation" in out
    print(
        "[verify] forward OK — shapes:",
        {k: tuple(v.shape) for k, v in out.items()},
        flush=True,
    )
    print("[verify] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
