"""Download classifier/localizer/unet checkpoints from Google Drive links or IDs.

Usage examples:
    python download_pth.py --classifier "<drive_link_or_id>" --localizer "<drive_link_or_id>" --unet "<drive_link_or_id>"
    python download_pth.py --classifier_id "<id>" --localizer_id "<id>" --unet_id "<id>"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gdown

DEFAULT_CLASSIFIER = "https://drive.google.com/file/d/1DhWJjRW15qBmdcNxAI2-xsMkAP84jQV8/view?usp=drive_link"
DEFAULT_LOCALIZER = "https://drive.google.com/file/d/14wFEgDXByLkNBXJBrgVZtgffa7urRWFB/view?usp=drive_link"
DEFAULT_UNET = "https://drive.google.com/file/d/1ZqD5HSLu3rPvoGlVhqQjxNm3rvMG72QB/view?usp=drive_link"


def _extract_drive_id(value: str) -> str:
    """Accept either raw file ID or common Drive URL forms and return file ID."""
    value = value.strip()
    if not value:
        raise ValueError("Empty Google Drive value provided.")

    # If it looks like a raw ID, keep it.
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", value):
        return value

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, value)
        if m:
            return m.group(1)

    raise ValueError(f"Could not parse Google Drive file id from: {value}")


def _download_one(id_or_link: str, output_path: Path) -> None:
    file_id = _extract_drive_id(id_or_link)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {output_path.name} ...", flush=True)
    gdown.download(id=file_id, output=str(output_path), quiet=False)
    if not output_path.is_file():
        raise RuntimeError(f"Download failed for {output_path}")
    print(f"Saved: {output_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--classifier",
        type=str,
        default=DEFAULT_CLASSIFIER,
        help="Drive link or ID for classifier.pth",
    )
    p.add_argument(
        "--localizer",
        type=str,
        default=DEFAULT_LOCALIZER,
        help="Drive link or ID for localizer.pth",
    )
    p.add_argument(
        "--unet",
        type=str,
        default=DEFAULT_UNET,
        help="Drive link or ID for unet.pth",
    )
    p.add_argument("--classifier_id", type=str, default="", help="Raw Drive ID for classifier.pth")
    p.add_argument("--localizer_id", type=str, default="", help="Raw Drive ID for localizer.pth")
    p.add_argument("--unet_id", type=str, default="", help="Raw Drive ID for unet.pth")
    p.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory (default current directory).",
    )
    args = p.parse_args()

    classifier_src = args.classifier_id.strip() or args.classifier.strip()
    localizer_src = args.localizer_id.strip() or args.localizer.strip()
    unet_src = args.unet_id.strip() or args.unet.strip()

    if not classifier_src or not localizer_src or not unet_src:
        raise ValueError(
            "Provide all three checkpoints via --classifier/--localizer/--unet "
            "or --classifier_id/--localizer_id/--unet_id."
        )

    out_dir = Path(args.out_dir).resolve()
    _download_one(classifier_src, out_dir / "classifier.pth")
    _download_one(localizer_src, out_dir / "localizer.pth")
    _download_one(unet_src, out_dir / "unet.pth")

    print("All checkpoints downloaded successfully.", flush=True)


if __name__ == "__main__":
    main()

