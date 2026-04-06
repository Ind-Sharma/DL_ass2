#!/usr/bin/env python3
"""Build list.txt from Oxford-IIIT Pet trainval.txt and test.txt (stdlib only)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_split(path: Path, split_flag: int) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: need at least 4 fields, got {len(parts)}: {line!r}")
        stem, cls_id, species = parts[0], parts[1], parts[2]
        if stem in seen:
            raise ValueError(f"{path}: duplicate image name {stem!r}")
        seen.add(stem)
        rows.append((stem, cls_id, species, split_flag))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge trainval.txt + test.txt into list.txt")
    ap.add_argument("trainval_txt", type=Path, help="Path to trainval.txt")
    ap.add_argument("test_txt", type=Path, help="Path to test.txt")
    ap.add_argument("-o", "--output", type=Path, default=Path("list.txt"), help="Output list.txt")
    args = ap.parse_args()

    train_rows = _read_split(args.trainval_txt, 1)
    test_rows = _read_split(args.test_txt, 0)

    train_stems = {r[0] for r in train_rows}
    test_stems = {r[0] for r in test_rows}
    overlap = train_stems & test_stems
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise SystemExit(f"Same image in trainval and test (first examples): {sample}")

    # Order: all trainval lines, then all test lines (matches common list.txt layouts)
    all_rows = train_rows + test_rows
    out_lines = [f"{a} {b} {c} {d}" for a, b, c, d in all_rows]
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    n_train = len(train_rows)
    n_test = len(test_rows)
    n_total = len(all_rows)
    print(f"Number of train samples: {n_train}")
    print(f"Number of test samples: {n_test}")
    print(f"Total samples written to list.txt: {n_total}")
    if n_total != 7349:
        print(f"Note: full Oxford-IIIT Pet usually has 7349 images; you have {n_total}.")


if __name__ == "__main__":
    main()
