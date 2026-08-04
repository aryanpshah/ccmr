#!/usr/bin/env python3
"""
Generate fixed train/test splits and nested label budgets for HVSMR-2.0.

Reproduces: Paper split (train pool n=40, test n=20, budgets L5/L10/L20/L40).
Uses seed 1337 for shuffle; budgets are prefix subsets of shuffled train pool.

Usage:
    python generate_splits.py --data_root /path/to/HVSMR-2.0 --splits_dir splits --out_dir .
"""
import argparse
import re
from pathlib import Path

import numpy as np

SEED = 1337
TRAIN_N = 40
TEST_N = 20
BUDGETS = (5, 10, 20, 40)


def discover_case_ids(data_root: Path) -> list:
    """Discover case IDs from NIfTI files (pat0, pat1, ...)."""
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    ids = set()
    for subdir in ["imagesTr", "cropped_norm", "images"]:
        d = data_root / subdir
        if d.exists():
            for p in d.glob("*.nii*"):
                base = p.stem.replace(".nii", "")
                m = re.search(r"pat(\d+)", base, re.I)
                if m:
                    ids.add(f"pat{m.group(1)}")
    if not ids:
        for p in data_root.rglob("*.nii*"):
            base = p.stem.replace(".nii", "")
            m = re.search(r"pat(\d+)", base, re.I)
            if m:
                ids.add(f"pat{m.group(1)}")
    if not ids:
        raise RuntimeError(f"No case IDs found under {data_root}")
    return sorted(ids, key=lambda x: int(re.search(r"\d+", x).group()))


def main():
    ap = argparse.ArgumentParser(description="Generate fixed splits for HVSMR-2.0")
    ap.add_argument("--data_root", type=str, required=True, help="Path to HVSMR-2.0 data")
    ap.add_argument("--splits_dir", type=str, default="splits", help="Output directory for split files")
    ap.add_argument("--out_dir", type=str, default=".", help="Base output directory")
    ap.add_argument("--train_n", type=int, default=TRAIN_N)
    ap.add_argument("--test_n", type=int, default=TEST_N)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / args.splits_dir
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "budgets").mkdir(exist_ok=True)

    ids = discover_case_ids(args.data_root)
    if len(ids) < args.train_n + args.test_n:
        raise RuntimeError(
            f"Need at least {args.train_n + args.test_n} cases, found {len(ids)}"
        )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ids))
    train_idx = perm[: args.train_n]
    test_idx = perm[args.train_n : args.train_n + args.test_n]
    train_ids = [ids[i] for i in sorted(train_idx)]
    test_ids = [ids[i] for i in sorted(test_idx)]

    rng2 = np.random.default_rng(args.seed)
    train_shuf = list(train_ids)
    rng2.shuffle(train_shuf)

    def write_ids(path, id_list):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(id_list) + "\n", encoding="utf-8")

    write_ids(splits_dir / "train_pool_ids.txt", train_ids)
    write_ids(splits_dir / "test_ids.txt", test_ids)
    for b in BUDGETS:
        if b <= len(train_shuf):
            write_ids(splits_dir / "budgets" / f"L{b}_ids.txt", train_shuf[:b])

    print("[OK] Wrote splits to", splits_dir)
    print("  train_pool:", len(train_ids), ", test:", len(test_ids))


if __name__ == "__main__":
    main()
