#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--raw-root", default="data/nnunet/nnUNet_raw")
    ap.add_argument("--pre-root", default="data/nnunet/nnUNet_preprocessed")
    args = ap.parse_args()

    raw = next(Path(args.raw_root).glob(f"Dataset{args.dataset_id}_*"))
    pre = next(Path(args.pre_root).glob(f"Dataset{args.dataset_id}_*"))

    tr = (raw/"budget_train_ids.txt").read_text().strip().splitlines()
    va = (raw/"budget_val_ids.txt").read_text().strip().splitlines()

    out = pre/"splits_final.json"
    out.write_text(json.dumps([{"train": tr, "val": va}], indent=2))
    print("Wrote", out)
    print("Train:", len(tr), "Val:", len(va))

if __name__ == "__main__":
    main()
