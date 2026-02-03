import argparse, json
from pathlib import Path
import json
import shutil

ROOT = Path("data/raw/HVSMR2/cropped_norm")  # contains directories ending with .nii
SPLIT_DIR = Path("data/splits")              # your train/val/test txt files
NNUNET_RAW = Path("data/nnunet/nnUNet_raw")

def inner_nii(case: str, kind: str) -> Path:
    # kind: "cropped_norm" or "cropped_seg"
    d = ROOT / f"{case}_{kind}.nii" / f"{case}_{kind}.nii"
    if not d.exists():
        raise FileNotFoundError(f"Missing: {d}")
    return d

def read_ids(p: Path):
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--dataset-name", type=str, default=None)
    ap.add_argument("--label-budget", type=int, default=None, help="5/10/20/40; uses train_L{B}.txt if provided")
    ap.add_argument("--seed", type=int, default=0, help="only used if you implement shuffling; kept for interface compat")
    args = ap.parse_args()

    label_budget = args.label_budget
    dataset_id = args.dataset_id
    # choose train list
    if args.label_budget:
        train_file = SPLIT_DIR / f"train_L{args.label_budget}.txt"
    else:
        train_file = SPLIT_DIR / "train_ids.txt"
    val_file  = SPLIT_DIR / "val_ids.txt"
    test_file = SPLIT_DIR / "test_ids.txt"

    train_ids = read_ids(train_file)
    val_ids   = read_ids(val_file)
    test_ids  = read_ids(test_file)

    dataset_name = args.dataset_name or f"Dataset{args.dataset_id:03d}_HVSMR_CROPPEDNORM_L{args.label_budget or 'FULL'}"
    out = NNUNET_RAW / dataset_name

    # fresh dirs
    if out.exists():
        raise RuntimeError(f"{out} already exists. Pick a new dataset-id/name or delete it.")
    (out/"imagesTr").mkdir(parents=True)
    (out/"labelsTr").mkdir(parents=True)
    (out/"imagesTs").mkdir(parents=True)
    (out/"labelsTs").mkdir(parents=True)  # nnU-Net ignores, but useful for local eval

    def copy_case(cid: str, split: str):
        img_src = inner_nii(cid, "cropped_norm")
        lab_src = inner_nii(cid, "cropped_seg")
        if split == "Tr":
            img_dst = out/"imagesTr"/f"{cid}_0000.nii.gz"
            lab_dst = out/"labelsTr"/f"{cid}.nii.gz"
        else:
            img_dst = out/"imagesTs"/f"{cid}_0000.nii.gz"
            lab_dst = out/"labelsTs"/f"{cid}.nii.gz"
        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(lab_src, lab_dst)
    # --- label-budget pool (size=L) + fixed 80/20 split within the budget ---
    budget_ids = list(train_ids)[:label_budget]
    n_train = max(1, int(round(0.8 * label_budget)))
    n_train = min(n_train, label_budget - 1) if label_budget >= 2 else 1
    budget_train_ids = budget_ids[:n_train]
    budget_val_ids = budget_ids[n_train:]

    # Copy ONLY labeled pool cases into labelsTr/imagesTr
    for cid in budget_ids:
        copy_case(cid, "Tr")

    NL = chr(10)
    (out/"budget_train_ids.txt").write_text(NL.join(budget_train_ids) + NL)
    (out/"budget_val_ids.txt").write_text(NL.join(budget_val_ids) + NL)

    # Raw split file (copy to preprocessed before training)
    splits = [{"train": budget_train_ids, "val": budget_val_ids}]
    (out/"splits_final.json").write_text(json.dumps(splits, indent=2) + NL)
    print(f"Budget split -> Train: {len(budget_train_ids)}  Val: {len(budget_val_ids)}  Test: {len(test_ids)}")
    for cid in test_ids:
        copy_case(cid, "Ts")

    # dataset.json (simple, correct)
    dataset_json = {
        "name": dataset_name,
        "description": "HVSMR2 cropped_norm volumes with aligned cropped_seg labels (no masking; nnU-Net handles preprocessing).",
        "tensorImageSize": "3D",
        "reference": "HVSMR 2016 / HVSMR2.0",
        "licence": "see HVSMR",
        "release": "1.0",
        "channel_names": {"0": "MR"},
        "file_ending": ".nii.gz",
        "labels": {"background": 0, "blood_pool": 1, "myocardium": 2},
        "numTraining": len(train_ids),
        "numTest": len(test_ids),
        "training": [
            {"image": f"./imagesTr/{cid}_0000.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
            for cid in train_ids
        ],
        "test": [f"./imagesTs/{cid}_0000.nii.gz" for cid in test_ids],
    }
    (out/"dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print("Wrote", out)
    print("Train:", len(train_ids), " Val:", len(val_ids), " Test:", len(test_ids))
    print("Example image:", (out/'imagesTr').glob('pat*_0000.nii.gz').__next__())

if __name__ == "__main__":
    main()
