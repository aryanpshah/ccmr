import json
import os
import re
from pathlib import Path
import shutil

# Paths
repo_root = Path(__file__).resolve().parents[1]
ds_root = repo_root / "data" / "nnunet" / "nnUNet_raw" / "Dataset940_HVSMR_L40"
dataset_json = ds_root / "dataset.json"
test_ids_file = repo_root / "data" / "splits" / "test_ids.txt"
labelsTs_dir = ds_root / "labelsTs"

print(f"Dataset root: {ds_root}")
print(f"dataset.json: {dataset_json}")
print(f"test_ids: {test_ids_file}")

# Load dataset.json
with open(dataset_json, "r") as f:
    ds = json.load(f)

# Load test IDs (e.g. pat6, pat31, ...)
with open(test_ids_file, "r") as f:
    test_ids = [line.strip() for line in f if line.strip()]

test_id_set = set(test_ids)
print("Test IDs:", test_ids)

labelsTs_dir.mkdir(parents=True, exist_ok=True)

# Regex to find things like pat06, pat6, pat031, etc.
pat_re = re.compile(r"(pat)(\d+)")

copied = 0

for tr in ds.get("training", []):
    image_rel = tr.get("image")
    label_rel = tr.get("label")
    if image_rel is None or label_rel is None:
        continue

    img_name = os.path.basename(image_rel)  # e.g. pat06_0000.nii.gz
    m = pat_re.search(img_name)
    if not m:
        # Skip entries that don't follow patXX naming
        continue

    prefix, num_str = m.groups()
    canonical_id = f"{prefix}{int(num_str)}"  # pat06 -> pat6, pat031 -> pat31
    # Debug print so we can see what is happening
    # print(f"From image {img_name} -> canonical_id {canonical_id}")

    if canonical_id in test_id_set:
        src = ds_root / label_rel.lstrip("./")
        dst = labelsTs_dir / f"{canonical_id}.nii.gz"
        print(f"Copying {src} -> {dst}")
        if not src.is_file():
            print(f"  WARNING: source label not found: {src}")
            continue
        shutil.copy(src, dst)
        copied += 1

print(f"Done. Copied {copied} label files into {labelsTs_dir}")
