import re
import json
import sys
from pathlib import Path

# Regex patterns for parsing nnU-Net logs
EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
TRAIN_RE = re.compile(r"train_loss\s+([-\d\.eE]+)")
VAL_RE = re.compile(r"val_loss\s+([-\d\.eE]+)")
PD_RE = re.compile(
    r"Pseudo dice\s*\[\s*np\.float32\(([-\d\.eE]+)\)\s*,\s*np\.float32\(([-\d\.eE]+)\)\s*\]"
)

def parse_log(path: Path):
    data = {}
    current_epoch = None

    with path.open("r", errors="ignore") as f:
        for line in f:
            # -----------------
            # Detect epoch start
            # -----------------
            m = EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1))
                if current_epoch not in data:
                    data[current_epoch] = {}
                continue

            if current_epoch is None:
                continue

            # -----------------
            # train_loss
            # -----------------
            m = TRAIN_RE.search(line)
            if m:
                try:
                    data[current_epoch]["train_loss"] = float(m.group(1))
                except ValueError:
                    pass
                continue

            # -----------------
            # val_loss
            # -----------------
            m = VAL_RE.search(line)
            if m:
                try:
                    data[current_epoch]["val_loss"] = float(m.group(1))
                except ValueError:
                    pass
                continue

            # -----------------
            # pseudo dice
            # -----------------
            m = PD_RE.search(line)
            if m:
                try:
                    data[current_epoch]["pseudo_dice_1"] = float(m.group(1))
                    data[current_epoch]["pseudo_dice_2"] = float(m.group(2))
                except ValueError:
                    pass
                continue

    # Convert dictionary → list sorted by epoch
    records = []
    for ep in sorted(data.keys()):
        rec = {"epoch": ep}
        rec.update(data[ep])
        records.append(rec)

    return records

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/parse_nnunet_log.py <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    records = parse_log(log_path)

    out_path = log_path.with_suffix(".metrics.json")
    out_path.write_text(json.dumps(records, indent=4))

    print(f"Parsed {len(records)} epochs → {out_path}")
