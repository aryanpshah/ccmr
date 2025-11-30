import pickle
import json
import sys
from pathlib import Path

def load_progress(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_metrics(ds_path: Path):
    progress_file = ds_path / "progress.pkl"
    if not progress_file.exists():
        print(f"Missing: {progress_file}")
        return None

    p = load_progress(progress_file)

    metrics = {
        "train_losses": p.get("train_losses", []),
        "val_losses": p.get("val_losses", []),
        "pseudo_dice": p.get("ema_pseudo_dice", []),
        "epochs_ran": len(p.get("val_losses", [])),
        "best_epoch": p.get("best_epoch", None),
        "best_pseudo_dice": p.get("best_ema_pseudo_dice", None),
        "timestamps": p.get("timestamp", []),
    }
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_nnunet_metrics.py <fold_0_dir>")
        sys.exit(1)

    fold_dir = Path(sys.argv[1])
    metrics = extract_metrics(fold_dir)

    if metrics is None:
        sys.exit(1)

    out = fold_dir / "metrics_export.json"
    out.write_text(json.dumps(metrics, indent=4))
    print(f"Saved metrics to {out}")
