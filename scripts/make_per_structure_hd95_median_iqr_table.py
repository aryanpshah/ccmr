import csv
from pathlib import Path
import numpy as np

PAIRS = [("L5", 985), ("L10", 986), ("L20", 987), ("L40", 988)]
OUT_PATH = Path("outputs/final_report/per_structure_hd95_median_iqr_by_budget.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

STRUCT = {1:"LV",2:"RV",3:"LA",4:"RA",5:"Aorta",6:"PA",7:"SVC",8:"IVC"}

def median_q1_q3_iqr(x):
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    q1  = float(np.percentile(x, 25))
    q3  = float(np.percentile(x, 75))
    return med, q1, q3, (q3 - q1)

def read_per_case_hd95(did: int):
    p = Path(f"outputs/metrics/D{did}_per_case.csv")
    if not p.exists():
        raise SystemExit(f"Missing per-case CSV: {p}")

    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        fns = r.fieldnames or []
        lower_map = {name.lower(): name for name in fns}

        # Find hd95_1_mm ... hd95_8_mm columns (case-insensitive)
        cols = {}
        for k in range(1, 9):
            key = f"hd95_{k}_mm"
            if key in lower_map:
                cols[k] = lower_map[key]
            else:
                # fallback if someone wrote hd95_1 without _mm
                key2 = f"hd95_{k}"
                if key2 in lower_map:
                    cols[k] = lower_map[key2]

        missing = [k for k in range(1, 9) if k not in cols]
        if missing:
            raise SystemExit(
                f"D{did}: missing HD95 columns for classes {missing}. "
                f"Available columns: {fns}"
            )

        vals = {k: [] for k in range(1, 9)}
        for row in r:
            for k in range(1, 9):
                v = str(row.get(cols[k], "")).strip()
                if v == "" or v.lower() == "nan":
                    continue
                vals[k].append(float(v))

        # sanity: expect 20 cases, but do not hard fail
        for k in range(1, 9):
            if len(vals[k]) == 0:
                raise SystemExit(f"D{did}: class {k} has 0 valid HD95 entries (col={cols[k]}).")

    return vals

def main():
    data = {lab: read_per_case_hd95(did) for lab, did in PAIRS}

    with OUT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["Structure"]
        for lab, _ in PAIRS:
            header += [f"{lab}_Median_HD95_mm", f"{lab}_Q1_mm", f"{lab}_Q3_mm", f"{lab}_IQR_mm"]
        w.writerow(header)

        for k in range(1, 9):
            row = [STRUCT[k]]
            for lab, _ in PAIRS:
                med, q1, q3, iqr = median_q1_q3_iqr(data[lab][k])
                row += [f"{med:.2f}", f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}"]
            w.writerow(row)

    print("[OK] wrote", OUT_PATH)

if __name__ == "__main__":
    main()
