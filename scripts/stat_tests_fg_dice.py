#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

pairs = [("L5",985),("L10",986),("L20",987),("L40",988)]

def load_fg(did):
    p = Path("outputs/metrics")/f"D{did}_per_case.csv"
    lines = p.read_text().strip().splitlines()
    hdr = lines[0].split(",")
    i_case = hdr.index("case_id")
    i_fg   = hdr.index("fg_mean_dice")
    m = {}
    for L in lines[1:]:
        parts = L.split(",")
        m[parts[i_case]] = float(parts[i_fg])
    return m

data = {label: load_fg(did) for label,did in pairs}
cases = sorted(set.intersection(*[set(data[label].keys()) for label,_ in pairs]))
print("n_cases:", len(cases))

def vec(label):
    return np.array([data[label][c] for c in cases], dtype=float)

comparisons = [("L5","L10"), ("L10","L20"), ("L20","L40"), ("L5","L40")]
out = []
for a,b in comparisons:
    x = vec(a); y = vec(b)
    td = ttest_rel(y, x, nan_policy="omit")  # improvement b - a
    try:
        wd = wilcoxon(y, x, zero_method="wilcox", alternative="greater")
        wp = wd.pvalue
    except Exception:
        wp = float("nan")
    diff = float(np.mean(y - x))
    out.append((a,b,diff,td.pvalue,wp))

print("comparison, mean_improvement, paired_t_p, wilcoxon_p (H1: second > first)")
for a,b,diff,tp,wp in out:
    print(f"{a}->{b}, {diff:.4f}, {tp:.4g}, {wp:.4g}")
