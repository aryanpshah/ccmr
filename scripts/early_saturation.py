#!/usr/bin/env python3
import json
from pathlib import Path
import math

pairs = [("L5",985),("L10",986),("L20",987),("L40",988)]
K = 8
THRESH = 0.90

summ = {}
for label,did in pairs:
    summ[label] = json.loads((Path("outputs/metrics")/f"D{did}_summary.json").read_text())

l40 = summ["L40"]

print("class,L40,L5,L10,L20,ratio_L5,ratio_L10,ratio_L20,reach_90_at")
for k in range(1, K+1):
    d40 = l40["mean"][f"dice_{k}"]
    d5  = summ["L5"]["mean"][f"dice_{k}"]
    d10 = summ["L10"]["mean"][f"dice_{k}"]
    d20 = summ["L20"]["mean"][f"dice_{k}"]

    def ratio(a,b):
        if b == 0:
            return float("nan")
        return a / b

    r5, r10, r20 = ratio(d5,d40), ratio(d10,d40), ratio(d20,d40)

    reach = "none"
    if not math.isnan(r5) and r5 >= THRESH: reach = "L5"
    elif not math.isnan(r10) and r10 >= THRESH: reach = "L10"
    elif not math.isnan(r20) and r20 >= THRESH: reach = "L20"
    else: reach = "L40_only"

    print(f"{k},{d40:.4f},{d5:.4f},{d10:.4f},{d20:.4f},{r5:.3f},{r10:.3f},{r20:.3f},{reach}")
