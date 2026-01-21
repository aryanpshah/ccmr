#!/usr/bin/env bash
set -euo pipefail

printf '%-4s | %-6s | %-14s | %-14s | %-9s\n' "L" "epochs" "final_val_dice" "best_val_dice" "best_epoch"
printf '%s\n' '---- | ------ | -------------- | -------------- | ---------'

for L in 5 10 20 40; do
  LOG="runs/scratch_L${L}_FIXED/train.log"

  if [[ ! -f "$LOG" ]]; then
    printf '%-4s | %-6s | %-14s | %-14s | %-9s\n' "$L" "NA" "MISSING_LOG" "NA" "NA"
    continue
  fi

  awk '
    BEGIN{
      maxep=0; cur=0;
      final="NA";
      best=-1; bestep="NA";
    }

    # epoch lines like: "Epoch 12/150" or "epoch 12/150"
    $0 ~ /[Ee]poch[^0-9]*[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+/ {
      s=$0
      sub(/.*[Ee]poch[^0-9]*/, "", s)   # remove up to the epoch number
      split(s, a, "/")                  # a[1] is epoch number
      cur = a[1] + 0
      if (cur > maxep) maxep = cur
    }

    # dice line like: "Val mean Dice (fg): 0.1234"
    /Val mean Dice \(fg\):/ {
      v = $NF + 0
      final = v
      if (v > best) { best = v; bestep = (cur > 0 ? cur : "NA") }
    }

    END{
      if (best < 0) best = "NA"
      printf "%s %s %s %s\n", maxep, final, best, bestep
    }
  ' "$LOG" | {
    read -r epochs final best bestep
    printf '%-4s | %-6s | %-14s | %-14s | %-9s\n' "$L" "$epochs" "$final" "$best" "$bestep"
  }

done
