#!/usr/bin/env python3
"""
rerun_cfa.py — Re-run CFA from saved per-model predictions (outputs/preds.npz).
Skips retraining. Detects and drops pathological models (NaN/Inf, exploding
magnitudes).  Runs greedy selection and the full k-way combinatorial search
(k = 2 … 5).

Writes:
  outputs/metrics.json
  outputs/cfa_greedy.json
  outputs/cfa_kway.json          — all k=2..5 results, sorted by val R²
  outputs/cfa_pairwise.json      — backward-compat: top-100 k=2 results
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfa import cfa_greedy, cfa_k_way

OUT = ROOT / "outputs"
data  = np.load(OUT / "preds.npz")
names = [str(n) for n in data["names"]]
val   = data["val"]      # (n_models, n_val)
test  = data["test"]     # (n_models, n_test)
y_val = data["y_val"]
y_te  = data["y_te"]

print(f"Loaded {len(names)} models from preds.npz")
print(f"y_val: n={len(y_val)}, range=[{y_val.min():.1f}, {y_val.max():.1f}]")
print(f"y_te : n={len(y_te)},  range=[{y_te.min():.1f}, {y_te.max():.1f}]")
print()

# ── Health check ────────────────────────────────────────────────────────────
def health(name, vp, tp):
    issues = []
    for label, p in (("val", vp), ("test", tp)):
        if not np.all(np.isfinite(p)):
            issues.append(f"{label} has NaN/Inf")
        if np.abs(p).max() > 1e3:
            issues.append(f"{label} max|pred|={np.abs(p).max():.2e}")
    return issues


known_broken = {"SGD", "PassiveAggressive", "TanimotoKNN"}

keep_names, drop_names = [], []
metrics = {}
for i, n in enumerate(names):
    if n in known_broken:
        drop_names.append((n, "deleted from model pool"))
        continue
    issues = health(n, val[i], test[i])
    if issues:
        drop_names.append((n, "; ".join(issues)))
        continue
    keep_names.append(n)
    metrics[n] = {
        "r2":   float(r2_score(y_te,  test[i])),
        "mae":  float(mean_absolute_error(y_te, test[i])),
        "rmse": float(np.sqrt(mean_squared_error(y_te, test[i]))),
    }

print(f"Kept    {len(keep_names)} healthy models")
print(f"Dropped {len(drop_names)}:")
for n, why in drop_names:
    print(f"  - {n}: {why}")
print()

val_preds = {n: val[names.index(n)]  for n in keep_names}
tes_preds = {n: test[names.index(n)] for n in keep_names}

# ── Save metrics ─────────────────────────────────────────────────────────────
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
print(f"Wrote {OUT / 'metrics.json'}  ({len(metrics)} models)")

# ── CFA greedy ───────────────────────────────────────────────────────────────
greedy = cfa_greedy(val_preds, tes_preds, y_val, y_te, time_limit=120.0, max_rounds=200)
(OUT / "cfa_greedy.json").write_text(json.dumps(greedy, indent=2))
print(f"Wrote {OUT / 'cfa_greedy.json'}")

# ── CFA k-way (k = 2 … 5) ────────────────────────────────────────────────────
kway = cfa_k_way(
    val_preds, tes_preds, y_val, y_te,
    max_k=5,
    top_n_k3=25,
    top_n_k4=20,
    top_n_k5=15,
)

# Full results sorted by val R²
all_sorted = sorted(kway, key=lambda r: -r["val_r2"])
(OUT / "cfa_kway.json").write_text(json.dumps(all_sorted[:500], indent=2))
print(f"Wrote {OUT / 'cfa_kway.json'}  (top 500 across k=2..5)")

# Backward-compat: cfa_pairwise.json = top-100 k=2 results in old format
pairs_old = [
    {"pair": r["combo"], "method": r["method"], "cd": r["group_cd"],
     "val_r2": r["val_r2"], "test_r2": r["test_r2"]}
    for r in kway if r["k"] == 2
]
top_pairs = sorted(pairs_old, key=lambda r: -r["val_r2"])[:100]
(OUT / "cfa_pairwise.json").write_text(json.dumps(top_pairs, indent=2))
print(f"Wrote {OUT / 'cfa_pairwise.json'}  (top {len(top_pairs)} pairs, backward-compat)")

# Summary per k
print("\n── Best result per k ──────────────────────────────────────────")
for k in range(2, 6):
    best = next((r for r in all_sorted if r["k"] == k), None)
    if best:
        print(f"  k={k}  val R²={best['val_r2']:+.4f}  test R²={best['test_r2']:+.4f}"
              f"  method={best['method']:6s}  {best['combo'][:60]}")
