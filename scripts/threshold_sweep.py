#!/usr/bin/env python3
"""threshold_sweep.py — sweep the binary-success cut-off from 20 % to 80 % yield.

Tests robustness of the classification problem to the choice of threshold.
The 50 % cut-off is the synthetic-chemistry literature's "preparatively useful"
line, but downstream users may care about other cut-offs (30 % for early-stage
triage; 70 % for scale-up readiness).  This sweep:

  1. For each threshold t in {20, 25, ..., 80} %,
  2. Re-binarise y_val and y_te at t,
  3. For each model in the pool, find the val-optimal score cut-off,
  4. Report test accuracy / MCC / F1 / AUC.

Plot interpretation: a flat or gently decaying curve = the model is robust to
threshold choice.  A cliff near 50 % = the model is overfit to the
binarisation.

Reads the merged classification pool (regression preds + native classifiers).

Outputs:
  outputs/classify_<split>/threshold_sweep.json

Usage:
  uv run python scripts/threshold_sweep.py --split random
  uv run python scripts/threshold_sweep.py --split scaffold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cfa_classify import _best_threshold

DEFAULT_THRESHOLDS = list(range(20, 81, 5))   # 20, 25, ..., 80


def load_pool(split):
    src = ROOT / "outputs" / split / "preds.npz"
    d   = np.load(src, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    val   = np.array(d["val"],  dtype=np.float64)
    test  = np.array(d["test"], dtype=np.float64)
    y_val = np.array(d["y_val"], dtype=np.float64)
    y_te  = np.array(d["y_te"],  dtype=np.float64)
    nat = ROOT / "outputs" / split / "native_clf_preds.npz"
    if nat.exists():
        nd = np.load(nat, allow_pickle=False)
        for i, n in enumerate(str(x) for x in nd["names"]):
            if n in names:
                continue
            names.append(n)
            val  = np.vstack([val,  nd["val"][i:i+1].astype(np.float64)])
            test = np.vstack([test, nd["test"][i:i+1].astype(np.float64)])
    keep = [
        i for i in range(len(names))
        if np.all(np.isfinite(val[i])) and np.all(np.isfinite(test[i]))
        and np.abs(val[i]).max() <= 1e3 and np.abs(test[i]).max() <= 1e3
    ]
    names = [names[i] for i in keep]
    val   = val[keep]
    test  = test[keep]
    return names, val, test, y_val, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",       choices=["random", "scaffold"], default="random")
    ap.add_argument("--thresholds",  type=int, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--top",         type=int, default=10,
                    help="report top-N models per threshold (by val accuracy)")
    args = ap.parse_args()

    names, val, test, y_val, y_te = load_pool(args.split)
    print(f"[threshold_sweep] split={args.split} pool={len(names)} "
          f"thresholds={args.thresholds}")

    sweep = {"split": args.split, "thresholds": args.thresholds, "test_n": int(len(y_te)),
             "n_pos_per_threshold": {}, "results": {}}

    for t in args.thresholds:
        y_val_bin = (y_val >= t).astype(int)
        y_te_bin  = (y_te  >= t).astype(int)
        n_pos_te  = int(y_te_bin.sum())
        sweep["n_pos_per_threshold"][str(t)] = {
            "n_pos_te":  n_pos_te,
            "frac_te":   round(n_pos_te / len(y_te_bin), 4),
            "n_pos_val": int(y_val_bin.sum()),
        }
        if y_te_bin.sum() == 0 or y_te_bin.sum() == len(y_te_bin):
            print(f"  threshold={t}%  test pos={n_pos_te} — skipping (degenerate)")
            continue
        if y_val_bin.sum() == 0 or y_val_bin.sum() == len(y_val_bin):
            print(f"  threshold={t}%  val pos={int(y_val_bin.sum())} — skipping (no val signal)")
            continue

        per_model = {}
        for i, n in enumerate(names):
            try:
                thr = _best_threshold(val[i], y_val_bin)
                ya  = (test[i] >= thr).astype(int)
                yv  = (val[i]  >= thr).astype(int)
                m = {
                    "score_threshold": float(thr),
                    "val_accuracy":    float(accuracy_score(y_val_bin, yv)),
                    "test_accuracy":   float(accuracy_score(y_te_bin, ya)),
                    "test_f1":         float(f1_score(y_te_bin, ya, average="macro", zero_division=0)),
                    "test_mcc":        float(matthews_corrcoef(y_te_bin, ya))
                                       if len(set(y_te_bin)) > 1 else 0.0,
                }
                try:
                    m["test_auc"] = float(roc_auc_score(y_te_bin, test[i]))
                except Exception:
                    m["test_auc"] = float("nan")
                per_model[n] = m
            except Exception as e:
                continue

        # Top models at this threshold (by val accuracy)
        top = sorted(per_model.items(), key=lambda kv: -kv[1]["val_accuracy"])[:args.top]
        print(f"\n  threshold={t}%  test pos={n_pos_te}/{len(y_te_bin)} "
              f"({n_pos_te/len(y_te_bin)*100:.0f}%)")
        print(f"    {'Model':22s}  {'val acc':>7s}  {'test acc':>8s}  {'test MCC':>8s}  {'test AUC':>8s}")
        for name, m in top[:5]:
            print(f"    {name:22s}  {m['val_accuracy']:7.3f}  {m['test_accuracy']:8.3f}  "
                  f"{m['test_mcc']:+8.3f}  {m['test_auc']:8.3f}")
        sweep["results"][str(t)] = per_model

    outdir = ROOT / "outputs" / f"classify_{args.split}"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "threshold_sweep.json"
    json.dump(sweep, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
