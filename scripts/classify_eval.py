#!/usr/bin/env python3
"""classify_eval.py — Evaluate the v2 binary success/failure classifier
from saved regression predictions in outputs/{random,scaffold}/preds.npz.

For each split:
  1. Binarise y at the 50% threshold.
  2. For every model, find the val-optimal cut-off on its continuous score
     and report val/test accuracy, F1-macro, MCC, ROC-AUC.
  3. Run greedy + k-way CFA in classification mode (cfa_classify).
  4. Write outputs to outputs/classify_<split>/.

Usage:
  uv run python scripts/classify_eval.py --split random
  uv run python scripts/classify_eval.py --split scaffold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfa_classify import (
    cfa_greedy_clf, cfa_k_way_clf, per_model_metrics,
)


THRESHOLD = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="yield threshold for binary success label (default 50)")
    ap.add_argument("--skip-native", action="store_true",
                    help="ignore native_clf_preds.npz even if present")
    args = ap.parse_args()

    src    = ROOT / "outputs" / args.split / "preds.npz"
    nat    = ROOT / "outputs" / args.split / "native_clf_preds.npz"
    outdir = ROOT / "outputs" / f"classify_{args.split}"
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[error] {src} not found")
        sys.exit(1)

    data  = np.load(src, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    val   = data["val"]
    test  = data["test"]
    y_val = data["y_val"]
    y_te  = data["y_te"]
    y_val_bin = (y_val >= args.threshold).astype(int)
    y_te_bin  = (y_te  >= args.threshold).astype(int)

    print(f"[{args.split}] loaded {len(names)} thresholded-regressor models")
    print(f"[{args.split}] val: n={len(y_val_bin)} pos={y_val_bin.sum()} ({y_val_bin.mean()*100:.1f}%)")
    print(f"[{args.split}] te : n={len(y_te_bin)}  pos={y_te_bin.sum()}  ({y_te_bin.mean()*100:.1f}%)")
    print()

    # Health filter: drop models that produced NaN/Inf or absurd magnitudes
    keep_idx, drop = [], []
    for i, n in enumerate(names):
        bad = (not np.all(np.isfinite(val[i])) or not np.all(np.isfinite(test[i]))
               or np.abs(val[i]).max() > 1e3 or np.abs(test[i]).max() > 1e3)
        (drop if bad else keep_idx).append(i if not bad else (i, n))
        if bad:
            drop[-1] = (i, n)
    keep_idx = [i for i in range(len(names)) if not any(i == d[0] for d in drop)]
    print(f"[{args.split}] keeping {len(keep_idx)} healthy / dropping {len(drop)}")

    val_preds = {names[i]: val[i]  for i in keep_idx}
    tes_preds = {names[i]: test[i] for i in keep_idx}
    model_kind = {names[i]: "thresh-reg" for i in keep_idx}

    # ── Merge native classifier predictions if present ──────────────────────
    if nat.exists() and not args.skip_native:
        nd = np.load(nat, allow_pickle=False)
        n_names = [str(n) for n in nd["names"]]
        for i, n in enumerate(n_names):
            if n in val_preds:
                print(f"[warn] native classifier {n!r} collides with regressor — skipping")
                continue
            val_preds[n]  = nd["val"][i]
            tes_preds[n]  = nd["test"][i]
            model_kind[n] = "native-clf"
        print(f"[{args.split}] merged {len(n_names)} native classifiers "
              f"(pool now {len(val_preds)} models)")
    elif args.skip_native:
        print(f"[{args.split}] --skip-native: ignoring native classifier preds")
    else:
        print(f"[{args.split}] no {nat.name} found — run "
              f"scripts/train_native_classifiers.py to add native classifiers.")

    # ── Per-model classification metrics ─────────────────────────────────────
    pmm = per_model_metrics(val_preds, tes_preds, y_val_bin, y_te_bin)
    for n, m in pmm.items():
        m["kind"] = model_kind.get(n, "unknown")
    json.dump(pmm, open(outdir / "metrics.json", "w"), indent=2, sort_keys=True)

    top = sorted(pmm.items(), key=lambda kv: -kv[1]["test_accuracy"])[:12]
    print(f"\n[{args.split}] Top 12 by test accuracy:")
    print(f"  {'Kind':10s}  {'Model':25s}  {'thr':>5s}  {'val acc':>7s}  {'val MCC':>7s}  {'test acc':>8s}  {'test F1':>7s}  {'test MCC':>8s}  {'test AUC':>8s}")
    for n, m in top:
        print(f"  {m['kind']:10s}  {n:25s}  {m['threshold']:5.2f}  {m['val_accuracy']:7.3f}  "
              f"{m['val_mcc']:+7.3f}  {m['test_accuracy']:8.3f}  "
              f"{m['test_f1']:7.3f}  {m['test_mcc']:+8.3f}  {m['test_auc']:8.3f}")

    # ── Direct head-to-head comparison: regressor vs native classifier ──────
    pairs = [
        ("LightGBM",   "Native_LGBClf"),
        ("XGBoost",    "Native_XGBClf"),
        ("skMLP_2",    "Native_MLPClf"),
        ("RandomForest","Native_RFClf"),
        ("Ridge",      "Native_Logistic"),
    ]
    comparison = []
    print(f"\n[{args.split}] Head-to-head: thresholded regressor vs. native classifier")
    print(f"  {'Family':14s}  {'Reg test acc':>13s}  {'Reg MCC':>8s}  {'Clf test acc':>13s}  {'Clf MCC':>8s}  {'Δ acc':>7s}  {'Δ MCC':>7s}")
    for reg, clf in pairs:
        if reg in pmm and clf in pmm:
            r, c = pmm[reg], pmm[clf]
            d_acc = c["test_accuracy"] - r["test_accuracy"]
            d_mcc = c["test_mcc"]      - r["test_mcc"]
            comparison.append({
                "family": reg, "regressor": reg, "classifier": clf,
                "reg_test_acc": r["test_accuracy"], "reg_test_mcc": r["test_mcc"],
                "clf_test_acc": c["test_accuracy"], "clf_test_mcc": c["test_mcc"],
                "delta_acc": d_acc, "delta_mcc": d_mcc,
            })
            print(f"  {reg:14s}  {r['test_accuracy']:13.3f}  {r['test_mcc']:+8.3f}  "
                  f"{c['test_accuracy']:13.3f}  {c['test_mcc']:+8.3f}  "
                  f"{d_acc:+7.3f}  {d_mcc:+7.3f}")
        else:
            missing = [m for m in (reg, clf) if m not in pmm]
            print(f"  {reg:14s}  (skipped — missing: {', '.join(missing)})")
    if comparison:
        json.dump(comparison, open(outdir / "native_vs_regressor.json", "w"), indent=2)

    # ── CFA greedy ───────────────────────────────────────────────────────────
    print(f"\n[{args.split}] CFA greedy classification …")
    g = cfa_greedy_clf(val_preds, tes_preds, y_val_bin, y_te_bin,
                       time_limit=120.0, max_rounds=200)
    json.dump(g, open(outdir / "cfa_greedy.json", "w"), indent=2)
    if g:
        print(f"  best combo: {g['best_combo']}")
        print(f"  method: {g['best_method']}  thr={g['threshold']:.3f}")
        print(f"  val acc={g['val_accuracy']:.3f}  test acc={g['test_accuracy']:.3f}  "
              f"test F1={g['test_f1']:.3f}  test MCC={g['test_mcc']:+.3f}")

    # ── k-way ───────────────────────────────────────────────────────────────
    print(f"\n[{args.split}] CFA k-way (k=2..5) …")
    kway = cfa_k_way_clf(val_preds, tes_preds, y_val_bin, y_te_bin,
                         max_k=5, top_n_k3=20, top_n_k4=15, top_n_k5=12)
    sorted_kway = sorted(kway, key=lambda r: -r["val_accuracy"])
    json.dump(sorted_kway[:500], open(outdir / "cfa_kway.json", "w"), indent=2)

    # Best per k by val accuracy
    print(f"\n[{args.split}] Best per k by val accuracy:")
    for k in range(2, 6):
        subs = [r for r in sorted_kway if r["k"] == k]
        if not subs:
            continue
        best = subs[0]
        print(f"  k={k}: val acc={best['val_accuracy']:.3f}  test acc={best['test_accuracy']:.3f}  "
              f"test F1={best['test_f1']:.3f}  test MCC={best['test_mcc']:+.3f}  "
              f"method={best['method']:6s}  {best['combo'][:80]}")

    # Best per k by test accuracy (informational ceiling)
    print(f"\n[{args.split}] Best per k by test accuracy (ceiling):")
    for k in range(2, 6):
        subs = sorted([r for r in kway if r["k"] == k], key=lambda r: -r["test_accuracy"])
        if not subs:
            continue
        best = subs[0]
        print(f"  k={k}: test acc={best['test_accuracy']:.3f}  test MCC={best['test_mcc']:+.3f}  "
              f"method={best['method']:6s}  {best['combo'][:80]}")


if __name__ == "__main__":
    main()
