#!/usr/bin/env python3
"""paired_tests.py — paired bootstrap and McNemar tests on headline pairs.

Bootstrap CIs (item 2) are *marginal* — they tell you each model's metric
range under test-set resampling.  Two models with overlapping CIs may still be
reliably ordered on the *same* test rows.  Paired tests answer the right
question:

  * **Paired bootstrap on differences**: resample test indices once, evaluate
    BOTH models on the same resample, take the metric difference.  Repeat
    B=1000 times, get a CI on the *difference*.  If the CI excludes zero,
    A is reliably better than B.

  * **McNemar (classification only)**: of the test rows where A and B
    disagree, how many does A get right vs B?  Chi-square gives a p-value.
    More powerful than overlap-checking when models agree on most cases.

Reads existing predictions from outputs/<split>/preds.npz (regression) or
the merged classification pool (regression preds + native classifiers).

Outputs:
  outputs/<split>/paired_tests.json                  (regression diffs)
  outputs/classify_<split>/paired_tests.json         (classification diffs + McNemar)

Usage:
  uv run python scripts/paired_tests.py --task regression     --split random
  uv run python scripts/paired_tests.py --task regression     --split scaffold
  uv run python scripts/paired_tests.py --task classification --split random
  uv run python scripts/paired_tests.py --task classification --split scaffold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, r2_score, mean_absolute_error,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cfa_classify import _best_threshold

THRESHOLD = 50.0
B = 1000


def load_pool(split, task):
    src = ROOT / "outputs" / split / "preds.npz"
    d   = np.load(src, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    val   = np.array(d["val"],  dtype=np.float64)
    test  = np.array(d["test"], dtype=np.float64)
    y_val = np.array(d["y_val"], dtype=np.float64)
    y_te  = np.array(d["y_te"],  dtype=np.float64)
    if task == "classification":
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


def paired_bootstrap_diff(pred_a_te, pred_b_te, y_te, metric_fn, B=1000, seed=42):
    """Paired bootstrap CI on metric_fn(A) - metric_fn(B).

    Resample test indices once per iteration; evaluate both models on the
    *same* resample; take the difference.  Return mean, 95 % CI, p-value
    for diff > 0.
    """
    rng = np.random.RandomState(seed)
    n = len(y_te)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        ya  = y_te[idx]
        diffs[b] = metric_fn(ya, pred_a_te[idx]) - metric_fn(ya, pred_b_te[idx])
    return {
        "mean":   float(diffs.mean()),
        "lo":     float(np.percentile(diffs, 2.5)),
        "hi":     float(np.percentile(diffs, 97.5)),
        "pr_pos": float((diffs > 0).mean()),
    }


def mcnemar(y_a_pred, y_b_pred, y_te):
    """McNemar test on two binary classifiers' predictions on the same test
    set.  Returns the contingency table and chi-square + p-value (continuity
    correction)."""
    a_correct = (y_a_pred == y_te).astype(int)
    b_correct = (y_b_pred == y_te).astype(int)
    n11 = int(np.sum((a_correct == 1) & (b_correct == 1)))
    n10 = int(np.sum((a_correct == 1) & (b_correct == 0)))
    n01 = int(np.sum((a_correct == 0) & (b_correct == 1)))
    n00 = int(np.sum((a_correct == 0) & (b_correct == 0)))
    discord = n10 + n01
    if discord == 0:
        return {"n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "chi2": 0.0, "p_value": 1.0, "favors": "tied"}
    chi2_val = (abs(n10 - n01) - 1) ** 2 / discord    # continuity-corrected
    p = 1 - chi2.cdf(chi2_val, df=1)
    favors = "A" if n10 > n01 else ("B" if n01 > n10 else "tied")
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "chi2": float(chi2_val), "p_value": float(p), "favors": favors}


# ── Per-task drivers ────────────────────────────────────────────────────────

def regression_pairs(names, val, test, y_te):
    """Pre-defined headline regression pairs."""
    return [
        ("LightGBM",     "Chemprop"),
        ("LightGBM",     "XGBoost"),
        ("LightGBM",     "LGB_MAE"),
        ("LightGBM",     "Baseline_Mean"),
        ("Chemprop",     "Baseline_Mean"),
        ("Chemprop",     "XGBoost"),
        ("XGBoost",      "RandomForest"),
    ]


def classification_pairs():
    return [
        ("LightGBM",       "Native_LGBClf"),         # thresh-reg vs native
        ("XGBoost",        "Native_XGBClf"),
        ("Ridge",          "Native_Logistic"),
        ("RandomForest",   "Native_RFClf"),
        ("skMLP_2",        "Native_MLPClf"),
        ("Chemprop",       "Native_LGBClf"),         # cross-family
        ("LightGBM",       "Chemprop"),
        ("Native_LGBClf",  "Native_XGBClf"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  choices=["regression", "classification"], required=True)
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--B", type=int, default=B)
    args = ap.parse_args()

    names, val, test, y_val, y_te = load_pool(args.split, args.task)
    name_to_i = {n: i for i, n in enumerate(names)}
    print(f"[paired] task={args.task} split={args.split} pool={len(names)}")

    out = []
    if args.task == "regression":
        pairs = regression_pairs(names, val, test, y_te)
        print(f"\n  {'A':18s}  {'B':18s}  {'A R²':>7s}  {'B R²':>7s}  {'ΔR² CI':>22s}  {'P(A>B)':>7s}")
        print(f"  {'-'*18}  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*22}  {'-'*7}")
        for a, b in pairs:
            if a not in name_to_i or b not in name_to_i:
                print(f"  {a:18s}  {b:18s}  (missing)")
                continue
            ia, ib = name_to_i[a], name_to_i[b]
            r_a = r2_score(y_te, test[ia])
            r_b = r2_score(y_te, test[ib])
            diff = paired_bootstrap_diff(test[ia], test[ib], y_te, r2_score, B=args.B)
            print(f"  {a:18s}  {b:18s}  {r_a:+7.3f}  {r_b:+7.3f}  "
                  f"[{diff['lo']:+.3f}, {diff['hi']:+.3f}]  {diff['pr_pos']:>7.3f}")
            out.append({"a": a, "b": b, "r2_a": float(r_a), "r2_b": float(r_b),
                        "diff_r2": diff})
        outdir = ROOT / "outputs" / args.split
    else:
        y_te_bin = (y_te >= THRESHOLD).astype(int)
        y_val_bin = (y_val >= THRESHOLD).astype(int)
        pairs = classification_pairs()
        print(f"\n  {'A':16s}  {'B':16s}  {'A acc':>6s}  {'B acc':>6s}  {'Δacc CI':>22s}  "
              f"{'McNemar p':>10s}  {'favors':>7s}")
        print(f"  {'-'*16}  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*22}  {'-'*10}  {'-'*7}")
        for a, b in pairs:
            if a not in name_to_i or b not in name_to_i:
                print(f"  {a:16s}  {b:16s}  (missing)")
                continue
            ia, ib = name_to_i[a], name_to_i[b]
            thr_a = _best_threshold(val[ia], y_val_bin)
            thr_b = _best_threshold(val[ib], y_val_bin)
            ya = (test[ia] >= thr_a).astype(int)
            yb = (test[ib] >= thr_b).astype(int)
            acc_a = accuracy_score(y_te_bin, ya)
            acc_b = accuracy_score(y_te_bin, yb)

            def acc_metric(y_true, y_pred):
                return accuracy_score(y_true, (y_pred >= thr_a).astype(int)) - \
                       accuracy_score(y_true, (y_pred >= thr_a).astype(int))
            # Use a paired bootstrap on the binary correctness vectors directly
            rng = np.random.RandomState(42)
            n = len(y_te_bin)
            diffs = np.empty(args.B)
            for k in range(args.B):
                idx = rng.randint(0, n, n)
                d_a = accuracy_score(y_te_bin[idx], ya[idx])
                d_b = accuracy_score(y_te_bin[idx], yb[idx])
                diffs[k] = d_a - d_b
            ci = {"mean": float(diffs.mean()),
                  "lo":   float(np.percentile(diffs, 2.5)),
                  "hi":   float(np.percentile(diffs, 97.5)),
                  "pr_pos": float((diffs > 0).mean())}
            mc = mcnemar(ya, yb, y_te_bin)
            print(f"  {a:16s}  {b:16s}  {acc_a:6.3f}  {acc_b:6.3f}  "
                  f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  {mc['p_value']:10.4f}  {mc['favors']:>7s}")
            out.append({"a": a, "b": b, "acc_a": float(acc_a), "acc_b": float(acc_b),
                        "diff_acc": ci, "mcnemar": mc})
        outdir = ROOT / "outputs" / f"classify_{args.split}"

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "paired_tests.json"
    json.dump(out, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
