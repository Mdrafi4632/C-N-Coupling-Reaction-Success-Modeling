#!/usr/bin/env python3
"""cfa_ablation.py — Does CFA earn its complexity?

Compares the CFA-selected ensemble (greedy + k-way) against simpler
alternatives on the **same model pool**:

  * single-best          — best individual model by val metric
  * equal-weight top-k   — unweighted mean of top-k continuous scores
  * majority vote top-k  — classification only; per-model val-optimal
                           threshold, then average the binary votes

Reads existing predictions from outputs/<split>/preds.npz (regression)
and merges outputs/<split>/native_clf_preds.npz when --task classification.

Outputs:
  outputs/<split>/cfa_ablation.json                  (regression)
  outputs/classify_<split>/cfa_ablation.json         (classification)

Usage:
  uv run python scripts/cfa_ablation.py --task regression     --split random
  uv run python scripts/cfa_ablation.py --task regression     --split scaffold
  uv run python scripts/cfa_ablation.py --task classification --split random
  uv run python scripts/cfa_ablation.py --task classification --split scaffold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cfa_classify import _best_threshold

THRESHOLD = 50.0


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_pool(split: str, task: str):
    """Load the full prediction pool for the requested task.

    Regression: outputs/<split>/preds.npz (includes baselines if appended).
    Classification: same, plus native classifiers from native_clf_preds.npz.
    Health filter: drop NaN/Inf and |pred| > 1e3 (matches bootstrap_metrics.py).
    """
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


# ── Metrics ─────────────────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred):
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def clf_metrics(scores_val, scores_te, y_val_bin, y_te_bin):
    thr = _best_threshold(scores_val, y_val_bin)
    y_pred  = (scores_te  >= thr).astype(int)
    y_predv = (scores_val >= thr).astype(int)
    out = {
        "threshold":    float(thr),
        "val_accuracy": float(accuracy_score(y_val_bin, y_predv)),
        "accuracy":     float(accuracy_score(y_te_bin, y_pred)),
        "f1":           float(f1_score(y_te_bin, y_pred, average="macro", zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_te_bin, y_pred)) if len(set(y_te_bin)) > 1 else 0.0,
    }
    try:
        out["auc"] = float(roc_auc_score(y_te_bin, scores_te))
    except Exception:
        out["auc"] = float("nan")
    return out


def _minmax(x, lo, hi):
    d = hi - lo
    if d == 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / d


# ── Strategies ──────────────────────────────────────────────────────────────

def regression_ablation(names, val, test, y_val, y_te):
    """Rank by val R²; drop trivial baselines from ranking but keep them
    visible in the pool."""
    val_r2 = np.array([r2_score(y_val, val[i]) for i in range(len(names))])
    is_baseline = np.array([n.startswith("Baseline_") for n in names])
    rank_r2 = np.where(is_baseline, -np.inf, val_r2)
    order = np.argsort(-rank_r2)

    out = {}

    # 1. Single best
    i = order[0]
    out["single_best"] = {
        "model":   names[i],
        "members": [names[i]],
        "val":     reg_metrics(y_val, val[i]),
        "test":    reg_metrics(y_te,  test[i]),
    }

    # 2. Equal-weight mean of top-k (and median for top-5)
    for k in (2, 3, 5):
        idx = order[:k]
        members = [names[i] for i in idx]
        out[f"equal_weight_top{k}"] = {
            "members": members,
            "val":     reg_metrics(y_val, val[idx].mean(axis=0)),
            "test":    reg_metrics(y_te,  test[idx].mean(axis=0)),
        }
    idx = order[:5]
    out["median_top5"] = {
        "members": [names[i] for i in idx],
        "val":     reg_metrics(y_val, np.median(val[idx],  axis=0)),
        "test":    reg_metrics(y_te,  np.median(test[idx], axis=0)),
    }

    # 3. Trivial baseline floor
    base_idx = [i for i, n in enumerate(names) if n == "Baseline_Mean"]
    if base_idx:
        i = base_idx[0]
        out["baseline_mean"] = {
            "model": names[i], "members": [names[i]],
            "val":   reg_metrics(y_val, val[i]),
            "test":  reg_metrics(y_te,  test[i]),
        }
    return out


def classification_ablation(names, val, test, y_val, y_te, threshold):
    y_val_bin = (y_val >= threshold).astype(int)
    y_te_bin  = (y_te  >= threshold).astype(int)

    # Rank by val MCC at val-optimal threshold
    val_mcc = np.zeros(len(names))
    for i in range(len(names)):
        thr = _best_threshold(val[i], y_val_bin)
        if len(set(y_val_bin)) > 1:
            val_mcc[i] = matthews_corrcoef(y_val_bin, (val[i] >= thr).astype(int))
        else:
            val_mcc[i] = 0.0
    order = np.argsort(-val_mcc)

    out = {}

    # 1. Single best
    i = order[0]
    out["single_best"] = {
        "model":   names[i],
        "members": [names[i]],
        **clf_metrics(val[i], test[i], y_val_bin, y_te_bin),
    }

    # 2. Equal-weight mean (per-model min-max normalised on val range)
    for k in (2, 3, 5):
        idx = order[:k]
        members = [names[i] for i in idx]
        v_n = np.stack([_minmax(val[i],  val[i].min(), val[i].max()) for i in idx])
        t_n = np.stack([_minmax(test[i], val[i].min(), val[i].max()) for i in idx])
        out[f"equal_weight_top{k}"] = {
            "members": members,
            **clf_metrics(v_n.mean(axis=0), t_n.mean(axis=0), y_val_bin, y_te_bin),
        }

    # 3. Majority vote (each model uses its own val-optimal threshold)
    for k in (3, 5):
        idx = order[:k]
        members = [names[i] for i in idx]
        v_b = np.zeros((len(idx), len(y_val_bin)))
        t_b = np.zeros((len(idx), len(y_te_bin)))
        for j, i in enumerate(idx):
            thr = _best_threshold(val[i], y_val_bin)
            v_b[j] = (val[i]  >= thr).astype(int)
            t_b[j] = (test[i] >= thr).astype(int)
        # Score = fraction of models voting positive; threshold at 0.5 for plurality
        v_s = v_b.mean(axis=0)
        t_s = t_b.mean(axis=0)
        thr_vote = 0.5
        y_pred = (t_s >= thr_vote).astype(int)
        m = {
            "threshold": thr_vote,
            "accuracy":  float(accuracy_score(y_te_bin, y_pred)),
            "f1":        float(f1_score(y_te_bin, y_pred, average="macro", zero_division=0)),
            "mcc":       float(matthews_corrcoef(y_te_bin, y_pred)) if len(set(y_te_bin)) > 1 else 0.0,
        }
        try:
            m["auc"] = float(roc_auc_score(y_te_bin, t_s))
        except Exception:
            m["auc"] = float("nan")
        out[f"majority_vote_top{k}"] = {"members": members, **m}

    # 4. Majority-class baseline floor
    maj = int(y_val_bin.sum() >= len(y_val_bin) / 2)
    y_pred_maj = np.full_like(y_te_bin, maj)
    out["majority_class"] = {
        "members": ["majority_class"],
        "threshold": float("nan"),
        "accuracy": float(accuracy_score(y_te_bin, y_pred_maj)),
        "f1":       float(f1_score(y_te_bin, y_pred_maj, average="macro", zero_division=0)),
        "mcc":      0.0,
        "auc":      0.5,
    }
    return out


# ── CFA reference (read from existing JSONs) ────────────────────────────────

def cfa_reference_regression(split):
    refdir = ROOT / "outputs" / split
    g = json.load(open(refdir / "cfa_greedy.json"))
    k = json.load(open(refdir / "cfa_kway.json"))
    out = {}
    if g and "best_combo" in g:
        out["cfa_greedy"] = {
            "members": g["best_combo"],
            "method":  g["best_method"],
            "val_r2":  g["val_r2"],
            "test_r2": g["test_r2"],
        }
    if k:
        best = sorted(k, key=lambda r: -r["val_r2"])[0]
        out["cfa_kway_val_selected"] = {
            "k":       best["k"],
            "members": best["combo"].split("+"),
            "method":  best["method"],
            "val_r2":  best["val_r2"],
            "test_r2": best["test_r2"],
        }
        ceiling = sorted(k, key=lambda r: -r["test_r2"])[0]
        out["cfa_kway_test_ceiling"] = {
            "k":       ceiling["k"],
            "members": ceiling["combo"].split("+"),
            "method":  ceiling["method"],
            "val_r2":  ceiling["val_r2"],
            "test_r2": ceiling["test_r2"],
        }
    return out


def cfa_reference_classification(split):
    refdir = ROOT / "outputs" / f"classify_{split}"
    g = json.load(open(refdir / "cfa_greedy.json"))
    k = json.load(open(refdir / "cfa_kway.json"))
    out = {}
    if g and "best_combo" in g:
        out["cfa_greedy"] = {
            "members":       g["best_combo"],
            "method":        g["best_method"],
            "val_accuracy":  g["val_accuracy"],
            "test_accuracy": g["test_accuracy"],
            "val_mcc":       g["val_mcc"],
            "test_mcc":      g["test_mcc"],
        }
    if k:
        best = sorted(k, key=lambda r: -r["val_accuracy"])[0]
        out["cfa_kway_val_selected"] = {
            "k":             best["k"],
            "members":       best["combo"].split("+"),
            "method":        best["method"],
            "val_accuracy":  best["val_accuracy"],
            "test_accuracy": best["test_accuracy"],
            "val_mcc":       best["val_mcc"],
            "test_mcc":      best["test_mcc"],
        }
        ceiling = sorted(k, key=lambda r: -r["test_accuracy"])[0]
        out["cfa_kway_test_ceiling"] = {
            "k":             ceiling["k"],
            "members":       ceiling["combo"].split("+"),
            "method":        ceiling["method"],
            "val_accuracy":  ceiling["val_accuracy"],
            "test_accuracy": ceiling["test_accuracy"],
            "val_mcc":       ceiling["val_mcc"],
            "test_mcc":      ceiling["test_mcc"],
        }
    return out


# ── Pretty printer ──────────────────────────────────────────────────────────

def print_regression_table(results):
    print(f"\n{'Strategy':28s}  {'Members':40s}  {'Val R²':>8s}  {'Test R²':>8s}  {'Δ vs single':>11s}")
    print(f"{'-'*28}  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*11}")
    base = results.get("single_best", {}).get("test", {}).get("r2", 0.0)
    order = ["baseline_mean", "single_best",
             "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
             "median_top5", "cfa_greedy", "cfa_kway_val_selected",
             "cfa_kway_test_ceiling"]
    for k in order:
        r = results.get(k)
        if not r:
            continue
        m = r.get("members", [])
        m_str = "+".join(m)[:40]
        if "test" in r:
            v_r2  = r["val"]["r2"]
            t_r2  = r["test"]["r2"]
        else:
            v_r2  = r.get("val_r2", float("nan"))
            t_r2  = r.get("test_r2", float("nan"))
        delta = t_r2 - base
        print(f"{k:28s}  {m_str:40s}  {v_r2:+8.4f}  {t_r2:+8.4f}  {delta:+11.4f}")


def print_classification_table(results):
    print(f"\n{'Strategy':28s}  {'Members':40s}  {'Val acc':>7s}  {'Test acc':>8s}  {'Test MCC':>8s}  {'Δ vs single':>11s}")
    print(f"{'-'*28}  {'-'*40}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*11}")
    base = results.get("single_best", {}).get("accuracy", 0.0)
    order = ["majority_class", "single_best",
             "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
             "majority_vote_top3", "majority_vote_top5",
             "cfa_greedy", "cfa_kway_val_selected", "cfa_kway_test_ceiling"]
    for k in order:
        r = results.get(k)
        if not r:
            continue
        m = r.get("members", [])
        m_str = "+".join(m)[:40]
        if "accuracy" in r:
            v_acc = r.get("val_accuracy", float("nan"))
            t_acc = r["accuracy"]
            t_mcc = r.get("mcc",  float("nan"))
        else:
            v_acc = r.get("val_accuracy", float("nan"))
            t_acc = r.get("test_accuracy", float("nan"))
            t_mcc = r.get("test_mcc", float("nan"))
        delta = t_acc - base
        print(f"{k:28s}  {m_str:40s}  {v_acc:7.3f}  {t_acc:8.3f}  {t_mcc:+8.3f}  {delta:+11.4f}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  choices=["regression", "classification"], required=True)
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    names, val, test, y_val, y_te = load_pool(args.split, args.task)
    print(f"[ablation] task={args.task} split={args.split} pool={len(names)} models")

    if args.task == "regression":
        results = regression_ablation(names, val, test, y_val, y_te)
        results.update(cfa_reference_regression(args.split))
        outdir  = ROOT / "outputs" / args.split
        print_regression_table(results)
    else:
        results = classification_ablation(names, val, test, y_val, y_te, args.threshold)
        results.update(cfa_reference_classification(args.split))
        outdir  = ROOT / "outputs" / f"classify_{args.split}"
        print_classification_table(results)

    outpath = outdir / "cfa_ablation.json"
    json.dump(results, open(outpath, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {outpath}")


if __name__ == "__main__":
    main()
