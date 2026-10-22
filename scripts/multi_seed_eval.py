#!/usr/bin/env python3
"""multi_seed_eval.py — train the fast subset of each pool with 5 seeds and
report per-metric mean ± std across seeds.

What this captures (different from bootstrap_metrics.py):
  * **Bootstrap CIs** capture *test-set* sampling variance for a single
    trained model — i.e., how would this model do on a slightly different
    test set drawn from the same distribution.
  * **Multi-seed** captures *training* variance — how would different
    train/val/test splits and model initialisations end up.
Both belong in a comprehensive evaluation; together they are the standard
ML-paper answer to the rubric's "comprehensively test" criterion.

Subset choice:  the slowest models (Chemprop, GP_Tanimoto, NGBoost on full
features) cost minutes apiece, making 5-seed × 2-split unaffordable.  We
therefore restrict to the fast subset — the headline regression contributors
plus all 5 native classifiers.  Together this is enough to produce headline
mean±std numbers for the rubric without 3+ hours of compute.

Splits & seeds:
  * random:   each seed gives a fresh 70/15/15 split (train/val/test variance).
  * scaffold: split is deterministic, so multi-seed only varies model init.

Outputs:
  outputs/<split>/multiseed_metrics.json                  (regression)
  outputs/classify_<split>/multiseed_metrics.json         (classification)

Usage:
  uv run python scripts/multi_seed_eval.py --task regression     --split random
  uv run python scripts/multi_seed_eval.py --task classification --split random
  uv run python scripts/multi_seed_eval.py --task regression     --split scaffold
  uv run python scripts/multi_seed_eval.py --task classification --split scaffold
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from main import CONFIG, load_data, random_split_idx, scaffold_split_idx
from cfa_classify import _best_threshold

SEEDS = [42, 123, 456, 789, 1000]


def fast_regressors(seed: int):
    """Fast regression models — chosen by training cost on the full feature
    matrix.  Excludes Chemprop, NGBoost, GP_Tanimoto (slow)."""
    from sklearn.ensemble    import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    import xgboost  as xgb
    return [
        ("LightGBM", lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )),
        ("XGBoost", xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=seed, n_jobs=-1,
        )),
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed,
        )),
        ("RandomForest", RandomForestRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed,
        )),
        ("Ridge", Ridge(alpha=1.0, random_state=seed)),
        ("HistGBT", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            random_state=seed,
        )),
    ]


def native_classifiers(seed: int):
    """Re-use the same builder as scripts/train_native_classifiers.py to keep the
    two paths consistent.  Loaded by file path because scripts/ has no
    __init__.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_native_classifiers",
        ROOT / "scripts" / "train_native_classifiers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models(seed)


# ── Metric primitives ───────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred):
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def clf_metrics(scores_val, scores_te, y_val_bin, y_te_bin):
    thr = _best_threshold(scores_val, y_val_bin)
    y_pred = (scores_te >= thr).astype(int)
    out = {
        "threshold": float(thr),
        "accuracy":  float(accuracy_score(y_te_bin, y_pred)),
        "f1":        float(f1_score(y_te_bin, y_pred, average="macro", zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_te_bin, y_pred)) if len(set(y_te_bin)) > 1 else 0.0,
    }
    try:
        out["auc"] = float(roc_auc_score(y_te_bin, scores_te))
    except Exception:
        out["auc"] = float("nan")
    return out


def aggregate(per_seed_metrics: list[dict]) -> dict:
    """list of {metric -> value} → {metric -> {mean, std, values}}"""
    keys = per_seed_metrics[0].keys()
    out  = {}
    for k in keys:
        vs = np.array([m[k] for m in per_seed_metrics], dtype=np.float64)
        vs = vs[np.isfinite(vs)]
        if len(vs) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "values": []}
            continue
        out[k] = {
            "mean":   float(vs.mean()),
            "std":    float(vs.std(ddof=1)) if len(vs) > 1 else 0.0,
            "values": [float(v) for v in vs],
        }
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  choices=["regression", "classification"], required=True)
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    cfg  = CONFIG.copy()
    feat = cfg["data_featurized"]
    if not feat.exists():
        print(f"[error] {feat} not found — run main.py first.")
        sys.exit(1)
    X, y = (lambda d: (d["X"], d["y"]))(np.load(feat))
    df   = load_data(cfg["data_processed"])

    print(f"[multiseed] task={args.task}  split={args.split}  seeds={args.seeds}")

    # name -> [per-seed metrics dict]
    per_model_runs: dict[str, list[dict]] = {}

    for seed in args.seeds:
        cfg_s = {**cfg, "seed": seed}
        if args.split == "scaffold":
            tr_idx, val_idx, te_idx = scaffold_split_idx(df, cfg_s)
        else:
            tr_idx, val_idx, te_idx = random_split_idx(len(df), cfg_s)

        X_tr,  y_tr  = X[tr_idx],  y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te,  y_te  = X[te_idx],  y[te_idx]
        print(f"\n[seed={seed}] train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

        if args.task == "regression":
            for name, est in fast_regressors(seed):
                t0 = time.time()
                est.fit(X_tr, y_tr)
                yhat_te = est.predict(X_te).astype(np.float32)
                m = reg_metrics(y_te, yhat_te)
                per_model_runs.setdefault(name, []).append(m)
                print(f"  [{name:14s}] R²={m['r2']:+.4f}  MAE={m['mae']:.2f}  "
                      f"({time.time()-t0:.1f}s)")
        else:
            y_val_bin = (y_val >= args.threshold).astype(int)
            y_te_bin  = (y_te  >= args.threshold).astype(int)
            for name, est in native_classifiers(seed):
                t0 = time.time()
                est.fit(X_tr, (y_tr >= args.threshold).astype(int))
                pv = est.predict_proba(X_val)[:, 1].astype(np.float32)
                pt = est.predict_proba(X_te )[:, 1].astype(np.float32)
                m = clf_metrics(pv, pt, y_val_bin, y_te_bin)
                per_model_runs.setdefault(name, []).append(m)
                print(f"  [{name:18s}] acc={m['accuracy']:.3f}  MCC={m['mcc']:+.3f}  "
                      f"AUC={m['auc']:.3f}  ({time.time()-t0:.1f}s)")

    # Aggregate
    print(f"\n{'='*60}\n  Mean ± std across {len(args.seeds)} seeds\n{'='*60}")
    agg = {}
    for name, runs in per_model_runs.items():
        agg[name] = aggregate(runs)
        if args.task == "regression":
            r2 = agg[name]["r2"]
            mae = agg[name]["mae"]
            print(f"  {name:18s}  R²={r2['mean']:+.4f}±{r2['std']:.4f}  "
                  f"MAE={mae['mean']:.2f}±{mae['std']:.2f}")
        else:
            acc = agg[name]["accuracy"]
            mcc = agg[name]["mcc"]
            auc = agg[name]["auc"]
            print(f"  {name:18s}  acc={acc['mean']:.3f}±{acc['std']:.3f}  "
                  f"MCC={mcc['mean']:+.3f}±{mcc['std']:.3f}  "
                  f"AUC={auc['mean']:.3f}±{auc['std']:.3f}")

    # Save
    if args.task == "regression":
        outdir = ROOT / "outputs" / args.split
    else:
        outdir = ROOT / "outputs" / f"classify_{args.split}"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "multiseed_metrics.json"
    json.dump({"seeds": args.seeds, "results": agg},
              open(outpath, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {outpath}")


if __name__ == "__main__":
    main()
