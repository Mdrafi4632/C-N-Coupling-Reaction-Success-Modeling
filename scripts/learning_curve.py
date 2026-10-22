#!/usr/bin/env python3
"""learning_curve.py — train-fraction learning curves for both pipelines.

Tests the "more data would help" hypothesis by retraining the fast subset at
5 train-set fractions (10/30/60/90/100 %) and plotting the resulting test
metric.  If the curve is still rising at 100 %, more data would help; if it
has plateaued, the bottleneck is coverage/bias, not quantity.

Two splits supported:
  random    — in-distribution.  Same 70/15/15 split as main.py.
  substrate — substrate-level OOD.  Uses fold 0 from scaffold_kfold as the
              held-out test; subsamples the remaining 4 folds for train.

Outputs:
  outputs/learning_curve_<task>_<split>.json

Usage:
  uv run python scripts/learning_curve.py --task regression     --split random
  uv run python scripts/learning_curve.py --task regression     --split substrate
  uv run python scripts/learning_curve.py --task classification --split random
  uv run python scripts/learning_curve.py --task classification --split substrate
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
from main import CONFIG, load_data, random_split_idx
from cfa_classify import _best_threshold

FRACTIONS = [0.1, 0.3, 0.6, 0.9, 1.0]


def load_data_features():
    cfg  = CONFIG.copy()
    feat = cfg["data_featurized"]
    X, y = (lambda d: (d["X"], d["y"]))(np.load(feat))
    df   = load_data(cfg["data_processed"])
    return X, y, df, cfg


def fast_regressors(seed):
    from sklearn.ensemble    import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    import xgboost  as xgb
    return [
        ("LightGBM",     lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=seed, n_jobs=-1, verbose=-1)),
        ("XGBoost",      xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.8,
                                           tree_method="hist", random_state=seed, n_jobs=-1)),
        ("ExtraTrees",   ExtraTreesRegressor(n_estimators=400, max_features="sqrt",
                                              n_jobs=-1, random_state=seed)),
        ("RandomForest", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                                n_jobs=-1, random_state=seed)),
        ("Ridge",        Ridge(alpha=1.0, random_state=seed)),
        ("HistGBT",      HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                        max_leaf_nodes=63, random_state=seed)),
    ]


def native_classifiers(seed):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_native_classifiers",
        ROOT / "scripts" / "train_native_classifiers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models(seed)


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


# ── Splits ──────────────────────────────────────────────────────────────────

def get_split(args, X, y, df, cfg):
    """Return tr_idx_full, val_idx, test_idx for the chosen split.

    For 'substrate' split, use fold 0 from scaffold_kfold as test, the union
    of folds 1-4 minus a 15 % val slice as the train pool.
    """
    if args.split == "random":
        tr_idx, val_idx, te_idx = random_split_idx(len(df), {**cfg, "seed": args.seed})
        return tr_idx, val_idx, te_idx
    # substrate split: use the same fold logic as scaffold_kfold.py
    sys.path.insert(0, str(ROOT / "scripts"))
    from scaffold_kfold import scaffold_kfolds, random_train_val
    folds   = scaffold_kfolds(df, k=5, group_by="substrate")
    te_idx  = folds[args.test_fold]
    rest    = np.concatenate([f for j, f in enumerate(folds) if j != args.test_fold])
    tr_idx, val_idx = random_train_val(rest, val_frac=0.15, seed=args.seed)
    return tr_idx, val_idx, te_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",      choices=["regression", "classification"], required=True)
    ap.add_argument("--split",     choices=["random", "substrate"], default="random")
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--test-fold", type=int, default=0, dest="test_fold",
                    help="Substrate-level fold to hold out as test (default 0).")
    args = ap.parse_args()

    X, y, df, cfg = load_data_features()
    tr_idx_full, val_idx, te_idx = get_split(args, X, y, df, cfg)

    print(f"[learning_curve] task={args.task} split={args.split} seed={args.seed}")
    print(f"[learning_curve] full train={len(tr_idx_full)} val={len(val_idx)} test={len(te_idx)}")
    print(f"[learning_curve] fractions={FRACTIONS}\n")

    rng = np.random.RandomState(args.seed)
    shuffled_tr = rng.permutation(tr_idx_full)

    if args.task == "classification":
        y_val_bin = (y[val_idx] >= args.threshold).astype(int)
        y_te_bin  = (y[te_idx]  >= args.threshold).astype(int)

    # name -> {fraction -> metric dict, "n" -> {...}}
    results: dict = {"split": args.split, "task": args.task,
                     "test_fold": args.test_fold if args.split == "substrate" else None,
                     "fractions": FRACTIONS, "test_n": int(len(te_idx)),
                     "models": {}}

    for frac in FRACTIONS:
        n_tr = int(round(len(shuffled_tr) * frac))
        sel  = shuffled_tr[:n_tr]
        X_tr, y_tr  = X[sel],     y[sel]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te,  y_te  = X[te_idx],  y[te_idx]
        print(f"\n=== fraction={frac:.0%}  n_train={n_tr} ===")

        if args.task == "regression":
            for name, est in fast_regressors(args.seed):
                t0 = time.time()
                est.fit(X_tr, y_tr)
                yhat = est.predict(X_te).astype(np.float32)
                m = reg_metrics(y_te, yhat)
                results["models"].setdefault(name, {})[str(frac)] = {**m, "n_train": n_tr}
                print(f"  [{name:14s}] R²={m['r2']:+.4f}  MAE={m['mae']:.2f}  ({time.time()-t0:.1f}s)")
        else:
            for name, est in native_classifiers(args.seed):
                t0 = time.time()
                est.fit(X_tr, (y_tr >= args.threshold).astype(int))
                pv = est.predict_proba(X_val)[:, 1].astype(np.float32)
                pt = est.predict_proba(X_te )[:, 1].astype(np.float32)
                m = clf_metrics(pv, pt, y_val_bin, y_te_bin)
                results["models"].setdefault(name, {})[str(frac)] = {**m, "n_train": n_tr}
                print(f"  [{name:18s}] acc={m['accuracy']:.3f}  MCC={m['mcc']:+.3f}  "
                      f"AUC={m['auc']:.3f}  ({time.time()-t0:.1f}s)")

    out_path = ROOT / "outputs" / f"learning_curve_{args.task}_{args.split}.json"
    json.dump(results, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
