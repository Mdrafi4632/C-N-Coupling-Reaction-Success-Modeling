#!/usr/bin/env python3
"""train_native_classifiers.py — Train binary classifiers on the same
features and split as the regression pool, save their predicted probabilities
to outputs/<split>/native_clf_preds.npz so classify_eval.py can compare them
side-by-side with the thresholded regressors.

Why: the report's classification pool is currently 100% thresholded regressors.
This script adds the missing alternative-approach test — native classifiers
trained directly on binary labels with log-loss.

Models:
  Native_Logistic    — LogisticRegression(L2)
  Native_LGBClf      — LGBMClassifier (direct counterpart to LightGBM regressor)
  Native_XGBClf      — XGBClassifier  (direct counterpart to XGBoost regressor)
  Native_RFClf       — RandomForestClassifier
  Native_MLPClf      — sklearn MLPClassifier (direct counterpart to skMLP_2)

Usage:
  uv run python scripts/train_native_classifiers.py --split random
  uv run python scripts/train_native_classifiers.py --split scaffold
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse main.py's data loader and split functions to guarantee identical splits
from main import (
    CONFIG, load_data, random_split_idx, scaffold_split_idx,
)

THRESHOLD = 50.0


def build_models(seed: int):
    """Return (name, estimator) pairs. Each estimator has predict_proba."""
    from sklearn.linear_model      import LogisticRegression
    from sklearn.ensemble          import RandomForestClassifier
    from sklearn.neural_network    import MLPClassifier
    import lightgbm as lgb
    import xgboost  as xgb

    return [
        ("Native_Logistic", LogisticRegression(
            C=1.0, max_iter=2000, solver="liblinear", random_state=seed,
        )),
        ("Native_LGBClf", lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )),
        ("Native_XGBClf", xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=seed, n_jobs=-1,
            eval_metric="logloss",
        )),
        ("Native_RFClf", RandomForestClassifier(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed,
        )),
        ("Native_MLPClf", MLPClassifier(
            hidden_layer_sizes=(512, 256), activation="relu",
            alpha=1e-4, max_iter=80, early_stopping=True,
            random_state=seed,
        )),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    cfg = CONFIG.copy()
    cfg["seed"] = args.seed

    # ── Load cached features ────────────────────────────────────────────────
    feat = cfg["data_featurized"]
    if not feat.exists():
        print(f"[error] {feat} not found — run main.py first to build features.")
        sys.exit(1)
    data = np.load(feat)
    X, y = data["X"], data["y"]
    df   = load_data(cfg["data_processed"])
    print(f"[load] X={X.shape}  y={y.shape}  df={len(df)}")

    # ── Replicate the split ─────────────────────────────────────────────────
    if args.split == "scaffold":
        tr_idx, val_idx, te_idx = scaffold_split_idx(df, cfg)
    else:
        tr_idx, val_idx, te_idx = random_split_idx(len(df), cfg)
    X_tr, y_tr  = X[tr_idx],  y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te  = X[te_idx],  y[te_idx]
    print(f"[split={args.split}] train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    y_tr_bin  = (y_tr  >= args.threshold).astype(int)
    y_val_bin = (y_val >= args.threshold).astype(int)
    y_te_bin  = (y_te  >= args.threshold).astype(int)
    print(f"[binarise@{args.threshold}] tr +={y_tr_bin.sum()}/{len(y_tr_bin)}  "
          f"val +={y_val_bin.sum()}/{len(y_val_bin)}  "
          f"te +={y_te_bin.sum()}/{len(y_te_bin)}")

    # ── Train each native classifier ────────────────────────────────────────
    models = build_models(args.seed)
    val_probs, test_probs, names = [], [], []
    for name, est in models:
        t0 = time.time()
        try:
            est.fit(X_tr, y_tr_bin)
            pv = est.predict_proba(X_val)[:, 1].astype(np.float32)
            pt = est.predict_proba(X_te )[:, 1].astype(np.float32)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            continue
        val_probs.append(pv)
        test_probs.append(pt)
        names.append(name)
        # Quick val accuracy at 0.5 (for log only — CFA picks its own threshold)
        acc = float(((pv >= 0.5).astype(int) == y_val_bin).mean())
        print(f"[{name:18s}] val_acc@0.5={acc:.3f}  ({time.time()-t0:.1f}s)")

    if not names:
        print("[error] no native classifiers trained")
        sys.exit(2)

    # ── Save in the same format as outputs/<split>/preds.npz ────────────────
    out = ROOT / "outputs" / args.split / "native_clf_preds.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        names=np.array(names),
        val=np.stack(val_probs),
        test=np.stack(test_probs),
        y_val=y_val.astype(np.float32),    # keep continuous for downstream consistency
        y_te=y_te.astype(np.float32),
    )
    print(f"\n[save] {out}  ({len(names)} native classifiers)")


if __name__ == "__main__":
    main()
