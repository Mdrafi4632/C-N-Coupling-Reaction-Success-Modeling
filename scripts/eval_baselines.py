#!/usr/bin/env python3
"""eval_baselines.py — Compute trivial-baseline regression metrics
(predict-mean, predict-median) on the same train/val/test split as the model
pool, and append them to outputs/<split>/preds.npz so they appear in the
regression metrics table.

Why: every reported R^2 should be benchmarked against the predict-mean floor
(R^2 = 0 by construction) so the reader knows what the architecture actually
buys above the trivial baseline. The pool currently lacks this floor anchor.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.dummy   import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from main import CONFIG, load_data, random_split_idx, scaffold_split_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    args = ap.parse_args()

    cfg  = CONFIG.copy()
    feat = cfg["data_featurized"]
    if not feat.exists():
        print(f"[error] {feat} not found — run main.py first.")
        sys.exit(1)
    X, y = (lambda d: (d["X"], d["y"]))(np.load(feat))
    df   = load_data(cfg["data_processed"])

    if args.split == "scaffold":
        tr_idx, val_idx, te_idx = scaffold_split_idx(df, cfg)
    else:
        tr_idx, val_idx, te_idx = random_split_idx(len(df), cfg)
    X_tr, y_tr   = X[tr_idx],  y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te   = X[te_idx],  y[te_idx]

    baselines = {
        "Baseline_Mean":   DummyRegressor(strategy="mean"),
        "Baseline_Median": DummyRegressor(strategy="median"),
    }
    new_val, new_test, new_names = [], [], []
    print(f"[{args.split}] train mean={y_tr.mean():.2f}  median={np.median(y_tr):.2f}  "
          f"std={y_tr.std():.2f}")
    print(f"  {'Model':18s}  {'val R²':>8s}  {'test R²':>8s}  {'test MAE':>9s}")
    for name, est in baselines.items():
        est.fit(X_tr, y_tr)
        pv = est.predict(X_val).astype(np.float32)
        pt = est.predict(X_te ).astype(np.float32)
        new_val.append(pv); new_test.append(pt); new_names.append(name)
        print(f"  {name:18s}  {r2_score(y_val, pv):+8.4f}  {r2_score(y_te, pt):+8.4f}  "
              f"{mean_absolute_error(y_te, pt):9.2f}")

    # Append to existing preds.npz so the metrics table picks them up
    src = ROOT / "outputs" / args.split / "preds.npz"
    if src.exists():
        d = np.load(src)
        existing_names = [str(n) for n in d["names"]]
        keep = [i for i, n in enumerate(existing_names) if n not in new_names]
        names = [existing_names[i] for i in keep] + new_names
        val   = np.concatenate([d["val"][keep],  np.stack(new_val)],  axis=0)
        test  = np.concatenate([d["test"][keep], np.stack(new_test)], axis=0)
        np.savez_compressed(src, names=np.array(names), val=val, test=test,
                            y_val=d["y_val"], y_te=d["y_te"])
        print(f"[save] appended {len(new_names)} baseline(s) to {src}")
    else:
        print(f"[warn] {src} not found — baselines not persisted.")


if __name__ == "__main__":
    main()
