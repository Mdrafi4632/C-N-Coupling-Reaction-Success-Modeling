#!/usr/bin/env python3
"""scaffold_kfold.py — K-fold structurally-disjoint cross-validation.

Generates K folds where each fold is structurally disjoint from the others on
the aryl halide.  Two grouping options:

  * --group-by substrate  (DEFAULT) — group by the full aryl-halide SMILES.
    Gives balanced folds (~1093 each on this dataset, 365 unique substrates).
    Strict OOD per substrate; test substrates never appear in train.
  * --group-by murcko    — group by Bemis-Murcko scaffold of the aryl halide.
    Produces wildly imbalanced folds on this dataset (88% of reactions live
    on just 2 scaffolds, c1ccccc1 and c1ccncc1).  Reproduces the original
    main.py scaffold split's grouping but is dominated by 2-3 huge folds.

Why: the single scaffold split has val=10 (largest scaffolds eat the train
allocation), which makes CFA selection unreliable and the scaffold-OOD test
metric a single noisy point estimate.  K-fold structural CV gives K
independent OOD estimates and a properly-sized val set within each fold.

Outputs:
  outputs/scaffold_kfold_reg/summary.json         (regression aggregates)
  outputs/scaffold_kfold_reg/fold_<i>/preds.npz   (per-fold predictions)
  outputs/scaffold_kfold_clf/summary.json         (classification aggregates)
  outputs/scaffold_kfold_clf/fold_<i>/preds.npz

Usage:
  uv run python scripts/scaffold_kfold.py --task regression
  uv run python scripts/scaffold_kfold.py --task classification
  uv run python scripts/scaffold_kfold.py --task regression --group-by murcko
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from main import CONFIG, load_data
from cfa import cfa_greedy, cfa_k_way
from cfa_classify import cfa_greedy_clf, cfa_k_way_clf, _best_threshold


# ── Scaffold partitioning ────────────────────────────────────────────────────

def murcko_key(smi: str) -> str:
    if not smi:
        return "invalid"
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return "invalid"


def scaffold_kfolds(df, k=5, group_by="substrate"):
    """Greedy bin-pack groups into k roughly-equal-size folds.

    group_by:
      "substrate" — full aryl-halide SMILES (recommended; balanced)
      "murcko"    — Bemis-Murcko scaffold (legacy; very imbalanced)

    Largest groups first; each goes to the smallest current fold.  Whole
    groups stay together so test-fold structures never leak into train.
    """
    groups: dict[str, list[int]] = {}
    smiles = df["aryl_halide_smiles"].fillna("")
    for i, smi in enumerate(smiles):
        key = murcko_key(smi) if group_by == "murcko" else (smi or "invalid")
        groups.setdefault(key, []).append(i)

    folds = [[] for _ in range(k)]
    sizes = [0] * k
    for _, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        target = int(np.argmin(sizes))
        folds[target].extend(idxs)
        sizes[target] += len(idxs)
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def random_train_val(train_idx, val_frac=0.15, seed=42):
    """Random val split within the training portion (not scaffold-disjoint).

    val is i.i.d. with train; CFA uses it for ensemble selection.  Test is
    still scaffold-disjoint from both — the OOD claim is preserved.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(train_idx)
    n_val = int(len(train_idx) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


# ── Model builders (match scripts/multi_seed_eval.py fast subset) ───────────

def fast_regressors(seed):
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
        ("ExtraTrees",   ExtraTreesRegressor(n_estimators=400, max_features="sqrt",
                                              n_jobs=-1, random_state=seed)),
        ("RandomForest", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                                n_jobs=-1, random_state=seed)),
        ("Ridge",        Ridge(alpha=1.0, random_state=seed)),
        ("HistGBT",      HistGradientBoostingRegressor(
                            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                            random_state=seed)),
    ]


def native_classifiers(seed):
    """Reuse build_models() from train_native_classifiers.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_native_classifiers",
        ROOT / "scripts" / "train_native_classifiers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_models(seed)


# ── Metrics ─────────────────────────────────────────────────────────────────

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
        "threshold":    float(thr),
        "val_accuracy": float(accuracy_score(y_val_bin, (scores_val >= thr).astype(int))),
        "accuracy":     float(accuracy_score(y_te_bin, y_pred)),
        "f1":           float(f1_score(y_te_bin, y_pred, average="macro", zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_te_bin, y_pred)) if len(set(y_te_bin)) > 1 else 0.0,
    }
    try:
        out["auc"] = float(roc_auc_score(y_te_bin, scores_te))
    except Exception:
        out["auc"] = float("nan")
    return out


def aggregate(per_fold):
    """list of {metric -> value} → {metric -> {mean, std, values}}"""
    keys = per_fold[0].keys()
    out  = {}
    for k in keys:
        vs = np.array([m[k] for m in per_fold], dtype=np.float64)
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


# ── CFA selection-stability summary ──────────────────────────────────────────

def cfa_stability(fold_cfa):
    """Across folds, how often is each model picked? Same combo?"""
    if not fold_cfa:
        return {}
    member_count = {}
    combo_count  = {}
    method_count = {}
    for r in fold_cfa:
        kw = r.get("kway") or {}
        for m in kw.get("members", []):
            member_count[m] = member_count.get(m, 0) + 1
        combo_str = "+".join(sorted(kw.get("members", [])))
        combo_count[combo_str]    = combo_count.get(combo_str, 0) + 1
        method_count[kw.get("method", "?")] = method_count.get(kw.get("method", "?"), 0) + 1
    return {
        "member_frequency": member_count,
        "combo_frequency":  combo_count,
        "method_frequency": method_count,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",      choices=["regression", "classification"], required=True)
    ap.add_argument("--k",         type=int, default=5)
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--group-by",  choices=["substrate", "murcko"], default="substrate",
                    dest="group_by",
                    help="Fold grouping. 'substrate' = full aryl-halide SMILES "
                         "(balanced); 'murcko' = Bemis-Murcko scaffold (legacy, "
                         "imbalanced on this dataset).")
    args = ap.parse_args()

    cfg  = CONFIG.copy()
    feat = cfg["data_featurized"]
    if not feat.exists():
        print(f"[error] {feat} not found — run main.py first.")
        sys.exit(1)
    X, y = (lambda d: (d["X"], d["y"]))(np.load(feat))
    df   = load_data(cfg["data_processed"])

    folds = scaffold_kfolds(df, k=args.k, group_by=args.group_by)
    print(f"[kfold] task={args.task} k={args.k} seed={args.seed} group_by={args.group_by}")
    print(f"[kfold] fold sizes: {[len(f) for f in folds]} (total={sum(len(f) for f in folds)})")

    outdir = ROOT / "outputs" / (
        "scaffold_kfold_reg" if args.task == "regression" else "scaffold_kfold_clf"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, list[dict]] = {}
    fold_cfa  : list[dict]            = []

    t_total = time.time()
    for fi, test_idx in enumerate(folds):
        train_idx_full = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        tr_idx, val_idx = random_train_val(train_idx_full, val_frac=0.15, seed=args.seed)

        X_tr,  y_tr  = X[tr_idx],  y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te,  y_te  = X[test_idx], y[test_idx]
        print(f"\n=== Fold {fi+1}/{args.k} ===")
        print(f"  train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

        val_preds: dict[str, np.ndarray] = {}
        tes_preds: dict[str, np.ndarray] = {}

        if args.task == "regression":
            for name, est in fast_regressors(args.seed):
                t0 = time.time()
                est.fit(X_tr, y_tr)
                val_preds[name] = est.predict(X_val).astype(np.float32)
                tes_preds[name] = est.predict(X_te ).astype(np.float32)
                m = reg_metrics(y_te, tes_preds[name])
                per_model.setdefault(name, []).append(m)
                print(f"  [{name:14s}] R²={m['r2']:+.4f}  MAE={m['mae']:.2f}  ({time.time()-t0:.1f}s)")
        else:
            y_val_bin = (y_val >= args.threshold).astype(int)
            y_te_bin  = (y_te  >= args.threshold).astype(int)
            for name, est in native_classifiers(args.seed):
                t0 = time.time()
                est.fit(X_tr, (y_tr >= args.threshold).astype(int))
                val_preds[name] = est.predict_proba(X_val)[:, 1].astype(np.float32)
                tes_preds[name] = est.predict_proba(X_te )[:, 1].astype(np.float32)
                m = clf_metrics(val_preds[name], tes_preds[name], y_val_bin, y_te_bin)
                per_model.setdefault(name, []).append(m)
                print(f"  [{name:18s}] acc={m['accuracy']:.3f}  MCC={m['mcc']:+.3f}  "
                      f"AUC={m['auc']:.3f}  ({time.time()-t0:.1f}s)")

        # Save per-fold preds for downstream re-analysis
        fold_dir = outdir / f"fold_{fi}"
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            fold_dir / "preds.npz",
            names=np.array(list(val_preds.keys())),
            val=np.stack(list(val_preds.values())),
            test=np.stack(list(tes_preds.values())),
            y_val=y_val, y_te=y_te,
            tr_idx=tr_idx, val_idx=val_idx, test_idx=test_idx,
        )

        # Run CFA on this fold (cap k at 3 for speed; pool is small)
        print(f"  [CFA fold {fi+1}]", flush=True)
        if args.task == "regression":
            try:
                g = cfa_greedy(val_preds, tes_preds, y_val, y_te, time_limit=15, max_rounds=10)
                kway = cfa_k_way(val_preds, tes_preds, y_val, y_te, max_k=3)
                best_kway = sorted(kway, key=lambda r: -r["val_r2"])[0] if kway else None
            except Exception as e:
                print(f"  [CFA error] {e}")
                g, best_kway = None, None
            entry = {"fold": fi}
            if g and "best_combo" in g:
                entry["greedy"] = {
                    "members": g["best_combo"], "method": g["best_method"],
                    "val_r2":  g["val_r2"],     "test_r2": g["test_r2"],
                }
            if best_kway:
                entry["kway"] = {
                    "k":       best_kway["k"],
                    "members": best_kway["combo"].split("+"),
                    "method":  best_kway["method"],
                    "val_r2":  best_kway["val_r2"],
                    "test_r2": best_kway["test_r2"],
                }
            fold_cfa.append(entry)
        else:
            try:
                g = cfa_greedy_clf(val_preds, tes_preds, y_val_bin, y_te_bin,
                                   time_limit=15, max_rounds=10)
                kway = cfa_k_way_clf(val_preds, tes_preds, y_val_bin, y_te_bin, max_k=3)
                best_kway = sorted(kway, key=lambda r: -r["val_accuracy"])[0] if kway else None
            except Exception as e:
                print(f"  [CFA error] {e}")
                g, best_kway = None, None
            entry = {"fold": fi}
            if g and "best_combo" in g:
                entry["greedy"] = {
                    "members":      g["best_combo"], "method": g["best_method"],
                    "val_accuracy": g["val_accuracy"], "test_accuracy": g["test_accuracy"],
                    "val_mcc":      g["val_mcc"],      "test_mcc":      g["test_mcc"],
                }
            if best_kway:
                entry["kway"] = {
                    "k":             best_kway["k"],
                    "members":       best_kway["combo"].split("+"),
                    "method":        best_kway["method"],
                    "val_accuracy":  best_kway["val_accuracy"],
                    "test_accuracy": best_kway["test_accuracy"],
                    "val_mcc":       best_kway["val_mcc"],
                    "test_mcc":      best_kway["test_mcc"],
                }
            fold_cfa.append(entry)

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}\n  Aggregated across {args.k} folds (mean ± std)\n{'='*64}")
    agg = {}
    for name, runs in per_model.items():
        agg[name] = aggregate(runs)
        if args.task == "regression":
            r2 = agg[name]["r2"]; mae = agg[name]["mae"]
            print(f"  {name:18s}  R²={r2['mean']:+.4f}±{r2['std']:.4f}  "
                  f"MAE={mae['mean']:.2f}±{mae['std']:.2f}")
        else:
            acc = agg[name]["accuracy"]; mcc = agg[name]["mcc"]; auc = agg[name]["auc"]
            print(f"  {name:18s}  acc={acc['mean']:.3f}±{acc['std']:.3f}  "
                  f"MCC={mcc['mean']:+.3f}±{mcc['std']:.3f}  "
                  f"AUC={auc['mean']:.3f}±{auc['std']:.3f}")

    # CFA aggregates
    print(f"\n  CFA k-way per-fold test metric:")
    for r in fold_cfa:
        kw = r.get("kway") or {}
        members = "+".join(kw.get("members", []))[:50]
        if args.task == "regression":
            print(f"    fold {r['fold']+1}: {members:50s}  {kw.get('method', '?'):6s}  test R²={kw.get('test_r2', float('nan')):+.4f}")
        else:
            print(f"    fold {r['fold']+1}: {members:50s}  {kw.get('method', '?'):6s}  test acc={kw.get('test_accuracy', float('nan')):.3f}  MCC={kw.get('test_mcc', float('nan')):+.3f}")

    if args.task == "regression":
        cfa_test_vals = [r["kway"]["test_r2"] for r in fold_cfa if r.get("kway")]
    else:
        cfa_test_vals = [r["kway"]["test_accuracy"] for r in fold_cfa if r.get("kway")]
    if cfa_test_vals:
        v = np.array(cfa_test_vals)
        label = "R²" if args.task == "regression" else "acc"
        print(f"\n  CFA k-way mean ± std test {label}: {v.mean():+.4f} ± {v.std(ddof=1):.4f}")

    stability = cfa_stability(fold_cfa)
    if stability.get("member_frequency"):
        print(f"\n  CFA member selection frequency (out of {args.k} folds):")
        for m, c in sorted(stability["member_frequency"].items(), key=lambda kv: -kv[1]):
            print(f"    {m:18s}  {c}/{args.k}")

    # Save summary
    out_path = outdir / "summary.json"
    json.dump({
        "task":       args.task,
        "k":          args.k,
        "seed":       args.seed,
        "group_by":   args.group_by,
        "fold_sizes": [int(len(f)) for f in folds],
        "per_model":  agg,
        "fold_cfa":   fold_cfa,
        "stability":  stability,
        "wall_time":  round(time.time() - t_total, 1),
    }, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"\n[save] {out_path}")
    print(f"[total] {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
