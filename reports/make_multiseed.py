#!/usr/bin/env python3
"""multi-seed CFA pipeline aligned with Patten et al.

For each (split, seed) cell in {random,scaffold} x {0,1,42,101,256}:
  * Train ~12 fast headline regressors on the cached feature matrix.
  * Threshold-binarise predictions at val-optimal cut-off.
  * Run pairwise CD-kneedle finalist selection (clamp K=5 by val MCC).
  * Enumerate all subsets of size 2..K x 6 CFA methods; pick best by
    (a) test-argmax (Patten convention) and (b) val-argmax (prospective).
  * Track which CFA method wins per cell; selection frequency per model.

Emits:
  report/multiseed_results.json
  report/tables/multiseed_paired_wilcoxon.tex
  report/tables/multiseed_per_method.tex
  report/tables/multiseed_selection_freq.tex
  report/tables/multiseed_prospective.tex
"""
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import wilcoxon
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "cn_reactions_raw.csv"
TBL  = ROOT / "report" / "tables"
TBL.mkdir(exist_ok=True, parents=True)

# ── Featurisation ────────────────────────────────────────────────────────────

SMILES_COLS = ["aryl_halide_smiles", "amine_smiles", "catalyst_smiles",
               "base_smiles", "additive_smiles", "solvent_smiles"]
NBITS = 2048
RADIUS = 2

def smiles_to_morgan(s):
    if not isinstance(s, str) or not s:
        return np.zeros(NBITS, dtype=np.float32)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(NBITS, dtype=np.float32)
    arr = np.zeros(NBITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS), arr)
    return arr

def featurize(df):
    parts = [np.stack([smiles_to_morgan(s) for s in df[c].fillna("")])
             for c in SMILES_COLS if c in df.columns]
    X = np.concatenate(parts, axis=1)
    if "temperature_c" in df.columns:
        t = df["temperature_c"].where(df["temperature_c"].between(-100, 500))
        t = t.fillna(t.median())
        t = (t - t.mean()) / (t.std() + 1e-8)
        X = np.concatenate([X, t.values.reshape(-1, 1).astype(np.float32)], axis=1)
    return X

# ── Splits ───────────────────────────────────────────────────────────────────

def random_split(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    nv, nt = int(0.15*n), int(0.15*n)
    return idx[: n-nv-nt], idx[n-nv-nt: n-nt], idx[n-nt:]

def scaffold_split(df, seed):
    groups = {}
    for i, s in enumerate(df["aryl_halide_smiles"].fillna("")):
        try:
            mol = Chem.MolFromSmiles(s) if s else None
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "invalid"
        except Exception:
            key = "invalid"
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)  # randomise group ordering by seed
    n = len(df)
    train, val, test = [], [], []
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        f = len(train) / n
        if f < 0.70:
            train.extend(groups[k])
        elif f < 0.85:
            val.extend(groups[k])
        else:
            test.extend(groups[k])
    return np.array(train), np.array(val if val else train[:10]), np.array(test if test else train[:10])

# ── Model pool (fast subset, all-sklearn for portability) ────────────────────

import lightgbm as lgb
import xgboost as xgb

def make_pool(seed):
    """Fast pool only (boosted trees + Ridge + projection methods).
    Total per-cell wall-clock target: <60s."""
    from sklearn.decomposition import TruncatedSVD
    return {
        "LightGBM":      lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                           num_leaves=63, subsample=0.8,
                                           colsample_bytree=0.8,
                                           random_state=seed, n_jobs=-1, verbose=-1),
        "LGB_MAE":       lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                           num_leaves=63, subsample=0.8,
                                           colsample_bytree=0.8, objective="mae",
                                           random_state=seed, n_jobs=-1, verbose=-1),
        "XGBoost":       xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, tree_method="hist",
                                          random_state=seed, n_jobs=-1, verbosity=0),
        "HistGBT":       HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                       random_state=seed),
        "ExtraTrees":    ExtraTreesRegressor(n_estimators=100, max_depth=None,
                                             min_samples_leaf=2, n_jobs=-1,
                                             random_state=seed),
        "Ridge":         Ridge(alpha=1.0, random_state=seed),
        "TSVD_Ridge_100":Pipeline([("svd", TruncatedSVD(n_components=100, random_state=seed)),
                                   ("rd", Ridge(alpha=1.0))]),
        "KernelRidge":   Pipeline([("svd", TruncatedSVD(n_components=100, random_state=seed)),
                                   ("sc", StandardScaler()),
                                   ("kr", KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3))]),
    }

# ── CFA primitives (regression scores → classification by val-opt threshold) ─

METHODS = ["ASC", "ARC", "WSCDS", "WRCDS", "WSCP", "WRCP"]

def normalize_minmax(v, lo=None, hi=None):
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    d = hi - lo
    return np.zeros_like(v, dtype=np.float64) if d == 0 else (v - lo) / d

def best_threshold(scores, y_bin):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_bin, dtype=np.int8)
    n = len(s)
    if n == 0 or np.std(s) < 1e-12:
        return 0.5
    order = np.argsort(-s)
    s_s, y_s = s[order], y[order]
    cum_pos = np.cumsum(y_s); total_pos = cum_pos[-1]; total_neg = n - total_pos
    fp = (np.arange(1, n+1) - cum_pos)
    tp = cum_pos
    tn = total_neg - fp
    correct = tp + tn
    best = int(np.argmax(correct))
    if total_neg > correct[best]:
        return float(s_s[0]) + 1e-6
    return float(s_s[best])

def cd_matrix(norm_val):
    cols = list(norm_val.columns)
    rsc = {c: np.sort(norm_val[c].values)[::-1] for c in cols}
    cd = {}
    for a, b in itertools.combinations(cols, 2):
        cd[tuple(sorted([a, b]))] = float(np.sqrt(np.mean((rsc[a]-rsc[b])**2)))
    ds = {c: float(np.mean([cd[tuple(sorted([c, o]))] for o in cols if o != c]))
          for c in cols}
    return cd, ds

def combine(preds_df, method, ds, perf):
    cols = list(preds_df.columns)
    n = len(preds_df)
    ranks = preds_df.rank(ascending=True).values / n
    vals  = preds_df.values
    if method == "ASC": return vals.mean(axis=1)
    if method == "ARC": return ranks.mean(axis=1)
    raw = (np.array([ds.get(c, 1e-9) for c in cols]) if "DS" in method
           else np.array([max(perf.get(c, 0.0), 1e-9) for c in cols]))
    w = raw / (raw.sum() + 1e-12)
    return vals @ w if method in ("WSCDS", "WSCP") else ranks @ w

# CD-kneedle elbow: find argmax of (1 - i/(n-1) - normalised_y_i) on sorted-desc CD curve
def cd_kneedle(cd_dict, K_cap=5, perf_dict=None):
    pairs = sorted(cd_dict.items(), key=lambda kv: -kv[1])
    if not pairs:
        return []
    vals = np.array([v for _, v in pairs])
    n = len(vals)
    lo, hi = vals.min(), vals.max()
    yn = (vals - lo) / (hi - lo + 1e-12)  # normalised y
    xn = np.arange(n) / max(1, n-1)       # normalised x in [0,1]
    diff = (1 - xn) - yn
    elbow = int(np.argmax(diff))
    selected = set()
    for (a, b), v in pairs[: max(1, elbow)]:
        selected.add(a); selected.add(b)
    sel = list(selected)
    if perf_dict is not None and len(sel) > K_cap:
        sel = sorted(sel, key=lambda c: -perf_dict.get(c, 0.0))[:K_cap]
    return sel

def cfa_classify_cell(val_preds, tes_preds, y_val_bin, y_te_bin, K_cap=5):
    """Run CFA enumeration for a single (split, seed) cell.
    Returns dict with best-individual, val-argmax fused, test-argmax fused,
    per-method best, finalist set, and best-subset model membership."""
    usable = list(val_preds.keys())
    if len(usable) < 2:
        return None

    norm_val = pd.DataFrame(index=range(len(y_val_bin)))
    norm_tes = pd.DataFrame(index=range(len(y_te_bin)))
    for c in usable:
        lo, hi = float(val_preds[c].min()), float(val_preds[c].max())
        norm_val[c] = normalize_minmax(val_preds[c], lo, hi)
        norm_tes[c] = normalize_minmax(tes_preds[c], lo, hi)

    perf = {}
    for c in usable:
        thr = best_threshold(norm_val[c].values, y_val_bin)
        perf[c] = matthews_corrcoef(y_val_bin,
                                    (norm_val[c].values >= thr).astype(int))
    eligible = [c for c in usable if perf[c] > 0]
    if len(eligible) < 2:
        return None

    cd, ds = cd_matrix(norm_val[eligible])

    # CD-kneedle finalist selection
    finalists = cd_kneedle(cd, K_cap=K_cap, perf_dict=perf)
    if len(finalists) < 2:
        # fallback to top-K by perf
        finalists = sorted(eligible, key=lambda c: -perf[c])[:K_cap]

    # Best individual on test (using val-optimal thr for that model)
    best_ind = max(usable, key=lambda c: perf[c])
    val_thr  = best_threshold(norm_val[best_ind].values, y_val_bin)
    best_ind_test_acc = accuracy_score(y_te_bin,
                                       (norm_tes[best_ind].values >= val_thr).astype(int))
    val_best_ind = max(usable, key=lambda c: accuracy_score(y_val_bin,
        (norm_val[c].values >= best_threshold(norm_val[c].values, y_val_bin)).astype(int)))
    val_thr2 = best_threshold(norm_val[val_best_ind].values, y_val_bin)
    val_best_ind_test_acc = accuracy_score(y_te_bin,
                                           (norm_tes[val_best_ind].values >= val_thr2).astype(int))

    # Enumerate all subsets r=2..K of finalists × 6 methods
    enumerated = []
    K = len(finalists)
    for r in range(2, K+1):
        for combo in itertools.combinations(finalists, r):
            sub_v = norm_val[list(combo)]; sub_t = norm_tes[list(combo)]
            sd = {c: ds.get(c, 1e-9) for c in combo}
            sp = {c: perf.get(c, 0.0) for c in combo}
            for method in METHODS:
                cv = combine(sub_v, method, sd, sp)
                ct = combine(sub_t, method, sd, sp)
                thr = best_threshold(cv, y_val_bin)
                v_acc = accuracy_score(y_val_bin, (cv >= thr).astype(int))
                t_acc = accuracy_score(y_te_bin,  (ct >= thr).astype(int))
                v_mcc = matthews_corrcoef(y_val_bin, (cv >= thr).astype(int))
                t_mcc = matthews_corrcoef(y_te_bin,  (ct >= thr).astype(int))
                enumerated.append({
                    "combo":  list(combo),
                    "method": method,
                    "val_acc": float(v_acc),
                    "test_acc": float(t_acc),
                    "val_mcc": float(v_mcc),
                    "test_mcc": float(t_mcc),
                })

    # Test-argmax (Patten convention)
    test_best = max(enumerated, key=lambda r: r["test_acc"])
    # Prospective val-argmax
    prosp_best = max(enumerated, key=lambda r: r["val_acc"])
    # Per-method best
    per_method = {}
    for m in METHODS:
        ms = [r for r in enumerated if r["method"] == m]
        if ms:
            per_method[m] = max(ms, key=lambda r: r["test_acc"])

    return {
        "best_ind":              val_best_ind,
        "best_ind_test_acc":     float(val_best_ind_test_acc),
        "best_ind_val_perf":     float(perf[val_best_ind]),
        "test_argmax":           test_best,
        "prospective":           prosp_best,
        "per_method":            per_method,
        "finalists":             finalists,
        "n_finalists":           K,
        "n_eligible":            len(eligible),
    }

# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = pd.read_csv(DATA).dropna(subset=["yield"]).reset_index(drop=True)
    df["yield"] = df["yield"].clip(0, 100)
    y = df["yield"].values.astype(np.float32)
    print(f"  n={len(df)}")

    print("Featurising...")
    X_path = ROOT / "data" / "processed" / "cn_features_quick.npz"
    if X_path.exists():
        X = np.load(X_path)["X"]
        print(f"  loaded cache {X.shape}")
    else:
        X = featurize(df)
        np.savez_compressed(X_path, X=X)
        print(f"  saved {X.shape}")

    SEEDS = [0, 1, 42, 101, 256]
    SPLITS = ["random", "scaffold"]
    cells = []
    t0 = time.time()
    for split in SPLITS:
        for seed in SEEDS:
            print(f"\n=== split={split} seed={seed} ===")
            if split == "random":
                tr, val, te = random_split(len(df), seed)
            else:
                tr, val, te = scaffold_split(df, seed)
            print(f"  sizes: tr={len(tr)} val={len(val)} te={len(te)}")
            y_val_bin = (y[val] >= 50).astype(int)
            y_te_bin  = (y[te]  >= 50).astype(int)

            pool = make_pool(seed)
            val_preds, tes_preds = {}, {}
            for name, mdl in pool.items():
                t1 = time.time()
                try:
                    mdl.fit(X[tr], y[tr])
                    val_preds[name] = mdl.predict(X[val])
                    tes_preds[name] = mdl.predict(X[te])
                except Exception as e:
                    print(f"    [skip] {name}: {e}")
                    continue
                t2 = time.time()
                print(f"  {name:14s}  {t2-t1:5.1f}s")

            cell = cfa_classify_cell(val_preds, tes_preds, y_val_bin, y_te_bin, K_cap=5)
            if cell is None:
                print("  [skip] no eligible models")
                continue
            cell["split"] = split
            cell["seed"]  = seed
            cells.append(cell)
            ti = cell["test_argmax"]; pi = cell["prospective"]
            print(f"  best_ind={cell['best_ind']:14s} test={cell['best_ind_test_acc']:.4f}  "
                  f"test-argmax={ti['test_acc']:.4f} ({ti['method']})  "
                  f"prosp={pi['test_acc']:.4f} ({pi['method']})  "
                  f"|fin|={cell['n_finalists']}")
    print(f"\nTotal time: {(time.time()-t0):.0f}s for {len(cells)} cells")

    # ── Aggregation ──────────────────────────────────────────────────────────
    out = {"cells": cells, "seeds": SEEDS, "splits": SPLITS}

    # Paired Wilcoxon
    rows = []
    for split in SPLITS:
        cs = [c for c in cells if c["split"] == split]
        bi   = np.array([c["best_ind_test_acc"]    for c in cs])
        ta   = np.array([c["test_argmax"]["test_acc"] for c in cs])
        pa   = np.array([c["prospective"]["test_acc"] for c in cs])
        d_ta = ta - bi
        d_pa = pa - bi
        if len(d_ta) >= 2 and not np.all(d_ta == 0):
            wta = wilcoxon(d_ta, alternative="greater", zero_method="wilcox")
            p_ta = float(wta.pvalue)
        else:
            p_ta = float("nan")
        if len(d_pa) >= 2 and not np.all(d_pa == 0):
            wpa = wilcoxon(d_pa, alternative="greater", zero_method="wilcox")
            p_pa = float(wpa.pvalue)
        else:
            p_pa = float("nan")
        rows.append({
            "split": split,
            "n_cells": len(cs),
            "mean_best_ind":  float(bi.mean()),
            "mean_test_argmax": float(ta.mean()),
            "mean_prospective": float(pa.mean()),
            "lift_test_argmax_pp": float((ta - bi).mean() * 100),
            "lift_prospective_pp": float((pa - bi).mean() * 100),
            "p_test_argmax_wilcoxon": p_ta,
            "p_prospective_wilcoxon": p_pa,
        })
    out["wilcoxon"] = rows
    print("\n=== Wilcoxon ===")
    for r in rows:
        print(f"  {r['split']:9s}  test-argmax +{r['lift_test_argmax_pp']:.2f}pp p={r['p_test_argmax_wilcoxon']:.4f}  "
              f"prosp +{r['lift_prospective_pp']:.2f}pp p={r['p_prospective_wilcoxon']:.4f}")

    # Per-method matrix
    pm_rows = []
    for split in SPLITS:
        cs = [c for c in cells if c["split"] == split]
        for m in METHODS:
            accs = [c["per_method"].get(m, {}).get("test_acc", np.nan) for c in cs]
            pm_rows.append({
                "split":  split, "method": m,
                "mean_test_acc": float(np.nanmean(accs)),
                "std_test_acc":  float(np.nanstd(accs)),
                "n_cells": int(np.sum(~np.isnan(accs))),
            })
    out["per_method"] = pm_rows

    # Per-model selection frequency (best-test-argmax membership)
    sel = {}
    for c in cells:
        for m in c["test_argmax"]["combo"]:
            sel[m] = sel.get(m, 0) + 1
    freq = {}
    for c in cells:
        for m in c["finalists"]:
            freq[m] = freq.get(m, 0) + 1
    out["selection_freq"] = freq
    out["best_membership"] = sel

    json_out = ROOT / "report" / "multiseed_results.json"
    json_out.write_text(json.dumps(out, indent=2, default=lambda x: list(x) if isinstance(x, np.ndarray) else x))
    print(f"\nSaved {json_out}")

    # ── LaTeX tables ─────────────────────────────────────────────────────────

    # Wilcoxon table
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Split & cells & best-ind & test-argmax & $\Delta_{ta}$ & prospective & $\Delta_{prosp}$ \\",
             r"\midrule"]
    for r in rows:
        p_ta = r["p_test_argmax_wilcoxon"]
        p_pa = r["p_prospective_wilcoxon"]
        p_ta_s = "$<\\!10^{-4}$" if (p_ta == p_ta and p_ta < 1e-4) else (f"{p_ta:.4f}" if p_ta == p_ta else "n/a")
        p_pa_s = "$<\\!10^{-4}$" if (p_pa == p_pa and p_pa < 1e-4) else (f"{p_pa:.4f}" if p_pa == p_pa else "n/a")
        lines.append(
            f"{r['split']} & {r['n_cells']} & {r['mean_best_ind']:.4f} & "
            f"{r['mean_test_argmax']:.4f} & "
            f"+{r['lift_test_argmax_pp']:.2f}\\,pp ($p\\!=\\!{p_ta_s}$) & "
            f"{r['mean_prospective']:.4f} & "
            f"+{r['lift_prospective_pp']:.2f}\\,pp ($p\\!=\\!{p_pa_s}$) \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL / "multiseed_paired_wilcoxon.tex").write_text("\n".join(lines))
    print(f"  wrote {TBL/'multiseed_paired_wilcoxon.tex'}")

    # Per-method matrix
    lines = [r"\begin{tabular}{l" + "r"*len(METHODS) + "}", r"\toprule",
             r"Split & " + " & ".join(METHODS) + r" \\", r"\midrule"]
    for split in SPLITS:
        row = [split]
        ms = [r for r in pm_rows if r["split"] == split]
        ms_d = {r["method"]: r for r in ms}
        # mark best
        accs = {m: ms_d[m]["mean_test_acc"] for m in METHODS}
        best = max(accs, key=lambda m: accs[m])
        for m in METHODS:
            v = ms_d[m]["mean_test_acc"]
            cell_s = f"\\textbf{{{v:.4f}}}" if m == best else f"{v:.4f}"
            row.append(cell_s)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL / "multiseed_per_method.tex").write_text("\n".join(lines))
    print(f"  wrote {TBL/'multiseed_per_method.tex'}")

    # Per-model selection-freq table (top 10)
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Model & finalist count & best-fused count \\", r"\midrule"]
    n_total = len(cells)
    all_models = sorted(set(list(freq.keys()) + list(sel.keys())),
                        key=lambda m: -(freq.get(m, 0) + sel.get(m, 0)))
    for m in all_models[:10]:
        f = freq.get(m, 0); s = sel.get(m, 0)
        lines.append(f"{m.replace('_', ' ')} & {f}/{n_total} & {s}/{n_total} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL / "multiseed_selection_freq.tex").write_text("\n".join(lines))
    print(f"  wrote {TBL/'multiseed_selection_freq.tex'}")

    print("\nAll multi-seed artefacts written.")

if __name__ == "__main__":
    main()
