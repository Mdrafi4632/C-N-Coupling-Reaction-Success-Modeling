"""cfa_classify.py — Combinatorial Fusion Analysis for binary classification.

Reverts the regression CFA (cfa.py) to its classification form. We treat each
existing regressor's continuous output as a *score*: high score = predicted
high yield = predicted "success" (class 1). The score is min-max normalised on
the validation set's range and combined across models with the standard 6 CFA
methods (ASC, ARC, WSCDS, WRCDS, WSCP, WRCP), then thresholded at the
validation-optimal cut-off (Youden's J or accuracy-maximising — we use the
accuracy-maximising threshold here for parity with the reference run.py).

Eligibility filter: MCC > 0 on validation (model is at least correlated with
the truth).  Models below this filter are excluded from the pool (they would
add noise without signal).

Public API:
  cfa_greedy_clf(val_preds, tes_preds, y_val_bin, y_te_bin, time_limit, max_rounds)
  cfa_pairwise_clf(val_preds, tes_preds, y_val_bin, y_te_bin)
  cfa_k_way_clf(val_preds, tes_preds, y_val_bin, y_te_bin, max_k=5, ...)
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

METHODS = ["ASC", "ARC", "WSCDS", "WRCDS", "WSCP", "WRCP"]


# ── Primitive operations (shared with regression cfa) ────────────────────────

def normalize_minmax(values, ref_min=None, ref_max=None):
    lo = float(np.min(values)) if ref_min is None else ref_min
    hi = float(np.max(values)) if ref_max is None else ref_max
    d = hi - lo
    return np.zeros_like(values, dtype=np.float64) if d == 0 else (values - lo) / d


def rank_score_curve(values):
    return np.sort(normalize_minmax(values))[::-1]


def cognitive_diversity(rsc_a, rsc_b):
    n = len(rsc_a)
    return 0.0 if n == 0 else float(np.sqrt(np.sum((rsc_a - rsc_b) ** 2) / n))


def _cd_matrix(norm_val: pd.DataFrame):
    cols = list(norm_val.columns)
    rsc = {c: rank_score_curve(norm_val[c].values) for c in cols}
    cd, ds = {}, {}
    for a, b in itertools.combinations(cols, 2):
        cd[tuple(sorted([a, b]))] = cognitive_diversity(rsc[a], rsc[b])
    for c in cols:
        others = [cd[tuple(sorted([c, o]))] for o in cols if o != c]
        ds[c] = float(np.mean(others)) if others else 0.0
    return cd, ds, rsc


def _combine(preds_df, method, ds, perf):
    cols = list(preds_df.columns)
    n    = len(preds_df)
    ranks = preds_df.rank(ascending=True).values / n
    vals  = preds_df.values
    if method == "ASC": return vals.mean(axis=1)
    if method == "ARC": return ranks.mean(axis=1)
    raw_w = np.array([ds.get(c, 1e-9) for c in cols]) if "DS" in method \
            else np.array([max(perf.get(c, 0.0), 1e-9) for c in cols])
    w = raw_w / (raw_w.sum() + 1e-12)
    return vals @ w if method in ("WSCDS", "WSCP") else ranks @ w


def _best_threshold(scores, y_bin):
    """Return the threshold on `scores` that maximises binary accuracy.

    O(N log N) — sort by score descending, sweep the cumulative TP/FP
    curve once.  Equivalent to the old quantile sweep but ~50× faster.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_bin,  dtype=np.int8)
    n = len(s)
    if n == 0 or np.std(s) < 1e-12:
        return 0.5
    order = np.argsort(-s)              # high → low
    s_sorted = s[order]
    y_sorted = y[order]
    cum_pos  = np.cumsum(y_sorted)      # TP if we predict positive at this rank
    total_pos = cum_pos[-1]
    total_neg = n - total_pos
    fp = (np.arange(1, n + 1) - cum_pos)  # cumulative FP
    tp = cum_pos
    tn = total_neg - fp                 # rest of negatives below threshold
    fn = total_pos - tp
    correct = tp + tn                   # accuracy * n
    # Threshold "at rank k" means predict positive for top-k by score.
    # The corresponding cut value is s_sorted[k-1] (the k-th largest score).
    best = int(np.argmax(correct))
    thr  = float(s_sorted[best])
    # Guard against k=0 (predict-all-negative): correct = total_neg
    if total_neg > correct[best]:
        return float(s_sorted[0]) + 1e-6
    return thr


def _classify_metrics(scores, y_bin, threshold=None):
    if threshold is None:
        threshold = _best_threshold(scores, y_bin)
    y_pred = (scores >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(y_bin, y_pred)),
        "f1_macro":  float(f1_score(y_bin, y_pred, average="macro", zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_bin, y_pred)) if len(set(y_bin)) > 1 else 0.0,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_bin, scores))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


# ── Per-model evaluation ─────────────────────────────────────────────────────

def per_model_metrics(val_preds, tes_preds, y_val_bin, y_te_bin):
    """Return dict[model -> {val_*, test_*, threshold_val}]."""
    out = {}
    for name in val_preds:
        thr = _best_threshold(val_preds[name], y_val_bin)
        m_v = _classify_metrics(val_preds[name], y_val_bin, thr)
        m_t = _classify_metrics(tes_preds[name], y_te_bin,  thr)
        out[name] = {
            "threshold": thr,
            "val_accuracy":  m_v["accuracy"], "val_f1":   m_v["f1_macro"],
            "val_mcc":       m_v["mcc"],       "val_auc":  m_v["roc_auc"],
            "test_accuracy": m_t["accuracy"], "test_f1":  m_t["f1_macro"],
            "test_mcc":      m_t["mcc"],       "test_auc": m_t["roc_auc"],
        }
    return out


# ── Greedy forward selection ──────────────────────────────────────────────────

def cfa_greedy_clf(val_preds, tes_preds, y_val_bin, y_te_bin,
                   time_limit=60.0, max_rounds=200):
    usable = list(val_preds.keys())
    if len(usable) < 2:
        return {}

    norm_val = pd.DataFrame(index=range(len(y_val_bin)))
    norm_tes = pd.DataFrame(index=range(len(y_te_bin)))
    for c in usable:
        lo, hi = float(val_preds[c].min()), float(val_preds[c].max())
        norm_val[c] = normalize_minmax(val_preds[c], lo, hi)
        norm_tes[c] = normalize_minmax(tes_preds[c], lo, hi)

    _, ds, _ = _cd_matrix(norm_val)
    perf     = {c: matthews_corrcoef(y_val_bin,
                  (norm_val[c].values >= _best_threshold(norm_val[c].values, y_val_bin)).astype(int))
                if len(set(y_val_bin)) > 1 else 0.0
                for c in usable}

    eligible = [c for c, m in perf.items() if m > 0]
    if not eligible:
        return {}
    ranked = sorted([(c, perf[c]) for c in eligible], key=lambda x: -x[1])
    best_model = ranked[0][0]
    best_combo = [best_model]
    best_thr   = _best_threshold(norm_val[best_model].values, y_val_bin)
    best_acc   = accuracy_score(y_val_bin,
                                (norm_val[best_model].values >= best_thr).astype(int))
    best_method = "ASC"
    pool = [c for c, _ in ranked if c != best_model]

    rounds = []
    t0 = time.time()
    for rnd in range(1, max_rounds + 1):
        if time.time() - t0 > time_limit or not pool:
            break
        gain, add_name, add_method, add_acc, add_thr = 0.0, None, None, best_acc, best_thr
        for cand in pool:
            subset   = best_combo + [cand]
            sub_val  = norm_val[subset]
            sub_ds   = {m: ds.get(m, 1e-9) for m in subset}
            sub_perf = {m: perf.get(m, 0.0)  for m in subset}
            for method in METHODS:
                combined = _combine(sub_val, method, sub_ds, sub_perf)
                thr = _best_threshold(combined, y_val_bin)
                acc = accuracy_score(y_val_bin, (combined >= thr).astype(int))
                if acc - best_acc > gain:
                    gain, add_name, add_method, add_acc, add_thr = (
                        acc - best_acc, cand, method, acc, thr
                    )
        if gain <= 0 or add_name is None:
            break
        best_combo.append(add_name)
        best_acc, best_method, best_thr = add_acc, add_method, add_thr
        pool.remove(add_name)
        sub_tes  = norm_tes[best_combo]
        sub_ds   = {m: ds.get(m, 1e-9) for m in best_combo}
        sub_perf = {m: perf.get(m, 0.0)  for m in best_combo}
        t_combined = _combine(sub_tes, best_method, sub_ds, sub_perf)
        t_acc = accuracy_score(y_te_bin, (t_combined >= best_thr).astype(int))
        rounds.append({
            "round": rnd, "added": add_name, "combo": "+".join(best_combo),
            "method": best_method, "n": len(best_combo),
            "val_accuracy": round(best_acc, 6),
            "test_accuracy": round(t_acc, 6), "threshold": round(best_thr, 6),
        })

    sub_ds   = {m: ds.get(m, 1e-9) for m in best_combo}
    sub_perf = {m: perf.get(m, 0.0)  for m in best_combo}
    val_score = _combine(norm_val[best_combo], best_method, sub_ds, sub_perf)
    tes_score = _combine(norm_tes[best_combo], best_method, sub_ds, sub_perf)
    val_pred = (val_score >= best_thr).astype(int)
    tes_pred = (tes_score >= best_thr).astype(int)
    return {
        "rounds": rounds,
        "best_combo": best_combo, "best_method": best_method,
        "threshold": round(best_thr, 6),
        "val_accuracy":  round(accuracy_score(y_val_bin, val_pred), 6),
        "val_mcc":       round(matthews_corrcoef(y_val_bin, val_pred), 6),
        "test_accuracy": round(accuracy_score(y_te_bin,  tes_pred), 6),
        "test_mcc":      round(matthews_corrcoef(y_te_bin,  tes_pred), 6),
        "test_f1":       round(f1_score(y_te_bin, tes_pred, average="macro"), 6),
    }


# ── k-way exhaustive ──────────────────────────────────────────────────────────

def cfa_k_way_clf(val_preds, tes_preds, y_val_bin, y_te_bin,
                  max_k=5, top_n_k3=25, top_n_k4=20, top_n_k5=15):
    usable = list(val_preds.keys())
    if len(usable) < 2:
        return []

    norm_val = pd.DataFrame(index=range(len(y_val_bin)))
    norm_tes = pd.DataFrame(index=range(len(y_te_bin)))
    for c in usable:
        lo, hi = float(val_preds[c].min()), float(val_preds[c].max())
        norm_val[c] = normalize_minmax(val_preds[c], lo, hi)
        norm_tes[c] = normalize_minmax(tes_preds[c], lo, hi)

    perf = {}
    for c in usable:
        thr = _best_threshold(norm_val[c].values, y_val_bin)
        perf[c] = matthews_corrcoef(y_val_bin,
                                    (norm_val[c].values >= thr).astype(int))
    eligible = sorted([c for c in usable if perf[c] > 0], key=lambda c: -perf[c])
    if len(eligible) < 2:
        return []

    cd, ds, _ = _cd_matrix(norm_val[eligible])
    results = []

    def _eval_subset(subset, methods=METHODS):
        sub_v   = norm_val[list(subset)]
        sub_t   = norm_tes[list(subset)]
        sd      = {c: ds.get(c, 1e-9)   for c in subset}
        sp      = {c: perf.get(c, 0.0) for c in subset}
        pair_cd = np.mean([cd.get(tuple(sorted([a, b])), 0.0)
                           for a, b in itertools.combinations(subset, 2)]) if len(subset) > 1 else 0.0
        for method in methods:
            cv = _combine(sub_v, method, sd, sp)
            ct = _combine(sub_t, method, sd, sp)
            thr = _best_threshold(cv, y_val_bin)
            v_pred = (cv >= thr).astype(int)
            t_pred = (ct >= thr).astype(int)
            results.append({
                "k": len(subset), "combo": "+".join(subset), "method": method,
                "group_cd": round(float(pair_cd), 6),
                "threshold": round(float(thr), 6),
                "val_accuracy":  round(accuracy_score(y_val_bin, v_pred), 6),
                "val_mcc":       round(matthews_corrcoef(y_val_bin, v_pred), 6),
                "test_accuracy": round(accuracy_score(y_te_bin,  t_pred), 6),
                "test_mcc":      round(matthews_corrcoef(y_te_bin,  t_pred), 6),
                "test_f1":       round(f1_score(y_te_bin, t_pred, average="macro"), 6),
            })

    # k=2: exhaustive
    for a, b in itertools.combinations(eligible, 2):
        _eval_subset((a, b))

    # k=3: top_n_k3 by val MCC; combinations within
    if max_k >= 3:
        for combo in itertools.combinations(eligible[:top_n_k3], 3):
            _eval_subset(combo)
    if max_k >= 4:
        for combo in itertools.combinations(eligible[:top_n_k4], 4):
            _eval_subset(combo)
    if max_k >= 5:
        for combo in itertools.combinations(eligible[:top_n_k5], 5):
            _eval_subset(combo)

    return results
