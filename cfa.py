"""
cfa.py — Combinatorial Fusion Analysis for continuous yield prediction.
Adapted from the classification version in run.py (stock price direction).
Core theory: Lin et al. (2007), "Rank-Based Combinatorial Fusion".

Key changes vs. the classification original:
  - Metric is R² (scale-invariant, continuous) instead of accuracy on binarized predictions.
  - Rank direction is ascending=True so rank n = highest predicted yield (positive R² alignment).
  - "MCC > 0" eligibility filter replaced by "R² > 0".
  - Combination methods return [0,1]-normalised scores; R² is scale-invariant so this
    is directly comparable to individual-model R² computed on the same y_true.

Public API:
  cfa_greedy(val_preds, tes_preds, y_val, y_te, time_limit, max_rounds) -> dict
  cfa_k_way(val_preds, tes_preds, y_val, y_te, max_k, ...)               -> list[dict]
  cfa_pairwise(val_preds, tes_preds, y_val, y_te)                        -> list[dict]  (wrapper: k=2)

All functions accept:
  val_preds / tes_preds : dict[model_name -> np.ndarray(float32)]  continuous yield predictions
  y_val / y_te          : np.ndarray  ground-truth yields (any scale; R² is scale-invariant)

k-way generalisation notes:
  - Group cognitive diversity for k > 2 = mean of all C(k,2) pairwise CDs within the group.
  - Combination formulas (ASC, ARC, WSCDS, WRCDS, WSCP, WRCP) are unchanged: they operate
    on (n_samples × k) matrices via column-wise means / weighted means, naturally extending
    from k=2 to any k.
  - Pool is capped at top-N eligible models for k ≥ 3 to keep evaluation count tractable.
"""

import itertools
import time
import warnings
from math import comb as _comb

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Constants ────────────────────────────────────────────────────────────────

METHODS = ["ASC", "ARC", "WSCDS", "WRCDS", "WSCP", "WRCP"]

# ── Primitive operations ─────────────────────────────────────────────────────

def normalize_minmax(values: np.ndarray,
                     ref_min: float | None = None,
                     ref_max: float | None = None) -> np.ndarray:
    """Min-max normalise to [0, 1]. Uses values' own range if refs not given."""
    lo = float(np.min(values)) if ref_min is None else ref_min
    hi = float(np.max(values)) if ref_max is None else ref_max
    d = hi - lo
    if d == 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / d


def rank_score_curve(values: np.ndarray) -> np.ndarray:
    """RSC: sort the normalised scores in descending order.
    Encodes the overall score distribution of a model, independent of
    which sample gets which score — used purely for diversity measurement."""
    return np.sort(normalize_minmax(values))[::-1]


def cognitive_diversity(rsc_a: np.ndarray, rsc_b: np.ndarray) -> float:
    """Root-mean-square difference between two rank-score curves.
    Zero = identical score distributions; higher = more complementary predictions."""
    n = len(rsc_a)
    if n == 0:
        return 0.0
    return float(np.sqrt(np.sum((rsc_a - rsc_b) ** 2) / n))


def _cd_matrix(norm_val: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Compute pairwise cognitive diversity and per-model mean diversity score.

    Returns:
        cd  : {(model_a, model_b) sorted tuple -> float}  pairwise CD
        ds  : {model_name -> float}  mean CD vs all other models
        rsc : {model_name -> np.ndarray}  rank-score curves (for reference)
    """
    cols = list(norm_val.columns)
    rsc = {c: rank_score_curve(norm_val[c].values) for c in cols}
    cd: dict = {}
    for a, b in itertools.combinations(cols, 2):
        key = tuple(sorted([a, b]))
        cd[key] = cognitive_diversity(rsc[a], rsc[b])
    ds: dict = {}
    for c in cols:
        others = [cd[tuple(sorted([c, o]))] for o in cols if o != c]
        ds[c] = float(np.mean(others)) if others else 0.0
    return cd, ds, rsc


# ── Combination methods ───────────────────────────────────────────────────────

def _combine(preds_df: pd.DataFrame, method: str,
             ds: dict, perf: dict) -> np.ndarray:
    """Combine a subset of normalised predictions using one of the 6 CFA methods.

    Rank-based methods use ascending=True so rank=n corresponds to the highest
    predicted yield (positive alignment with y_true for accurate models).
    Ranks are divided by n so they lie in (0, 1], keeping the same scale as
    score-based methods when evaluating with R².

    Args:
        preds_df : (n_samples × n_models) DataFrame of normalised predictions [0,1]
        method   : one of METHODS
        ds       : per-model diversity score (from _cd_matrix)
        perf     : per-model R² on validation set
    Returns:
        1-D array of combined scores, in approximately [0, 1]
    """
    cols = list(preds_df.columns)
    n    = len(preds_df)

    # Normalised ranks in (0, 1]: rank n = highest predicted yield
    ranks = preds_df.rank(ascending=True).values / n   # shape (n, k)
    vals  = preds_df.values                             # shape (n, k)

    if method == "ASC":
        return vals.mean(axis=1)

    if method == "ARC":
        return ranks.mean(axis=1)

    # Weighted methods — choose weights
    if "DS" in method:
        raw_w = np.array([ds.get(c, 1e-9) for c in cols])
    else:  # "P" suffix → performance-weighted
        raw_w = np.array([max(perf.get(c, 0.0), 1e-9) for c in cols])

    w = raw_w / (raw_w.sum() + 1e-12)

    if method in ("WSCDS", "WSCP"):
        return vals @ w
    else:   # WRCDS, WRCP
        return ranks @ w


def _evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Scale-invariant R²: signed Pearson r².

    Why: CFA combination methods produce scores in [0,1] but y_true is in [0,100].
    Standard R² = 1 - SS_res/SS_tot would be wildly negative just from the scale
    mismatch. Pearson r² captures the underlying *correlation* — exactly what we
    want for ranking ensembles. We multiply by sign(r) to keep anti-correlated
    models below the eligibility threshold (R² > 0).
    """
    if np.std(y_pred) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = float(np.corrcoef(y_pred, y_true)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return float(np.sign(r) * r * r)


def _calibrate(val_pred: np.ndarray, te_pred: np.ndarray,
               y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly rescale val/test predictions onto y_val's scale.

    Fit y_val = a * val_pred + b on validation, apply same (a, b) to test.
    After this, sklearn's r2_score gives a meaningful number on the original
    yield scale.
    """
    if np.std(val_pred) < 1e-12:
        m = float(np.mean(y_val))
        return np.full_like(val_pred, m), np.full_like(te_pred, m)
    a, b = np.polyfit(val_pred, y_val, 1)
    return a * val_pred + b, a * te_pred + b


def _eval_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# ── Public functions ──────────────────────────────────────────────────────────

def cfa_greedy(
    val_preds:  dict,
    tes_preds:  dict,
    y_val:      np.ndarray,
    y_te:       np.ndarray,
    time_limit: float = 60.0,
    max_rounds: int   = 200,
) -> dict:
    """Greedy forward CFA ensemble selection.

    Algorithm:
      1. Start with the single model with the highest validation R².
      2. Each round: try adding every remaining model with all 6 combination
         methods; keep the (candidate, method) pair that yields the biggest
         validation R² gain.  Stop when no gain is found or time is up.

    Returns a dict with:
      rounds      : list of per-round result dicts
      best_combo  : list of model names in the final ensemble
      best_method : combination method used
      val_r2      : final validation R²
      test_r2     : final test R²
      val_pred    : combined validation predictions (normalised [0,1])
      test_pred   : combined test predictions (normalised [0,1])
    """
    usable = list(val_preds.keys())
    if len(usable) < 2:
        print("[CFA-Greedy] Need ≥2 models.")
        return {}

    # ── Normalise predictions to [0, 1] using each model's own val range ──
    norm_val = pd.DataFrame(index=range(len(y_val)))
    norm_tes = pd.DataFrame(index=range(len(y_te)))
    for c in usable:
        lo, hi = float(val_preds[c].min()), float(val_preds[c].max())
        norm_val[c] = normalize_minmax(val_preds[c], lo, hi)
        norm_tes[c] = normalize_minmax(tes_preds[c], lo, hi)

    _, ds, _ = _cd_matrix(norm_val)
    perf     = {c: _evaluate(norm_val[c].values, y_val) for c in usable}

    print(f"\n{'='*40}")
    print(f"  CFA GREEDY SELECTION")
    print(f"{'='*40}")
    print(f"  Pool: {len(usable)} models  |  time limit: {time_limit:.0f}s  |  max rounds: {max_rounds}")
    print()

    # Individual model scores
    ranked = sorted(perf.items(), key=lambda x: -x[1])
    print("  Individual model scores (val R²  →  test R²):")
    for name, v_r2 in ranked:
        t_r2 = _evaluate(norm_tes[name].values, y_te)
        flag = "  ←─ best" if name == ranked[0][0] else ""
        print(f"    {name:25s}  val={v_r2:+.4f}  test={t_r2:+.4f}{flag}")
    print()

    # ── Greedy forward selection ───────────────────────────────────────────
    best_model  = ranked[0][0]
    best_combo  = [best_model]
    best_score  = ranked[0][1]
    best_method = "ASC"
    pool        = [n for n, _ in ranked if n != best_model]

    t_score = _evaluate(norm_tes[best_model].values, y_te)
    print(f"  Starting with: [{best_model}]  val={best_score:+.4f}  test={t_score:+.4f}")

    rounds  = []
    t0      = time.time()

    for rnd in range(1, max_rounds + 1):
        if time.time() - t0 > time_limit or not pool:
            break

        best_gain       = 0.0
        best_add        = None
        best_add_method = None
        best_add_score  = best_score

        for candidate in pool:
            subset   = best_combo + [candidate]
            sub_val  = norm_val[subset]
            sub_ds   = {m: ds.get(m, 1e-9) for m in subset}
            sub_perf = {m: perf.get(m, 0.0)  for m in subset}

            for method in METHODS:
                combined = _combine(sub_val, method, sub_ds, sub_perf)
                score    = _evaluate(combined, y_val)
                gain     = score - best_score
                if gain > best_gain:
                    best_gain       = gain
                    best_add        = candidate
                    best_add_method = method
                    best_add_score  = score

        if best_gain <= 0:
            print(f"\n  [Round {rnd}] No improvement — stopping greedy search.")
            break

        best_combo.append(best_add)
        best_score  = best_add_score
        best_method = best_add_method
        pool.remove(best_add)

        # Evaluate on test
        sub_tes  = norm_tes[best_combo]
        sub_ds   = {m: ds.get(m, 1e-9) for m in best_combo}
        sub_perf = {m: perf.get(m, 0.0)  for m in best_combo}
        t_score  = _evaluate(_combine(sub_tes, best_method, sub_ds, sub_perf), y_te)

        elapsed = time.time() - t0
        print(f"  [Round {rnd:3d}] +{best_add:25s}  method={best_method:6s}  "
              f"val={best_score:+.4f} (+{best_gain:.4f})  "
              f"test={t_score:+.4f}  "
              f"[{len(best_combo)} models, {len(pool)} in pool, {elapsed:.1f}s]")

        rounds.append({
            "round":       rnd,
            "added":       best_add,
            "combo":       "+".join(best_combo),
            "method":      best_method,
            "n":           len(best_combo),
            "val_r2":      round(best_score, 6),
            "test_r2":     round(t_score, 6),
            "gain":        round(best_gain, 6),
        })

    # ── Final ensemble predictions ────────────────────────────────────────
    sub_ds   = {m: ds.get(m, 1e-9) for m in best_combo}
    sub_perf = {m: perf.get(m, 0.0)  for m in best_combo}
    raw_val_pred = _combine(norm_val[best_combo], best_method, sub_ds, sub_perf)
    raw_tes_pred = _combine(norm_tes[best_combo], best_method, sub_ds, sub_perf)
    final_val_pred, final_tes_pred = _calibrate(raw_val_pred, raw_tes_pred, y_val)
    final_val_r2  = float(r2_score(y_val, final_val_pred))
    final_test_r2 = float(r2_score(y_te,  final_tes_pred))
    final_test_mae = float(mean_absolute_error(y_te, final_tes_pred))

    print(f"\n  Final ensemble ({len(best_combo)} model{'s' if len(best_combo) > 1 else ''}):")
    print(f"    Models : {' + '.join(best_combo)}")
    print(f"    Method : {best_method}")
    print(f"    Val R² : {final_val_r2:+.4f}")
    print(f"    Test R²: {final_test_r2:+.4f}  MAE={final_test_mae:.2f}")

    return {
        "rounds":      rounds,
        "best_combo":  best_combo,
        "best_method": best_method,
        "val_r2":      round(final_val_r2, 6),
        "test_r2":     round(final_test_r2, 6),
        "test_mae":    round(final_test_mae, 4),
        "val_pred":    final_val_pred.tolist(),
        "test_pred":   final_tes_pred.tolist(),
    }


def cfa_k_way(
    val_preds:        dict,
    tes_preds:        dict,
    y_val:            np.ndarray,
    y_te:             np.ndarray,
    max_k:            int = 5,
    top_n_k3:         int = 25,
    top_n_k4:         int = 20,
    top_n_k5:         int = 15,
) -> list:
    """Exhaustive k-way CFA for k = 2 … max_k (Lin et al. 2007, generalised).

    For k = 2: all eligible pairs — same as the original pairwise analysis.
    For k ≥ 3: all combinations from the top-N eligible models (by val R²) to
    keep computation tractable.

    Group cognitive diversity for k > 2 models is the mean of all C(k,2)
    pairwise cognitive-diversity values within the group.  The six combination
    methods (ASC, ARC, WSCDS, WRCDS, WSCP, WRCP) generalise to k models without
    formula change: the per-model diversity scores (ds) and per-model R² weights
    (perf) are computed from the full eligible pool, then applied to the k-column
    prediction matrix inside _combine().

    Args:
        val_preds / tes_preds : dict[model_name -> np.ndarray(float32)]
        y_val / y_te          : np.ndarray  ground-truth yields
        max_k                 : largest combination size to evaluate (≤ 5 recommended)
        top_n_k3/k4/k5        : pool size cap for k = 3, 4, 5

    Returns:
        list of dicts: {k, combo, method, group_cd, val_r2, test_r2}
    """
    usable = list(val_preds.keys())
    if len(usable) < 2:
        print("[CFA-k-way] Need ≥2 models.")
        return []

    # ── Normalise ──────────────────────────────────────────────────────────
    norm_val = pd.DataFrame(index=range(len(y_val)))
    norm_tes = pd.DataFrame(index=range(len(y_te)))
    for c in usable:
        lo, hi = float(val_preds[c].min()), float(val_preds[c].max())
        norm_val[c] = normalize_minmax(val_preds[c], lo, hi)
        norm_tes[c] = normalize_minmax(tes_preds[c], lo, hi)

    perf     = {c: _evaluate(norm_val[c].values, y_val) for c in usable}
    eligible = sorted([c for c in usable if perf[c] > 0], key=lambda c: -perf[c])

    # Pairwise CD matrix over ALL eligible models (used for group-CD reporting
    # and for the diversity-weighted combination methods).
    cd, ds, _ = _cd_matrix(norm_val[eligible] if eligible else norm_val)

    print(f"\n{'='*40}")
    print(f"  CFA K-WAY  (k = 2 … {max_k})")
    print(f"{'='*40}")
    print(f"  {len(usable)} models  →  {len(eligible)} eligible (R²>0)")

    # Estimate total evaluations
    pool_sizes = {
        2: len(eligible),
        3: min(len(eligible), top_n_k3),
        4: min(len(eligible), top_n_k4),
        5: min(len(eligible), top_n_k5),
    }
    total_evals = sum(
        _comb(pool_sizes[k], k) * len(METHODS)
        for k in range(2, max_k + 1)
        if k <= len(eligible)
    )
    print(f"  Estimated evaluations: {total_evals:,}")
    print()

    results = []
    t0 = time.time()

    for k in range(2, max_k + 1):
        if k > len(eligible):
            break
        top_n   = {2: len(eligible), 3: top_n_k3, 4: top_n_k4, 5: top_n_k5}.get(k, top_n_k5)
        pool    = eligible[:min(len(eligible), top_n)]
        n_combos = _comb(len(pool), k)
        print(f"  k={k}: {len(pool)} models → {n_combos:,} combos × {len(METHODS)} methods = "
              f"{n_combos * len(METHODS):,} evaluations")

        for combo in itertools.combinations(pool, k):
            combo = list(combo)

            # Group CD = mean of all C(k,2) pairwise CDs within the combo
            pair_cds = [
                cd.get(tuple(sorted([a, b])), 0.0)
                for a, b in itertools.combinations(combo, 2)
            ]
            group_cd = float(np.mean(pair_cds)) if pair_cds else 0.0

            sub_v = norm_val[combo]
            sub_t = norm_tes[combo]
            sd    = {m: ds.get(m, 1e-9) for m in combo}
            sp    = {m: perf[m]          for m in combo}

            for method in METHODS:
                cv = _combine(sub_v, method, sd, sp)
                ct = _combine(sub_t, method, sd, sp)
                cv_cal, ct_cal = _calibrate(cv, ct, y_val)
                v = float(r2_score(y_val, cv_cal))
                t = float(r2_score(y_te,  ct_cal))
                results.append({
                    "k":        k,
                    "combo":    "+".join(combo),
                    "method":   method,
                    "group_cd": round(group_cd, 6),
                    "val_r2":   round(v, 6),
                    "test_r2":  round(t, 6),
                })

        # Top 5 for this k
        top_k = sorted([r for r in results if r["k"] == k], key=lambda r: -r["val_r2"])
        seen_c: set = set()
        print(f"\n  Top 5 (k={k}) by val R²:")
        print(f"  {'Combo':{50 if k > 2 else 45}s}  {'Method':6s}  {'GrpCD':6s}  {'Val R²':8s}  {'Test R²':8s}")
        print(f"  {'-'*(50 if k > 2 else 45)}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")
        shown = 0
        for r in top_k:
            key = (r["combo"], r["method"])
            if key in seen_c:
                continue
            seen_c.add(key)
            combo_str = r["combo"]
            w = 50 if k > 2 else 45
            if len(combo_str) > w:
                combo_str = combo_str[:w-3] + "..."
            print(f"  {combo_str:{w}s}  {r['method']:6s}  {r['group_cd']:6.4f}  "
                  f"{r['val_r2']:+8.4f}  {r['test_r2']:+8.4f}")
            shown += 1
            if shown >= 5:
                break
        print()

    # ── Lin 2007: most cognitively diverse k=2 pair, ARC ──────────────────
    if cd:
        most_diverse_key = max(cd, key=lambda kk: cd[kk])
        a, b = most_diverse_key
        lin = next(
            (r for r in results
             if r["k"] == 2 and r["combo"] in (f"{a}+{b}", f"{b}+{a}") and r["method"] == "ARC"),
            None,
        )
        print(f"  [Lin 2007] Most diverse pair: {a} + {b}  (CD={cd[most_diverse_key]:.4f})")
        if lin:
            print(f"  [Lin 2007] ARC combination:  val R²={lin['val_r2']:+.4f}  "
                  f"test R²={lin['test_r2']:+.4f}")

    # ── Overall top 10 across all k ────────────────────────────────────────
    top_all = sorted(results, key=lambda r: -r["val_r2"])
    seen_a: set = set()
    print(f"\n  Overall top 10 (all k) by val R²:")
    print(f"  {'k':2s}  {'Combo':50s}  {'Method':6s}  {'Val R²':8s}  {'Test R²':8s}")
    print(f"  {'-'*2}  {'-'*50}  {'-'*6}  {'-'*8}  {'-'*8}")
    shown = 0
    for r in top_all:
        key = (r["combo"], r["method"])
        if key in seen_a:
            continue
        seen_a.add(key)
        combo_str = r["combo"][:50]
        print(f"  {r['k']:2d}  {combo_str:50s}  {r['method']:6s}  "
              f"{r['val_r2']:+8.4f}  {r['test_r2']:+8.4f}")
        shown += 1
        if shown >= 10:
            break

    elapsed = time.time() - t0
    print(f"\n  Total k-way CFA time: {elapsed:.1f}s")
    return results


def cfa_pairwise(
    val_preds: dict,
    tes_preds: dict,
    y_val:     np.ndarray,
    y_te:      np.ndarray,
) -> list:
    """Backward-compatible wrapper: exhaustive pairwise (k=2) CFA.

    Calls cfa_k_way(max_k=2) and returns results in the original format
    {pair, method, cd, val_r2, test_r2} for compatibility with callers that
    expect the old structure.
    """
    raw = cfa_k_way(val_preds, tes_preds, y_val, y_te, max_k=2)
    # Rename fields to match old API
    return [
        {
            "pair":    r["combo"],
            "method":  r["method"],
            "cd":      r["group_cd"],
            "val_r2":  r["val_r2"],
            "test_r2": r["test_r2"],
        }
        for r in raw
    ]
