#!/usr/bin/env python3
"""make_figures.py — figures and tables for the v2 (classification) report.

Reads:
  outputs/classify_random/{metrics.json, cfa_greedy.json, cfa_kway.json}
  outputs/classify_scaffold/{metrics.json, cfa_greedy.json, cfa_kway.json}
  data/processed/cn_reactions_raw.csv

Writes:
  report/figures/clf_random_per_model.png
  report/figures/clf_scaffold_per_model.png
  report/figures/clf_random_kway.png
  report/figures/clf_split_compare.png
  report/figures/dataset_overview.png  (re-emit with 50% line)
  report/tables/{clf_top_random,clf_top_scaffold,clf_cfa_top_random,clf_cfa_top_scaffold}.tex
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
OUT_RND  = ROOT / "outputs" / "classify_random"
OUT_SCF  = ROOT / "outputs" / "classify_scaffold"
FIG_DIR  = ROOT / "report"  / "figures"
TBL_DIR  = ROOT / "report"  / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# Colour scheme
CAT_COLOR = {
    "boost":  "#3b6ea8", "tree":   "#8a3b3b",
    "linear": "#3b8a55", "kernel": "#a06f3b",
    "knn":    "#7a4ba0", "nn":     "#c46b3a",
    "proj":   "#777777", "chem":   "#1a3a6e",
}
CATEGORY = {
    "boost":   {"LightGBM", "LGB_MAE", "LGB_Dart", "XGBoost", "XGB_MAE",
                "XGB_Shallow", "HistGBT", "HistGBT_MAE", "GradientBoosting",
                "AdaBoost", "NGBoost", "rxnfp_LGB"},
    "tree":    {"RandomForest", "ExtraTrees", "DecisionTree"},
    "linear":  {"Ridge", "Lasso", "ElasticNet", "BayesianRidge", "Huber",
                "LassoLars", "OrthogonalMP", "Tweedie", "SGDHuber", "LinearSVR"},
    "kernel":  {"SVR_RBF", "KernelRidge", "GP_Tanimoto"},
    "knn":     {"KNN", "KNN_3", "KNN_15"},
    "nn":      {"skMLP_1", "skMLP_2", "skMLP_Tanh", "MLP"},
    "proj":    {"PCR", "PLS", "GRP_Ridge_100", "GRP_Ridge_200",
                "SRP_Ridge_100", "SRP_Ridge_200",
                "TruncSVD_Ridge_50", "TruncSVD_Ridge_100"},
    "chem":    {"Chemprop"},
}


def _cat(name):
    for c, members in CATEGORY.items():
        if name in members:
            return c
    return "linear"


# ── Dataset figure with 50% threshold line ──────────────────────────────────

def fig_dataset_overview():
    df = pd.read_csv(ROOT / "data" / "processed" / "cn_reactions_raw.csv")
    df = df.dropna(subset=["yield"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    axes[0].hist(df["yield"].clip(0, 100), bins=40, color="#3b6ea8", edgecolor="white")
    axes[0].set_xlabel("Yield (%)"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Yield distribution ($n = {len(df):,}$)")
    axes[0].axvline(50, color="crimson", linestyle="--", lw=1.4,
                    label="success threshold (50%)")
    axes[0].axvline(df["yield"].mean(), color="black", linestyle=":", lw=1.0,
                    label=f"mean = {df['yield'].mean():.1f}%")
    axes[0].legend()
    if "temperature_c" in df.columns:
        t = df["temperature_c"].dropna()
        t = t[(t > -100) & (t < 500)]
        axes[1].hist(t, bins=30, color="#c46b3a", edgecolor="white")
        axes[1].set_xlabel("Temperature (°C)"); axes[1].set_ylabel("Count")
        axes[1].set_title(f"Reaction temperature ($n = {len(t):,})$")
    plt.tight_layout()
    out = FIG_DIR / "dataset_overview.png"
    plt.savefig(out); plt.close(); print(f"  wrote {out}")


# ── Per-model bar chart (per split) ──────────────────────────────────────────

def fig_per_model(split_dir, out_name, title_suffix, baseline_acc):
    metrics = json.loads((split_dir / "metrics.json").read_text())
    items = sorted(metrics.items(), key=lambda kv: kv[1]["test_accuracy"])
    names = [k for k, _ in items]
    acc   = [v["test_accuracy"] for _, v in items]
    mcc   = [v["test_mcc"]      for _, v in items]
    cats  = [_cat(n) for n in names]
    cols  = [CAT_COLOR[c] for c in cats]

    height = max(5.0, 0.27 * len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, height), sharey=True)
    axes[0].barh(names, acc, color=cols, edgecolor="white")
    axes[0].axvline(baseline_acc, color="black", lw=0.8, ls=":",
                    alpha=0.7, label=f"majority-class baseline ({baseline_acc:.3f})")
    axes[0].set_xlabel("Test accuracy"); axes[0].set_title(f"Test accuracy {title_suffix}")
    axes[0].grid(axis="x", alpha=0.25); axes[0].legend(loc="lower right")
    axes[1].barh(names, mcc, color=cols, edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Test MCC"); axes[1].set_title("Test MCC")
    axes[1].grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out); plt.close(); print(f"  wrote {out}")


# ── k-way bar chart for a split ─────────────────────────────────────────────

def fig_kway(split_dir, out_name, baseline_acc):
    metrics = json.loads((split_dir / "metrics.json").read_text())
    kway    = json.loads((split_dir / "cfa_kway.json").read_text()) \
              if (split_dir / "cfa_kway.json").exists() else []
    if not kway:
        print(f"  [skip] {out_name}: no kway data"); return

    best_test_acc = max(
        max(m["test_accuracy"] for m in metrics.values()),
        max(r["test_accuracy"] for r in kway),
    )

    solo_top = sorted(metrics.items(), key=lambda kv: -kv[1]["test_accuracy"])[:10]
    by_k = {}
    for r in kway:
        by_k.setdefault(r["k"], []).append(r)
    for k in by_k:
        by_k[k] = sorted(by_k[k], key=lambda r: -r["test_accuracy"])

    fig, axes = plt.subplots(5, 1, figsize=(10, 13))
    col_test = "#c46b3a"; BAR_H = 0.55

    def _line(ax, val):
        ax.axvline(val, color="red", lw=1.4, ls=":",
                   label=f"best observed test acc={val:.3f}")
        ax.axvline(baseline_acc, color="black", lw=0.8, ls=":",
                   alpha=0.7, label=f"baseline={baseline_acc:.3f}")

    ax = axes[0]
    n1 = [n for n, _ in solo_top]
    a1 = [m["test_accuracy"] for _, m in solo_top]
    y  = np.arange(len(n1))
    ax.barh(y, a1, BAR_H, color=col_test, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in n1], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Test accuracy")
    ax.set_title("k = 1 (top-10 individual classifiers by test accuracy)")
    _line(ax, best_test_acc)
    lo = min(a1) - 0.005; hi = max(max(a1), best_test_acc) + 0.006
    ax.set_xlim(max(0.5, lo), hi); ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)

    for idx, k in enumerate(range(2, 6)):
        ax = axes[idx + 1]
        entries = by_k.get(k, [])
        seen, top = set(), []
        for r in entries:
            key = (r["combo"], r["method"])
            if key not in seen:
                seen.add(key); top.append(r)
            if len(top) == 10: break
        labels = [f"{r['combo'][:60]} ({r['method']})" for r in top]
        accs   = [r["test_accuracy"] for r in top]
        y = np.arange(len(top))
        ax.barh(y, accs, BAR_H, color=col_test, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("Test accuracy")
        ax.set_title(f"k = {k} (top-10 ensembles by test accuracy)")
        _line(ax, best_test_acc)
        if accs:
            lo = min(accs) - 0.005; hi = max(max(accs), best_test_acc) + 0.006
            ax.set_xlim(max(0.5, lo), hi)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout(h_pad=1.8)
    out = FIG_DIR / out_name
    plt.savefig(out, bbox_inches="tight"); plt.close(); print(f"  wrote {out}")


# ── Random vs scaffold comparison ───────────────────────────────────────────

def fig_split_compare():
    rd = json.loads((OUT_RND / "metrics.json").read_text())
    sc = json.loads((OUT_SCF / "metrics.json").read_text()) \
         if (OUT_SCF / "metrics.json").exists() else None
    if not sc:
        print("  [skip] split compare: scaffold metrics missing"); return
    common = sorted(set(rd) & set(sc), key=lambda n: -rd[n]["test_accuracy"])
    top = common[:20]
    names    = [n.replace("_", " ") for n in top]
    rnd_acc  = [rd[n]["test_accuracy"] for n in top]
    scf_acc  = [sc[n]["test_accuracy"] for n in top]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top))
    ax.barh(y - 0.2, rnd_acc, 0.4, color="#3b6ea8", label="Random split test acc", edgecolor="white")
    ax.barh(y + 0.2, scf_acc, 0.4, color="#a04040", label="Scaffold-OOD test acc", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.6)
    ax.axvline(0.681, color="green", lw=0.8, ls=":", alpha=0.7,
               label="majority-class baseline (random=0.681)")
    ax.set_xlabel("Test accuracy")
    ax.set_title(f"Random vs scaffold-OOD test accuracy (top-{len(top)} models)")
    ax.grid(axis="x", alpha=0.25); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "clf_split_compare.png"
    plt.savefig(out); plt.close(); print(f"  wrote {out}")


# ── Tables ──────────────────────────────────────────────────────────────────

def table_top(split_dir, out_name, n=10):
    metrics = json.loads((split_dir / "metrics.json").read_text())
    rows = sorted(metrics.items(), key=lambda kv: -kv[1]["test_accuracy"])[:n]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Threshold & Acc & F1 & MCC & AUC \\", r"\midrule"]
    for name, m in rows:
        safe = name.replace("_", r"\_")
        auc  = m.get("test_auc", float("nan"))
        lines.append(f"{safe} & {m['threshold']:.2f} & "
                     f"{m['test_accuracy']:.4f} & {m['test_f1']:.4f} & "
                     f"{m['test_mcc']:+.4f} & {auc:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL_DIR / out_name).write_text("\n".join(lines))
    print(f"  wrote {TBL_DIR / out_name}")


def table_cfa_top(split_dir, out_name):
    if not (split_dir / "cfa_kway.json").exists():
        return
    metrics = json.loads((split_dir / "metrics.json").read_text())
    kway    = json.loads((split_dir / "cfa_kway.json").read_text())
    # k=1 entry: best individual by test accuracy
    best1   = max(metrics.items(), key=lambda kv: kv[1]["test_accuracy"])
    rows = [{
        "k": 1, "combo": best1[0], "method": "N/A",
        "test_accuracy": best1[1]["test_accuracy"],
        "test_f1":       best1[1]["test_f1"],
        "test_mcc":      best1[1]["test_mcc"],
    }]
    # Best per k by val accuracy (the principled selection)
    by_k = {}
    for r in kway:
        by_k.setdefault(r["k"], []).append(r)
    for k in sorted(by_k):
        best = max(by_k[k], key=lambda r: r["val_accuracy"])
        rows.append({
            "k": k, "combo": best["combo"], "method": best["method"],
            "test_accuracy": best["test_accuracy"],
            "test_f1":       best["test_f1"],
            "test_mcc":      best["test_mcc"],
        })

    lines = [r"\begin{tabular}{p{8cm}cclrr}", r"\toprule",
             r"Models & $k$ & Method & Acc & F1 & MCC \\", r"\midrule"]
    for r in rows:
        safe = r["combo"].replace("_", r"\_").replace("+", " + ")
        method = r["method"] if r["method"] != "N/A" else "N/A"
        lines.append(f"{safe} & {r['k']} & {method} & "
                     f"{r['test_accuracy']:.4f} & {r['test_f1']:.4f} & "
                     f"{r['test_mcc']:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL_DIR / out_name).write_text("\n".join(lines))
    print(f"  wrote {TBL_DIR / out_name}")


def main():
    fig_dataset_overview()
    fig_per_model(OUT_RND, "clf_random_per_model.png", "(random split)", baseline_acc=0.681)
    table_top(OUT_RND, "clf_top_random.tex")
    table_cfa_top(OUT_RND, "clf_cfa_top_random.tex")
    fig_kway(OUT_RND, "clf_random_kway.png", baseline_acc=0.681)
    if (OUT_SCF / "metrics.json").exists():
        sc_metrics = json.loads((OUT_SCF / "metrics.json").read_text())
        scf_baseline = max(sum(1 for n,m in sc_metrics.items() if False),  # placeholder
                           0.5)
        # compute majority class on scaffold test from any model's threshold predictions
        # Easier: just use 0.5 as fallback or load y_te_bin majority
        from numpy import load
        try:
            d = load(ROOT / "outputs/scaffold/preds.npz", allow_pickle=False)
            y = d["y_te"]
            scf_baseline = float(max((y < 50).mean(), (y >= 50).mean()))
        except Exception:
            scf_baseline = 0.5
        fig_per_model(OUT_SCF, "clf_scaffold_per_model.png",
                      "(scaffold-OOD)", baseline_acc=scf_baseline)
        table_top(OUT_SCF, "clf_top_scaffold.tex")
        table_cfa_top(OUT_SCF, "clf_cfa_top_scaffold.tex")
        fig_split_compare()


if __name__ == "__main__":
    main()
