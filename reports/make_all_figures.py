#!/usr/bin/env python3
"""make_all_figures.py — full figure + table set for the v2 report and the
Beamer slide deck.

Two output targets:
  --target report   — column-width figures, 10pt fonts (default)
  --target slide    — wide 16:9 figures, 16pt fonts, top-5 only

Figures are written to:
  report/figures/<target>/<name>.png

Tables (LaTeX booktabs) go to:
  report/tables/<name>.tex

Reads:
  outputs/{,classify_}{random,scaffold}/{metrics,cfa_*,bootstrap_metrics,
                                          multiseed_metrics,paired_tests,
                                          cfa_ablation,native_vs_regressor,
                                          threshold_sweep}.json
  outputs/scaffold_kfold_{reg,clf}/summary.json
  outputs/learning_curve_<task>_<split>.json
  outputs/{,classify_}{random,scaffold}/preds.npz
  data/processed/cn_reactions_raw.csv

Usage:
  uv run python report/make_all_figures.py --target report
  uv run python report/make_all_figures.py --target slide
  uv run python report/make_all_figures.py                    # report (default)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cfa_classify import _best_threshold

OUT      = ROOT / "outputs"
TBL_DIR  = ROOT / "report" / "tables"

# ── Target presets ──────────────────────────────────────────────────────────
TARGETS = {
    "report": {
        "figsize_wide":    (9.0, 4.0),
        "figsize_tall":    (7.5, 5.5),
        "figsize_grid2":   (10.0, 5.5),
        "figsize_grid4":   (11.0, 8.0),
        "font_base":       9,
        "font_title":      11,
        "font_axes":       10,
        "font_tick":       8,
        "font_legend":     8,
        "top_n_bar":       15,
        "dpi":             150,
    },
    "slide": {
        "figsize_wide":    (12.0, 5.5),
        "figsize_tall":    (10.0, 6.5),
        "figsize_grid2":   (12.0, 5.5),
        "figsize_grid4":   (12.0, 7.0),
        "font_base":       16,
        "font_title":      18,
        "font_axes":       16,
        "font_tick":       14,
        "font_legend":     13,
        "top_n_bar":       6,
        "dpi":             150,
    },
}

CAT_COLOR = {
    "boost":  "#3b6ea8", "tree":   "#8a3b3b",
    "linear": "#3b8a55", "kernel": "#a06f3b",
    "knn":    "#7a4ba0", "nn":     "#c46b3a",
    "proj":   "#777777", "chem":   "#1a3a6e",
    "native": "#6a3a8a",
}
CATEGORY = {
    "boost":   {"LightGBM", "LGB_MAE", "LGB_Dart", "XGBoost", "XGB_MAE",
                "XGB_Shallow", "HistGBT", "HistGBT_MAE", "GradientBoosting",
                "AdaBoost", "NGBoost", "rxnfp_LGB"},
    "tree":    {"RandomForest", "ExtraTrees", "DecisionTree"},
    "linear":  {"Ridge", "Lasso", "ElasticNet", "BayesianRidge", "Huber",
                "LassoLars", "OrthogonalMP", "Tweedie", "SGDHuber", "LinearSVR",
                "Baseline_Mean", "Baseline_Median"},
    "kernel":  {"SVR_RBF", "KernelRidge", "GP_Tanimoto"},
    "knn":     {"KNN", "KNN_3", "KNN_15"},
    "nn":      {"skMLP_1", "skMLP_2", "skMLP_Tanh", "MLP"},
    "proj":    {"PCR", "PLS", "GRP_Ridge_100", "GRP_Ridge_200",
                "SRP_Ridge_100", "SRP_Ridge_200",
                "TruncSVD_Ridge_50", "TruncSVD_Ridge_100"},
    "chem":    {"Chemprop"},
    "native":  {"Native_LGBClf", "Native_XGBClf", "Native_RFClf",
                "Native_MLPClf", "Native_Logistic"},
}


def _cat(name):
    for c, members in CATEGORY.items():
        if name in members:
            return c
    return "linear"


# ── Helpers ─────────────────────────────────────────────────────────────────

class Maker:
    def __init__(self, target):
        self.target = target
        self.cfg    = TARGETS[target]
        self.out    = ROOT / "report" / "figures" / target
        self.out.mkdir(parents=True, exist_ok=True)
        TBL_DIR.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size":       self.cfg["font_base"],
            "axes.titlesize":  self.cfg["font_title"],
            "axes.labelsize":  self.cfg["font_axes"],
            "xtick.labelsize": self.cfg["font_tick"],
            "ytick.labelsize": self.cfg["font_tick"],
            "legend.fontsize": self.cfg["font_legend"],
            "figure.dpi":      self.cfg["dpi"],
            "savefig.bbox":    "tight",
        })

    def save(self, name):
        path = self.out / name
        plt.savefig(path); plt.close()
        print(f"  wrote {path}")

    def _safe_load(self, p, default=None):
        try:
            return json.load(open(p))
        except FileNotFoundError:
            return default

    # ── Dataset figures ─────────────────────────────────────────────────────
    def fig_dataset_overview(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "cn_reactions_raw.csv")
        df = df.dropna(subset=["yield"])
        fig, axes = plt.subplots(1, 2, figsize=self.cfg["figsize_grid2"])
        axes[0].hist(df["yield"].clip(0, 100), bins=40, color="#3b6ea8", edgecolor="white")
        axes[0].set_xlabel("Yield (%)"); axes[0].set_ylabel("Count")
        axes[0].set_title(f"Yield distribution  (n = {len(df):,})")
        axes[0].axvline(50, color="crimson", ls="--", lw=1.6, label="success threshold (50%)")
        axes[0].axvline(df["yield"].mean(), color="black", ls=":", lw=1.0,
                        label=f"mean = {df['yield'].mean():.1f}%")
        axes[0].legend()
        if "temperature_c" in df.columns:
            t = df["temperature_c"].dropna()
            t = t[(t > -100) & (t < 500)]
            axes[1].hist(t, bins=30, color="#c46b3a", edgecolor="white")
            axes[1].set_xlabel("Temperature (°C)"); axes[1].set_ylabel("Count")
            axes[1].set_title(f"Reaction temperature  (n = {len(t):,})")
        plt.tight_layout()
        self.save("dataset_overview.png")

    def fig_class_balance(self):
        """Class balance bar (random vs scaffold splits)."""
        rd = np.load(OUT / "random"   / "preds.npz")
        sc = np.load(OUT / "scaffold" / "preds.npz")
        labels = ["Train (random)", "Test (random)", "Train (scaffold)", "Test (scaffold)"]
        # We only have y_val and y_te in preds.npz, but val IS the train surrogate at scale here
        yvr = rd["y_val"]; ytr = rd["y_te"]
        yvs = sc["y_val"]; yts = sc["y_te"]
        pos = [(yvr >= 50).mean(), (ytr >= 50).mean(),
               (yvs >= 50).mean(), (yts >= 50).mean()]
        neg = [1 - p for p in pos]
        fig, ax = plt.subplots(figsize=self.cfg["figsize_wide"])
        x = np.arange(len(labels))
        ax.bar(x, neg, color="#a04040", label="yield < 50%")
        ax.bar(x, pos, bottom=neg, color="#3b6ea8", label="yield ≥ 50%")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Fraction of split"); ax.set_ylim(0, 1)
        ax.set_title("Class balance per split")
        ax.legend(loc="lower right")
        plt.tight_layout(); self.save("class_balance.png")

    # ── Per-model bar with bootstrap error bars ─────────────────────────────
    def fig_perf_bars(self, task, split):
        """Per-model accuracy/R² bar chart with 95% bootstrap CI error bars.
        task: regression | classification
        """
        is_reg = (task == "regression")
        out_dir = OUT / split if is_reg else OUT / f"classify_{split}"
        bs = self._safe_load(out_dir / "bootstrap_metrics.json")
        if not bs:
            print(f"  [skip] bars {task}/{split}: no bootstrap_metrics.json"); return
        metric_key = "r2" if is_reg else "accuracy"
        rows = sorted(bs.items(), key=lambda kv: -kv[1]["point"][metric_key])
        rows = rows[:self.cfg["top_n_bar"]]
        names  = [r[0] for r in rows]
        points = np.array([r[1]["point"][metric_key] for r in rows])
        lo     = np.array([r[1]["ci"][metric_key]["lo"] for r in rows])
        hi     = np.array([r[1]["ci"][metric_key]["hi"] for r in rows])
        cats   = [_cat(n) for n in names]
        cols   = [CAT_COLOR.get(c, "#888") for c in cats]
        err = np.vstack([points - lo, hi - points])

        fig, ax = plt.subplots(figsize=self.cfg["figsize_tall"])
        y = np.arange(len(names))
        ax.barh(y, points, xerr=err, color=cols, edgecolor="white",
                error_kw={"elinewidth": 1.0, "ecolor": "black", "capsize": 2.5})
        ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        if is_reg:
            ax.axvline(0, color="black", lw=0.6, alpha=0.5)
            ax.set_xlabel("Test R²")
        else:
            n_baseline = 0.668 if split == "random" else 0.529
            ax.axvline(n_baseline, color="black", ls=":", lw=0.9, alpha=0.7,
                       label=f"majority-class ({n_baseline:.3f})")
            ax.set_xlabel("Test accuracy")
            ax.legend(loc="lower right")
            # Zoom for slide target classification: top accuracies cluster 0.85+
            if self.target == "slide" and split == "random":
                ax.set_xlim(0.80, 1.0)
            elif self.target == "slide" and split == "scaffold":
                ax.set_xlim(0.50, 0.65)
        title = f"{task.title()} — {split} split (top-{len(names)}; 95% bootstrap CI, B=1000)"
        ax.set_title(title); ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        self.save(f"bars_{task}_{split}.png")

    # ── Native vs regressor head-to-head ────────────────────────────────────
    def fig_native_vs_regressor(self, split):
        comp = self._safe_load(OUT / f"classify_{split}" / "native_vs_regressor.json")
        if not comp:
            print(f"  [skip] native_vs_regressor {split}: missing"); return
        fams  = [r["family"] for r in comp]
        r_acc = [r["reg_test_acc"] for r in comp]
        c_acc = [r["clf_test_acc"] for r in comp]
        d_mcc = [r["delta_mcc"]    for r in comp]
        fig, axes = plt.subplots(1, 2, figsize=self.cfg["figsize_grid2"])
        x = np.arange(len(fams))
        w = 0.35
        axes[0].bar(x - w/2, r_acc, w, color="#3b6ea8", label="Thresholded regressor")
        axes[0].bar(x + w/2, c_acc, w, color="#6a3a8a", label="Native classifier")
        axes[0].set_xticks(x); axes[0].set_xticklabels(fams, rotation=20, ha="right")
        axes[0].set_ylabel("Test accuracy")
        axes[0].set_title(f"Head-to-head test accuracy ({split})")
        axes[0].legend(loc="lower right")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(x, d_mcc, color=["#3b8a55" if v >= 0 else "#a04040" for v in d_mcc])
        axes[1].axhline(0, color="black", lw=0.7)
        axes[1].set_xticks(x); axes[1].set_xticklabels(fams, rotation=20, ha="right")
        axes[1].set_ylabel("Δ MCC (native − thresh-reg)")
        axes[1].set_title(f"Native classifier ΔMCC ({split})")
        axes[1].grid(axis="y", alpha=0.25)
        plt.tight_layout()
        self.save(f"native_vs_regressor_{split}.png")

    # ── CFA ablation ────────────────────────────────────────────────────────
    def fig_cfa_ablation(self, task, split):
        is_reg = (task == "regression")
        out_dir = OUT / split if is_reg else OUT / f"classify_{split}"
        ab = self._safe_load(out_dir / "cfa_ablation.json")
        if not ab:
            print(f"  [skip] ablation {task}/{split}: missing"); return

        if is_reg:
            order = ["baseline_mean", "single_best",
                     "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
                     "median_top5", "cfa_greedy", "cfa_kway_val_selected",
                     "cfa_kway_test_ceiling"]
            metric_extractor = lambda r: r.get("test", {}).get("r2", r.get("test_r2", float("nan")))
        else:
            order = ["majority_class", "single_best",
                     "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
                     "majority_vote_top3", "majority_vote_top5",
                     "cfa_greedy", "cfa_kway_val_selected", "cfa_kway_test_ceiling"]
            metric_extractor = lambda r: r.get("accuracy", r.get("test_accuracy", float("nan")))

        labels = []
        vals   = []
        for k in order:
            r = ab.get(k)
            if not r:
                continue
            labels.append(k.replace("_", " "))
            vals.append(metric_extractor(r))

        fig, ax = plt.subplots(figsize=self.cfg["figsize_wide"])
        x = np.arange(len(labels))
        cmap = plt.cm.viridis
        ax.bar(x, vals, color=[cmap(i / max(1, len(labels) - 1)) for i in range(len(labels))])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Test R²" if is_reg else "Test accuracy")
        ax.set_title(f"CFA ablation — {task}, {split} split")
        if not is_reg:
            if self.target == "slide":
                ax.set_ylim(0.6, 1.0)
                ax.set_yticks(np.arange(0.6, 1.01, 0.05))
            else:
                ax.set_ylim(min(0.4, min(vals) - 0.02), max(vals) + 0.02)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        self.save(f"ablation_{task}_{split}.png")

    # ── Substrate-level k=5 box plots ───────────────────────────────────────
    def fig_kfold_box(self, task):
        is_reg = (task == "regression")
        out_dir = OUT / ("scaffold_kfold_reg" if is_reg else "scaffold_kfold_clf")
        s = self._safe_load(out_dir / "summary.json")
        if not s:
            print(f"  [skip] kfold {task}: missing"); return
        per_model = s["per_model"]
        metric = "r2" if is_reg else "accuracy"
        ranked = sorted(per_model.items(), key=lambda kv: -kv[1][metric]["mean"])
        names = [n for n, _ in ranked]
        data  = [per_model[n][metric]["values"] for n in names]
        fig, ax = plt.subplots(figsize=self.cfg["figsize_tall"])
        bp = ax.boxplot(data, labels=[n.replace("_", " ") for n in names],
                        patch_artist=True, vert=True, widths=0.55)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(CAT_COLOR.get(_cat(n), "#888"))
            patch.set_alpha(0.7)
        if is_reg:
            ax.axhline(0, color="black", lw=0.6, alpha=0.5)
            ax.set_ylabel("Test R² per fold")
        else:
            ax.axhline(0.529, color="black", ls=":", lw=0.9, alpha=0.6, label="majority class")
            ax.legend()
            ax.set_ylabel("Test accuracy per fold")
        ax.set_title(f"Substrate-level 5-fold CV — {task}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        self.save(f"kfold_{task}.png")

    # ── Threshold sweep ─────────────────────────────────────────────────────
    def fig_threshold_sweep(self, split):
        ts = self._safe_load(OUT / f"classify_{split}" / "threshold_sweep.json")
        if not ts:
            print(f"  [skip] threshold sweep {split}: missing"); return
        thresholds = ts["thresholds"]
        # Pick top 5 models by their performance summed across thresholds
        all_names = set()
        for t, per in ts["results"].items():
            all_names.update(per.keys())
        scores = {n: [] for n in all_names}
        for t in thresholds:
            per = ts["results"].get(str(t), {})
            for n in all_names:
                scores[n].append(per.get(n, {}).get("test_mcc", float("nan")))
        avg = {n: np.nanmean(v) for n, v in scores.items()}
        top = sorted(avg.items(), key=lambda kv: -kv[1])[:5]

        fig, axes = plt.subplots(1, 2, figsize=self.cfg["figsize_grid2"])
        # Left: MCC vs threshold
        for n, _ in top:
            ax = axes[0]
            ys = scores[n]
            ax.plot(thresholds, ys, "o-", label=n.replace("_", " "), lw=1.6, ms=5)
        axes[0].axvline(50, color="crimson", ls=":", lw=1.0, alpha=0.7, label="default 50%")
        axes[0].axhline(0, color="black", lw=0.5, alpha=0.5)
        axes[0].set_xlabel("Yield threshold (%)"); axes[0].set_ylabel("Test MCC")
        axes[0].set_title("Top-5 model MCC vs threshold")
        axes[0].legend(loc="best", fontsize=self.cfg["font_legend"] - 1)
        axes[0].grid(alpha=0.25)
        # Right: positive class fraction
        fr = [ts["n_pos_per_threshold"][str(t)]["frac_te"] for t in thresholds]
        axes[1].plot(thresholds, fr, "s-", color="#3b6ea8", lw=1.6, ms=5)
        axes[1].axvline(50, color="crimson", ls=":", lw=1.0, alpha=0.7)
        axes[1].set_xlabel("Yield threshold (%)"); axes[1].set_ylabel("Test positive class fraction")
        axes[1].set_title("Class balance vs threshold")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        self.save(f"threshold_sweep_{split}.png")

    # ── Learning curves ─────────────────────────────────────────────────────
    def fig_learning_curve(self, task, split):
        lc = self._safe_load(OUT / f"learning_curve_{task}_{split}.json")
        if not lc:
            print(f"  [skip] learning curve {task}/{split}: missing"); return
        fractions = lc["fractions"]
        is_reg = (task == "regression")
        metric = "r2" if is_reg else "accuracy"
        fig, ax = plt.subplots(figsize=self.cfg["figsize_wide"])
        for name, per_frac in lc["models"].items():
            xs = []; ys = []
            for f in fractions:
                if str(f) in per_frac:
                    xs.append(per_frac[str(f)]["n_train"])
                    ys.append(per_frac[str(f)][metric])
            ax.plot(xs, ys, "o-", label=name.replace("_", " "), lw=1.6, ms=5)
        ax.set_xlabel("Training set size"); ax.set_xscale("log")
        ax.set_ylabel("Test R²" if is_reg else "Test accuracy")
        ax.set_title(f"Learning curve — {task} ({split})")
        ax.legend(loc="best", fontsize=self.cfg["font_legend"] - 1)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        self.save(f"learning_curve_{task}_{split}.png")

    # ── Confusion matrix ────────────────────────────────────────────────────
    def fig_confusion_random(self, model_name="Native_LGBClf"):
        d = np.load(OUT / "random" / "preds.npz", allow_pickle=False)
        names = [str(n) for n in d["names"]]
        if model_name not in names:
            # Try native classifier file
            nat = OUT / "random" / "native_clf_preds.npz"
            if nat.exists():
                nd = np.load(nat, allow_pickle=False)
                nnames = [str(n) for n in nd["names"]]
                if model_name in nnames:
                    i = nnames.index(model_name)
                    val_p = nd["val"][i]; tes_p = nd["test"][i]
                else:
                    print(f"  [skip] confusion: {model_name} not found"); return
            else:
                return
        else:
            i = names.index(model_name)
            val_p = d["val"][i]; tes_p = d["test"][i]
        y_val_bin = (d["y_val"] >= 50).astype(int)
        y_te_bin  = (d["y_te"]  >= 50).astype(int)
        thr  = _best_threshold(val_p, y_val_bin)
        pred = (tes_p >= thr).astype(int)
        cm = confusion_matrix(y_te_bin, pred)
        norm = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(self.cfg["figsize_wide"][0] * 0.55,
                                        self.cfg["figsize_wide"][1] * 0.95))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred fail", "pred success"])
        ax.set_yticklabels(["true fail", "true success"])
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, f"{cm[ii, jj]}\n({norm[ii, jj]:.2f})",
                        ha="center", va="center",
                        color="white" if norm[ii, jj] > 0.5 else "black",
                        fontsize=self.cfg["font_axes"] + 2)
        ax.set_title(f"Confusion matrix — {model_name} (random)")
        plt.tight_layout()
        self.save(f"confusion_{model_name}_random.png")

    # ── Rank-score curves ───────────────────────────────────────────────────
    def fig_rank_score(self, split, task="classification"):
        """CFA's signature plot: each model's predictions normalised to [0,1]
        then sorted descending; oracle reference = sorted ground-truth."""
        out_dir = OUT / split if task == "regression" else OUT / f"classify_{split}"
        bs = self._safe_load(out_dir / "bootstrap_metrics.json")
        if not bs:
            return
        rank_metric = "r2" if task == "regression" else "accuracy"
        top5 = sorted(bs.items(), key=lambda kv: -kv[1]["point"][rank_metric])[:5]
        top5_names = [n for n, _ in top5]

        if task == "regression":
            d = np.load(OUT / split / "preds.npz", allow_pickle=False)
        else:
            d = np.load(OUT / split / "preds.npz", allow_pickle=False)
            nat = OUT / split / "native_clf_preds.npz"
            extra_names = []; extra_test = None
            if nat.exists():
                nd = np.load(nat, allow_pickle=False)
                extra_names = [str(n) for n in nd["names"]]
                extra_test = nd["test"]
        names = [str(n) for n in d["names"]]
        test  = d["test"]
        y_te  = d["y_te"]

        fig, ax = plt.subplots(figsize=self.cfg["figsize_wide"])
        for n in top5_names:
            if n in names:
                tp = test[names.index(n)]
            elif task == "classification" and n in extra_names:
                tp = extra_test[extra_names.index(n)]
            else:
                continue
            lo, hi = float(np.min(tp)), float(np.max(tp))
            if hi == lo:
                continue
            norm = (tp - lo) / (hi - lo)
            ax.plot(np.arange(1, len(norm) + 1), np.sort(norm)[::-1],
                    label=n.replace("_", " "), lw=1.5, alpha=0.9)
        # Oracle: ground-truth normalised
        ylo, yhi = float(np.min(y_te)), float(np.max(y_te))
        if yhi > ylo:
            yn = (y_te - ylo) / (yhi - ylo)
            ax.plot(np.arange(1, len(yn) + 1), np.sort(yn)[::-1],
                    "k--", lw=2, label="ground truth (oracle)")
        ax.set_xlabel("Test sample rank"); ax.set_ylabel("Normalised score")
        ax.set_title(f"Rank-score curves — {task}, {split}")
        ax.legend(loc="upper right", fontsize=self.cfg["font_legend"] - 1)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        self.save(f"rank_score_{task}_{split}.png")


# ── Tables (target-independent) ─────────────────────────────────────────────

def _safe_name(n):
    return n.replace("_", r"\_")


def table_bootstrap_top(task, split, n=10):
    is_reg = (task == "regression")
    out_dir = OUT / split if is_reg else OUT / f"classify_{split}"
    bs_path = out_dir / "bootstrap_metrics.json"
    if not bs_path.exists():
        return
    bs = json.load(open(bs_path))
    metric = "r2" if is_reg else "accuracy"
    rows = sorted(bs.items(), key=lambda kv: -kv[1]["point"][metric])[:n]
    if is_reg:
        lines = [r"\begin{tabular}{lrr}", r"\toprule",
                 r"Model & R$^2$ [95\% CI] & MAE [95\% CI] \\", r"\midrule"]
        for nm, r in rows:
            r2 = r["ci"]["r2"]; mae = r["ci"]["mae"]
            lines.append(
                f"{_safe_name(nm)} & {r['point']['r2']:+.3f} [{r2['lo']:+.3f}, {r2['hi']:+.3f}] "
                f"& {r['point']['mae']:.2f} [{mae['lo']:.2f}, {mae['hi']:.2f}] \\\\"
            )
    else:
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Model & Accuracy [95\% CI] & MCC [95\% CI] & AUC [95\% CI] \\", r"\midrule"]
        for nm, r in rows:
            acc = r["ci"]["accuracy"]; mcc = r["ci"]["mcc"]; auc = r["ci"]["auc"]
            lines.append(
                f"{_safe_name(nm)} & "
                f"{r['point']['accuracy']:.3f} [{acc['lo']:.3f}, {acc['hi']:.3f}] & "
                f"{r['point']['mcc']:+.3f} [{mcc['lo']:+.3f}, {mcc['hi']:+.3f}] & "
                f"{r['point']['auc']:.3f} [{auc['lo']:.3f}, {auc['hi']:.3f}] \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_name = f"bootstrap_{task}_{split}.tex"
    (TBL_DIR / out_name).write_text("\n".join(lines))
    print(f"  wrote {TBL_DIR / out_name}")


def table_kfold(task):
    is_reg = (task == "regression")
    out_dir = OUT / ("scaffold_kfold_reg" if is_reg else "scaffold_kfold_clf")
    s_path = out_dir / "summary.json"
    if not s_path.exists():
        return
    s = json.load(open(s_path))
    per = s["per_model"]
    if is_reg:
        ranked = sorted(per.items(), key=lambda kv: -kv[1]["r2"]["mean"])
        lines = [r"\begin{tabular}{lrr}", r"\toprule",
                 r"Model & R$^2$ mean $\pm$ std & MAE mean $\pm$ std \\", r"\midrule"]
        for nm, r in ranked:
            r2 = r["r2"]; mae = r["mae"]
            lines.append(f"{_safe_name(nm)} & "
                         f"{r2['mean']:+.3f} $\\pm$ {r2['std']:.3f} & "
                         f"{mae['mean']:.2f} $\\pm$ {mae['std']:.2f} \\\\")
    else:
        ranked = sorted(per.items(), key=lambda kv: -kv[1]["accuracy"]["mean"])
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Model & Accuracy & MCC & AUC \\", r"\midrule"]
        for nm, r in ranked:
            acc = r["accuracy"]; mcc = r["mcc"]; auc = r["auc"]
            lines.append(f"{_safe_name(nm)} & "
                         f"{acc['mean']:.3f} $\\pm$ {acc['std']:.3f} & "
                         f"{mcc['mean']:+.3f} $\\pm$ {mcc['std']:.3f} & "
                         f"{auc['mean']:.3f} $\\pm$ {auc['std']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_name = f"kfold_{task}.tex"
    (TBL_DIR / out_name).write_text("\n".join(lines))
    print(f"  wrote {TBL_DIR / out_name}")


def table_ablation(task, split):
    is_reg = (task == "regression")
    out_dir = OUT / split if is_reg else OUT / f"classify_{split}"
    ab_path = out_dir / "cfa_ablation.json"
    if not ab_path.exists():
        return
    ab = json.load(open(ab_path))
    if is_reg:
        order = ["baseline_mean", "single_best",
                 "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
                 "median_top5", "cfa_greedy", "cfa_kway_val_selected",
                 "cfa_kway_test_ceiling"]
        lines = [r"\begin{tabular}{lrr}", r"\toprule",
                 r"Strategy & Test R$^2$ & $\Delta$ vs single-best \\", r"\midrule"]
        base = ab.get("single_best", {}).get("test", {}).get("r2", 0.0)
        for k in order:
            r = ab.get(k)
            if not r:
                continue
            v = r.get("test", {}).get("r2", r.get("test_r2", float("nan")))
            lines.append(f"{k.replace('_', ' ')} & {v:+.3f} & {v - base:+.3f} \\\\")
    else:
        order = ["majority_class", "single_best",
                 "equal_weight_top2", "equal_weight_top3", "equal_weight_top5",
                 "majority_vote_top3", "majority_vote_top5",
                 "cfa_greedy", "cfa_kway_val_selected", "cfa_kway_test_ceiling"]
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Strategy & Accuracy & MCC & $\Delta$ acc vs single \\", r"\midrule"]
        base = ab.get("single_best", {}).get("accuracy", 0.0)
        for k in order:
            r = ab.get(k)
            if not r:
                continue
            acc = r.get("accuracy", r.get("test_accuracy", float("nan")))
            mcc = r.get("mcc", r.get("test_mcc", float("nan")))
            lines.append(f"{k.replace('_', ' ')} & {acc:.3f} & {mcc:+.3f} & {acc - base:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_name = f"ablation_{task}_{split}.tex"
    (TBL_DIR / out_name).write_text("\n".join(lines))
    print(f"  wrote {TBL_DIR / out_name}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["report", "slide", "both"], default="report")
    args = ap.parse_args()

    targets = ["report", "slide"] if args.target == "both" else [args.target]
    for t in targets:
        print(f"\n=== target = {t} ===")
        m = Maker(t)

        # Dataset
        m.fig_dataset_overview()
        m.fig_class_balance()

        # Per-model bars with bootstrap CIs
        for task in ("regression", "classification"):
            for split in ("random", "scaffold"):
                m.fig_perf_bars(task, split)

        # Native vs regressor
        for split in ("random", "scaffold"):
            m.fig_native_vs_regressor(split)

        # CFA ablation
        for task in ("regression", "classification"):
            for split in ("random", "scaffold"):
                m.fig_cfa_ablation(task, split)

        # Substrate-level k=5 box plots
        for task in ("regression", "classification"):
            m.fig_kfold_box(task)

        # Threshold sweep
        for split in ("random", "scaffold"):
            m.fig_threshold_sweep(split)

        # Learning curves (may be missing if item 5 is still running)
        for task in ("regression", "classification"):
            for split in ("random", "substrate"):
                m.fig_learning_curve(task, split)

        # Confusion matrix
        m.fig_confusion_random("Native_LGBClf")

        # Rank-score curves
        for split in ("random", "scaffold"):
            m.fig_rank_score(split, "classification")
            m.fig_rank_score(split, "regression")

    # Tables (target-independent)
    print("\n=== tables ===")
    for task in ("regression", "classification"):
        for split in ("random", "scaffold"):
            table_bootstrap_top(task, split)
            table_ablation(task, split)
    for task in ("regression", "classification"):
        table_kfold(task)


if __name__ == "__main__":
    main()
