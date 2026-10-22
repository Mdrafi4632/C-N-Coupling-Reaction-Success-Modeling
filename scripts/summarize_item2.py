#!/usr/bin/env python3
"""
Reads:
  outputs/<split>/bootstrap_metrics.json                  (regression)
  outputs/<split>/multiseed_metrics.json                  (regression)
  outputs/classify_<split>/bootstrap_metrics.json         (classification)
  outputs/classify_<split>/multiseed_metrics.json         (classification)
Usage:
  uv run python scripts/summarize_item2.py            # both tasks, both splits
  uv run python scripts/summarize_item2.py --task classification --split random
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fmt_ci(point, lo, hi, sign=False, prec=3):
    pfmt = f"{{:+.{prec}f}}" if sign else f"{{:.{prec}f}}"
    return f"{pfmt.format(point)} [{pfmt.format(lo)}, {pfmt.format(hi)}]"


def _fmt_ms(mean, std, sign=False, prec=3):
    pfmt = f"{{:+.{prec}f}}" if sign else f"{{:.{prec}f}}"
    return f"{pfmt.format(mean)}±{pfmt.format(std)}"


def regression_section(split):
    bs_path = ROOT / "outputs" / split / "bootstrap_metrics.json"
    ms_path = ROOT / "outputs" / split / "multiseed_metrics.json"
    out = []
    out.append(f"## Regression — {split} split\n")

    if bs_path.exists():
        bs = json.load(open(bs_path))
        rows = sorted(bs.items(), key=lambda kv: -kv[1]["point"]["r2"])[:10]
        out.append("**Top 10 by point R² with 95% bootstrap CI** (B=1000):")
        out.append("")
        out.append("| Model | R² | MAE |")
        out.append("|---|---|---|")
        for name, r in rows:
            r2  = r["ci"]["r2"]
            mae = r["ci"]["mae"]
            out.append(f"| {name} | {_fmt_ci(r['point']['r2'],  r2['lo'],  r2['hi'],  sign=True)} | "
                       f"{_fmt_ci(r['point']['mae'], mae['lo'], mae['hi'])} |")
        out.append("")
    else:
        out.append(f"_(missing: {bs_path.relative_to(ROOT)})_\n")

    if ms_path.exists():
        ms = json.load(open(ms_path))["results"]
        out.append(f"**Multi-seed mean±std across {len(json.load(open(ms_path))['seeds'])} seeds** "
                   f"(fast subset only):\n")
        out.append("| Model | R² | MAE |")
        out.append("|---|---|---|")
        for name, r in sorted(ms.items(), key=lambda kv: -kv[1]["r2"]["mean"]):
            out.append(f"| {name} | {_fmt_ms(r['r2']['mean'],  r['r2']['std'],  sign=True, prec=4)} | "
                       f"{_fmt_ms(r['mae']['mean'], r['mae']['std'], prec=2)} |")
        out.append("")
    else:
        out.append(f"_(missing: {ms_path.relative_to(ROOT)})_\n")
    return "\n".join(out)


def classification_section(split):
    bs_path = ROOT / "outputs" / f"classify_{split}" / "bootstrap_metrics.json"
    ms_path = ROOT / "outputs" / f"classify_{split}" / "multiseed_metrics.json"
    out = []
    out.append(f"## Classification — {split} split\n")

    if bs_path.exists():
        bs = json.load(open(bs_path))
        rows = sorted(bs.items(), key=lambda kv: -kv[1]["point"]["accuracy"])[:10]
        out.append("**Top 10 by point accuracy with 95% bootstrap CI** (B=1000):\n")
        out.append("| Model | Accuracy | MCC | AUC |")
        out.append("|---|---|---|---|")
        for name, r in rows:
            acc = r["ci"]["accuracy"]
            mcc = r["ci"]["mcc"]
            auc = r["ci"]["auc"]
            out.append(f"| {name} | {_fmt_ci(r['point']['accuracy'], acc['lo'], acc['hi'])} | "
                       f"{_fmt_ci(r['point']['mcc'], mcc['lo'], mcc['hi'], sign=True)} | "
                       f"{_fmt_ci(r['point']['auc'], auc['lo'], auc['hi'])} |")
        out.append("")
    else:
        out.append(f"_(missing: {bs_path.relative_to(ROOT)})_\n")

    if ms_path.exists():
        ms = json.load(open(ms_path))["results"]
        out.append(f"**Multi-seed mean±std across {len(json.load(open(ms_path))['seeds'])} seeds** "
                   f"(native classifiers only):\n")
        out.append("| Model | Accuracy | MCC | AUC |")
        out.append("|---|---|---|---|")
        for name, r in sorted(ms.items(), key=lambda kv: -kv[1]["accuracy"]["mean"]):
            out.append(f"| {name} | {_fmt_ms(r['accuracy']['mean'], r['accuracy']['std'])} | "
                       f"{_fmt_ms(r['mcc']['mean'], r['mcc']['std'], sign=True)} | "
                       f"{_fmt_ms(r['auc']['mean'], r['auc']['std'])} |")
        out.append("")
    else:
        out.append(f"_(missing: {ms_path.relative_to(ROOT)})_\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  choices=["regression", "classification", "both"], default="both")
    ap.add_argument("--split", choices=["random", "scaffold", "both"], default="both")
    args = ap.parse_args()

    splits = ["random", "scaffold"] if args.split == "both" else [args.split]
    tasks  = ["regression", "classification"] if args.task == "both" else [args.task]

    print("# Item-2 statistical-reliability summary\n")
    for task in tasks:
        for split in splits:
            if task == "regression":
                print(regression_section(split))
            else:
                print(classification_section(split))
            print()


if __name__ == "__main__":
    main()
