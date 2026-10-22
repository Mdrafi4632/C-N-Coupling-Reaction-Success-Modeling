#!/usr/bin/env python3
"""
main.py — C-N Coupling Reaction Yield Prediction Pipeline

Pipeline:
  1. Load data      data/processed/cn_reactions_raw.csv
  2. Featurize      Morgan FP (per component) + DRFP + temperature
  3. Split          random 70/15/15  or  scaffold-based
  4. Run models     all *.py files in models/ that export MODEL_NAME
  5. CFA ensemble   run.py greedy + pairwise selection
  6. Save metrics   outputs/metrics.json

Usage:
  python main.py                          # all models, random split
  python main.py --models XGBoost Ridge   # specific models
  python main.py --split scaffold         # scaffold-based split
  python main.py --no-cfa                 # skip CFA
  python main.py --download               # fetch ORD data first
"""

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from drfp import DrfpEncoder
    DRFP_AVAILABLE = True
except ImportError:
    DRFP_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from cfa import cfa_greedy, cfa_k_way, cfa_pairwise
    CFA_AVAILABLE = True
except ImportError as e:
    CFA_AVAILABLE = False
    warnings.warn(f"CFA unavailable: {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# -- Configuration ------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "data_processed":  _ROOT / "data" / "processed" / "cn_reactions_raw.csv",
    "data_featurized": _ROOT / "data" / "processed" / "cn_features.npz",
    "output_dir":      _ROOT / "outputs",
    "models_dir":      _ROOT / "models",
    "seed": 42,
    # Featurization
    "morgan_radius": 2,
    "morgan_nbits":  2048,
    "drfp_nbits":    2048,
    # Split fractions
    "train_frac": 0.70,
    "val_frac":   0.15,
    "test_frac":  0.15,
    # Model hyperparameters passed to train(cfg=...)
    "xgb_params": {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "seed": 42, "n_jobs": -1,
    },
    "lgb_params": {
        "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    },
    "mlp_params": {
        "hidden_sizes": [1024, 512, 256], "dropout": 0.3,
        "lr": 1e-3, "epochs": 100, "batch_size": 256, "patience": 20,
    },
    # CFA
    "cfa_time_limit": 60.0,
    "cfa_max_rounds": 200,
}

# -- Data loading -------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["yield"])
    df["yield"] = df["yield"].clip(0, 100)
    log.info(f"Loaded {len(df)} reactions from {path}")
    return df

# -- Featurization ------------------------------------------------------------

SMILES_COLS = ["aryl_halide_smiles", "amine_smiles", "catalyst_smiles",
               "base_smiles", "additive_smiles", "solvent_smiles"]

def smiles_to_morgan(smiles: str, radius: int, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), arr)
    return arr

def featurize_morgan(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    r, n = cfg["morgan_radius"], cfg["morgan_nbits"]
    cols = [c for c in SMILES_COLS if c in df.columns]
    fps = [
        np.stack([smiles_to_morgan(s, r, n) for s in tqdm(df[c].fillna(""), desc=c, leave=False)])
        for c in cols
    ]
    return np.concatenate(fps, axis=1)

def featurize_drfp(df: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    if not DRFP_AVAILABLE or "reaction_smiles" not in df.columns:
        return None
    try:
        fps = DrfpEncoder().encode(df["reaction_smiles"].fillna(">>").tolist(),
                                   n_folded_length=cfg["drfp_nbits"])
        return np.array(fps, dtype=np.float32)
    except Exception as e:
        log.warning(f"DRFP failed: {e}")
        return None

def featurize_conditions(df: pd.DataFrame) -> np.ndarray:
    if "temperature_c" not in df.columns:
        return np.zeros((len(df), 0), dtype=np.float32)
    t = df["temperature_c"].where(df["temperature_c"].between(-100, 500))
    t = t.fillna(t.median())
    t_norm = (t - t.mean()) / (t.std() + 1e-8)
    return t_norm.values.reshape(-1, 1).astype(np.float32)

def featurize(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    log.info("Featurizing...")
    parts = [featurize_morgan(df, cfg)]
    if (X_drfp := featurize_drfp(df, cfg)) is not None:
        parts.append(X_drfp)
    if (X_cond := featurize_conditions(df)).shape[1] > 0:
        parts.append(X_cond)
    X = np.concatenate(parts, axis=1)
    y = df["yield"].values.astype(np.float32)
    log.info(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")
    return X, y

# -- Splits -------------------------------------------------------------------

def random_split_idx(n: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = cfg["seed"]
    idx = np.arange(n)
    tr_val, te = train_test_split(idx, test_size=cfg["test_frac"], random_state=seed)
    val_size = cfg["val_frac"] / (1 - cfg["test_frac"])
    tr, val = train_test_split(tr_val, test_size=val_size, random_state=seed)
    return tr, val, te


def scaffold_split_idx(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "aryl_halide_smiles" not in df.columns:
        log.warning("No aryl_halide_smiles — falling back to random split.")
        return random_split_idx(len(df), cfg)

    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(df["aryl_halide_smiles"].fillna("")):
        mol = Chem.MolFromSmiles(smi) if smi else None
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else "invalid"
        except Exception:
            key = "invalid"
        scaffolds.setdefault(key, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    n = len(df)
    for group in sorted(scaffolds.values(), key=len, reverse=True):
        frac = len(train_idx) / n
        if frac < cfg["train_frac"]:
            train_idx.extend(group)
        elif frac < cfg["train_frac"] + cfg["val_frac"]:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)  if val_idx  else train_idx[:10]
    test_idx  = np.array(test_idx) if test_idx else train_idx[:10]
    log.info(f"Scaffold split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return train_idx, val_idx, test_idx

# -- Model discovery and execution --------------------------------------------
# Contract for models/*.py:
#   MODEL_NAME: str
#   make_model() -> sklearn regressor          (main.py calls fit/predict)
#   train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg) -> (val_pred, tes_pred)

def discover_models(models_dir: Path) -> list[tuple[str, Any]]:
    models_dir = Path(models_dir)
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    found = []
    for fpath in sorted(models_dir.glob("*.py")):
        if fpath.stem.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(fpath.stem, fpath)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "MODEL_NAME"):
                found.append((mod.MODEL_NAME, mod))
                log.info(f"  [discover] {mod.MODEL_NAME}")
        except Exception as e:
            log.warning(f"  [skip] {fpath.stem}: {e}")
    return found

def run_model(name, mod, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    log.info(f"\n===== {name} =====")
    t0 = time.time()
    try:
        if hasattr(mod, "train"):
            val_p, tes_p = mod.train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg)
        elif hasattr(mod, "make_model"):
            from sklearn.base import clone
            est = clone(mod.make_model())
            est.fit(X_tr, y_tr)
            val_p, tes_p = est.predict(X_val), est.predict(X_te)
        else:
            log.warning(f"[{name}] no train() or make_model() — skipped")
            return None
        m = metrics(y_te, tes_p)
        log.info(f"  [{name}] R²={m['r2']:.4f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  ({time.time()-t0:.1f}s)")
        return val_p, tes_p
    except Exception as e:
        log.error(f"[{name}] ERROR: {e}")
        return None

# -- Metrics ------------------------------------------------------------------

def metrics(y_true, y_pred) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def log_metrics(name, val_m, tes_m):
    log.info(f"  {name:20s}  val R²={val_m['r2']:.4f} MAE={val_m['mae']:.2f} | "
             f"test R²={tes_m['r2']:.4f} MAE={tes_m['mae']:.2f}")

# -- CFA ----------------------------------------------------------------------

def run_cfa(val_preds, tes_preds, y_val, y_te, cfg):
    if not CFA_AVAILABLE:
        log.warning("CFA unavailable — skipping.")
        return None, None

    out_dir = cfg["output_dir"]
    log.info(f"\nRunning CFA over {len(val_preds)} models...")

    greedy = cfa_greedy(
        val_preds, tes_preds, y_val, y_te,
        time_limit=cfg["cfa_time_limit"],
        max_rounds=cfg["cfa_max_rounds"],
    )
    (out_dir / "cfa_greedy.json").write_text(json.dumps(greedy, indent=2))

    # k-way exhaustive search (k=2..5)
    kway = cfa_k_way(val_preds, tes_preds, y_val, y_te, max_k=5)
    kway_sorted = sorted(kway, key=lambda r: -r["val_r2"])
    (out_dir / "cfa_kway.json").write_text(json.dumps(kway_sorted[:500], indent=2))

    # Backward-compat pairwise file
    top_pairs = [
        {"pair": r["combo"], "method": r["method"], "cd": r["group_cd"],
         "val_r2": r["val_r2"], "test_r2": r["test_r2"]}
        for r in kway_sorted if r["k"] == 2
    ][:100]
    (out_dir / "cfa_pairwise.json").write_text(json.dumps(top_pairs, indent=2))

    if greedy:
        log.info(
            f"[CFA] Greedy: {' + '.join(greedy['best_combo'])} "
            f"({greedy['best_method']})  "
            f"val R²={greedy['val_r2']:.4f}  test R²={greedy['test_r2']:.4f}"
        )
    if kway_sorted:
        best = kway_sorted[0]
        log.info(
            f"[CFA] Best k-way (k={best['k']}): {best['combo'][:60]} "
            f"({best['method']})  "
            f"val R²={best['val_r2']:.4f}  test R²={best['test_r2']:.4f}"
        )
    return greedy, kway

# -- Main pipeline ------------------------------------------------------------

def main(args):
    t0  = time.time()
    cfg = CONFIG.copy()
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    if args.download:
        ret = subprocess.run([sys.executable, str(_ROOT / "scripts" / "download_data.py")], check=False)
        if ret.returncode != 0:
            log.warning("Download script had errors — continuing.")

    # 1. Load
    df = load_data(Path(args.data) if args.data else cfg["data_processed"])
    if df.empty:
        log.error("No data. Run: python main.py --download")
        sys.exit(1)

    # 2. Featurize (cache to disk)
    cache = cfg["data_featurized"]
    if cache.exists() and not args.refeaturize:
        data = np.load(cache)
        X, y = data["X"], data["y"]
        log.info(f"Loaded cached features {X.shape}")
    else:
        X, y = featurize(df, cfg)
        np.savez_compressed(cache, X=X, y=y)

    # 3. Split (compute indices first so SMILES-aware models can index into df)
    if args.split == "scaffold":
        tr_idx, val_idx, te_idx = scaffold_split_idx(df, cfg)
    else:
        tr_idx, val_idx, te_idx = random_split_idx(len(df), cfg)
    X_tr, y_tr  = X[tr_idx],  y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te  = X[te_idx],  y[te_idx]
    cfg["df_train"] = df.iloc[tr_idx].reset_index(drop=True)
    cfg["df_val"]   = df.iloc[val_idx].reset_index(drop=True)
    cfg["df_test"]  = df.iloc[te_idx].reset_index(drop=True)
    log.info(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    # 4. Discover models
    models_dir  = Path(args.models_dir) if args.models_dir else cfg["models_dir"]
    all_models  = discover_models(models_dir)
    if not all_models:
        log.error(f"No models found in {models_dir}")
        sys.exit(1)

    if args.models:
        filt = {n.lower() for n in args.models}
        all_models = [(n, m) for n, m in all_models
                      if n.lower() in filt or any(f in n.lower() for f in filt)]
        if not all_models:
            log.error(f"--models {args.models} matched nothing")
            sys.exit(1)

    log.info(f"\nRunning {len(all_models)} model(s): {[n for n, _ in all_models]}")

    val_preds, tes_preds, all_metrics = {}, {}, {}
    for name, mod in all_models:
        result = run_model(name, mod, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg)
        if result:
            val_p, tes_p = result
            val_preds[name]  = val_p
            tes_preds[name]  = tes_p
            all_metrics[name] = metrics(y_te, tes_p)
            log_metrics(name, metrics(y_val, val_p), all_metrics[name])

    # 5. Persist per-model predictions so CFA can be re-run without retraining
    if val_preds:
        preds_path = cfg["output_dir"] / "preds.npz"
        np.savez_compressed(
            preds_path,
            names=np.array(list(val_preds.keys())),
            val=np.stack(list(val_preds.values())),
            test=np.stack(list(tes_preds.values())),
            y_val=y_val, y_te=y_te,
        )
        log.info(f"Saved predictions → {preds_path}")

    # 6. CFA
    if not args.no_cfa and len(val_preds) >= 2:
        run_cfa(val_preds, tes_preds, y_val, y_te, cfg)

    # 7. Save
    out = cfg["output_dir"] / "metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2))
    log.info(f"\nSaved metrics → {out}  ({time.time()-t0:.1f}s total)")


def parse_args():
    p = argparse.ArgumentParser(description="C-N Coupling Yield Prediction")
    p.add_argument("--data",        help="Path to reactions CSV")
    p.add_argument("--models",      nargs="+", metavar="NAME",
                   help="Filter models by MODEL_NAME (default: all)")
    p.add_argument("--models-dir",  metavar="DIR", help="Model files directory")
    p.add_argument("--split",       choices=["random", "scaffold"], default="random")
    p.add_argument("--no-cfa",      action="store_true")
    p.add_argument("--download",    action="store_true", help="Fetch ORD data first")
    p.add_argument("--refeaturize", action="store_true", help="Ignore feature cache")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["seed"] = args.seed
    np.random.seed(args.seed)
    main(args)
