#!/usr/bin/env python3
"""train_deployable.py — train and persist the deployable triage models.

Why this exists: the main pipeline only saves *predictions* (preds.npz), not
the trained models themselves. To run inference on user-supplied SMILES we
need fitted estimators on disk. This script trains the small subset of models
we want in the Streamlit app, plus the per-model val-optimal threshold and a
calibrator (isotonic regression on val).

Saved artefacts (under `models_saved/`):
  lightgbm.pkl              joblib-pickled LightGBM regressor
  chemprop_dir/             Chemprop checkpoint + featurizer state
  thresholds.json           {model -> val-optimal yield threshold (in %), val accuracy}
  calibrators.pkl           {model -> isotonic regressor mapping score -> P(success)}
  meta.json                 feature dimensions, success threshold, training rows used

The app loads these to predict on new SMILES.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import (
    CONFIG, featurize, load_data, random_split_idx,
)

warnings.filterwarnings("ignore")
SUCCESS_THRESHOLD = 50.0  # yield % cutoff for class 1


def _best_threshold(scores, y_bin):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_bin,  dtype=np.int8)
    n = len(s)
    if n == 0 or np.std(s) < 1e-12:
        return float(np.median(s))
    order = np.argsort(-s)
    s_sort = s[order]; y_sort = y[order]
    cum_pos = np.cumsum(y_sort)
    total_pos = cum_pos[-1]; total_neg = n - total_pos
    fp = np.arange(1, n + 1) - cum_pos
    tp = cum_pos
    correct = tp + (total_neg - fp)
    best = int(np.argmax(correct))
    if total_neg > correct[best]:
        return float(s_sort[0]) + 1e-6
    return float(s_sort[best])


def main():
    out_dir = ROOT / "models_saved"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CONFIG.copy()
    df = load_data(cfg["data_processed"])
    cache = cfg["data_featurized"]
    if cache.exists():
        data = np.load(cache)
        X, y = data["X"], data["y"]
    else:
        X, y = featurize(df, cfg)
    print(f"Loaded features {X.shape}; n={len(y)}")

    tr, val, te = random_split_idx(len(df), cfg)
    X_tr, y_tr   = X[tr],  y[tr]
    X_val, y_val = X[val], y[val]
    X_te,  y_te  = X[te],  y[te]
    print(f"Split: train={len(tr)} val={len(val)} test={len(te)}")

    df_tr  = df.iloc[tr ].reset_index(drop=True)
    df_val = df.iloc[val].reset_index(drop=True)
    df_te  = df.iloc[te ].reset_index(drop=True)

    thresholds = {}
    calibrators = {}

    # ── LightGBM ────────────────────────────────────────────────────────────
    print("\n[LightGBM] training…")
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    val_p_lgb = lgb_model.predict(X_val).astype(np.float32)
    te_p_lgb  = lgb_model.predict(X_te).astype(np.float32)

    y_val_bin = (y_val >= SUCCESS_THRESHOLD).astype(int)
    y_te_bin  = (y_te  >= SUCCESS_THRESHOLD).astype(int)
    thr_lgb = _best_threshold(val_p_lgb, y_val_bin)
    val_acc_lgb = accuracy_score(y_val_bin, (val_p_lgb >= thr_lgb).astype(int))
    te_acc_lgb  = accuracy_score(y_te_bin,  (te_p_lgb  >= thr_lgb).astype(int))
    print(f"  LightGBM: val acc={val_acc_lgb:.3f}  test acc={te_acc_lgb:.3f}  thr={thr_lgb:.2f}")
    thresholds["LightGBM"] = {"threshold": float(thr_lgb),
                              "val_accuracy": float(val_acc_lgb)}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_p_lgb, y_val_bin)
    calibrators["LightGBM"] = iso

    joblib.dump(lgb_model, out_dir / "lightgbm.pkl")
    print(f"  saved → {out_dir / 'lightgbm.pkl'}")

    # ── Chemprop ────────────────────────────────────────────────────────────
    print("\n[Chemprop] training…")
    import torch
    from lightning import pytorch as pl
    from chemprop import data as cd, featurizers as cf, models as cm, nn as cn

    def _build_smi(d):
        out = []
        for _, row in d.iterrows():
            ah = str(row.get("aryl_halide_smiles") or "")
            am = str(row.get("amine_smiles") or "")
            smi = ".".join(s for s in (ah, am) if s and s != "nan")
            out.append(smi if smi else "C")
        return out

    def _xd(X_block):
        return X_block[:, 4096:].astype(np.float32)

    smi_tr  = _build_smi(df_tr);  smi_val = _build_smi(df_val);  smi_te = _build_smi(df_te)
    xd_tr   = _xd(X_tr);          xd_val  = _xd(X_val);          xd_te  = _xd(X_te)

    def _dps(smi_list, y_, xd):
        out = []
        for s, yi, xi in zip(smi_list, y_, xd):
            try:
                out.append(cd.MoleculeDatapoint.from_smi(
                    s, y=np.array([float(yi)], dtype=np.float32),
                    x_d=xi.astype(np.float32),
                ))
            except Exception:
                continue
        return out

    feat = cf.SimpleMoleculeMolGraphFeaturizer()
    ds_tr  = cd.MoleculeDataset(_dps(smi_tr,  y_tr,  xd_tr),  feat)
    ds_val = cd.MoleculeDataset(_dps(smi_val, y_val, xd_val), feat)

    y_scaler  = ds_tr.normalize_targets()
    from sklearn.preprocessing import StandardScaler
    xd_scaler = StandardScaler().fit(xd_tr)
    ds_val.normalize_targets(y_scaler)
    ds_tr.normalize_inputs("X_d", xd_scaler)
    ds_val.normalize_inputs("X_d", xd_scaler)

    extra_dim = xd_tr.shape[1]
    mp  = cn.BondMessagePassing(d_h=300, depth=3)
    agg = cn.MeanAggregation()
    output_unscale = cn.UnscaleTransform.from_standard_scaler(y_scaler)
    ffn = cn.RegressionFFN(
        input_dim=mp.output_dim + extra_dim, hidden_dim=300, n_layers=2,
        output_transform=output_unscale,
    )
    cp_model = cm.MPNN(mp, agg, ffn, batch_norm=True,
                       metrics=[cn.metrics.RMSE()])

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=30, accelerator=accel, devices=1,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, logger=False,
    )
    loader_tr  = cd.build_dataloader(ds_tr,  batch_size=64, num_workers=0)
    loader_val = cd.build_dataloader(ds_val, batch_size=64, num_workers=0, shuffle=False)
    trainer.fit(cp_model, loader_tr, loader_val)

    # Predict on val/test
    ds_val_pred = cd.MoleculeDataset(_dps(smi_val, y_val, xd_val), feat)
    ds_te_pred  = cd.MoleculeDataset(_dps(smi_te,  y_te,  xd_te),  feat)
    ds_val_pred.normalize_inputs("X_d", xd_scaler)
    ds_te_pred.normalize_inputs("X_d",  xd_scaler)
    loader_val_p = cd.build_dataloader(ds_val_pred, batch_size=64, num_workers=0, shuffle=False)
    loader_te_p  = cd.build_dataloader(ds_te_pred,  batch_size=64, num_workers=0, shuffle=False)

    def _predict(loader):
        out = trainer.predict(cp_model, loader)
        return torch.cat([t.detach().cpu() for t in out]).reshape(-1).numpy().astype(np.float32)

    val_p_cp = _predict(loader_val_p)
    te_p_cp  = _predict(loader_te_p)
    thr_cp = _best_threshold(val_p_cp, y_val_bin)
    val_acc_cp = accuracy_score(y_val_bin, (val_p_cp >= thr_cp).astype(int))
    te_acc_cp  = accuracy_score(y_te_bin,  (te_p_cp  >= thr_cp).astype(int))
    print(f"  Chemprop: val acc={val_acc_cp:.3f}  test acc={te_acc_cp:.3f}  thr={thr_cp:.2f}")
    thresholds["Chemprop"] = {"threshold": float(thr_cp),
                              "val_accuracy": float(val_acc_cp)}
    iso_cp = IsotonicRegression(out_of_bounds="clip")
    iso_cp.fit(val_p_cp, y_val_bin)
    calibrators["Chemprop"] = iso_cp

    # Save Chemprop model + scalers
    cp_dir = out_dir / "chemprop_dir"
    cp_dir.mkdir(exist_ok=True)
    torch.save(cp_model.state_dict(), cp_dir / "weights.pt")
    joblib.dump(y_scaler,  cp_dir / "y_scaler.pkl")
    joblib.dump(xd_scaler, cp_dir / "xd_scaler.pkl")
    cp_meta = {
        "d_h": 300, "depth": 3, "ffn_hidden": 300, "ffn_layers": 2,
        "extra_dim": int(extra_dim), "batch_norm": True,
    }
    json.dump(cp_meta, open(cp_dir / "meta.json", "w"), indent=2)
    print(f"  saved Chemprop → {cp_dir}")

    # ── Save thresholds + calibrators + meta ────────────────────────────────
    json.dump(thresholds, open(out_dir / "thresholds.json", "w"), indent=2)
    joblib.dump(calibrators, out_dir / "calibrators.pkl")
    json.dump({
        "feature_dim":      int(X.shape[1]),
        "morgan_radius":    cfg["morgan_radius"],
        "morgan_nbits":     cfg["morgan_nbits"],
        "drfp_nbits":       cfg.get("drfp_nbits"),
        "success_threshold": SUCCESS_THRESHOLD,
        "n_train":          int(len(tr)),
    }, open(out_dir / "meta.json", "w"), indent=2)
    print(f"\nAll artefacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
