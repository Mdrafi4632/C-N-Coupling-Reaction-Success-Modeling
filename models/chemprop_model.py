"""Chemprop — Directed Message-Passing Neural Network (Yang et al. JCIM 2019).

The aryl halide + amine are concatenated into a single multi-fragment SMILES
input; the remaining condition fingerprints (catalyst, base, solvent, additive)
plus the temperature scalar are passed as molecule-level extra features (`x_d`).
The MP module produces a learned embedding which is concatenated with x_d and
fed to a regression FFN head.
"""
import logging
import warnings

import numpy as np

MODEL_NAME = "Chemprop"

# Suppress noisy lightning logging
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def _build_reactant_smiles(df):
    out = []
    for _, row in df.iterrows():
        ah = str(row.get("aryl_halide_smiles") or "")
        am = str(row.get("amine_smiles") or "")
        smi = ".".join(s for s in (ah, am) if s and s != "nan")
        out.append(smi if smi else "C")
    return out


def _condition_features(X_block):
    """X_block is the part of the feature matrix after the aryl+amine FPs.
    We give Chemprop the catalyst/base/solvent/additive bits + temperature
    as molecule-level extra descriptors."""
    return X_block.astype(np.float32)


def _condition_split(X):
    """Aryl+amine FPs occupy the first 2 * 2048 = 4096 cols. Everything else
    (catalyst, base, solvent, additive, optional DRFP, temp) is passed as x_d."""
    return X[:, 4096:]


def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import torch
    from lightning import pytorch as pl
    from chemprop import data, featurizers, models, nn

    df_tr = cfg["df_train"]; df_val = cfg["df_val"]; df_te = cfg["df_test"]
    smi_tr  = _build_reactant_smiles(df_tr)
    smi_val = _build_reactant_smiles(df_val)
    smi_te  = _build_reactant_smiles(df_te)

    xd_tr  = _condition_split(X_tr)
    xd_val = _condition_split(X_val)
    xd_te  = _condition_split(X_te)

    def _make_dps(smi_list, y, xd):
        dps = []
        for s, yi, xi in zip(smi_list, y, xd):
            try:
                dps.append(data.MoleculeDatapoint.from_smi(
                    s, y=np.array([float(yi)], dtype=np.float32),
                    x_d=xi.astype(np.float32),
                ))
            except Exception:
                continue
        return dps

    dps_tr  = _make_dps(smi_tr,  y_tr,  xd_tr)
    dps_val = _make_dps(smi_val, y_val, xd_val)
    dps_te  = _make_dps(smi_te,  y_te,  xd_te)

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    ds_tr  = data.MoleculeDataset(dps_tr,  feat)
    ds_val = data.MoleculeDataset(dps_val, feat)
    ds_te  = data.MoleculeDataset(dps_te,  feat)

    # Per the chemprop README we standardise BOTH the targets and the
    # extra molecule-level features on the training set, then attach
    # an UnscaleTransform to the FFN so its outputs come back in the
    # original yield scale.
    y_scaler  = ds_tr.normalize_targets()
    xd_scaler = _fit_xd_scaler(xd_tr)
    ds_val.normalize_targets(y_scaler)
    ds_te.normalize_targets(y_scaler)
    ds_tr.normalize_inputs("X_d", xd_scaler)
    ds_val.normalize_inputs("X_d", xd_scaler)
    ds_te.normalize_inputs("X_d", xd_scaler)

    loader_tr  = data.build_dataloader(ds_tr,  batch_size=64, num_workers=0)
    loader_val = data.build_dataloader(ds_val, batch_size=64, num_workers=0, shuffle=False)
    loader_te  = data.build_dataloader(ds_te,  batch_size=64, num_workers=0, shuffle=False)

    extra_dim = xd_tr.shape[1]
    mp  = nn.BondMessagePassing(d_h=300, depth=3)
    agg = nn.MeanAggregation()
    output_unscale = nn.UnscaleTransform.from_standard_scaler(y_scaler)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + extra_dim, hidden_dim=300, n_layers=2,
        output_transform=output_unscale,
    )
    model = models.MPNN(mp, agg, ffn, batch_norm=True,
                        metrics=[nn.metrics.RMSE()])

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=30, accelerator=accel, devices=1,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, logger=False,
    )
    trainer.fit(model, loader_tr, loader_val)

    def _predict(loader):
        out = trainer.predict(model, loader)
        # out: list[Tensor] (one per batch).  Each tensor is (batch, 1).
        return torch.cat([t.detach().cpu() for t in out]).reshape(-1).numpy().astype(np.float32)

    return _predict(loader_val), _predict(loader_te)


def _fit_xd_scaler(xd_tr):
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler()
    s.fit(xd_tr)
    return s
