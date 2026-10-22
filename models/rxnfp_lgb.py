"""rxnfp_LGB — LightGBM trained on rxnfp embeddings.

rxnfp encodes a full reaction SMILES through a BERT pretrained on 1M USPTO
reactions (Schwaller et al. Nat. Mach. Intell. 2021). The 256-dim reaction
embedding is concatenated with the temperature scalar and fed to LightGBM.
"""
import numpy as np

MODEL_NAME = "rxnfp_LGB"

_CACHE = {}


def _build_rxn_smiles(df):
    """Build a reaction SMILES from per-component columns."""
    cols = ["aryl_halide_smiles", "amine_smiles", "catalyst_smiles",
            "base_smiles", "solvent_smiles", "additive_smiles"]
    out = []
    for _, row in df.iterrows():
        reactants = ".".join(str(row[c]) for c in cols if row.get(c) and str(row[c]) != "nan")
        if not reactants:
            reactants = "C"  # placeholder
        out.append(f"{reactants}>>")
    return out


def _embed(smiles_list):
    if "gen" not in _CACHE:
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator, get_default_model_and_tokenizer,
        )
        model, tokenizer = get_default_model_and_tokenizer()
        _CACHE["gen"] = RXNBERTFingerprintGenerator(model, tokenizer)
    gen = _CACHE["gen"]
    embs = []
    for smi in smiles_list:
        try:
            embs.append(gen.convert(smi))
        except Exception:
            embs.append([0.0] * 256)
    return np.array(embs, dtype=np.float32)


def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import lightgbm as lgb

    df_tr = cfg["df_train"]; df_val = cfg["df_val"]; df_te = cfg["df_test"]
    rxn_tr  = _build_rxn_smiles(df_tr)
    rxn_val = _build_rxn_smiles(df_val)
    rxn_te  = _build_rxn_smiles(df_te)

    E_tr  = _embed(rxn_tr)
    E_val = _embed(rxn_val)
    E_te  = _embed(rxn_te)

    # Append temperature (the trailing scalar feature in X)
    t_tr  = X_tr[:,  -1:].astype(np.float32)
    t_val = X_val[:, -1:].astype(np.float32)
    t_te  = X_te[:,  -1:].astype(np.float32)
    F_tr  = np.concatenate([E_tr,  t_tr],  axis=1)
    F_val = np.concatenate([E_val, t_val], axis=1)
    F_te  = np.concatenate([E_te,  t_te],  axis=1)

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(F_tr, y_tr, eval_set=[(F_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.predict(F_val).astype(np.float32), model.predict(F_te).astype(np.float32)
