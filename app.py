#!/usr/bin/env python3
"""app.py — Streamlit triage tool for Buchwald--Hartwig success prediction.

Run with:
    uv run streamlit run app.py

Loads the LightGBM and Chemprop artefacts from `models_saved/`, accepts six
component SMILES + a temperature, and returns:

  * the calibrated probability that yield will exceed 50%,
  * the per-model raw scores,
  * a binary recommendation OR an ``uncertain`` abstention when the two
    models disagree (reject-option).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models_saved"


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all saved artefacts once per session."""
    import torch
    from chemprop import models as cm, nn as cn

    meta        = json.loads((MODELS_DIR / "meta.json").read_text())
    thresholds  = json.loads((MODELS_DIR / "thresholds.json").read_text())
    calibrators = joblib.load(MODELS_DIR / "calibrators.pkl")
    lgb_model   = joblib.load(MODELS_DIR / "lightgbm.pkl")

    cp_dir = MODELS_DIR / "chemprop_dir"
    cp_meta   = json.loads((cp_dir / "meta.json").read_text())
    y_scaler  = joblib.load(cp_dir / "y_scaler.pkl")
    xd_scaler = joblib.load(cp_dir / "xd_scaler.pkl")

    mp  = cn.BondMessagePassing(d_h=cp_meta["d_h"], depth=cp_meta["depth"])
    agg = cn.MeanAggregation()
    output_unscale = cn.UnscaleTransform.from_standard_scaler(y_scaler)
    ffn = cn.RegressionFFN(
        input_dim=mp.output_dim + cp_meta["extra_dim"],
        hidden_dim=cp_meta["ffn_hidden"], n_layers=cp_meta["ffn_layers"],
        output_transform=output_unscale,
    )
    cp_model = cm.MPNN(mp, agg, ffn, batch_norm=cp_meta["batch_norm"],
                       metrics=[cn.metrics.RMSE()])
    cp_model.load_state_dict(torch.load(cp_dir / "weights.pt", map_location="cpu",
                                        weights_only=True))
    cp_model.eval()

    return {
        "meta": meta, "thresholds": thresholds, "calibrators": calibrators,
        "lgb_model": lgb_model, "cp_model": cp_model,
        "xd_scaler": xd_scaler,
    }


# ── Featurisation ────────────────────────────────────────────────────────────
SMILES_COMPONENT_ORDER = [
    "aryl_halide_smiles", "amine_smiles", "catalyst_smiles",
    "base_smiles", "additive_smiles", "solvent_smiles",
]
MORGAN_RADIUS = 2
MORGAN_BITS   = 2048
TEMP_MEAN     = 95.0   # rough training-set centre for normalisation
TEMP_STD      = 25.0


def smi_to_morgan(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi or "") if smi else None
    arr = np.zeros(MORGAN_BITS, dtype=np.float32)
    if mol is None:
        return arr
    DataStructs.ConvertToNumpyArray(
        GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS), arr,
    )
    return arr


def featurise(comp_smiles: dict, temperature_c: float, feature_dim: int) -> np.ndarray:
    fps = [smi_to_morgan(comp_smiles.get(c, "")) for c in SMILES_COMPONENT_ORDER]
    t_norm = (temperature_c - TEMP_MEAN) / (TEMP_STD + 1e-8)
    x = np.concatenate(fps + [np.array([t_norm], dtype=np.float32)])
    if x.shape[0] < feature_dim:                    # pad to match cached shape
        x = np.concatenate([x, np.zeros(feature_dim - x.shape[0], dtype=np.float32)])
    return x.reshape(1, -1)


# ── Inference ────────────────────────────────────────────────────────────────
def chemprop_predict(cp_model, xd_scaler, comp_smiles, X_full):
    """Predict yield with Chemprop on a single example."""
    import torch
    from chemprop import data as cd, featurizers as cf

    ah = comp_smiles.get("aryl_halide_smiles", "") or ""
    am = comp_smiles.get("amine_smiles", "") or ""
    smi = ".".join(s for s in (ah, am) if s) or "C"
    xd  = X_full[0, 4096:].astype(np.float32)
    xd_norm = xd_scaler.transform(xd.reshape(1, -1))[0]
    try:
        dp = cd.MoleculeDatapoint.from_smi(
            smi, y=np.array([0.0], dtype=np.float32),
            x_d=xd_norm.astype(np.float32),
        )
    except Exception as e:
        raise ValueError(f"Could not parse aryl_halide+amine SMILES: {e}")

    feat = cf.SimpleMoleculeMolGraphFeaturizer()
    ds = cd.MoleculeDataset([dp], feat)
    loader = cd.build_dataloader(ds, batch_size=1, num_workers=0, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = cp_model(batch.bmg, batch.V_d, batch.X_d)
    return float(pred.detach().cpu().numpy().reshape(-1)[0])


def predict_all(comp_smiles, temperature_c, models):
    feature_dim = models["meta"]["feature_dim"]
    X = featurise(comp_smiles, temperature_c, feature_dim)
    raw = {}
    raw["LightGBM"] = float(models["lgb_model"].predict(X)[0])
    raw["Chemprop"] = chemprop_predict(
        models["cp_model"], models["xd_scaler"], comp_smiles, X,
    )

    cal = {n: float(np.clip(models["calibrators"][n].predict([raw[n]])[0], 0.0, 1.0))
           for n in raw}
    classes = {n: int(raw[n] >= models["thresholds"][n]["threshold"]) for n in raw}
    return raw, cal, classes


# ── Reject-option logic ──────────────────────────────────────────────────────
def reject_option(cal_probs: dict, abstain_gap: float = 0.20):
    """If models disagree by more than `abstain_gap` in calibrated probability,
    abstain. Otherwise return the mean."""
    p = np.array(list(cal_probs.values()))
    mean = float(p.mean())
    spread = float(p.max() - p.min())
    abstain = bool(spread > abstain_gap)
    return mean, spread, abstain


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="C–N Coupling Triage", page_icon="🧪", layout="centered")

st.title("🧪 C–N Coupling Reaction Triage")
st.caption("Predict whether a Buchwald–Hartwig amination will exceed 50 % yield.")

with st.spinner("Loading saved models …"):
    MODELS = load_models()
SUCCESS_THR = MODELS["meta"]["success_threshold"]

with st.expander("ℹ️  About this tool", expanded=False):
    st.markdown(f"""
This is a **success/failure classifier** for palladium-catalysed C–N coupling
(Buchwald–Hartwig amination), not a yield oracle.  It returns the calibrated
probability that the reaction will produce **≥ {SUCCESS_THR:.0f} %** yield
and a binary recommendation, with an *uncertain* abstention when the two
underlying models disagree.

**Models in the ensemble:** LightGBM (gradient-boosted trees on 14 k-dim Morgan
fingerprints) + Chemprop (directed message-passing neural network on the
molecular graph).  Both were trained on 3 825 reactions extracted from the
Open Reaction Database, validated on 820, and report 91–93 % accuracy on
in-distribution test data.

**Limitations:**
- Trained on Buchwald–Hartwig amination only.  Predictions on other
  reaction classes are not meaningful.
- On unseen aryl-halide scaffolds (out-of-distribution) accuracy drops to
  about 60 %.  The reject-option below is your friend.
- The reported probability is calibrated on the validation set.  At 0.7
  the empirical success rate is roughly 70 %, not 95 %.
""")

st.subheader("Reactants and conditions")
col1, col2 = st.columns(2)
with col1:
    aryl    = st.text_input("Aryl halide SMILES",
                            value="Brc1ccc(cc1)C(F)(F)F",
                            help="The C–X partner. Bromo, chloro, iodo, triflate.")
    catalyst = st.text_input("Catalyst SMILES (Pd source / ligand)",
                             value="CC1(C)c2ccccc2-c2ccccc21OP(c1ccccc1)c1ccccc1",
                             help="Phosphine ligand or pre-formed Pd complex.")
    solvent  = st.text_input("Solvent SMILES",
                             value="C1COCCO1",
                             help="e.g., dioxane (C1COCCO1), toluene (Cc1ccccc1), THF.")
with col2:
    amine    = st.text_input("Amine SMILES (HN<)",
                             value="N1CCOCC1",
                             help="The N–H partner.  Primary, secondary, or aryl amine.")
    base     = st.text_input("Base SMILES",
                             value="[Cs+].[O-]C(=O)O[Cs+]",
                             help="e.g., Cs2CO3, NaOtBu, K3PO4.")
    additive = st.text_input("Additive SMILES (optional)",
                             value="",
                             help="Optional. Leave blank if none.")

temperature = st.slider("Temperature (°C)", min_value=20, max_value=180, value=110)
abstain_gap = st.slider(
    "Reject-option threshold (max disagreement between models)",
    min_value=0.05, max_value=0.50, value=0.20, step=0.05,
    help="When the two models disagree by more than this in calibrated "
         "probability, the tool abstains rather than commit to a binary call.",
)

if st.button("🔬 Predict", type="primary", use_container_width=True):
    comp = {
        "aryl_halide_smiles": aryl,
        "amine_smiles":       amine,
        "catalyst_smiles":    catalyst,
        "base_smiles":        base,
        "additive_smiles":    additive,
        "solvent_smiles":     solvent,
    }
    if not aryl or not amine:
        st.error("Aryl halide and amine SMILES are both required.")
        st.stop()
    with st.spinner("Running inference …"):
        try:
            raw, cal, classes = predict_all(comp, float(temperature), MODELS)
            mean_p, spread, abstain = reject_option(cal, abstain_gap)
        except Exception as e:
            st.exception(e); st.stop()

    st.subheader("Result")
    if abstain:
        st.warning(
            f"**🤔 Uncertain — verify in lab.** "
            f"The two models disagree by {spread:.2f} in calibrated probability "
            f"(threshold {abstain_gap:.2f}).  Mean probability of success "
            f"is {mean_p:.2%}, but you should not rely on a binary call here."
        )
    elif mean_p >= 0.5:
        st.success(
            f"**✅ Likely to succeed.** "
            f"Estimated probability of yield ≥ {SUCCESS_THR:.0f} %: **{mean_p:.1%}**.  "
            f"Models agree to within {spread:.2f}."
        )
    else:
        st.error(
            f"**❌ Likely to fail.** "
            f"Estimated probability of yield ≥ {SUCCESS_THR:.0f} %: **{mean_p:.1%}**.  "
            f"Consider screening alternative conditions before committing bench time."
        )

    st.markdown("**Per-model breakdown**")
    for name in ("LightGBM", "Chemprop"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(name, "raw yield est.", f"{raw[name]:.1f} %")
        col_b.metric(name, "calibrated p(success)", f"{cal[name]:.2%}")
        col_c.metric(name, "binary call",
                     "✅ success" if classes[name] else "❌ fail")

    with st.expander("Raw scores and thresholds"):
        st.json({
            "raw_yield_estimates": raw,
            "calibrated_probabilities": cal,
            "binary_classes": classes,
            "model_thresholds": MODELS["thresholds"],
            "ensemble_mean_probability": mean_p,
            "ensemble_disagreement": spread,
            "abstained": abstain,
        })

st.caption(
    "Built on 5,465 ORD reactions • LightGBM + Chemprop ensemble • "
    "isotonic-calibrated probabilities • reject-option on model disagreement."
)
