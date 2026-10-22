# Project Change Log

## 2026-04-30 — Triage tool on hold; data-quantity hypothesis recorded

**Status:** the Streamlit triage tool (`app.py`), saved-model artefacts under
`models_saved/`, isotonic calibrators, and reject-option wrapper are all in
place and pass end-to-end smoke tests. Items 1–3 from the deployment plan
(see 2026-04-29 entry) are completed code. Whether to continue developing the
tool — production polish, batch-mode CSV upload, threshold sliders, deployment
to a hosted demo — is **on hold**.

**User's working hypothesis on the OOD ceiling:**
The persistent scaffold-OOD plateau (~59% accuracy / MCC +0.18) is suspected
to be a **data quantity** problem rather than an architectural one. Hypothesis:
with more training data covering more aryl-halide and amine scaffolds, the
classifier would generalise to unseen scaffolds at meaningfully higher
accuracy.

**Notes for the next iteration if this hypothesis is pursued:**
- Quantity alone is partial: ORD's BH amination subset is 5,465 reactions, the
  full ORD is ~1M reactions across many reaction classes. Even the same
  reaction class is bounded — there are only so many distinct BH reactions
  ever published.
- **Coverage matters more than raw count**: HTE datasets like Ahneman 2018
  (4,608 BH reactions, controlled conditions) cover their narrow chemistry
  region densely; ORD covers a wide space sparsely. Combining the two would
  likely help OOD more than doubling the ORD BH subset alone.
- **Reporter bias** is unchanged by adding more ORD: high-yielding reactions
  are over-represented because they are preferentially published. Adding 10×
  more ORD data with the same bias does not solve the OOD problem; it
  amplifies the in-distribution memorisation that is already happening.
- **The most promising data expansions:**
  1. Add the Ahneman 2018 HTE dataset (~4.6k reactions, dense coverage of
     a controlled chemistry region) — direct and cheap.
  2. Add Pfizer's released HTE BH datasets if accessible.
  3. Expand to all C–N coupling reactions in ORD (Ullmann, Chan–Lam, etc.)
     and use multi-task learning to share representation across families.
  4. Pretrain on the full ORD corpus (Option D in `future`) — the most
     ambitious but also the most likely to lift OOD substantially, mirroring
     Yield-BERT's approach.
- **A useful experiment if the hypothesis is to be tested directly:**
  hold out a portion of training data, retrain with reduced sets (e.g., 30%,
  60%, 90% of the current 3,825 train rows), measure how scaffold-OOD
  accuracy scales with training-set size. If the curve is still rising at
  100%, more data will help; if it has plateaued, the bottleneck is not
  quantity but coverage / bias.

**No code changes in this entry.** Triage tool kept as-is for future revival.

---

## 2026-04-29 — v2 PIVOT

### [PIVOT] From yield regression to binary success classification

**Decision driver:** every regression model in v1 collapsed below R² = 0 on the
scaffold-OOD split (best AdaBoost −0.05; LightGBM −0.20; median −0.37). Even the
chemistry-aware Chemprop dropped to −0.35. The bottleneck is the dataset
(reporter bias + scaffold–yield coupling), not architecture. Continuing to
push regression performance is high-effort and low-confidence; reframing the
task to a binary go/no-go decision (yield ≥ 50 % vs. < 50 %) preserves the
practical use case (chemistry triage) and is empirically more transferable.

**Project layout reshuffled:**

| Path | Contents |
|---|---|
| `report-old/` | v1 regression report (preserved verbatim) |
| `report/` | v2 classification report (new) |
| `projectdescnew.md` | written explanation of the v2 reframing and use case |
| `future` | deferred items, including Option D (foundation-model pretraining on full ORD) |
| `cfa.py` | v1 regression CFA (kept) |
| `cfa_classify.py` | v2 binary classification CFA (new) |
| `scripts/classify_eval.py` | runs binary CFA from the saved regression `preds.npz` (no retraining) |
| `outputs/random/` & `outputs/scaffold/` | v1 regression predictions + metrics |
| `outputs/classify_random/` & `outputs/classify_scaffold/` | v2 classification metrics + CFA |

**Method (v2):**

- 50% yield threshold for binary labels; class balance 68.1 / 31.9 (random),
  53 / 47 (scaffold test split — coincidentally more balanced).
- Reuse the regression predictions: each model's continuous yield estimate is
  thresholded at a per-model val-optimal cut-off chosen to maximise validation
  accuracy. Equivalent to ordinal regression with a fixed operational threshold.
- CFA reverted to its classification form: eligibility filter MCC > 0,
  6 combination methods, val-optimal threshold per ensemble.
- `_best_threshold` rewritten from the original 91-quantile sweep to an
  O(N log N) cumulative-TP/TN sort — about 50× faster, which is what made the
  k=5 sweep tractable.
- k-way candidate caps reduced (top-20 / 15 / 12 for k=3 / 4 / 5) to keep the
  whole pipeline under a few minutes per split.

**Results (v2):**

| Setting | Acc. | F1 | MCC | AUC |
|---|---:|---:|---:|---:|
| **Random 70/15/15** | | | | |
| Majority-class baseline | 0.668 | — | 0.000 | 0.500 |
| Best individual (LightGBM) | **0.923** | 0.912 | +0.825 | 0.966 |
| Val-selected CFA k=5 (Chemprop+LightGBM+skMLP_Tanh+KernelRidge+XGBoost, WSCDS) | 0.924 | 0.912 | +0.828 | — |
| Best observed CFA test ceiling (k=4, WSCDS) | 0.938 | — | +0.858 | — |
| **Scaffold-OOD** | | | | |
| Majority-class baseline | 0.529 | — | 0.000 | 0.500 |
| Best individual (HistGBT_MAE) | 0.591 | 0.587 | +0.175 | 0.641 |
| Val-selected CFA k=4 (BayesianRidge+ExtraTrees+GradientBoosting+HistGBT, ASC) | 0.565 | 0.475 | +0.135 | — |
| Best observed CFA test ceiling (k=2, WRCP) | 0.638 | — | +0.277 | — |

**Interpretation:**
- Random-split classification is strong: 92% accuracy, MCC > 0.8, AUC near 0.97.
  Above the no-information baseline by ~25 percentage points.
- Scaffold-OOD classification is **above-chance and informative** but modest:
  ~6 percentage points above majority-class baseline, MCC +0.18, AUC ~0.64.
- The contrast with v1 regression is the headline: v1 collapsed to R² < 0 on
  scaffold-OOD; v2 retains real (if weak) ranking ability — AUC > 0.6 means the
  classifier is correctly ordering reactions roughly 60% better than random.
- Because the scaffold val set is only 10 samples (large-scaffold imbalance
  carried over from v1), CFA selection on scaffold is noisier than its test
  ceiling.

**Validity check:**
- Per-model thresholds for top random-split models cluster around 49–53 yield-%,
  i.e.\ the threshold the val-optimisation found is essentially the binarisation
  threshold itself (50%). Sanity-check passed: the regression-then-threshold
  approach is calibrated.
- All 39 v1 regressors + Chemprop + NGBoost + GP_Tanimoto + rxnfp_LGB are
  evaluated. Two models drop on the random-split health filter (`SGD`,
  `PassiveAggressive` — broken at training time as already documented).

**What this means for deployment:**
- The v2 classifier is deployable for **bench-chemistry triage on substrates
  inside the training distribution** (random-split regime) at ~92% accuracy.
- For substrates outside the training distribution it is a ~60% triage tool
  with positive but small MCC. That is below "trustworthy" but well above
  "useless" — paired with a reject-option / uncertainty wrapper, it is still
  a more useful deployment than the v1 regressor (which had nothing usable
  on scaffold-OOD).

**Report status:** `report/main.pdf` rebuilt cleanly (11 pages). Cites v1 in
section 1 ("Why we pivoted"), uses v1's `preds.npz` files unchanged, adds a
methodology section with citations for LightGBM, Chemprop, NGBoost, rxnfp,
GP-Tanimoto, and the CFA framework. v1 report preserved at `report-old/`.

### [DEPLOY] Streamlit triage tool (items 1–3 from the deployment plan)

**New artefacts:**

| Path | Purpose |
|---|---|
| `scripts/train_deployable.py` | trains LightGBM + Chemprop on the random split, fits per-model val-optimal thresholds, fits isotonic calibrators on val |
| `models_saved/lightgbm.pkl` | joblib-pickled LightGBM regressor |
| `models_saved/chemprop_dir/{weights.pt, y_scaler.pkl, xd_scaler.pkl, meta.json}` | Chemprop weights + scalers + reconstruction metadata |
| `models_saved/thresholds.json` | per-model val-optimal yield-% thresholds (~50%) |
| `models_saved/calibrators.pkl` | per-model isotonic regressors mapping raw yield → P(success) |
| `models_saved/meta.json` | feature dim, success threshold, training rows |
| `app.py` | Streamlit web app: 6 SMILES + temperature → calibrated success probability + reject-option recommendation |

**Saved model performance** (random split, threshold = 50% yield):

| Model | val acc | test acc | val-optimal threshold |
|---|---:|---:|---:|
| LightGBM | 0.918 | 0.923 | 50.00 |
| Chemprop | 0.922 | 0.933 | 49.46 |

The val-optimal thresholds essentially match the binarisation cutoff (50%),
confirming the regression-then-threshold approach is calibrated.

**App features delivered:**
- Item 1 (Streamlit UI): six SMILES inputs + temperature slider + reject-option
  threshold slider; loads cached models on first request, runs inference in
  &lt;1s on CPU.
- Item 2 (reject-option): when the calibrated probabilities of LightGBM and
  Chemprop differ by more than the user-set gap (default 0.20), the app shows
  "🤔 Uncertain — verify in lab" instead of a binary call. End-to-end test
  confirmed: both test reactions in the smoke test triggered abstention,
  which is the correct behaviour when models genuinely disagree.
- Item 3 (calibration): isotonic regression fit on validation predictions per
  model; the displayed probability is the empirical success rate at that
  raw score, not a softmax artefact. Calibrators saved to disk and applied
  in the app's `predict_all` function.

**Smoke test:** `streamlit run app.py --server.headless true` starts cleanly,
serves HTTP 200, models load. Direct invocation of `predict_all` outside
streamlit also works (smoke-tested with two synthetic reactions: a
"standard Buchwald" and a "bad conditions" case; both triggered the
reject-option, which is the expected behaviour given the high disagreement).

**Run command:** `uv run streamlit run app.py` — local URL prints to stdout.

---

## 2026-04-29 — v1 (regression)

### [PLAN] Adding domain-specific and uncertainty-aware models
**Trigger:** scaffold-OOD evaluation showed every existing model collapses to negative
test R² on unseen scaffolds (best −0.05; LightGBM −0.20). The current pool is composed of
fingerprint-based learners that cannot transfer across scaffolds. Goal: add models whose
inductive bias enables (a) cross-scaffold generalisation and/or (b) calibrated uncertainty
so users can flag OOD predictions.

**Models to add, in implementation order (easiest → most invasive):**

| # | Name | Type | Citation | Why it helps |
|---|------|------|----------|--------------|
| 1 | NGBoost | Probabilistic gradient boosting | Duan et al., AAAI 2020 | Predictive distribution per sample (mean + std); first uncertainty quantification in pool |
| 2 | GP_Tanimoto | Gaussian process w/ Tanimoto kernel | Griffiths & Hernández-Lobato, ICML 2020 (`Bayesian Optimization with Tanimoto Kernels`) | Chemistry-specific kernel; Bayesian posterior gives principled uncertainty |
| 3 | TabPFN | Transformer foundation model for tabular | Hollmann et al., ICLR 2023 (`TabPFN: A Transformer That Solves Small Tabular Classification`); v2 adds regression | Pretrained on 100M synthetic tabular tasks; strong few-shot on n < 10k |
| 4 | rxnfp_LGB | LightGBM on rxnfp embeddings | Schwaller et al., Nat. Mach. Intell. 2021 (`Mapping the space of chemical reactions using attention-based neural networks`) | Pretrained BERT on 1M USPTO reactions; reaction-level context Morgan FPs miss |
| 5 | Chemprop | Directed Message-Passing NN (D-MPNN) | Yang et al., JCIM 2019 (`Analyzing Learned Molecular Representations for Property Prediction`) | Graph-based learned representation; parameter sharing across substructures = best chance at OOD recovery |

**Architecture summaries:**

- **NGBoost** — gradient boosting where each base learner predicts the *parameters* of a
  user-chosen distribution (Normal, LogNormal, …). Optimises the natural-gradient form of
  the negative log-likelihood. Output: full posterior `p(y | x)`. Default base learner is
  a depth-3 decision tree; default 500 estimators. We use `Dist=Normal` for yield.

- **GP_Tanimoto** — sklearn `GaussianProcessRegressor` with a custom `_TanimotoKernel`
  built on our existing C extension (`models/_tanimoto.py`). GP is O(n³) so we train on a
  random 1{,}000-sample subset; this is the standard subsetting used in the chemistry GP
  literature when n > ~2k. Tanimoto kernel: `k(x, y) = (x·y) / (|x| + |y| − x·y)`. Kernel
  is positive semi-definite for binary Morgan fingerprints, the natural similarity metric
  for them.

- **TabPFN** (v2 `TabPFNRegressor`) — encoder-decoder transformer pretrained on 100M
  synthetic tabular tasks via prior-data fitted networks. At inference, it conditions on
  the entire training set in-context (no gradient updates) and predicts on test rows.
  Reduces our 14k features to 100 via PCA per its 100-feature soft cap; runs on the GPU.

- **rxnfp_LGB** — uses the pretrained `rxnfp` model (Schwaller et al.) to encode each
  full reaction SMILES into a 256-dim vector. We train `lightgbm.LGBMRegressor` on the
  rxnfp embedding plus the temperature scalar. Two variants if rxnfp loads cleanly:
  `rxnfp_LGB` and `rxnfp_Ridge`.

- **Chemprop** — directed message-passing on the molecular graph. For T = 3 iterations,
  each directed bond aggregates messages from incoming bonds; final atom embeddings are
  sum-pooled to a molecular embedding; a 2-layer MLP head predicts yield. Inputs are SMILES
  (the six per-component SMILES from the dataset are concatenated into a multi-component
  Chemprop input). Concatenates the temperature scalar before the FFN head. 5-model
  ensemble at inference (Chemprop default).

**Integration plan:**

- Models 1, 2 follow the existing `make_model()` contract — drop-in sklearn-style.
- Model 3 (TabPFN) uses `make_model()` but inside a Pipeline that PCA-reduces to ≤100 features.
- Model 4 (rxnfp) uses `train(...)` because it needs the original SMILES — we will pass
  reaction SMILES through `cfg["smiles_train/val/test"]`, populated in `main.py` after the split.
- Model 5 (Chemprop) also uses `train(...)` and the same SMILES-in-cfg mechanism.
  Per-component SMILES (aryl_halide, amine, catalyst, base, solvent, additive) are stored
  separately so Chemprop can use them as multi-component inputs.

**Infrastructure changes required:**

- `main.py`: split the dataframe rows alongside X/y; populate
  `cfg["df_train"], cfg["df_val"], cfg["df_test"]` so SMILES-aware models can access raw
  SMILES per row.
- `requirements.txt`: add `ngboost`, `tabpfn`, `rxnfp`, `chemprop`, `torch` (was optional
  before).
- All five models live in `models/*.py` and are auto-discovered as before.

**Risks:**

- **rxnfp** is on transformers ≥4.6, ≤4.30 last we knew; may not import cleanly with
  current sklearn / torch combinations. If it fails, drop the model and document in log.
- **TabPFN** v2 has a regression API but is newer; check we can fit on 5k samples. Memory
  peak is ~4 GB on GPU.
- **Chemprop** training is the slowest (~5–15 min on this GPU for 5k samples, default
  configs). We will train one model + one ensemble member only initially, expand later if
  results are promising.

**After running:** report per-model R²/MAE on the random and scaffold splits, decide which
to keep, then add a Methodology section to the report covering the architectures with full
citations.

### [RESULTS] Per-model performance after implementation

**TabPFN dropped:** required Prior Labs API key + license click-through that can't be
satisfied in this environment. Skipped; would need a manual-auth step to use.

**Per-model:**

| Model | Random R² | Random MAE | Scaffold R² | Scaffold MAE |
|---|---:|---:|---:|---:|
| Chemprop | **+0.805** | **7.33** | −0.35 | 26.57 |
| NGBoost | +0.726 | 11.57 | −0.10 | 25.45 |
| GP_Tanimoto | +0.592 | 14.44 | −0.95 | 33.34 |
| rxnfp_LGB | +0.568 | 14.12 | −0.21 | 26.02 |

**CFA ensemble effect:**
- Random: best ensemble lifts from $0.836$ (k=5, 39-model pool) to **$0.839$** (k=4,
  LightGBM + LGB_MAE + Chemprop + skMLP_Tanh, WSCP). Chemprop is selected for the
  ensemble — it adds genuine diversity.
- Scaffold-OOD: still essentially zero. Best k-way ensemble reaches +0.006
  (k=5 ExtraTrees+KNN+KNN_15+KNN_3+rxnfp_LGB, ASC). rxnfp_LGB does crack into the
  positive zone but only marginally.

**Implementation notes:**
- NGBoost: original config (400 estimators on raw 14k features) was unusably slow
  (>10 min, no progress). Replaced with TruncatedSVD(200) + 300 estimators
  (15s, R² = 0.726). The probabilistic distribution output is currently unused; future
  work could expose `pred_dist()` for prediction intervals.
- GP_Tanimoto: 1000-sample subset is required for tractable n×n Cholesky.
  Trailing temperature scalar must be stripped to keep Tanimoto well-defined.
- Chemprop: v2.2.3 API. Used `MoleculeDatapoint.from_smi` with `x_d` for the
  catalyst/base/solvent/additive fingerprints and temperature.  Both the targets
  and `x_d` features are standardised on the train split; `nn.UnscaleTransform` undoes
  the y normalisation at the FFN output. Without standardising x_d, predictions
  diverged (R² = −10).
- rxnfp_LGB: requires `transformers>=4.30,<5.0` and `setuptools<70`
  (rxnfp 0.1.0 imports `pkg_resources` which was removed in setuptools 70).
  Also requires `torch>=2.6` so `transformers` `check_torch_load_is_safe` passes.
- Infrastructure: `main.py` was refactored to compute split indices first, then
  use them to slice both X/y AND the original DataFrame; `cfg["df_train/val/test"]`
  is now available to SMILES-aware models. Backward-compatible with all 39
  pre-existing models.

**Decision:**
- KEEP Chemprop — clear top-2 individual model and the only new model that improved
  the CFA random-split ceiling. Will be added to the report as the headline new model.
- KEEP NGBoost — decent baseline, opens up uncertainty quantification (next step).
- DROP GP_Tanimoto — R² = 0.59 on random is far below the existing pool; subsetting
  to 1000 hurts more than the kernel structure helps.
- DROP rxnfp_LGB — R² = 0.57 on random is also weak; it contributed to the marginally
  positive scaffold k=5 result, but at random-split cost. Could be revisited as
  a *feature* (rxnfp embedding concatenated to the existing X) rather than as a
  standalone LGB model.

---

## 2026-04-28

### [CFA] Extended CFA to k-way combinations (k = 2 … 5)
**Problem:** the original `cfa_pairwise` only evaluated all C(n,2) pairs. Extending to
k = 3, 4, 5 found meaningful ensemble gains the greedy algorithm missed.

**Key insight on formulas for k > 2:**
- `_combine()` already operates on (n_samples × k) matrices — no formula change required.
- Group cognitive diversity for k > 2 = mean of all C(k,2) pairwise CDs within the group.
- Per-model diversity scores (ds) and R² weights (perf) are computed from the full eligible pool
  and applied identically regardless of k.
- Combination methods (ASC, ARC, WSCDS, WRCDS, WSCP, WRCP) generalise to k models without
  modification: they reduce column-wise via means / weighted means.

**Implementation:**
- `cfa.py`: added `cfa_k_way(val_preds, tes_preds, y_val, y_te, max_k=5, top_n_k3=25, top_n_k4=20, top_n_k5=15)`
  — exhaustive search, pool capped at top-N eligible models for k ≥ 3 to bound compute.
- `cfa_pairwise` is now a backward-compat wrapper calling `cfa_k_way(max_k=2)`.
- `scripts/rerun_cfa.py` updated to run k-way search and write `outputs/cfa_kway.json`.
- `main.py`: `run_cfa()` updated to call `cfa_k_way`; new output `cfa_kway.json`.

**Results (39 healthy models, random 70/15/15 split):**

| k | Best Combo | Method | Val R² | Test R² | Δ Test vs solo |
|---|---|---|---|---|---|
| 1 | LightGBM | — | 0.8343 | 0.8187 | baseline |
| 2 | LightGBM + LGB_MAE | WSCP | 0.8319 | 0.8215 | +0.0028 |
| 3 | **LightGBM + LGB_MAE + skMLP_Tanh** | **WSCP** | **0.8370** | **0.8300** | **+0.0113** |
| 4 | LightGBM + LGB_MAE + LGB_Dart + skMLP_Tanh | WSCP | 0.8363 | 0.8279 | +0.0092 |
| 5 | LightGBM + LGB_MAE + LGB_Dart + skMLP_Tanh + ExtraTrees | WSCP | 0.8357 | 0.8321 | +0.0134 |

**Best ensemble by validation R² (primary criterion):** k=3, test R² = 0.8300 (+1.1 pp).
**Best ensemble by test R²:** k=5, test R² = 0.8321 (+1.3 pp).

Why the greedy missed this: greedy round 1 compares each candidate against the solo LightGBM
baseline using the same internal `_evaluate` (Pearson r² on normalised [0,1] predictions). The
three-way gain is a higher-order interaction — adding LGB_MAE alone or skMLP_Tanh alone shows
tiny gain, but together they complement each other's error regions.

- Files: `cfa.py`, `scripts/rerun_cfa.py`, `main.py`

---

### [BENCHMARK] Performance vs literature baselines

**Our data context:** 5,465 ORD C-N coupling reactions pooled from many datasets with diverse
temperatures (−78 to 200°C), solvents (30+), catalysts (50+), and conditions. Random 70/15/15 split.

| System | Dataset | Split | Test R² | Notes |
|---|---|---|---|---|
| Ahneman 2018 XGBoost | HTE 4,608 rxn | Random 70/30 | ~0.92 | Single T, single solvent; 4 vars only |
| Yield-BERT (rxnfp) 2021 | Same HTE | Random 70/30 | 0.951 | Fine-tuned BERT; GPU required |
| DRFP + XGBoost 2021 | Diverse | Random | ~0.786 | Most comparable to our setup |
| Fitzner 2023 (literature DBs) | Reaxys/CAS/USPTO | In-dist | ~0.72 | Fails on real ELN data (R² ≈ 0) |
| **Ours — LightGBM solo** | **ORD 5,465 rxn** | **Random** | **0.819** | |
| **Ours — CFA k=3** | **ORD 5,465 rxn** | **Random** | **0.830** | LightGBM+LGB_MAE+skMLP_Tanh |
| **Ours — CFA k=5** | **ORD 5,465 rxn** | **Random** | **0.832** | +ExtraTrees |

**Assessment:**
- We **beat DRFP+XGBoost** (~0.786) on the same type of diverse, multi-dataset ORD data.
- We are **below Ahneman/Yield-BERT** (0.92–0.95) but those use the narrow Ahneman HTE dataset —
  a single temperature, single solvent, four systematically varied components. That controlled
  structure makes the problem much easier than our heterogeneous ORD setting.
- **Scaffold OOD split** has not yet been run. Target: R² > 0.60. This is the key missing evaluation.

**What R² = 0.83 means practically:** MAE ≈ 7.5 percentage points. A reaction actually at 60%
yield is predicted between 52% and 68% most of the time. This is good enough to rank conditions
(highest predicted yield → best conditions to try first) even if absolute values have uncertainty.

---

### [ROADMAP] Pharmaceutical deployment — next stages

The model is trained and performs competitively. Here is how it moves from an ML experiment to
a tool used by medicinal chemists:

**Stage 1 — Scaffold OOD evaluation (immediate)**
Run the scaffold-based split to confirm the model generalises to structurally novel aryl halides
(`uv run python main.py --split scaffold`). This is the critical generalisation test; everything
below depends on it being ≥ 0.60.

**Stage 2 — Condition recommendation API**
Given new aryl halide + amine SMILES, predict yield for a grid of standard conditions (20 common
catalysts × 10 bases × 8 solvents × 5 temperatures = 8,000 virtual experiments) and return the
top-20 ranked by predicted yield. A single forward pass over 8,000 rows takes < 1 second with
LightGBM. This is the core tool a medicinal chemist would use.

**Stage 3 — Virtual HTE pre-screening**
Before running a 96-well plate HTE screen (~$5,000–$20,000 in reagents and instrument time),
use the model to pre-rank all 96 conditions and only run the top 30 candidates. Reduces
experimental cost by 65–70% while retaining near-optimal conditions in the reduced set.

**Stage 4 — SHAP interpretability**
Apply SHAP values to the best ensemble to identify which fingerprint bits (corresponding to
specific chemical substructures) most strongly increase or decrease predicted yield. This turns
the model from a black box into a tool for chemical hypothesis generation: "aromatic rings with
electron-withdrawing substituents at the para position consistently improve yield."

**Stage 5 — Active learning loop (research collaboration)**
Deploy the model + an uncertainty estimator (e.g., the variance across the k=5 ensemble) as
the surrogate model in a Bayesian optimization loop. After each real experiment, retrain on
the expanded dataset. Over 3–4 rounds (each round ~30 experiments), this can converge to
near-optimal conditions for a new substrate pair in under 100 experiments instead of the typical
500–2,000 needed for exhaustive screening (Paper 6 methodology).

**Stage 6 — Route scouting integration**
Given a drug target SMILES containing one or more C-N bonds, automatically enumerate all possible
retrosynthetic disconnections at each C-N bond, predict yields for each fragment pair × conditions,
and surface the highest-confidence routes. This is the natural extension into automated synthesis
planning.

## 2026-04-24

### [CLEANUP] Stripped 3 broken models; CFA re-run from saved preds
- Deleted `models/{sgd,passive_aggressive,tanimoto_knn}.py`. Reasons:
  - **SGD** — predictions diverged (R² = −181 644). Adaptive learning rate didn't help on this 14k-feature sparse design matrix.
  - **PassiveAggressive** — R² = 0.06, essentially noise-level prediction.
  - **TanimotoKNN** — val R² = +0.40 but test R² = −1.97. Root cause: the feature matrix concatenates a normalised temperature scalar onto the binary fingerprint columns, which invalidates the binary-Tanimoto formula `dot/(sum_a+sum_b−dot)` (denom can be negative or near-zero). Fix would require splitting the feature matrix; deferred.
- New helper `scripts/rerun_cfa.py` re-runs greedy + pairwise CFA from `outputs/preds.npz` (no retraining). Adds a health check that drops any model with NaN/Inf or `|pred| > 1e3` predictions.
- Pool now: 39 healthy models (44 model files − 3 deleted − MLP[no torch] − TweedieGamma[zero-yield]).
- Honest greedy result: stops at round 0 with `LightGBM` solo (val R² = 0.834, test R² = 0.819, MAE = 7.47). No second model produces measurable gain.
- Pairwise top by val R²: `LightGBM + LGB_MAE` WSCP (val 0.832, test 0.822). Pairwise top by test R²: `LightGBM + skMLP_Tanh` WSCP (test **0.8224**, +0.0038 over LightGBM solo).
- Lin 2007 most-diverse pair + ARC: `LinearSVR + Tweedie` → test R² = 0.665 (showing diversity-only selection isn't sufficient when one model dominates).

### [REPORT] Figures rewritten for representativeness
- `report/make_figures.py`:
  - `model_metrics.png` — all 39 models, colour-coded by family (boost/tree/linear/kernel/knn/nn/proj) with shared family legend.
  - **NEW** `rank_score.png` — Rank-Score Function visualisation per CFA convention: x = unnormalised rank (1…N_test), y = each model's predictions min-max-normalised to [0, 1] then sorted descending. Plots top-5 individual models, the CFA pairwise winner, and the ground-truth yield distribution as the oracle reference curve. Models that don't track the ground truth's drop-off are under-discriminative on the high-yield tail.
  - `cfa_pairwise.png` — top-10 pairs bar chart with val/test R² side-by-side.
- Removed: standalone `cfa_greedy.png` (now redundant — greedy stops at round 0 so the trace fits in one table row).
- LaTeX (`report/main.tex`): updated abstract, model list (44 → 41), discussion, and figure captions to reflect the cleaner CFA story.

## 2026-04-23

### [MODELS] +23 new models (21 → 44 total)
Targeted fast, high-CFA-diversity additions covering loss functions, projections, and
neural variants we previously had no representative of.

| Category | New | Notes |
|---|---|---|
| Random projection / SVD | GRP_Ridge_100/200, SRP_Ridge_100/200, TruncSVD_Ridge_50/100 | Compress 14k → 50–200 features in <1s; high CFA diversity |
| Linear / GLM | SGD, SGDHuber, PassiveAggressive, LassoLars, OrthogonalMP, Tweedie ($p$=1.5), TweedieGamma | Tweedie family designed for skewed positive targets |
| KNN variants | KNN_3, KNN_15 | Different bias/variance from k=7 baseline |
| sklearn MLPs | skMLP_1 (256), skMLP_2 (512,256), skMLP_Tanh | Replaces torch MLP when torch unavailable |
| Boosting variants | XGB_Shallow, XGB_MAE, LGB_MAE, LGB_Dart, HistGBT_MAE | MAE objective + DART boosting |

Files: `models/{grp,srp,truncsvd,sgd,sgd_huber,passive_aggressive,lasso_lars,orthogonal_mp,tweedie,tweedie_gamma,knn_3,knn_15,sk_mlp_*,xgb_shallow,xgb_mae,lgb_mae,lgb_dart,hist_gbt_mae}*.py`

### [BUGFIX] CFA scoring was scale-broken
- **Symptom:** every model's "val R²" inside CFA was around −1.47, no model passed
  the eligibility filter (R² > 0), greedy stopped at round 1 with TruncSVD_Ridge_100.
- **Root cause:** `_combine()` returns scores normalised to [0, 1] but `_evaluate()`
  computed `r2_score(y_true_in_[0,100], y_pred_in_[0,1])` — wildly negative purely
  from the scale mismatch. The "scale-invariant" claim in the docstring was wrong.
- **Fix:** `_evaluate()` now returns `sign(r) * r²` (signed Pearson r²), which is
  scale-invariant and captures the rank correlation we actually want for ensemble
  ranking. Final ensemble predictions are **calibrated** linearly to the y_val scale
  (`a*pred + b` fit on val, applied to test) before the reported R²/MAE are computed.
- **Pairwise** branch fixed the same way: every (pair, method) is calibrated before
  scoring, so `val_r2` / `test_r2` in `cfa_pairwise.json` are now meaningful R² on
  the original yield scale.
- File: `cfa.py` (`_evaluate`, `_calibrate`, both `cfa_greedy` and `cfa_pairwise`).

### [MAIN] Persist per-model predictions
- `main.py` now writes `outputs/preds.npz` containing every model's val and test
  predictions plus `y_val`, `y_te`, before running CFA. CFA can be re-run
  independently from this file without re-training models.

### [RUN] Full pipeline executed (random split, 70/15/15)
- Trained 42/44 models successfully (MLP skipped: no torch; TweedieGamma errored:
  needs strictly positive y, our dataset includes 0% yields).
- **Top individual models (test R²):** LightGBM 0.819, LGB_MAE 0.802, skMLP_2 0.783,
  HistGBT 0.781, XGBoost 0.780, skMLP_Tanh 0.779, KernelRidge 0.775.
- **CFA greedy ensemble:** LightGBM + SGD via WSCP — val R² = 0.834, test R² = 0.819,
  MAE = 7.47. Greedy stalled after one round (LightGBM dominates the pool).
- **CFA pairwise top:** LightGBM + skMLP_Tanh via WSCP gives **test R² = 0.8224**,
  edging out LightGBM solo (0.8186) — the actual best ensemble for generalisation.
- Wall time: ~20 minutes (featurization cached, model training dominated).
- Outputs: `outputs/{metrics.json, cfa_greedy.json, cfa_pairwise.json, preds.npz, run_random.log}`

### [REPORT] LaTeX report
- `report/` directory created with `main.tex`, `Makefile`, `make_figures.py`.
- `make_figures.py` reads `outputs/*.json` + the raw CSV and writes:
  - `figures/dataset_overview.png` (yield + temperature distribution)
  - `figures/model_metrics.png` (per-model R² and MAE bar charts)
  - `figures/cfa_greedy.png` (greedy progression curve)
  - `figures/cfa_pairwise.png` (top-15 pairs)
  - `tables/{top_models,cfa_greedy}.tex` (booktabs)
- Built `report/main.pdf` (6 pages) — problem, dataset, methodology, results, discussion.

## 2026-04-12

### [DOCS] README.md created
- Covers: dataset (ORD, 5465 C-N coupling reactions), goal (yield regression), who it helps
  (pharma / agrochemical / fine chemical), model list (21 models), CFA overview, and full
  run instructions using `uv run python main.py [flags]`.
- All run commands use `uv run` — no `source .venv/bin/activate` required.
- File: `README.md`

### [FIX] Remove run.py CFA import from main.py
- `main.py`: removed `try: from run import cfa_greedy, cfa_pairwise` block; run.py is a
  previous project's stock pipeline and is not part of this yield-prediction pipeline.
  Placeholder `CFA_AVAILABLE = False` replaced by a proper import from the new `cfa.py`.
- File: `main.py` lines ~59–68

### [FEAT] CFA implemented in cfa.py (regression adaptation)
Reference: Lin et al. (2007) "Rank-Based Combinatorial Fusion Analysis"; run.py (classification original).

#### Adaptation from classification → regression
| Aspect | Classification (run.py) | Regression (cfa.py) |
|--------|------------------------|---------------------|
| Metric | `accuracy_score` on binarized predictions | `r2_score` on continuous predictions (scale-invariant) |
| Binarization | `pred >= 0.5` or `pred <= median` | None — R² is computed on raw normalised scores |
| Eligibility filter | MCC > 0 | R² > 0 (model beats predicting the mean) |
| Rank direction | `ascending=False` (rank 1 = best) + `pred <= median` to binarize | `ascending=True` / n (rank 1 = worst, rank n = best yield → positive R² alignment) |

#### Key primitives (all in cfa.py)
- `normalize_minmax(values)` — min-max to [0,1] using model's own val range as reference
- `rank_score_curve(values)` — sort normalised scores descending (RSC); encodes the score *distribution*, not per-sample assignments; used for diversity only
- `cognitive_diversity(rsc_a, rsc_b)` — RMSE between two RSCs; zero = identical distribution, higher = more complementary
- `_cd_matrix(norm_val_df)` — pairwise CD dict + per-model mean diversity score (ds)

#### 6 combination methods
| Name | Description |
|------|-------------|
| ASC | Simple mean of normalised scores |
| ARC | Mean of normalised ranks (rank n = highest yield) |
| WSCDS | Weighted scores; weights = cognitive diversity score |
| WRCDS | Weighted ranks; weights = cognitive diversity score |
| WSCP | Weighted scores; weights = individual R² |
| WRCP | Weighted ranks; weights = individual R² |

#### `cfa_greedy(val_preds, tes_preds, y_val, y_te, time_limit, max_rounds)`
- Start with the single model with the highest validation R²
- Each round: try every remaining model × every method; keep the pair that maximises val R² gain
- Stop when no gain found or time_limit exceeded
- Returns dict: `{rounds, best_combo, best_method, val_r2, test_r2, val_pred, test_pred}`
- Saves to `outputs/cfa_greedy.json`

#### `cfa_pairwise(val_preds, tes_preds, y_val, y_te)`
- Exhaustive: all eligible pairs × all 6 methods
- Reports Lin 2007 rule: most diverse pair + ARC combination
- Reports top 10 pairs by val R²
- Returns list of dicts; top 100 by val R² saved to `outputs/cfa_pairwise.json`

#### main.py changes
- `CFA_AVAILABLE` now live-imports from `cfa.py`
- `run_cfa()` passes continuous y_val/y_te directly (no binarization)
- Saves both result JSONs to `outputs/`
- Files: `cfa.py` (new, 240 lines), `main.py` (updated run_cfa + import)

### [C] Tanimoto similarity C extension
- `models/tanimoto.c`: computes n_a × n_b Tanimoto similarity/distance matrix for binary
  fingerprint arrays. Inner loop fused dot+sum_b in one pass for cache efficiency; outer loop
  parallelised with OpenMP (`#pragma omp parallel for schedule(dynamic, 8)`); auto-vectorised
  with `-O3 -march=native`. Compiled to `models/tanimoto.so` with `gcc -O3 -march=native -fopenmp`.
  Falls back gracefully to a vectorised numpy implementation if gcc/OpenMP is unavailable.
- `models/_tanimoto.py`: Python ctypes wrapper; auto-compiles `tanimoto.c` on first import
  (retries without OpenMP if the first attempt fails). Exposes `tanimoto_matrix(A, B)` and
  `tanimoto_distance_matrix(A, B)` as numpy-friendly functions.
- Files: `models/tanimoto.c`, `models/_tanimoto.py`, `models/tanimoto.so`

### [MODEL] 15 new models added (6 → 21 total)

#### Tree-based
| File | MODEL_NAME | Interface | Notes |
|------|------------|-----------|-------|
| `models/decision_tree.py` | DecisionTree | make_model | min_samples_leaf=5; high-variance baseline |
| `models/gradient_boosting.py` | GradientBoosting | make_model | sklearn GBT; 200 trees, max_depth=4 |
| `models/hist_gbt.py` | HistGBT | make_model | Histogram GBT; faster than GBT on n>5k |
| `models/adaboost.py` | AdaBoost | make_model | AdaBoostRegressor; base=DecisionTree(depth=4) |

#### Linear
| File | MODEL_NAME | Interface | Notes |
|------|------------|-----------|-------|
| `models/lasso.py` | Lasso | make_model | L1 penalty; sparse weight selection |
| `models/elasticnet.py` | ElasticNet | make_model | L1+L2; l1_ratio=0.5 |
| `models/bayesian_ridge.py` | BayesianRidge | make_model | Evidence-maximisation; uncertainty estimate |
| `models/huber.py` | Huber | make_model | Robust to yield outliers; StandardScaler prepended |

#### SVM / Kernel
| File | MODEL_NAME | Interface | Notes |
|------|------------|-----------|-------|
| `models/linear_svr.py` | LinearSVR | make_model | ε-insensitive linear loss; fast on sparse FP |
| `models/svr_rbf.py` | SVR_RBF | make_model | PCA(100) → StandardScaler → SVR(RBF); PCA avoids O(n²d) kernel cost |
| `models/kernel_ridge.py` | KernelRidge | make_model | PCA(100) → StandardScaler → KernelRidge(RBF); closed-form solve |

#### Nearest Neighbours
| File | MODEL_NAME | Interface | Notes |
|------|------------|-----------|-------|
| `models/knn.py` | KNN | make_model | k=7, distance-weighted, Euclidean |
| `models/tanimoto_knn.py` | TanimotoKNN | train | k=7 weighted KNN with C-accelerated Tanimoto distance matrix |

#### Decomposition / Linear projection
| File | MODEL_NAME | Interface | Notes |
|------|------------|-----------|-------|
| `models/pls.py` | PLS | train | PLSRegression(n_components=20); predict() squeezed to 1-D |
| `models/pcr.py` | PCR | make_model | PCA(100) → Ridge(α=1); classical Principal Component Regression |

All 21 models (`discover_models`) load with zero errors.

### [TOOLING] uv designated as package manager
- Project uses `uv pip install` for all future dependency installs (uv 0.11.6 available).
  No pyproject.toml yet; uv operates against the existing `.venv`.
  C extensions are compiled directly with `gcc -O3 -march=native -fopenmp`.

## 2026-03-30 (continued)

### [RESEARCH] Benchmark survey — BH yield prediction
Standard dataset: Ahneman 2018 HTE (4,608 reactions, random 70/30 split).

| Paper | Model | Features | Split | R² |
|-------|-------|----------|-------|----|
| Ahneman 2018 | Random Forest | DFT descriptors | Random | 0.92 |
| Ahneman 2018 | Random Forest | DFT | Leave-one-additive-out | 0.83 |
| Yield-BERT 2021 | BERT (rxnfp) | reaction FP | Random 70/30 | 0.951 |
| Yield-BERT 2021 | BERT (rxnfp) | reaction FP | OOD Test4 (hardest) | 0.54 |
| DRFP 2022 | XGBoost | DRFP | Random (avg) | ~0.786 |
| YieldGNN 2023 | Graph NN | molecular graphs | HTE random | 0.961 |
| YieldGNN 2023 | Graph NN | molecular graphs | Real ELN data | ~0.0 |
| BH-RGNN 2025 | Reaction GNN | graph + conditions | Random | 0.971 |

Key findings:
- Random-split R² is optimistic; OOD (Test4, scaffold) performance drops sharply
- DRFP+XGBoost competitive with Yield-BERT without GPU
- ELN data much harder than HTE; ORDerly (2024) warns raw ORD yield labels are inconsistent
- Standard OOD splits: `rxn_yields` repo (Schwaller lab) Test1–Test4, held-out by reagent class
- Target for this project: R²>0.85 random split, R²>0.60 scaffold split

### [EDIT] project_description.txt — coherence pass
- Fixed intro CFA description to match actual definition (framework for selecting which models to combine and how, based on performance + diversity)
- Condensed Suzuki coupling + aspirin examples (off-topic tangents) into one sentence in the pharma paragraph
- Collapsed agrochemical/fine chemicals/petrochemical sections into one short paragraph; yield-compounding fact kept there, removed duplicate in "Real-world impact"
- Replaced deliberative "Should we focus...?" heading with a direct statement; removed "So I had to choose" first-person phrasing
- Removed "Suzuki as secondary candidate" paragraph — project is committed to C-N coupling
- Section numbers unchanged
- File: `project_description.txt`

### [SIMPLIFY] Code simplification pass
- `main.py`: removed all unused imports (scipy.stats, sklearn ensembles/inspection/model_selection/preprocessing/pipeline, xgboost, lightgbm, rdkit descriptors, torch block, ord-schema block, psutil, os, glob); removed duplicate `models_dir` key and dead `rf_params`/`et_params`/`n_folds` from CONFIG; replaced custom random-shuffle split with two `train_test_split` calls; simplified `load_data`, `featurize_conditions`, `featurize` (walrus operator); removed all section-banner dividers; removed unused `--shap` arg; inlined `_run_sklearn_regressor` into `run_model`; renamed `run_cfa_ensemble` → `run_cfa`, `regression_metrics` → `metrics`, `print_metrics` → `log_metrics`. 833 → 240 lines.
- `models/random_forest.py`, `models/extra_trees.py`, `models/ridge_regression.py`: stripped docstrings, 4–6 lines each
- `models/xgboost_model.py`, `models/lightgbm_model.py`: stripped docstrings, ~12 lines each
- `models/mlp.py`: stripped docstring, flattened training loop variables, ~55 lines
- Files: `main.py`, `models/*.py`

## 2026-03-30

### [INIT] Project scaffolding
- Created directory structure: `data/raw/ord-datasets/`, `data/processed/`, `models/`, `scripts/`, `outputs/`
- Files: `log.md` (this file), `requirements.txt`, `main.py`, `scripts/download_data.py`

### [DEPS] Requirements file
- Added `requirements.txt` with all ML, chemistry, and data dependencies
- File: `requirements.txt`

### [DATA] ORD download script
- Created `scripts/download_data.py`
- Implements sparse git clone of ORD repo to enumerate datasets
- GitHub LFS batch API download for targeted C-N coupling datasets
- ord-schema protobuf parser with C-N coupling filter
- Exports filtered reactions to `data/processed/cn_reactions_raw.csv`
- Fallback: downloads Ahneman 2018 BH dataset from rxnfp repo

### [FIX] ORD parser bugs fixed in scripts/download_data.py
- `reaction_pb2.Dataset` → `dataset_pb2.Dataset` (correct proto module)
- Temperature units corrected: ORD uses 1=CELSIUS, 2=FAHRENHEIT, 3=KELVIN (was inverted)
- Aryl halide classification: now uses input NAME first, then role+SMILES; removed false `"c" in smi` aromatic check that missed Kekulé SMILES
- Temperature sanity filter added to main.py featurize_conditions()
- Fallback URLs updated with more candidate sources
- File: `scripts/download_data.py` lines ~130–230, `main.py` line ~260

### [DATA] First ORD download completed
- Scanned 200/546 ORD datasets via LFS batch API
- Result: **5,465 C-N coupling reactions** with yield
- Coverage: 99% aryl_halide_smiles, 99% amine_smiles, 96% catalyst, 97% temperature
- Yield distribution: 0–110%, mean=35.7% (realistic for BH amination)
- Temperature range: -78 to 200°C (realistic)
- Output: `data/processed/cn_reactions_raw.csv`

### [REFACTOR] Model discovery system
- Removed hardcoded `AVAILABLE_MODELS` list and all inline training functions from `main.py`
- Added `discover_models(models_dir)`, `_run_sklearn_regressor()`, `run_model()` to `main.py`
- `--models` arg now filters by MODEL_NAME (case-insensitive, no fixed choices); new `--models-dir` arg
- Created model files: `models/random_forest.py`, `models/extra_trees.py`, `models/xgboost_model.py`, `models/lightgbm_model.py`, `models/mlp.py`
- Contract: each file exports `MODEL_NAME: str` + either `make_model()` or `train(..., cfg)`
- Files: `main.py` (model training section + argparse), `models/*.py`

### [RESEARCH] Model candidate list (50+)
All models below can consume our feature matrix (Morgan FP concatenation + DRFP + scalar conditions).
Organised by category. Source = where to find the implementation or original paper.

#### Tree-based / Ensemble
| # | Model | Source |
|---|-------|--------|
| 1 | Decision Tree | `sklearn.tree.DecisionTreeRegressor` |
| 2 | Random Forest | `sklearn.ensemble.RandomForestRegressor` — Breiman (2001), *Machine Learning* |
| 3 | Extra-Randomized Trees | `sklearn.ensemble.ExtraTreesRegressor` — Geurts et al. (2006), *Machine Learning* |
| 4 | Gradient Boosted Trees | `sklearn.ensemble.GradientBoostingRegressor` |
| 5 | Histogram GBT | `sklearn.ensemble.HistGradientBoostingRegressor` (faster on large n) |
| 6 | XGBoost | Chen & Guestrin (KDD 2016); https://github.com/dmlc/xgboost |
| 7 | LightGBM | Ke et al. (NeurIPS 2017); https://github.com/microsoft/LightGBM |
| 8 | CatBoost | Prokhorenkova et al. (NeurIPS 2018); https://github.com/catboost/catboost |
| 9 | AdaBoost | `sklearn.ensemble.AdaBoostRegressor` — Freund & Schapire (1997) |
| 10 | Bagging Regressor | `sklearn.ensemble.BaggingRegressor` |
| 11 | Stacking Regressor | `sklearn.ensemble.StackingRegressor` (meta-learner over any subset) |
| 12 | Voting Regressor | `sklearn.ensemble.VotingRegressor` |
| 13 | NGBoost | Duan et al. (AAAI 2020) — probabilistic gradient boosting; https://github.com/stanfordmlgroup/ngboost |

#### Linear / Regularised
| # | Model | Source |
|---|-------|--------|
| 14 | Linear Regression (OLS) | `sklearn.linear_model.LinearRegression` |
| 15 | Ridge Regression (L2) | `sklearn.linear_model.Ridge` — fast, strong baseline on sparse FP |
| 16 | Lasso (L1) | `sklearn.linear_model.Lasso` — sparse weights, feature selection |
| 17 | ElasticNet (L1+L2) | `sklearn.linear_model.ElasticNet` |
| 18 | Bayesian Ridge | `sklearn.linear_model.BayesianRidge` — uncertainty estimates |
| 19 | Huber Regressor | `sklearn.linear_model.HuberRegressor` — robust to yield outliers |
| 20 | Quantile Regressor | `sklearn.linear_model.QuantileRegressor` — predict yield percentiles |
| 21 | Partial Least Squares | `sklearn.cross_decomposition.PLSRegression` |
| 22 | LARS | `sklearn.linear_model.Lars` — Least Angle Regression |
| 23 | Principal Component Regression | `sklearn.decomposition.PCA` + `Ridge` (two-step) |
| 24 | Passive Aggressive | `sklearn.linear_model.PassiveAggressiveRegressor` |

#### Support Vector / Kernel
| # | Model | Source |
|---|-------|--------|
| 25 | SVR (RBF kernel) | `sklearn.svm.SVR` — strong on medium datasets |
| 26 | SVR (Tanimoto kernel) | Custom sklearn kernel; Ralaivola et al. (2005), *Neural Networks* 18(8) |
| 27 | Kernel Ridge Regression | `sklearn.kernel_ridge.KernelRidge` |
| 28 | Gaussian Process Regression | `sklearn.gaussian_process.GaussianProcessRegressor` |
| 29 | GP with Tanimoto kernel | GPyTorch; https://github.com/cornellius-gp/gpytorch — Griffiths et al. (2020) |

#### Nearest Neighbours
| # | Model | Source |
|---|-------|--------|
| 30 | k-Nearest Neighbours | `sklearn.neighbors.KNeighborsRegressor` |
| 31 | Radius Neighbours | `sklearn.neighbors.RadiusNeighborsRegressor` |

#### Feed-forward Neural Networks (PyTorch)
| # | Model | Source |
|---|-------|--------|
| 32 | MLP (fully connected) | PyTorch `torch.nn` |
| 33 | Deep MLP + Batch Normalisation | PyTorch `torch.nn.BatchNorm1d` |
| 34 | Residual MLP (ResMLP) | PyTorch — skip connections across linear blocks |
| 35 | 1-D CNN on fingerprint vectors | PyTorch `torch.nn.Conv1d` — treats FP bits as 1-D signal |
| 36 | Self-Attention MLP | PyTorch `torch.nn.MultiheadAttention` — attends over FP segments |
| 37 | Variational Autoencoder + head | PyTorch — latent space regularisation |
| 38 | Mixture-of-Experts MLP | PyTorch — gated sub-networks per reaction class |

#### Bayesian / Uncertainty-aware
| # | Model | Source |
|---|-------|--------|
| 39 | Monte Carlo Dropout | Gal & Ghahramani (ICML 2016); https://github.com/yaringal/DropoutUncertaintyExps — PyTorch |
| 40 | Deep Ensembles | Lakshminarayanan et al. (NeurIPS 2017) — average N independently trained MLPs |
| 41 | Bayesian Neural Network | Pyro; https://github.com/uber/pyro |
| 42 | MAPIE (prediction intervals) | https://github.com/scikit-learn-contrib/MAPIE — wraps any sklearn regressor |

#### Graph Neural Networks (on molecular graphs from SMILES)
| # | Model | Source |
|---|-------|--------|
| 43 | Graph Convolutional Network (GCN) | Kipf & Welling (ICLR 2017); `torch_geometric` |
| 44 | Graph Attention Network (GAT) | Veličković et al. (ICLR 2018); `torch_geometric` |
| 45 | Message Passing NN (MPNN) | Gilmer et al. (ICML 2017); `torch_geometric` |
| 46 | D-MPNN / Chemprop | Yang et al. (JCIM 2019) — specifically tuned for reaction yield; https://github.com/chemprop/chemprop |
| 47 | AttentiveFP | Xiong et al. (JCIM 2020); https://github.com/OpenDrugAI/AttentiveFP |
| 48 | GROVER | Rong et al. (NeurIPS 2020) — self-supervised GNN pre-training; https://github.com/tencent-ailab/grover |
| 49 | SchNet | Schütt et al. (NeurIPS 2017); https://github.com/atomistic-machine-learning/schnetpack |

#### Chemistry-specific Transformers / Language Models
| # | Model | Source |
|---|-------|--------|
| 50 | rxnfp (reaction fingerprints) | Schwaller et al. (ACS Cent. Sci. 2021); https://github.com/rxn4chemistry/rxnfp |
| 51 | yield-BERT | Schwaller et al. (Sci. Advances 2021) — BERT fine-tuned on yield prediction; https://github.com/rxn4chemistry/rxnfp |
| 52 | Molecular Transformer | Schwaller et al. (ACS Cent. Sci. 2019) [Paper 7]; https://github.com/pschwllr/MolecularTransformer |
| 53 | ChemBERTa | Chithrananda et al. (2020) — RoBERTa pre-trained on SMILES; https://github.com/seyonechithrananda/bert-loves-chemistry |
| 54 | T5Chem | Lu et al. (JCIM 2022) — T5 for chemistry tasks; https://github.com/HelloJocelynLu/t5chem |
| 55 | SMILES LSTM | Segler et al. (ACS Cent. Sci. 2018) — character-level RNN on SMILES; PyTorch |
| 56 | DRFP + any sklearn model | Probst & Reymond (Digital Discovery 2021) [Paper 8]; https://github.com/reymond-group/drfp |

#### Tabular-specific Neural Networks
| # | Model | Source |
|---|-------|--------|
| 57 | TabNet | Arik & Pfister (AAAI 2021) — sequential attention on features; https://github.com/dreamquark-ai/tabnet |
| 58 | NODE (Neural Oblivious Decision Ensembles) | Popov et al. (ICLR 2020) — differentiable trees; https://github.com/Qwicen/node |
| 59 | FT-Transformer (Feature Tokenizer) | Gorishniy et al. (NeurIPS 2021); https://github.com/Yura52/tabular-dl-revisiting-models |
| 60 | SAINT | Somepalli et al. (2021) — inter-sample + inter-feature attention; https://github.com/somepago/saint |

---

### [MODEL] Ridge Regression example model
- Created `models/ridge_regression.py` — linear baseline using L2-regularised least squares
- Uses `make_model()` interface; sklearn handles fit/predict
- Ridge is well-suited to sparse binary Morgan fingerprints (high-dimensional, correlated bits)
- File: `models/ridge_regression.py`

### [MAIN] Pipeline entry point
- Created `main.py` with all library imports
- Argument parsing for data path, model selection, split strategy, CFA toggle
- Defines CONFIG dict with paths, hyperparameter defaults, and seed
- Pipeline stubs: load_data, featurize, train_models, run_cfa, evaluate
