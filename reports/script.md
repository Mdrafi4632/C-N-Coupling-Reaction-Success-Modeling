# Presentation Script
---

## Title slide  *(~10 sec)*

> Hi, I'm **Gurpreet Singh**, this is **Md Rafi**. Our project predicts
> whether a chemical reaction will succeed or fail, using machine
> learning combined through an ensemble framework called **Combinatorial
> Fusion Analysis**.

**Background note (only if asked):**
Our model pool has 51 learners: 41 fingerprint-based regressors —
sklearn-style models like gradient-boosted trees, kernel methods, MLPs,
linear regressors — that predict continuous yield from the molecular
fingerprint, then we threshold at 50% to get a binary label. Plus 5
native classifiers trained directly on binary labels, plus the deep
graph-based chemistry model Chemprop.

---

## Slide 1 — The Problem  *(~60 sec)*

> The reactions we study are **palladium-catalyzed Buchwald-Hartwig
> aminations**, used heavily in pharmaceutical synthesis to form
> carbon-nitrogen bonds — they couple an **aryl halide** with an
> **amine** to make a new C-N bond.

**Definitions (deliver inline if needed):**
- An **aryl halide** is an aromatic ring (think benzene-like, six carbons
  in a hexagon) with a halogen atom — Cl, Br, I, or a triflate group —
  attached. The carbon-halogen bond is what gets broken in the reaction.
- A **primary amine** has nitrogen bonded to one carbon and two hydrogens
  (R-NH₂). A **secondary amine** has nitrogen bonded to two carbons and
  one hydrogen (R-NH-R'). The N-H bond is what reacts to form the new
  C-N bond.

> Each reaction has **six ingredients** we encode as SMILES strings —
> a textual representation of molecular structure — plus a reaction
> temperature. The six are: the **aryl halide**, the **amine**, the
> **palladium catalyst** (with its phosphine ligand), the **base** (e.g.
> cesium carbonate), the **solvent** (e.g. toluene or dioxane), and an
> optional **additive**.

> We define **success as yield ≥ 50%**. This is a working convention
> from process chemistry, not a hard scientific threshold — it's the
> "preparatively useful" line below which side reactions dominate and
> the synthesis isn't worth scaling up.

**Reference (deliver only if pressed):**
There's no single canonical paper for the 50% line; it's a convention.
Closely related: Carey et al. (2006), *Organic & Biomolecular Chemistry*,
"Analysis of the reactions used for the preparation of drug candidate
molecules" — a survey of pharma reaction yields. Anderson, N.G. (2012)
*Practical Process Research and Development* (textbook) discusses yield
thresholds in process development.

> The use case is bench-chemistry triage: predict which reactions are
> worth running before committing the lab time.

---

## Slide 2 — Why classification, not regression?  *(~55 sec)*

> The first version of this project predicted exact yield as a
> continuous number. On a random train/test split it worked great —
> R-squared 0.82, competitive with published benchmarks.

> But on **held-out scaffolds** every model collapsed below R-squared
> zero — meaning none of them did better than just predicting the mean.

**Definition (deliver inline):**
A **scaffold** is the core ring framework of a molecule with all the
hanging substituents stripped off. For example, the molecule
"4-bromobenzonitrile" is a benzene ring (the scaffold) with a bromine
and a cyano group attached. Two molecules sharing the same benzene
scaffold belong to the same scaffold group even if their substituents
are completely different. "Held-out" means we partitioned the data so
the test set has scaffolds the model never saw at training. The formal
definition we use is the **Bemis-Murcko scaffold** (Bemis & Murcko,
1996, *J. Med. Chem.* — "The properties of known drugs. 1. Molecular
frameworks").

> The pivot to binary classification is more honest because chemists
> making a go/no-go decision **might not** need exact yield precision —
> they need to decide whether the reaction is worth attempting.

---

## Slide 3 — Dataset  *(~75 sec)*

> 5,465 reactions from the **Open Reaction Database, or ORD** — an
> open-science chemistry repository where labs submit experimental
> reaction data in a standard format.

**Why we did feature construction (deliver inline):**
Sklearn-style models — gradient-boosted trees, linear models, MLPs —
work on fixed-length numeric vectors, not on molecular graphs. So we
need to convert each molecule into a numeric fingerprint that encodes
its structure. The choice of featurization determines what information
the model has access to.

> For feature construction we use **Morgan fingerprints**: 2,048-bit binary
> vectors that record which chemical substructures appear in a 2-atom
> radius around each atom. We compute one per reaction component — six
> total — and concatenate **DRFP**, a reaction-level fingerprint that
> encodes the difference between reactants and products as a single
> vector. About 14,000 features per reaction.

> Two splits. **Random** is the standard 70/15/15 shuffle —
> in-distribution. **Substrate-level 5-fold cross-validation** is our
> out-of-distribution test.

**Definitions (deliver inline):**
- A **substrate** in chemistry is a molecule that participates in a
  reaction — typically the molecule that gets modified. Here, the
  aryl halide is the substrate (the partner that gets aminated).
- **Substrate-level 5-fold CV vs regular k-fold**: regular k-fold just
  shuffles all rows and partitions them into 5 disjoint groups — the
  same substrate can appear with different reactions in both train and
  test, so the model has effectively seen the compound during training.
  *Substrate-level* k-fold groups by the full aryl-halide structure
  first (we have 365 unique substrates), then bin-packs whole groups
  into 5 folds. So if a substrate is in the test fold, *all* of its
  reactions are in test and *none* are in train — the model has truly
  never seen that compound. Both protocols give 5 train/test splits. **The difference is just how we group rows.**

- **OOD = out-of-distribution**, a general ML concept (popularized in
  deep learning by Hendrycks & Gimpel, 2017, *ICLR*). It means the test
  data is structurally different from the training data, so the model
  has to extrapolate rather than interpolate. ORD is just the dataset
  name; OOD is the test regime — they're unrelated.

> The substrate-level methodology is standard in chemistry ML — see
> Yang et al. 2019, the Chemprop paper, and Wu et al. 2018, MoleculeNet.

---

## Slide 4 — Approach  *(~30 sec)*

> SMILES and temperature go through featurization, hit a 51-model pool,
> and CFA selects the best ensemble. CFA is from Lin et al. 2007 — it
> scores models on individual accuracy AND **cognitive diversity**, which
> measures how differently two models distribute their prediction scores.
> CFA explores ensemble sizes 2 through 5 across 6 weighting methods.

---

## Slide 5 — Random-split results  *(~45 sec)*

> Top 15 models on the **random split** — meaning shuffled 70/15/15 —
> with **95% bootstrap confidence intervals**. A bootstrap CI works like
> this: we resample the test set 1000 times with replacement, recompute
> accuracy each time, and the 2.5th and 97.5th percentile give the
> interval. Translation: if we drew a slightly different test set, the
> accuracy would land between these bars 95% of the time.

**Reference (deliver only if asked):**
The bootstrap method is from Efron (1979), *Annals of Statistics*,
"Bootstrap Methods: Another Look at the Jackknife". A more accessible
treatment is Efron & Tibshirani (1993), *An Introduction to the
Bootstrap* (CRC Press textbook). It's now the standard non-parametric
way to put error bars on any test-set statistic.

> Top models cluster between 0.91 and 0.93. LightGBM and the native
> LightGBM (Light Gradient Boosting Machine) classifier are statistically
> tied — their CIs overlap completely.

---

## Slide 6 — Substrate-level OOD  *(~45 sec)*

> The harder test, 5-fold CV by substrate. Box plots show per-fold
> accuracy across the 5 folds. **Native_XGBClf**, the XGBoost classifier,
> is the best individual at 0.824 plus or minus 0.047 accuracy.
> **Native_LGBClf**, the LightGBM classifier, is just behind. The CFA
> ensemble lands at 0.804 with tighter spread.

> **Selection stability**: across the 5 folds, did CFA pick the same
> models? Yes — Native_LGBClf and Native_XGBClf chosen in 5 out of 5
> folds. The choice is robust, not luck we got from choosing some seed.

---

## Slide 7 — Does CFA earn its complexity?  *(~50 sec)*

> We compared CFA against simpler ensembling: single-best model,
> equal-weight top-k averaging, and majority voting. Bars go from worst
> on the left to best on the right.

> "Random regression" and "random classification" mean those tasks
> evaluated **on the random split**. On random regression, CFA wins by a
> small but real margin — 1.2 R-squared points over equal-weight top-5.
> On random classification, CFA ties majority voting almost exactly. On
> the scaffold splits, the validation set has only 10 reactions, so CFA
> can't reliably choose between candidates.

> For an honest takeaway: CFA earns its complexity on tighter regression
> problems; for large-validation classification, simple averaging works
> just as well. We don't oversell it.

---

## Slide 8 — Robustness to the success threshold  *(~30 sec)*

> The 50% success threshold is the chemistry literature's "preparatively
> useful" line — below 50%, side reactions consume too much material to
> scale up. To make sure our results aren't an artifact of that exact
> choice, we swept the threshold from 20% to 80% in 5% steps. The (MCC)
> Matthews correlation coefficient stays above 0.78 across the entire
> 50-75% range — the model is robust to where we draw the line.

---

## Slide 9 — Summary  *(~55 sec)*

> What we built — three lines:
> a 51-model pool combined through CFA;
> two evaluation regimes — random in-distribution and substrate-level
> cross-validation for OOD;
> a full reliability stack — bootstrap intervals, multi-seed
> reproducibility, McNemar paired tests, and an ablation. **92% accuracy
> in-distribution, 80% on unseen substrates.**

> Honest limitations:

> First, the **Murcko-scaffold OOD test**. This is a stricter test than
> substrate-level CV: instead of holding out specific compounds, we
> hold out entire ring systems. So the model trains on benzene
> derivatives and pyridine derivatives, then has to predict on
> completely new ring systems like quinoline or pyrazole. On our
> dataset, accuracy drops to 59% because 88% of the data uses just two
> scaffolds — benzene and pyridine — so the train set is dominated by
> them and the test set is everything exotic. The validation set in
> this regime ends up with only 10 reactions, which is too few to tune
> any hyperparameter.

> Second, **ORD has reporter bias** — high-yield reactions get published
> more often than low-yield ones, so any classifier trained on the
> database inherits that imbalance. Our dataset is roughly 67/33
> failure-to-success, but the real chemistry-space distribution is
> unknown.

> Thank you.

---

## Backup slides  *(used only if asked)*

- **Multi-seed reliability** — Native_LGBClf reproducible to ±0.01
  accuracy on random, ±0.06 on OOD.
- **Native vs thresholded regressor** — native classifiers tie tree
  families but reliably beat the linear family.
- **Rank-score curves** — CFA's signature plot. Each model normalized
  to [0,1] and sorted descending; dashed line is the oracle.
- **Paired statistical tests** — McNemar p-values; Native_Logistic
  reliably beats Ridge (p = 0.036).
- **Confusion matrix** — errors balanced across classes despite 67/33
  imbalance.
- **Learning curves** — substrate-OOD accuracy is still rising at 100%
  of available data — more data would help OOD.
