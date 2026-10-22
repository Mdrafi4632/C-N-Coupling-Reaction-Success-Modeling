# C-N Coupling Reaction Modeling

### From Yield Regression to Binary Reaction Triage with k-Way Combinatorial Fusion Analysis

**Authors:** Gurpreet Singh and Md Rafi  
**Institution:** Fordham University, Department of Computer Science

## Overview

This project investigates machine learning for **Buchwald-Hartwig C-N coupling reactions**, using **5,465 reactions extracted from the Open Reaction Database (ORD)**. It compares a pool of **41 regressors** and combines predictions through **k-way Combinatorial Fusion Analysis (CFA)**.

The study begins with continuous yield prediction and then reframes the task as binary classification: **will a reaction achieve at least 50% yield?** This shift supports a reaction-screening decision while exposing the difference between performance on familiar chemistry and generalization to unseen molecular scaffolds.

The project focuses on five tasks:

1. **Reaction Data Preparation** - extract, filter, and deduplicate ORD records with reported yields.
2. **Chemical Featurization** - encode molecular components and reaction temperature into numerical representations.
3. **Yield Regression** - compare heterogeneous models and investigate ensemble combinations.
4. **Binary Success Classification** - reuse regression outputs as scores and compare them with directly trained classifiers.
5. **Calibrated Triage** - provide success probabilities and an uncertainty-based abstention option through a Streamlit interface.

## Data

The paper describes an ORD subset selected for palladium-catalyzed reactions containing an aryl halide and amine pair, with non-null yield records.

| Property | Description |
| --- | --- |
| Reactions | 5,465 after extraction and deduplication |
| Molecular inputs | Aryl halide, amine, catalyst/ligand, base, solvent, optional additive |
| Condition input | Temperature in degrees Celsius |
| Regression target | Reported reaction yield (%) |
| Classification target | Success: yield >= 50%; failure: yield < 50% |

The paper reports approximately **15% zero-yield reactions** and a mean yield of **35.7%**. The random split contains approximately **68% negative examples**, making class-aware evaluation important.

## Methods and Tools

- **Language:** Python
- **Representations:** Morgan fingerprints, Differential Reaction Fingerprints (DRFP), and SMILES-based representations
- **Model families:** Linear and regularized regression, tree ensembles, gradient boosting, kernel methods, nearest neighbors, projection-based regression, and neural networks
- **Featured models:** LightGBM, XGBoost, Random Forest, ExtraTrees, Chemprop D-MPNN, NGBoost, rxnfp + LightGBM, and Gaussian Process regression with a Tanimoto kernel
- **Ensembling:** k-way Combinatorial Fusion Analysis
- **Probability calibration:** Isotonic regression
- **Interface:** Streamlit

## Project Pipeline

```text
Open Reaction Database
          |
Reaction Extraction, Filtering, and Deduplication
          |
Chemical Featurization and Feature Caching
          |
Random / Scaffold-Based Evaluation Splits
          |
41-Model Regression Pool
          |
k-Way Combinatorial Fusion Analysis
          |
Binary Success Scoring and Native Classifiers
          |
Calibration and Generalization Evaluation
          |
Streamlit Triage with an Abstention Option
```

## Feature Engineering

The shared fingerprint representation has **14,337 dimensions**:

| Component | Dimensions | Description |
| --- | ---: | --- |
| Six Morgan fingerprint blocks | 12,288 | 2,048 bits per molecular component, radius 2 |
| DRFP block | 2,048 | Reaction-level fingerprint |
| Temperature | 1 | Standardized numerical value |

Missing optional components are represented by zero vectors. Features are cached in an `.npz` file for reuse across experiments.

**DRFP availability:** The paper states that DRFP is optional and that its block was zero-filled in the reported feature-ablation run. The configured feature width should not be interpreted as proof that DRFP was active in every experiment.

## Combinatorial Fusion Analysis

CFA combines model predictions using both **individual validation performance** and **cognitive diversity**, calculated from differences between normalized rank-score curves.

The implementation explores combinations of **two to five models**, with candidate-pool caps for larger combinations and a greedy forward-search fallback.

| Method | Combination strategy |
| --- | --- |
| ASC | Average normalized scores |
| ARC | Average ranks |
| WSCDS | Scores weighted by diversity |
| WRCDS | Ranks weighted by diversity |
| WSCP | Scores weighted by individual performance |
| WRCP | Ranks weighted by individual performance |

For regression, combined scores are mapped back to the yield scale through validation-set least-squares calibration. For classification, the pipeline filters eligible models using positive validation MCC and selects decision thresholds on validation data.

## Evaluation Design

The report uses three evaluation settings:

- **Random split:** 70% training, 15% validation, and 15% test for in-distribution evaluation.
- **Scaffold-OOD split:** Groups reactions by the aryl halide's Murcko scaffold so held-out molecular cores are absent from training.
- **Substrate-grouped 5-fold cross-validation:** Keeps scaffold groups within folds to assess transfer across substrate groups.

**Regression metrics:** R² and MAE, with MAE measured in yield percentage points.  
**Classification metrics:** Accuracy, macro-F1, Matthews correlation coefficient (MCC), ROC AUC, and Brier score where reported.

Additional analyses include 1,000-resample bootstrap confidence intervals, learning curves, threshold sensitivity, feature ablation, reliability diagrams, and error analysis.

## Results Reported in the Paper

### Yield Regression: Random Split

| Individual model | Test R² | MAE (yield percentage points) |
| --- | ---: | ---: |
| LightGBM | 0.819 | 7.47 |
| Chemprop | 0.805 | 7.33 |
| LightGBM - MAE objective | 0.802 | 7.70 |
| skMLP_2 | 0.783 | 8.10 |
| HistGradientBoosting | 0.781 | 9.13 |
| XGBoost | 0.780 | 8.98 |

The paper's regression ablation table reports **test R² = 0.836** for the validation-selected CFA ensemble. Its results narrative instead reports **validation R² = 0.836 and test R² = 0.832**. This discrepancy should be reconciled against saved predictions before selecting a definitive ensemble score.

### Generalization: Regression

Every individual regressor has negative test R² on the single scaffold-OOD split. The validation-selected CFA ensemble scores **-0.022** in the ablation table, showing that fusion does not resolve the generalization problem on this split.

Grouped cross-validation gives a less severe picture: XGBoost achieves **R² = 0.579 ± 0.061** and **MAE = 13.28 ± 1.57** across folds. The single scaffold split and grouped cross-validation therefore describe different levels of difficulty.

### Binary Classification

| Setting / model | Accuracy | MCC | ROC AUC |
| --- | ---: | ---: | ---: |
| Random split - majority baseline | 66.8% | 0.000 | 0.500 |
| Random split - thresholded LightGBM | 92.32% | 0.8250 | 0.9664 |
| Random split - native LightGBM classifier | 92.3% | 0.826 | 0.979 |
| Random split - validation-selected 5-model CFA | 92.44% | 0.8275 | Not reported |
| Grouped 5-fold CV - native XGBoost | 82.4% ± 4.7 pp | 0.579 ± 0.064 | 0.901 ± 0.039 |
| Scaffold-OOD - majority baseline | 52.9% | 0.000 | 0.500 |
| Scaffold-OOD - HistGBT MAE | 59.06% | 0.1754 | 0.6405 |
| Scaffold-OOD - TruncSVD Ridge 100 | 59.06% | 0.1769 | 0.6421 |

**Interpretation:** Classification retains a modest signal on unseen scaffolds even when exact yield regression performs poorly. The approximately 92% random-split accuracy should not be treated as expected accuracy on unfamiliar chemistry.

The paper also explores ensembles ranked using test outcomes. Those results are **retrospective test-set ceilings**, not validation-selected performance estimates, and are excluded from the headline table above. All numbers here are transcribed from the report and have not been independently reproduced.

## Feature Ablation

The LightGBM ablation reports:

| Available inputs | Accuracy | MCC | ROC AUC |
| --- | ---: | ---: | ---: |
| Aryl halide and amine only | 74.1% | 0.378 | 0.801 |
| All six molecular components | 89.7% | 0.761 | 0.951 |
| Components plus temperature | 91.6% | 0.799 | 0.959 |
| Plus zero-filled DRFP block | 91.6% | 0.799 | 0.959 |

These results highlight the predictive contribution of reaction conditions beyond the substrate pair. The unchanged final row reflects an inactive DRFP block, not a test of active DRFP features.

## Streamlit Triage Tool

The report describes an interface that accepts **six component SMILES strings and temperature**, then returns calibrated success probabilities and a binary decision.

- LightGBM and Chemprop outputs are calibrated using validation-fitted isotonic regression.
- The reported LightGBM calibration experiment achieves a **Brier score of 0.045** on the random test split.
- A reject-option wrapper abstains when calibrated model probabilities differ by more than **0.20**.
- Disagreement returns an **uncertain - verify in lab** result.

This interface supports computational prioritization; experimental validation remains necessary to establish a reaction outcome.

## Repository Structure

The following files and directories are described in the paper's reproducibility section. Their presence and exact interfaces should be checked against the repository.

```text
main.py                         Experiment pipeline and configuration
cfa.py                          Regression fusion
cfa_classify.py                 Classification fusion
app.py                          Streamlit triage interface
models/                         Model implementations
scripts/
    download_data.py            ORD download script
models_saved/
    lightgbm.pkl                Saved LightGBM model
    chemprop_dir/               Chemprop artifacts
    calibrators.pkl             Isotonic calibrators
report/
    make_all_figures.py          Figure generation
    make_extra_figures.py        Additional figure generation
```

## Limitations

- **Distribution shift:** Performance falls substantially on unseen scaffolds.
- **Small validation slice:** The scaffold-OOD validation set contains approximately ten reactions, making selection and calibration unstable.
- **Reporting bias:** The ORD subset reflects the reactions and yields available in its sources.
- **Threshold dependence:** The 50% success boundary is an operational choice; different applications may require different thresholds.
- **Coverage:** Error analysis identifies difficult substrate combinations, including aryl-chloride cases.
- **Ensemble evidence:** A reported paired-bootstrap comparison of a three-seed LightGBM ensemble with a single model does not establish significant improvement at the 5% level; it is not a direct significance test of every CFA combination.
- **Reproducibility:** Ensemble-score inconsistencies and optional-feature settings require verification against code and cached outputs.

## Future Work

- Pretrain molecular or reaction representations on a larger ORD corpus.
- Evaluate active learning against held-out experimental data.
- Improve coverage through substrate-aware sampling and high-throughput data augmentation.
- Explore shared models across additional coupling reaction families.
- Strengthen scaffold-aware validation and uncertainty evaluation.

## Running the Project

The source code, dependency manifest, and model artifacts were not supplied with this README request. Exact installation commands, download options, and execution flags should be added after checking those files.

The paper reports `seed=42` and an experimental workstation with a 16-core CPU, NVIDIA RTX 3090, and 64 GB RAM. These describe the reported run rather than verified minimum requirements.

## Technical Report

**Predicting Buchwald-Hartwig C-N Coupling Reaction Success: From Yield Regression to Binary Triage with k-Way Combinatorial Fusion Analysis** documents the methodology, comparisons, and findings. Include `CN_Coupling_Reaction_Modeling.pdf` alongside the project documentation.

## Acknowledgments

- Open Reaction Database maintainers
- Chemprop, NGBoost, rxnfp, and DRFP authors and contributors
- Fordham University
