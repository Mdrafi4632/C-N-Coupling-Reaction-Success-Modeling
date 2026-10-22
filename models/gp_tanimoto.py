"""GP_Tanimoto — Gaussian-process regression with a Tanimoto kernel on Morgan FPs.

Tanimoto similarity is the natural metric for binary fingerprints. We use only
the binary fingerprint columns (the trailing scalar feature, e.g. temperature,
breaks Tanimoto's denominator), and subsample to 1000 training rows to keep the
n×n kernel solve tractable. Reference: Griffiths et al. ICML 2020.
"""
import numpy as np

MODEL_NAME = "GP_Tanimoto"

_SUBSET = 1000


def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _tanimoto import tanimoto_matrix

    rng = np.random.RandomState(42)
    n_tr = len(X_tr)
    if n_tr > _SUBSET:
        idx = rng.choice(n_tr, _SUBSET, replace=False)
        X_sub, y_sub = X_tr[idx], y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    # Strip trailing temperature scalar so we operate on the binary FP block
    n_bits = X_sub.shape[1] - 1 if X_sub.shape[1] % 2048 == 1 else X_sub.shape[1]
    Xb_sub = (X_sub[:, :n_bits] > 0.5).astype(np.float32)
    Xb_val = (X_val[:, :n_bits] > 0.5).astype(np.float32)
    Xb_te  = (X_te[:,  :n_bits] > 0.5).astype(np.float32)

    K_train = tanimoto_matrix(Xb_sub, Xb_sub)
    K_val   = tanimoto_matrix(Xb_val, Xb_sub)
    K_test  = tanimoto_matrix(Xb_te,  Xb_sub)

    # Closed-form GP regression: μ = K_*(K + σ²I)^{-1} y
    sigma2 = 1.0
    n = K_train.shape[0]
    L = np.linalg.cholesky(K_train + sigma2 * np.eye(n, dtype=np.float32))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sub.astype(np.float32)))
    val_p = (K_val @ alpha).astype(np.float32)
    tes_p = (K_test @ alpha).astype(np.float32)
    return val_p, tes_p
