import numpy as np

MODEL_NAME = "PLS"

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    from sklearn.cross_decomposition import PLSRegression
    # PLSRegression.predict() returns shape (n, 1) — squeeze to 1-D
    model = PLSRegression(n_components=20, max_iter=500)
    model.fit(X_tr, y_tr)
    val_p = model.predict(X_val).ravel().astype(np.float32)
    tes_p = model.predict(X_te).ravel().astype(np.float32)
    return val_p, tes_p
