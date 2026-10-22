import numpy as np

MODEL_NAME = "NGBoost"

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.decomposition import TruncatedSVD
    # 14k binary features × default ngboost depth-3 trees is impossibly slow;
    # PCA-reduce to 200 first (Ridge over the SVD basis already worked well).
    svd = TruncatedSVD(n_components=200, random_state=42)
    Xs_tr  = svd.fit_transform(X_tr)
    Xs_val = svd.transform(X_val)
    Xs_te  = svd.transform(X_te)
    model = NGBRegressor(
        Dist=Normal, n_estimators=300, learning_rate=0.05,
        minibatch_frac=0.5, col_sample=0.5,
        random_state=42, verbose=False,
    )
    model.fit(Xs_tr, y_tr, X_val=Xs_val, Y_val=y_val, early_stopping_rounds=30)
    return model.predict(Xs_val).astype(np.float32), model.predict(Xs_te).astype(np.float32)
