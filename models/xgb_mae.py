import numpy as np

MODEL_NAME = "XGB_MAE"

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import xgboost as xgb
    # MAE objective — robust to yield outliers, different error geometry than squared loss.
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", seed=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(X_val).astype(np.float32), model.predict(X_te).astype(np.float32)
