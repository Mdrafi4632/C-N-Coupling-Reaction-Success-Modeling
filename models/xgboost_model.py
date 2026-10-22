import numpy as np

MODEL_NAME = "XGBoost"

_DEFAULTS = {
    "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "seed": 42, "n_jobs": -1,
}

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import xgboost as xgb
    model = xgb.XGBRegressor(**{**_DEFAULTS, **cfg.get("xgb_params", {})})
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(X_val).astype(np.float32), model.predict(X_te).astype(np.float32)
