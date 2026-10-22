import numpy as np

MODEL_NAME = "LightGBM"

_DEFAULTS = {
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**{**_DEFAULTS, **cfg.get("lgb_params", {})})
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return model.predict(X_val).astype(np.float32), model.predict(X_te).astype(np.float32)
