import numpy as np

MODEL_NAME = "XGB_Shallow"

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", seed=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(X_val).astype(np.float32), model.predict(X_te).astype(np.float32)
