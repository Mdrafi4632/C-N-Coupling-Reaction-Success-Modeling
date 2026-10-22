import numpy as np

MODEL_NAME = "LGB_Dart"

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import lightgbm as lgb
    # DART — drops random trees during boosting; different ensemble structure from GBDT.
    model = lgb.LGBMRegressor(
        boosting_type="dart", drop_rate=0.1,
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_val).astype(np.float32), model.predict(X_te).astype(np.float32)
