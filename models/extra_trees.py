from sklearn.ensemble import ExtraTreesRegressor

MODEL_NAME = "ExtraTrees"

def make_model():
    return ExtraTreesRegressor(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)
