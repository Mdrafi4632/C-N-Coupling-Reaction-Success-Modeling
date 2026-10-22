from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "RandomForest"

def make_model():
    return RandomForestRegressor(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1)
