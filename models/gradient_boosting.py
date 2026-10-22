from sklearn.ensemble import GradientBoostingRegressor

MODEL_NAME = "GradientBoosting"

def make_model():
    return GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=42,
    )
