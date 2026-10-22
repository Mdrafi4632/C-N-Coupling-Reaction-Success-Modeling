from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_NAME = "HistGBT"

def make_model():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=6,
        l2_regularization=0.1, random_state=42,
    )
