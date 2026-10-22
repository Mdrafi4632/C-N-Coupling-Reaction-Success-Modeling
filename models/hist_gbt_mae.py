from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_NAME = "HistGBT_MAE"

def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=500, learning_rate=0.05, max_depth=6,
        l2_regularization=0.1, random_state=42,
    )
