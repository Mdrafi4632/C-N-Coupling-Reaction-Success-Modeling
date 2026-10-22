from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "SGDHuber"

def make_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("sgd",    SGDRegressor(loss="huber", epsilon=0.1, penalty="l2",
                                alpha=0.001, max_iter=2000, random_state=42)),
    ])
