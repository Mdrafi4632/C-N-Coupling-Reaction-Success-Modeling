from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "Huber"

def make_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model",  HuberRegressor(max_iter=500)),
    ])
