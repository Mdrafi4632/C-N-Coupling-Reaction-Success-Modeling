from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "skMLP_2"

def make_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("mlp",    MLPRegressor(hidden_layer_sizes=(512, 256),
                                activation="relu", alpha=0.001,
                                max_iter=300, early_stopping=True,
                                random_state=42)),
    ])
