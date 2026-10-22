from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection

MODEL_NAME = "SRP_Ridge_100"

def make_model():
    return Pipeline([
        ("srp",   SparseRandomProjection(n_components=100, density=0.1, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])
