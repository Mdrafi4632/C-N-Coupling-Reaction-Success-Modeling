from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection

MODEL_NAME = "SRP_Ridge_200"

def make_model():
    return Pipeline([
        ("srp",   SparseRandomProjection(n_components=200, density=0.1, random_state=43)),
        ("ridge", Ridge(alpha=1.0)),
    ])
