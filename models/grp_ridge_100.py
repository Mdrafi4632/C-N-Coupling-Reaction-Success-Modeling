from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

MODEL_NAME = "GRP_Ridge_100"

def make_model():
    return Pipeline([
        ("grp",   GaussianRandomProjection(n_components=100, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])
