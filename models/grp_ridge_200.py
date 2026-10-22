from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

MODEL_NAME = "GRP_Ridge_200"

def make_model():
    return Pipeline([
        ("grp",   GaussianRandomProjection(n_components=200, random_state=43)),
        ("ridge", Ridge(alpha=1.0)),
    ])
