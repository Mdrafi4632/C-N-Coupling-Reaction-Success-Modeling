from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

MODEL_NAME = "TruncSVD_Ridge_50"

def make_model():
    return Pipeline([
        ("svd",   TruncatedSVD(n_components=50, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])
