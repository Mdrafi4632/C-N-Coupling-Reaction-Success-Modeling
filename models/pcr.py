from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

MODEL_NAME = "PCR"

def make_model():
    return Pipeline([
        ("pca",   PCA(n_components=100, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ])
