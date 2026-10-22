from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "KernelRidge"

# PCA reduces to 100 components before the RBF kernel to keep the
# n×n kernel matrix tractable.
def make_model():
    return Pipeline([
        ("pca",    PCA(n_components=100, random_state=42)),
        ("scaler", StandardScaler()),
        ("kr",     KernelRidge(kernel="rbf", alpha=0.1, gamma=0.01)),
    ])
