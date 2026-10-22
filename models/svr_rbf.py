from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

MODEL_NAME = "SVR_RBF"

# PCA reduces to 100 components first; without it the RBF kernel matrix
# (n_train × n_train × d) over 14 k features would be prohibitively slow.
def make_model():
    return Pipeline([
        ("pca",    PCA(n_components=100, random_state=42)),
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)),
    ])
