from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

MODEL_NAME = "LinearSVR"

def make_model():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svr",    LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, dual="auto")),
    ])
