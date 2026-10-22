from sklearn.linear_model import Lasso

MODEL_NAME = "Lasso"

def make_model():
    return Lasso(alpha=0.01, max_iter=5000)
