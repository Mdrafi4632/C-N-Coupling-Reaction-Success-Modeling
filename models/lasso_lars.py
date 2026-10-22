from sklearn.linear_model import LassoLars

MODEL_NAME = "LassoLars"

def make_model():
    return LassoLars(alpha=0.001, max_iter=500)
