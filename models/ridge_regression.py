from sklearn.linear_model import Ridge

MODEL_NAME = "Ridge"

def make_model():
    return Ridge(alpha=1.0)
