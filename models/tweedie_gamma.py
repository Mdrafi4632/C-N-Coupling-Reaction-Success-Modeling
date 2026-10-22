from sklearn.linear_model import TweedieRegressor

MODEL_NAME = "TweedieGamma"

def make_model():
    return TweedieRegressor(power=2.0, alpha=0.1, link="log", max_iter=500)
