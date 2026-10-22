from sklearn.linear_model import TweedieRegressor

MODEL_NAME = "Tweedie"

# power=1.5 — compound Poisson-Gamma, suited to non-negative skewed targets (yield).
def make_model():
    return TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=500)
