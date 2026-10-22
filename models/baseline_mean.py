from sklearn.dummy import DummyRegressor

MODEL_NAME = "Baseline_Mean"

def make_model():
    return DummyRegressor(strategy="mean")
