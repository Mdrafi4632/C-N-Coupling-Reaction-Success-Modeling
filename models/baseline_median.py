from sklearn.dummy import DummyRegressor

MODEL_NAME = "Baseline_Median"

def make_model():
    return DummyRegressor(strategy="median")
