from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

MODEL_NAME = "AdaBoost"

def make_model():
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=200, learning_rate=0.5, random_state=42,
    )
