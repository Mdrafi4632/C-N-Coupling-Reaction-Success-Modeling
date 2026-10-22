from sklearn.tree import DecisionTreeRegressor

MODEL_NAME = "DecisionTree"

def make_model():
    return DecisionTreeRegressor(max_depth=None, min_samples_leaf=5, random_state=42)
