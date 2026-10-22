from sklearn.linear_model import ElasticNet

MODEL_NAME = "ElasticNet"

def make_model():
    return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
