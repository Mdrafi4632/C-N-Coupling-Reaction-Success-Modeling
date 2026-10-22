from sklearn.neighbors import KNeighborsRegressor

MODEL_NAME = "KNN_15"

def make_model():
    return KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
