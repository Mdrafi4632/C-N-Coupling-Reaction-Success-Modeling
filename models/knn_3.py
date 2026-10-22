from sklearn.neighbors import KNeighborsRegressor

MODEL_NAME = "KNN_3"

def make_model():
    return KNeighborsRegressor(n_neighbors=3, weights="distance", n_jobs=-1)
