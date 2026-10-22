from sklearn.neighbors import KNeighborsRegressor

MODEL_NAME = "KNN"

def make_model():
    return KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1)
