from sklearn.linear_model import OrthogonalMatchingPursuit

MODEL_NAME = "OrthogonalMP"

def make_model():
    return OrthogonalMatchingPursuit(n_nonzero_coefs=200)
