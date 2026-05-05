from numpy.typing import NDArray
import numpy as np

def dropout(X: NDArray, p=0.2):
    mask = (np.random.rand(X.shape[0], X.shape[1]) > p).astype(float)
    return mask