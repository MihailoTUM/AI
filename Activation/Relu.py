from numpy.typing import NDArray
import numpy as np

def relu(X: NDArray):
    return np.maximum(0, X)

# def relu(X: NDArray):
#     return (X > 0).astype(float)

def relu2Derivative(X: NDArray):
    return (X > 0).astype(float)