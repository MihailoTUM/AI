import numpy as np
from numpy.typing import NDArray

class CrossEntropyLoss():
    def __init__(self):
        self.y_softmanx = None

    def softmax(self, X:NDArray) -> NDArray:
        X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def loss(self, y_logits, y_expected):
        y_softmax = self.softmax(y_logits)
        self.y_softmax = y_softmax
        self.y_expected = y_expected
        l = y_expected * np.log(y_softmax)
        return np.mean(-np.sum(l, axis=1, keepdims=True), axis=0, keepdims=True)
    
    def get_gradients(self):
        return self.y_softmax - self.y_expected