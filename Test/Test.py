from numpy.typing import NDArray
import numpy as np

class Test():
    def __init__(self, model, error):
        self.model = model
        self.error = error

    def test(self, X: NDArray, y: NDArray):
        out = self.model.forward(X)
        
        right = 0
        examples = 32
        for l in range(X.shape[0]):
            if out[l].argmax() == y[l].argmax():
                right += 1

        print(f"Accuracy: {(right/examples) * 100}%")
        # l = self.error.loss(out, y)
        # print(f"Loss: {l}")