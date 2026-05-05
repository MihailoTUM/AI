import numpy as np
from numpy.typing import NDArray

class DataLoader():
    def __init__(self, X: NDArray, y: NDArray, batch_size=32):
        self.X = X
        self.y = y 
        self.batch_size = batch_size
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if(self.index + self.batch_size > self.X.shape[0]):
            raise StopIteration

        start = self.index
        end = self.index + self.batch_size

        self.index = end
        return self.X[start:end], self.y[start:end]

    def reset(self):
        self.index = 0