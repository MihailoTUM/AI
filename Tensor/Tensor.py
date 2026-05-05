import numpy as np
from numpy.typing import NDArray

class Tensor():
    def __init__(self, n: int, m: int):
        self.data = np.random.rand(n, m) - 0.5
        self.n = n
        self.m = m

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, matrix):
        if self.n != matrix.n or self.m != matrix.m:
            return None

        add = Tensor(self.n, self.m)
        add.data = self.data + matrix.data

        return add
    
    def __matmul__(self, matrix):
        if self.m != matrix.n:
            return None
        
        matmul = Tensor(self.n, self.m)
        matmul.data = self.data @ matrix.data

        return matmul


x = Tensor(3, 3)
y = Tensor(3, 3)

sum = x + y
print(sum)

m = x @ y
print(m)

