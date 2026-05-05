import numpy as np
from numpy.typing import NDArray
from Activation.Relu import relu, relu2Derivative
from Activation.Dropout import dropout

class NN():
    def __init__(self, n_input, n_hidden, n_output):
        self.n_inpu = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.weights_1 = np.random.rand(n_input, n_hidden) - 0.5
        self.bias_1 =  np.random.rand(n_hidden)

        self.weights_2 = np.random.rand(n_hidden, n_output) - 0.5
        self.bias_2 = np.random.rand(n_output)

        self.weights_1_grads = None
        self.bias_1_grads = None

        self.weights_2_grads = None
        self.bias_2_grads = None

    def forward(self, X: NDArray) -> NDArray:
        self.X = X
        self.z1 = X @ self.weights_1 + self.bias_1
        self.a1 = relu(self.z1)

        self.mask = dropout(self.a1, p=0.2)
        self.a1 = self.mask * self.a1

        self.z2 = self.a1 @ self.weights_2 + self.bias_2
        return self.z2
    
    def softmax(self, X: NDArray) -> NDArray:
        X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def backward(self, gradients):
        self.weights_2_grads = (self.mask * self.a1).T @ gradients
        self.bias_2_grads = np.mean(gradients, axis=0)

        self.weights_1_grads = self.X.T @ ((gradients @ self.weights_2.T) * relu2Derivative(self.z1))
        self.bias_1_grads = np.mean(gradients @ self.weights_2.T * relu2Derivative(self.z1), axis=0)

    def update_parameters(self, lr=0.01):
        self.weights_1 = self.weights_1 - lr * self.weights_1_grads
        self.bias_1 = self.bias_1 - lr * self.bias_1_grads

        self.weights_2 = self.weights_2 - lr * self.weights_2_grads
        self.bias_2 = self.bias_2 - lr * self.bias_2_grads
