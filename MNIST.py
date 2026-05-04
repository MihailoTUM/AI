import numpy as np
from numpy.typing import NDArray
import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/fashion-mnist_train.csv")

data = df.drop(columns=["label"])
raw_data = data.to_numpy()
X = raw_data[:64] / 255

raw_data_y = df[["label"]].to_numpy()
y = np.eye(10)[raw_data_y[:64].squeeze()]

def relu(X: NDArray):
    return X > 0

def relu2Deriv(X: NDArray):
    return (X > 0).astype(float)

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
        self.z2 = self.a1 @ self.weights_2 + self.bias_2
        return self.z2
    
    def softmax(self, X: NDArray) -> NDArray:
        X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def backward(self, gradients):
        self.weights_2_grads = self.a1.T @ gradients
        self.bias_2_grads = np.mean(gradients, axis=0)

        self.weights_1_grads = self.X.T @ ((gradients @ self.weights_2.T) * relu2Deriv(self.z1))
        self.bias_1_grads = np.mean(gradients @ self.weights_2.T * relu2Deriv(self.z1), axis=0)

    def update_parameters(self, lr=0.01):
        self.weights_1 = self.weights_1 - lr * self.weights_1_grads
        self.bias_1 = self.bias_1 - lr * self.bias_1_grads

        self.weights_2 = self.weights_2 - lr * self.weights_2_grads
        self.bias_2 = self.bias_2 - lr * self.bias_2_grads

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
    
model = NN(784, 128, 10)
out = model.forward(X)
print(out.shape)

crossEntropy = CrossEntropyLoss()

epochs = 200

for epoch in range(epochs):
    out = model.forward(X)
    l = crossEntropy.loss(out, y)
    print(f"Iter: {epoch + 1}, Loss: {l}")
    model.backward(crossEntropy.get_gradients())
    model.update_parameters(lr=0.005)

