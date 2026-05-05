import numpy as np
from numpy.typing import NDArray
import kagglehub
import pandas as pd
from Model.NN import NN
from Error.CrossEntropyLoss import CrossEntropyLoss
from Optimizer.SGD import SGD
from DataLoader.DataLoader import DataLoader
from Test.Test import Test

TRAINING_EXAMPLES = 1032

# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/fashion-mnist_train.csv")

data = df.drop(columns=["label"])
raw_data = data.to_numpy()
X = raw_data[:TRAINING_EXAMPLES] / 255

raw_data_y = df[["label"]].to_numpy()
y = np.eye(10)[raw_data_y[:TRAINING_EXAMPLES].squeeze()]

dataloader = DataLoader(X[:1000], y[:1000], batch_size=32)

model = NN(784, 128, 10)
error = CrossEntropyLoss()
optimizer = SGD(model, error, dataloader)

optimizer.train(epochs=100, lr=0.005)

t = Test(model, error)

t.test(X[1000:], y[1000:])

