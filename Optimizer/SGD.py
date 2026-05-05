

class SGD():
    def __init__(self, model, error, data_loader):
        self.model = model
        self.error = error
        self.data_loader = iter(data_loader)

    def train(self, epochs=100, lr=0.01):
        for epoch in range(epochs):
            epoch_error = 0
            examples = 0
            self.data_loader.reset()
            for runtime in range(int(1000/32)):
                X, y = next(self.data_loader)
                out = self.model.forward(X)
                l = self.error.loss(out, y)
                epoch_error += l
                examples += X.shape[0]
                self.model.backward(self.error.get_gradients())
                self.model.update_parameters(lr)
            print(f"Epoch: {epoch}; Loss: {epoch_error / examples}")

    