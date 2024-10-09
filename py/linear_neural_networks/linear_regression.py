import torch
import utils
from syntetic_regression import SyntheticRegressionData
import matplotlib.pyplot as plt

class LinearRegressionScratch(utils.Module): #@save
    """ Linear regression model """
    def __init__(self, num_inputs, lr, sigma=.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma,
                              (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, X):
        # Here broadcasting allows us to add a scalar
        # to a vector, by adding the value to each
        # element of the vector
        return torch.matmul(X, self.w) + self.b
    def loss(self, y_hat, y):
        """ Mean squared error loss """
        l = ((y_hat - y) ** 2) / 2
        return l.mean()
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
class SGD(utils.HyperParameters): #@save
    """ Minibatch stochastic gradient descent. """
    def __init__(self, params, lr):
        self.save_hyperparameters()
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

if __name__ == "__main__":
    model = LinearRegressionScratch(2, lr=.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = utils.Trainer(max_epochs=3)
    trainer.fit(model, data)
    plt.show()
