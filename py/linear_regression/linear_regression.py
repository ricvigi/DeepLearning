import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from synthetic_regression import SyntheticRegressionData
from torch import nn

class LinearRegression(utils.Module): #@save
    """ LazyLinear regression model """
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, .01)
        self.net.bias.data.fill_(0)
    def forward(self, X):
        """ In forward we just invoke the built-in __call__
            method of the predefined layers to compute the
            outputs"""
        return self.net(X)
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    def configure_optimizers(self):
        """ When you initiate a SGD instance, specify the
            parameters to optimize over, and the learning
            rate"""
        return torch.optim.SGD(self.parameters(), self.lr)

if __name__=="__main__":
    model = LinearRegression(lr=.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = utils.Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()
    with torch.no_grad():
        print(f"error in estimating w: {data.w - model.net.weight.reshape(data.w.shape)}")
        print(f"error in estimating b: {data.b - model.net.bias}")
