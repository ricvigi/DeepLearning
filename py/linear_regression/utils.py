import time
import numpy as np
import torch
import inspect
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import collections
from IPython import display
from torch import nn

def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def add_to_class(Class): #@save
    """ This function allows us to register functions as methods
        in a class AFTER the class has been created. This also
        works after an instance of the class has been created """
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters: #@save
    """ The base class of hyperparameters. This is a utility
        class that saves all arguments in a class's __init__
        method as class attributes. This allows us to extend
        constructor call signatures implicitly without additional
        code. To use it, we define a class that inherits from
        HyperParameters and calls save_hyperparameters in the
        __init__ method. """
    def save_hyperparameters(self, ignore=[]):
        """ Save function arguments into class attributes """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+["_"])
                        and not k.startswith("_")}
        for k, v in self.hparams.items():
            super().__setattr__(k, v)


class ProgressBoard(HyperParameters): #@save
    """ A board that plots data points in animation """
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale="linear", yscale="linear",
                 ls=["-", "--", "-.", ":"],
                 colors=["C0", "C1", "C2", "C3"],
                 fig=None, axes=None, figsize=(3.5, 2.5),
                 display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple("Point", ["x", "y"])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls,
                                     self.colors):
            plt_lines.append(plt.plot([p.x for p in v],
                                           [p.y for p in v],
                                           linestyle=ls,
                                           color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        # toggle this if you're in an interactive environment
        # display.display(self.fig)
        # display.clear_output(wait=False)
        plt.draw()
        plt.pause(.001)


class Module(nn.Module, HyperParameters): #@save
    """ Base class of all models """
    def __init__(self,
                 plot_train_per_epoch=2,
                 plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    def loss(self, y_hat, y):
        raise NotImplementedError
    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)
    def plot(self, key, value, train):
        """ Plot a point in animation """
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x,
                        value.to(torch.device("cpu")).detach().numpy(),
                        ("train_" if train else "val_") + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=True)
        return l
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)
    def configure_optimizers(self):
        raise NotImplementedError

class DataModule(HyperParameters): #@save
    """ Base class of data """
    def __init__(self, root="../data", num_workers=4):
        self.save_hyperparameters()
    def get_dataloader(self, train):
        raise NotImplementedError
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    def val_dataloader(self):
        return self.get_dataloader(train=False)

class Trainer(HyperParameters): #@save
    """ Base class for training models with data """
    def __init__(self, max_epochs,
                 num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, "No GPU support yet"
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not
                                None else 0)
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model
    def prepare_batch(self, batch):
        return batch
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

if __name__=="__main__":
    """ Test add_to_class """
    # class A:
    #     def __init__(self):
    #         self.b = 1
    # a = A()
    # @add_to_class(A)
    # def do(self):
    #     print("Class attribue b is", self.b)
    # a.do()

    # class B(HyperParameters):
    #     def __init__(self, a, b, c):
    #         self.save_hyperparameters(ignore=['c'])
    #         print("self.a =", self.a, "self.b =", self.b)
    #         print("There is no self.c =", not hasattr(self,  "c"))
    # b = B(a=1, b=2, c=3)
    # board = ProgressBoard("x")
    # for x in np.arange(0, 10, 0.1):
    #     board.draw(x, np.sin(x), "sin", every_n=2)
    #     board.draw(x, np.cos(x), "cos", every_n=2)
    # plt.show()








