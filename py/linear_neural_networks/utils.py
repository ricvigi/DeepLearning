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
            setattr(self, k, v)

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
        display.display(self.fig)
        display.clear_output(wait=False)
        # plt.show()

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
    board = ProgressBoard("x")
    for x in np.arange(0, 10, 0.1):
        board.draw(x, np.sin(x), "sin", every_n=2)
        board.draw(x, np.cos(x), "cos", every_n=2)








