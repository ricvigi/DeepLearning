# An implementation of simple gates to show how backpropagation works

import torch
from typing import TypeAlias
from typing import Tuple

Tensor: TypeAlias = torch.Tensor

class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, y:Tensor) -> Tensor:
        ctx.save_for_backward(x, y) # We need to cache some values for use in backward propagation
        z = x * y   # multiply gate function
        return z
    @staticmethod
    def backward(ctx, grad_z:Tensor) -> Tuple[Tensor, Tensor]: # grad_z is the upstream gradient
        x, y = ctx.saved_tensors
        grad_x = y * grad_z
        grad_y = x * grad_z
        return grad_x, grad_y

class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, y:Tensor) -> Tensor:
        z = x + y
        return z
    @staticmethod
    def backward(ctx, grad_z:Tensor) -> Tuple[Tensor, Tensor]:
        grad_x = grad_z
        grad_y = grad_z
        return grad_x, grad_y

class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor) -> Tuple[Tensor, Tensor]:
        z = x
        return z, z
    @staticmethod
    def backward(ctx, grad_z1:Tensor, grad_z2:Tensor) -> Tuple[Tensor]:
        grad_x = grad_z1 + grad_z2
        return grad_x, # In PyTorch's autograd system, the backward method MUST return a tuple, even if there is only one element.
class Max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:Tensor, y:Tensor) -> Tensor:
        ctx.save_for_backward(x, y)
        z = torch.max(x, y)
        return z
    @staticmethod
    def backward(ctx, grad_z:Tensor) -> Tuple[Tensor, Tensor]:
        x, y = ctx.saved_tensors
        # What should we do if x == y? where should i propagate back the gradient?
        grad_x = grad_z * (x >= y).float()
        grad_y = grad_z * (y > x).float()
        return grad_x, grad_y

















