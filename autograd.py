import torch

from typing import Optional, List


class Operation:

    def forward(self, *args: "Array") -> "Array":
        raise NotImplementedError

    def backward(self, *grads: "Array") -> None:
        raise NotImplementedError

    def __call__(self, *args: "Array") -> "Array":
        out = self.forward(*args)
        out.prev.append(*args)
        return out


class Add(Operation):

    def forward(self, x: "Array", y: "Array") -> "Array":
        return Array(x.data + y.data, self)

    def backward(self, out: "Array") -> None:
        out.prev[0].grad += out.grad
        out.prev[1].grad += out.grad
        for child in out.prev:
            if child.op is not None:
                child.op.backward(child)


class Array:
    data: torch.Tensor
    grad: Optional[torch.Tensor]

    def __init__(self, data: torch.Tensor, op: Optional[Operation] = None):
        self.data = data
        self.grad = None
        self.prev: List["Array"] = []
        self.op: Optional[Operation] = op

    def __add__(self, other: "Array") -> "Array":
        return Add()(self, other)
