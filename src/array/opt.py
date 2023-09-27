import torch

from typing import Dict

from src.array.auto import Array
from src.array.mod import Module


class Optimizer:
    """Base class for optimizers."""
    model: Module
    lr: float

    def __init__(self, model: Module, lr: float = 0.01):
        self.model = model
        self.lr = lr

    def zero_grad(self):
        """Zero gradients."""
        for param in self.model.parameters():
            param.zero_grad()

    def step(self):
        """Update parameters."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def step(self):
        for param in self.model.parameters():
            param.data -= self.lr * param.grad


class Adam(Optimizer):
    """Adaptive Moment Estimation."""
    beta1: float
    beta2: float
    m: Dict[Array, torch.Tensor]
    v: Dict[Array, torch.Tensor]

    def __init__(self, model: Module, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {}
        self.v = {}
        for param in self.model.parameters():
            self.m[param] = torch.zeros_like(param.data)
            self.v[param] = torch.zeros_like(param.data)

    def step(self):
        for param in self.model.parameters():
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * param.grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * param.grad ** 2
            m_hat = self.m[param] / (1 - self.beta1)
            v_hat = self.v[param] / (1 - self.beta2)
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
