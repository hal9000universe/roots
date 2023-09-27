import math
import torch

from typing import Iterable, Callable, Union

from src.array.auto import Array


class Module:
    """Base class for modules."""

    def parameters(self) -> Union[Iterable[Array], Array]:
        """Get parameters."""
        raise NotImplementedError

    def __call__(self, *args: Array) -> Array:
        """Forward pass."""
        raise NotImplementedError


class Linear(Module):
    """Linear layer."""
    _weights: Array

    def __init__(self, in_features: int, out_features: int):
        """Initialize linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        # initialize weights
        # see: https://www.deeplearning.ai/ai-notes/initialization/index.html
        # He initialization
        self._weights = Array(torch.randn(in_features, out_features, requires_grad=False) * math.sqrt(2 / in_features))

    def parameters(self) -> Array:
        """Get parameters."""
        return self._weights

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input array. (batch_size, in_features)

        Returns:
            Output array. (batch_size, out_features)
        """
        return x @ self._weights


class Sequential(Module):
    """Sequential model."""
    _layers: Iterable[Callable[[Array], Array]]

    def __init__(self, layers: Iterable[Callable[[Array], Array]]):
        """Initialize sequential model.

        Args:
            layers: Layers.
        """
        self._layers = layers

    def parameters(self) -> Iterable[Array]:
        """Get parameters."""
        for layer in self._layers:
            if isinstance(layer, Linear):
                yield layer.parameters()

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input array. (batch_size, in_features)

        Returns:
            Output array. (batch_size, out_features)
        """
        for layer in self._layers:
            x = layer(x)
        return x
