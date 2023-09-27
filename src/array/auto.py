import torch

from typing import List, Optional


def one_hot(x: torch.Tensor) -> "Array":
    """One-hot encode the input tensor

    Args:
        x: Input tensor.

    Returns:
        One-hot encoded Array.
    """
    return Array(torch.nn.functional.one_hot(x, num_classes=10))


class Operation:
    """Base class for operations."""

    def forward(self, *args: "Array") -> "Array":
        """Forward pass.

        Args:
            *args: Input arrays.

        Returns:
            Output array.
        """
        raise NotImplementedError

    def backward(self, out: "Array") -> None:
        """Backward pass.

        Args:
            out: Output array.
        """
        raise NotImplementedError

    def __call__(self, *args: "Array") -> "Array":
        out = self.forward(*args)
        for arg in args:
            out.prev.append(arg)
        return out


class Add(Operation):
    """Addition operation."""

    def forward(self, x: "Array", y: "Array") -> "Array":
        """Forward pass.

        Args:
            x: First input array.
            y: Second input array.

        Returns:
            Output array.
        """
        return Array(x.data + y.data, self)

    def backward(self, out: "Array") -> None:
        """Backward pass.

        Args:
            out: Output array.
        """
        # initialize gradients if they are None
        if out.prev[0].grad is None:
            out.prev[0].grad = torch.zeros_like(out.prev[0].data)
        if out.prev[1].grad is None:
            out.prev[1].grad = torch.zeros_like(out.prev[1].data)
        # compute gradients
        out.prev[0].grad += out.grad
        out.prev[1].grad += out.grad
        # back-propagate gradients
        for child in out.prev:
            if child.op is not None:
                child.op.backward(child)


class MatMul(Operation):
    """Matrix multiplication operation."""

    def forward(self, x: "Array", y: "Array") -> "Array":
        """Forward pass.

        Args:
            x: First input array.
            y: Second input array.
        """
        return Array(x.data @ y.data, self)

    def backward(self, out: "Array") -> None:
        """Backward pass.

        Args:
            out: Output array.
        """
        # initialize gradients if they are None
        if out.prev[0].grad is None:
            out.prev[0].grad = torch.zeros_like(out.prev[0].data)
        if out.prev[1].grad is None:
            out.prev[1].grad = torch.zeros_like(out.prev[1].data)
        # compute gradients
        out.prev[0].grad += out.grad @ out.prev[1].data.T
        out.prev[1].grad += out.prev[0].data.T @ out.grad
        # back-propagate gradients
        for child in out.prev:
            if child.op is not None:
                child.op.backward(child)


class ReLU(Operation):
    """ReLU operation."""

    def forward(self, x: "Array") -> "Array":
        """Forward pass.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        return Array(torch.nn.functional.relu(x.data), self)

    def backward(self, out: "Array") -> None:
        """Backward pass.

        Args:
            out: Output array.
        """
        # initialize gradients if they are None
        if out.prev[0].grad is None:
            out.prev[0].grad = torch.zeros_like(out.prev[0].data)
        # compute gradients
        out.prev[0].grad += out.grad * (out.data > 0)
        # back-propagate gradients
        if out.prev[0].op is not None:
            out.prev[0].op.backward(out.prev[0])


class CrossEntropyLoss(Operation):
    """Cross entropy loss."""
    _y_true: Optional["Array"]

    def __init__(self, y_true: Optional["Array"] = None):
        """Initialize cross entropy loss."""
        self._y_true = y_true

    def forward(self, y_pred: "Array", y_true: "Array") -> "Array":
        """Forward pass.

        Args:
            y_pred: Predicted labels. (batch_size, num_classes)
            y_true: True labels. (batch_size)

        Returns:
            Loss.
        """
        return Array(torch.nn.functional.cross_entropy(y_pred.data, y_true.data), self)

    def backward(self, out: "Array") -> None:
        """Backward pass.

        Args:
            out: True labels.
        """
        # initialize gradients if they are None
        if out.prev[0].grad is None:
            out.prev[0].grad = torch.zeros_like(out.prev[0].data)
        # compute gradients according to s - y, where s is the softmax output and y is the true distribution
        y_true = one_hot(self._y_true.data)
        out.prev[0].grad += (torch.nn.functional.softmax(out.prev[0].data, dim=1) - y_true.data) * out.grad
        # back-propagate gradients
        if out.prev[0].op is not None:
            out.prev[0].op.backward(out.prev[0])


class Array:
    """Array class."""
    data: torch.Tensor
    grad: Optional[torch.Tensor]

    def __init__(self, data: torch.Tensor, operation: Optional[Operation] = None):
        """Initialize array.

        Args:
            data: Data.
            operation: Operation.
        """
        self.data = data
        self.grad = None
        self.prev: List["Array"] = []
        self.op: Optional[Operation] = operation

    def backward(self) -> None:
        """Backward pass."""
        if self.grad is None:
            self.grad = torch.ones_like(self.data)
        if self.op is not None:
            self.op.backward(self)

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.grad = None

    def __add__(self, other: "Array") -> "Array":
        """Addition."""
        return Add()(self, other)

    def __matmul__(self, other: "Array") -> "Array":
        """Matrix multiplication."""
        return MatMul()(self, other)


def relu(x: Array) -> Array:
    return ReLU()(x)


def cross_entropy_loss(y_pred: Array, y_true: Array) -> Array:
    return CrossEntropyLoss(y_true)(y_pred, y_true)
