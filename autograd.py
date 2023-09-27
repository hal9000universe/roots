import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from typing import Optional, List, Callable, Iterable


def reshape(x: torch.Tensor) -> "Array":
    """Reshape the input tensor to Array of shape (-1, 784).

    Args:
        x: Input tensor.

    Returns:
        Reshaped Array.
    """
    return Array(x.view(-1, 784))


def one_hot(x: torch.Tensor) -> "Array":
    """One-hot encode the input tensor

    Args:
        x: Input tensor.

    Returns:
        One-hot encoded Array.
    """
    return Array(torch.nn.functional.one_hot(x, num_classes=10))


def array_transform(x: torch.Tensor, y: torch.Tensor) -> ("Array", "Array"):
    """Transform the input tensor to (-1, 784)

    Args:
        x: Input tensor.
        y: Label tensor.

    Returns:
        Reshaped tensors.
    """
    return reshape(x), Array(y)


def load_data(batch_size: int) -> (DataLoader, DataLoader):
    """Load MNIST data.

    Args:
        batch_size: Batch size.

    Returns:
        Train and test data loaders.
    """
    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # download data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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


class Linear:
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


class Sequential:
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


def zero_grad(model: Sequential) -> None:
    """Zero gradients."""
    for param in model.parameters():
        param.zero_grad()


def train():
    # define model
    mlp = Sequential([
        Linear(784, 512),
        relu,
        Linear(512, 256),
        relu,
        Linear(256, 128),
        relu,
        Linear(128, 10),
    ])
    # define optimizer
    lr = 0.001
    # load data
    train_loader, test_loader = load_data(64)

    # define loss function
    loss_fn = cross_entropy_loss

    # train
    num_steps = 0
    for epoch in range(10):
        for x, y in train_loader:
            x, y = array_transform(x, y)
            y_pred = mlp(x)
            loss = loss_fn(y_pred, y)
            zero_grad(mlp)
            loss.backward()
            for param in mlp.parameters():
                param.data -= lr * param.grad

            num_steps += x.data.shape[0]
            if num_steps % 1000 == 0:
                print(f'epoch: {epoch}, num_steps: {num_steps}, loss: {loss.data.item()}')

    # test
    num_correct = 0
    num_examples = 0
    for x, y in test_loader:
        x, y = array_transform(x, y)
        y_pred = mlp(x)
        num_correct += (y_pred.data.argmax(dim=1) == y.data).sum()
        num_examples += x.data.shape[0]
    print(f'accuracy: {num_correct / num_examples}')
