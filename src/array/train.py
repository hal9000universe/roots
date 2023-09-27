from typing import Tuple, Callable

import torch
from torch.utils.data import DataLoader

from src.array.auto import Array
from src.array.mod import Module
from src.array.opt import Optimizer


def reshape(x: torch.Tensor, num_features: int) -> torch.Tensor:
    """Reshape the input tensor to Array of shape (-1, num_features).

    Args:
        x: Input tensor.
        num_features: Number of features.

    Returns:
        Reshaped Array.
    """
    return x.view(-1, num_features)


def array_transform(x: torch.Tensor, num_features: int, y: torch.Tensor) -> Tuple["Array", "Array"]:
    """Transform the input tensor to (-1, num_features) shape.

    Args:
        x: Input tensor.
        num_features: Number of features.
        y: Label tensor.

    Returns:
        Reshaped tensors.
    """
    return Array(reshape(x, num_features)), Array(y)


def load_data(train_dataset: torch.utils.data.Dataset,
              test_dataset: torch.utils.data.Dataset,
              batch_size: int) -> (DataLoader, DataLoader):
    """Load MNIST data.

    Args:
        train_dataset: Train dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.

    Returns:
        Train and test data loaders.
    """

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def training(
        model: Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_features: int,
        loss_fn: Callable[[Array, Array], Array],
        num_epochs: int,
):
    """Training loop.

    Args:
        model: Model.
        optimizer: Optimizer.
        train_loader: Train data loader.
        test_loader: Test data loader.
        num_features: Number of features.
        loss_fn: Loss function.
        num_epochs: Number of epochs.
    """
    # training loop
    num_steps = 0
    for epoch in range(0, num_epochs):
        for x, y in train_loader:
            # prepare data
            x, y = array_transform(x, num_features, y)
            # forward pass
            y_pred = model(x)
            # compute loss
            loss = loss_fn(y_pred, y)
            # zero gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()

            if num_steps % 1000 == 0:
                print(f'epoch: {epoch}, num_steps: {num_steps}, loss: {loss.data.item()}')
            num_steps += x.data.shape[0]

    # test
    num_correct = 0
    num_examples = 0
    for x, y in test_loader:
        x, y = array_transform(x, num_features, y)
        y_pred = model(x)
        num_correct += (y_pred.data.argmax(dim=1) == y.data).sum()
        num_examples += x.data.shape[0]
    print(f'accuracy: {num_correct / num_examples}')
