from src.array.auto import relu, cross_entropy_loss
from src.array.mod import Linear, Sequential
from src.array.opt import Adam
from src.array.train import training, load_data

import torchvision
import torchvision.transforms as transforms


def mnist_train():
    """Train MNIST."""
    # define model
    mlp = Sequential([
        Linear(784, 2048),
        relu,
        Linear(2048, 1024),
        relu,
        Linear(1024, 512),
        relu,
        Linear(512, 64),
        relu,
        Linear(64, 10),
    ])
    # define optimizer
    lr = 0.001
    optimizer = Adam(model=mlp, lr=lr)

    # define data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # download data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # create data loaders
    train_loader, test_loader = load_data(train_dataset, test_dataset, batch_size=64)
    num_features = 784

    # define loss function
    loss_fn = cross_entropy_loss

    # training loop
    training(
        model=mlp,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_features=num_features,
        loss_fn=loss_fn,
        num_epochs=100,
    )
