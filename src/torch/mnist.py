import torch
import torchvision.datasets.mnist as mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt


def show_image(x: torch.Tensor):
    """Show the input image"""
    plt.imshow(x.view(28, 28).numpy(), cmap='gray')
    plt.show()


def reshape(x: torch.Tensor) -> torch.Tensor:
    """Reshape the input tensor to (-1, 784)"""
    return x.view(-1)


# load data
def load_data(batch_size):
    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        reshape,
    ])

    # download data
    train_dataset = mnist.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = mnist.MNIST(root='./data', train=False, transform=transform, download=True)

    # load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# define model
class MLP(nn.Module):
    _linear1: nn.Linear
    _linear2: nn.Linear
    _linear3: nn.Linear
    _linear4: nn.Linear

    def __init__(self):
        super().__init__()
        self._linear1 = nn.Linear(784, 512)
        self._linear2 = nn.Linear(512, 256)
        self._linear3 = nn.Linear(256, 128)
        self._linear4 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1(x)
        x = torch.relu(x)
        x = self._linear2(x)
        x = torch.relu(x)
        x = self._linear3(x)
        x = torch.relu(x)
        x = self._linear4(x)
        return x


def train():
    # load data
    train_loader, test_loader = load_data(64)
    num_examples: int = 0

    # define model
    mlp = MLP()

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # train
    for epoch in range(10):
        for x, y in train_loader:
            y_pred = mlp(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_examples += x.shape[0]
            if num_examples % 1000 == 0:
                print(f'epoch: {epoch}, num_examples: {num_examples}, loss: {loss.item()}')

    # test
    mlp.eval()
    num_correct = 0
    num_examples = 0
    for x, y in test_loader:
        y_pred = mlp(x)
        num_correct += (y_pred.argmax(dim=1) == y).sum()
        num_examples += x.shape[0]

    print(f'accuracy: {num_correct / num_examples}')
