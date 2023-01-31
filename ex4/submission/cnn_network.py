import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    # def __init__(self):
    #     """Initialize layers."""
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(8, 8)
    #     self.fc1 = nn.Linear(6 * 5 * 5, 6)

    # def forward(self, x):
    #     """Forward pass of network."""
    #     x = self.pool(self.conv1(x))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = self.fc1(x)
    #     return x

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=250, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.LazyLinear(out_features=1024)
        self.fc2 = nn.LazyLinear(out_features=6)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv4(x)), 2))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
