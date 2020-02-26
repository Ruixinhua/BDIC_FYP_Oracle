import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class OracleModel(BaseModel):
    def __init__(self, num_classes=5645):
        super().__init__()
        # 1 input image channel with size of 96*96, 10 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 22 * 22, 120)  # 22*22 from image dimension ((96-2)/2-2)/2=22
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
