import torch
import torch.nn as nn


class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 1 × 28 × 28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        )

        # 8 × 26 × 26
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 8 × 13 × 13
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )

        # 16 × 11 × 11
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 16 × 5 × 5 = 400
        self.fc1 = nn.Linear(
            16 * 5 * 5,
            64
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            64,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x