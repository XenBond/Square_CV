import torch.nn as nn
import torch.nn.functional as F

class SquareModel(nn.Module):
    '''
    2 layer CNN model with 128x128 as the input size.
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16 ,5) # output size: 16 * 124 * 124
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2) # output size: 16 * 62 * 62
        self.conv2 = nn.Conv2d(16, 32, 5) # output size: 32 * 58 * 58
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2) # output size: 32 * 29 * 29
        self.fc1 = nn.Linear(32 * 29 * 29, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 29 * 29)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc3(x)
        return x

