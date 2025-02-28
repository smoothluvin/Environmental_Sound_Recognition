import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TARGET_CLASSES

class SoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(SoundCNN, self).__init__()

        num_classes = len(TARGET_CLASSES)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1) # Flatten the tensor

        if self.flatten_size is None:
            self.flatten_size = x.shape[1]
            self.fc1 = nn.Linear(self.flatten_size, 256).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x