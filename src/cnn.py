import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from config import TARGET_CLASSES_MUSIC

class SoundCNN(nn.Module):
    def __init__(self, input_shape=(1, 84, 336), num_classes=None):
        super(SoundCNN, self).__init__()
        
        if num_classes is None:
            num_classes = len(TARGET_CLASSES_MUSIC)

        self.input_shape = input_shape
        
        # Input shape: (channels, features, time)
        # For Mel: (1, 64, 336)
        # For MFCC: (1, 20, 336)
        # If using both Mel and MFCC: (1, 84, 336)

        # First, get the input dimensions
        in_channels = input_shape[0]  # Typically 1 for audio
        in_features = input_shape[1]  # 64 for Mel, 20 for MFCC, or 84 for both
        in_time = input_shape[2]      
        
        print(f"Building model with input shape: {input_shape}")
        
        # Determine if we're using small feature dimension (like MFCC)
        small_features = in_features <= 32
        
        # CNN layers with adjusted pooling for smaller feature dimensions The Kernel size and padding values were chosen to preserve the aspect ratio of the input
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 5), stride=1, padding=(1, 2))
        # Output after the first conv layer if using both mel and mfcc = (32, 84, 336)
        self.bn1 = nn.BatchNorm2d(32)
        # Use smaller frequency pooling for MFCCs
        self.pool1 = nn.MaxPool2d((1 if small_features else 2), 2)
        # Output after first pooling layer if using both mel and mfcc = (32, 42, 168)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=1, padding=(1, 2)) # Shape (64, 42, 168)
        self.bn2 = nn.BatchNorm2d(64)
        # Use smaller frequency pooling for MFCCs
        self.pool2 = nn.MaxPool2d((2 if small_features else 3), 3)
        # Output after second pooling layer if using both mel and mfcc= (64, 14, 56)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=1, padding=(1, 2)) # (128, 14, 56)
        self.bn3 = nn.BatchNorm2d(128)
        # Use adaptive pooling for the final layer to ensure consistent output size
        self.pool3 = nn.AdaptiveMaxPool2d((3, 12))  # Force output to be 3x12 regardless of input for preseving 1:4 aspect ratio (84:336)
        # Output after last pooling layer if using both mel and mfcc = (128, 3, 12)
        
        # Calculate the flattened size based on the adaptive pooling output
        self.flatten_size = 128 * 3 * 12  # 128 channels x 3 x 12 = 4608
        
        print(f"Flattened size will be: {self.flatten_size}")
        
        # FC layers (Dense Layers)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.dropout2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, num_classes)

        # Applying He Initialization 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Print shape for debugging (only during first forward pass)
        if self.training and x.size(0) > 1 and hasattr(self, 'first_pass'):
            print(f"After pool1: {x.shape}")
        
        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        if self.training and x.size(0) > 1 and hasattr(self, 'first_pass'):
            print(f"After pool2: {x.shape}")
        
        # Third conv block with adaptive pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Adaptive pooling ensures consistent output size
        
        if self.training and x.size(0) > 1 and hasattr(self, 'first_pass'):
            print(f"After pool3: {x.shape}")
            delattr(self, 'first_pass')  # Only print once
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # FC layers
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc1(x)))
        
        x = self.dropout2(x)
        x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)

        return x
        
    def set_debug(self):
        """Enable shape debugging for first forward pass"""
        self.first_pass = True

if __name__ == "__main__":
    model = SoundCNN()
    batch_size = 16
    summary(model, input_size=(batch_size, 1, 84, 336))