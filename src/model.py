import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TARGET_CLASSES

class SoundCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SoundCNN, self).__init__()
        num_classes = len(TARGET_CLASSES)

        # After doing some audio analysis and some math, I found that the shape of the the features coming in are as follows:
        # Spectrograms shape (1, 64, 800)
        # - 1 is the number of channels (mono audio)
        # - 64 is the number of Mel bands (frequency bins) in the spectrogram
        # - 800 is the number of time frames (time steps) in the spectrogram

        # In Conv2d, the parameters we input are as follows (in_channels, out_channels, kernel_size, stride, padding)
        # - In in_channels, we input the number of channels in the input data (1 for mono audio)
        # - Out_channels is the number of filters we want to apply (16, 32, 64 in our case) This is similar to how many neurons we defined in MNIST digit recognizer
        # - Kernel_size is the size of the filter (3, 5) in our case. This means we are applying a 3x5 filter to the input data (Like how our MNIST weights connected to localized regions)
        # - Stride is the step size of the filter as it moves across the input data (1 in our case, meaning we are moving one step at a time)
        # - Padding is the amount of zero-padding we add to the input data (1, 2 in our case). This is used to control the spatial dimensions of the output data
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)                                                         # Adding batch normalization to the first layer
        self.pool1 = nn.MaxPool2d(2, 4)                                                       # Pooling layer to reduce the spatial dimensions of the output data

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)                                                         # Adding batch normalization to the second layer
        self.pool2 = nn.MaxPool2d(2, 4)                                                       # Pooling layer to reduce the spatial dimensions of the output data

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(64)                                                         # Adding batch normalization to the third layer
        self.pool3 = nn.MaxPool2d(2, 4)                                                       # Pooling layer to reduce the spatial dimensions of the output data

        # Calculating flattened size based on known dimensions from the Mel Spectrogram analysis
        # Input shape: (1, 64, 800)
        self.flatten_size = 64 * 8 * 12 # 64 features x (64/8) frequency x (800/64) time = 64 x 8 x 12 = 6144

        # Calculate flattened size: 64 features x (64/8) frequency x (800/64) time = 64 x 8 x 12 = 6144
        self.dropout = nn.Dropout(0.5) # Adding dropout to reduce overfitting
        self.fc1 = nn.Linear(self.flatten_size, 256) # Fully connected layer to reduce the number of features to 256
        self.fc2 = nn.Linear(256, num_classes) # Fully connected layer to reduce the number of features to the number of classes


        # Applying He Initialization learned from creating MNIST digit recognizer in ENGR 492
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) # This initializes 0 for the biases

    # Forward propogation function for the model
    def forward(self, x):
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        ## Third conv block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1) # Flatten the tensor

        x = F.relu(self.fc1(x)) # Fully connected layer
        x = self.dropout(x)
        x = self.fc2(x)

        return x