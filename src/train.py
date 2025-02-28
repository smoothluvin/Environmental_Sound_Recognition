import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from audio_processing import AudioDataSet
from model import SoundCNN
from config import TARGET_CLASSES

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = len(TARGET_CLASSES) 

# Load dataset
train_dataset = AudioDataSet(root_dir="./data/Filtered_Dataset")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoundCNN(num_classes=NUM_CLASSES).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop where epochs refer to the number of iterations we go through the dataset
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# For saving the model
torch.save(model.state_dict(), "sound_cnn.pth")
print("Model saved to sound_cnn.pth")
print("Training complete")