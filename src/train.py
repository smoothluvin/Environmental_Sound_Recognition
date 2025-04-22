import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import argparse
from audio_processing import AudioDataSet
from cnn import SoundCNN
from config import TARGET_CLASSES_MUSIC

def train_model(args):
    # Create directories for saving results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Determine input shape based on feature configuration
    feature_dim = 0
    if args.use_mel:
        feature_dim += 64  # N_MELS
    if args.use_mfcc:
        feature_dim += 20  # N_MFCC
    
    input_shape = (1, feature_dim, 800)  # Channels, Features, Time frames
    print(f"Using features: Mel Spectrogram = {args.use_mel}, MFCC = {args.use_mfcc}")
    print(f"Input shape: {input_shape}")
    
    # Load training dataset only
    train_path = os.path.join(args.data_dir, 'train')
    print(f"Loading training data from: {train_path}")
    train_dataset = AudioDataSet(root_dir=train_path, use_mel=args.use_mel, use_mfcc=args.use_mfcc)
    
    # Print dataset information
    train_size = len(train_dataset)
    print(f"Training dataset loaded: {train_size} samples")
    
    # Create data loader for training with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Use single process loading to avoid multiprocessing issues
        drop_last = True
    )
    
    # Load validation dataset
    val_path = os.path.join(args.data_dir, 'val')
    print(f"Loading validation data from: {val_path}")
    val_dataset = AudioDataSet(root_dir=val_path, use_mel=args.use_mel, use_mfcc=args.use_mfcc)
    val_size = len(val_dataset)
    print(f"Validation dataset loaded: {val_size} samples")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=0,
        drop_last = True
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SoundCNN(input_shape=input_shape, num_classes=len(TARGET_CLASSES_MUSIC)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'input_shape': input_shape,
                'use_mel': args.use_mel,
                'use_mfcc': args.use_mfcc
            }, f"models/best_model.pth")
            
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Check for early stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save the final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'input_shape': input_shape,
        'use_mel': args.use_mel,
        'use_mfcc': args.use_mfcc
    }, f"models/final_model.pth")
    
    # Plot training and validation curves
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model

# Main function with argument parser so that when the script is called, you can update the different parameters in command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a sound classification model')
    parser.add_argument('--data_dir', type=str, default='./data/Music', 
                        help='Directory containing the dataset with train and val subdirectories')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--use_mel', action='store_true', help='Use Mel spectrogram features')
    parser.add_argument('--use_mfcc', action='store_true', help='Use MFCC features')
    
    args = parser.parse_args()
    
    # Default to using Mel spectrograms if nothing is specified
    if not (args.use_mel or args.use_mfcc):
        args.use_mel = True
        print("No feature type specified, defaulting to using Mel spectrogram features")
    
    train_model(args)