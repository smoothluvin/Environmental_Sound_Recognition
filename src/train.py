import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from audio_processing import AudioDataSet
from cnn import SoundCNN
from config import TARGET_CLASSES_MUSIC, MAX_FRAMES

def calculate_class_weights(class_counts):
    """Refined class weight calculation"""
    total_samples = sum(class_counts.values())
    weights = []
    
    for class_name in TARGET_CLASSES_MUSIC:
        count = class_counts.get(class_name, 0)
        if count == 0:
            weight = 1.0
        else:
            # Base weight calculation
            weight = total_samples / (len(TARGET_CLASSES_MUSIC) * count)
            
            # Adjust specific classes
            if class_name == "Acoustic_Guitar":
                weight *= 0.6  # Reduce Acoustic Guitar weight
            elif class_name == "background_noise":
                weight *= 1.0  # Increase background noise weight
            elif class_name == "piano":
                weight *= 0.7
        
        weights.append(weight)
    
    # Normalize weights to keep loss scale reasonable
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    return weights_tensor

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        tuple of (loss, accuracy, all_predictions, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
    
    return avg_loss, accuracy, all_predictions, all_labels

def plot_confusion_matrix(predictions, labels, class_names, output_path):
    """
    Plot and save confusion matrix
    
    Args:
        predictions: List of predicted class indices
        labels: List of true class indices
        class_names: List of class names
        output_path: Path to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Convert to percentage (normalized by true labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm_percent, index=class_names, columns=class_names)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt=".1f", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also print classification report
    report = classification_report(labels, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report to file
    with open(output_path.replace('.png', '.txt'), 'w') as f:
        f.write(report)

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
    
    input_shape = (1, feature_dim, MAX_FRAMES)  # Channels, Features, Time frames
    print(f"Using features: Mel Spectrogram = {args.use_mel}, MFCC = {args.use_mfcc}")
    print(f"Input shape: {input_shape}")
    
    # Load training dataset and extracting features based on terminal argument
    train_path = os.path.join(args.data_dir, 'train')
    print(f"Loading training data from: {train_path}")
    train_dataset = AudioDataSet(
        root_dir=train_path, 
        use_mel=args.use_mel, 
        use_mfcc=args.use_mfcc,
        mode='train'  # Enable augmentation for training
    )
    
    # Print dataset information
    train_size = len(train_dataset)
    print(f"Training dataset loaded: {train_size} samples")
    print(f"Class distribution: {train_dataset.class_counts}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset.class_counts)
    print(f"Class weights: {class_weights}")
    
    # Create data loader for training
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    # Load validation dataset
    val_path = os.path.join(args.data_dir, 'val')
    print(f"Loading validation data from: {val_path}")
    val_dataset = AudioDataSet(
        root_dir=val_path, 
        use_mel=args.use_mel, 
        use_mfcc=args.use_mfcc,
        mode='val'  # No augmentation for validation
    )
    val_size = len(val_dataset)
    print(f"Validation dataset loaded: {val_size} samples")
    print(f"Class distribution: {val_dataset.class_counts}")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SoundCNN(input_shape=input_shape, num_classes=len(TARGET_CLASSES_MUSIC)).to(device)
    
    # Loss and optimizer
    if args.use_class_weights:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted loss function with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard unweighted loss function")
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
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
        val_loss, val_acc, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion, device
        )
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log learning rate
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']:.6f}")
        
        # Check if this is the best model
        is_best = False
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            is_best = True
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'input_shape': input_shape,
                'use_mel': args.use_mel,
                'use_mfcc': args.use_mfcc,
                'class_weights': class_weights.cpu() if args.use_class_weights else None
            }, f"models/best_model.pth")
            
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
            early_stop_counter = 0
            
            # Plot confusion matrix for best model
            plot_confusion_matrix(
                val_preds, val_labels, TARGET_CLASSES_MUSIC,
                f"results/confusion_matrix_epoch_{epoch+1}.png"
            )
            
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
        'use_mfcc': args.use_mfcc,
        'class_weights': class_weights.cpu() if args.use_class_weights else None
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
    
    # Final evaluation on validation set
    val_loss, val_acc, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device
    )
    
    # Plot final confusion matrix
    plot_confusion_matrix(
        val_preds, val_labels, TARGET_CLASSES_MUSIC,
        f"results/confusion_matrix_final.png"
    )
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model

# Main function with argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a sound classification model')
    parser.add_argument('--data_dir', type=str, default='./data/Music', 
                        help='Directory containing the dataset with train and val subdirectories')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights in loss function')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--use_mel', action='store_true', help='Use Mel spectrogram features')
    parser.add_argument('--use_mfcc', action='store_true', help='Use MFCC features')
    
    args = parser.parse_args()
    
    # Default to using Mel spectrograms if nothing is specified
    if not (args.use_mel or args.use_mfcc):
        args.use_mel = True
        print("No feature type specified, defaulting to using Mel spectrogram features")
    
    train_model(args)