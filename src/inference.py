import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from audio_processing import load_audio, extract_mel_spectrogram, extract_mfcc, AudioDataSet
from cnn import SoundCNN
from config import TARGET_CLASSES_MUSIC
import argparse
import pandas as pd
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained audio classification model')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='./data/Music/val',
                        help='Directory containing validation audio files')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save test results')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Path to a single audio file for prediction')
    parser.add_argument('--use_mel', action='store_true', default=True,
                        help='Use Mel spectrogram features')
    parser.add_argument('--use_mfcc', action='store_true', default=False,
                        help='Use MFCC features')
    return parser.parse_args()

def load_model(model_path, device, use_mel=True, use_mfcc=False):
    """Load the trained model."""
    # Determine input shape based on feature configuration
    feature_dim = 0
    if use_mel:
        feature_dim += 64  # N_MELS
    if use_mfcc:
        feature_dim += 20  # N_MFCC
    
    input_shape = (1, feature_dim, 336)  # Channels, Features, Time frames
    print(f"Using features: Mel Spectrogram = {use_mel}, MFCC = {use_mfcc}")
    print(f"Input shape: {input_shape}")
    
    # Create model instance
    model = SoundCNN(input_shape=input_shape, num_classes=len(TARGET_CLASSES_MUSIC)).to(device)
    
    # Load the saved model state - handle both full checkpoints and state_dict only saves
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('val_acc', 'unknown')
        
        # Check if the checkpoint has feature configuration
        saved_use_mel = checkpoint.get('use_mel', use_mel)
        saved_use_mfcc = checkpoint.get('use_mfcc', use_mfcc)
        saved_input_shape = checkpoint.get('input_shape', input_shape)
        
        if saved_input_shape != input_shape:
            print(f"Warning: Current input shape {input_shape} doesn't match saved model input shape {saved_input_shape}")
            print(f"Using saved model configuration: Mel={saved_use_mel}, MFCC={saved_use_mfcc}")
            
            # Recreate the model with the correct input shape
            feature_dim = saved_input_shape[1]
            model = SoundCNN(input_shape=saved_input_shape, num_classes=len(TARGET_CLASSES_MUSIC)).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Update the feature flags to match the saved model
            return model, saved_use_mel, saved_use_mfcc
        
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy {val_acc}%")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.eval()
    return model, use_mel, use_mfcc

def predict_single_file(model, audio_path, device, use_mel=True, use_mfcc=False):
    """Make prediction for a single audio file."""
    waveform = load_audio(audio_path)
    
    features = []
    if use_mel:
        mel_spectrogram = extract_mel_spectrogram(waveform)
        features.append(mel_spectrogram)
    
    if use_mfcc:
        mfccs = extract_mfcc(waveform)
        features.append(mfccs)
    
    # Combine features if using both
    if len(features) > 1:
        combined_features = torch.cat(features, dim=1)
    else:
        combined_features = features[0]
    
    combined_features = combined_features.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(combined_features)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
    
    predicted_class_name = TARGET_CLASSES_MUSIC[predicted_class_idx]
    confidence_scores = {class_name: score.item() for class_name, score in zip(TARGET_CLASSES_MUSIC, probabilities[0])}
    
    return predicted_class_name, confidence_scores

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset."""
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy, all_preds, all_labels, all_probs

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_accuracy(y_true, y_pred, class_names, output_dir):
    """Plot and save per-class accuracy."""
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_indices = [j for j, label in enumerate(y_true) if label == i]
        if len(class_indices) > 0:
            correct = sum(1 for j in class_indices if y_pred[j] == i)
            class_accuracy[class_name] = 100 * correct / len(class_indices)
        else:
            class_accuracy[class_name] = 0
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_accuracy.keys()), y=list(class_accuracy.values()))
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'))
    plt.close()
    
    return class_accuracy

def save_metrics(accuracy, class_report, class_accuracy, all_probs, output_dir):
    """Save all metrics to files."""
    metrics = {
        'overall_accuracy': accuracy,
        'per_class_accuracy': class_accuracy
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics['overall_accuracy']], columns=['Overall Accuracy'])
    for class_name, acc in metrics['per_class_accuracy'].items():
        metrics_df[f"{class_name} Accuracy"] = acc
    
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, 'accuracy_metrics.csv'), index=False)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
    
    # Save mean confidence scores per class
    mean_probs = np.mean(all_probs, axis=0)
    prob_df = pd.DataFrame({
        'Class': TARGET_CLASSES_MUSIC,
        'Mean Confidence': mean_probs
    })
    prob_df.to_csv(os.path.join(output_dir, 'confidence_scores.csv'), index=False)

def plot_confidence_distribution(all_probs, all_preds, all_labels, class_names, output_dir):
    """Plot confidence score distributions for correct and incorrect predictions."""
    correct_probs = []
    incorrect_probs = []
    
    for i in range(len(all_preds)):
        pred = all_preds[i]
        label = all_labels[i]
        prob = all_probs[i][pred]  # Confidence score for the predicted class
        
        if pred == label:
            correct_probs.append(prob)
        else:
            incorrect_probs.append(prob)
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_probs, bins=20, alpha=0.5, label='Correct Predictions')
    plt.hist(incorrect_probs, bins=20, alpha=0.5, label='Incorrect Predictions')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Score Distribution')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, use_mel, use_mfcc = load_model(args.model_path, device, args.use_mel, args.use_mfcc)
    print(f"Model loaded from {args.model_path}")
    
    # Single file prediction
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} does not exist.")
            return
        
        start_time = time.time()
        predicted_class, confidence_scores = predict_single_file(model, args.single_file, device, use_mel, use_mfcc)
        inference_time = time.time() - start_time
        
        print(f"\nPredicted Sound Class: {predicted_class}")
        print(f"Inference Time: {inference_time*1000:.2f} ms")
        print("Confidence Scores:")
        for class_name, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {class_name}: {score:.4f}")
        
        return
    
    # Load validation dataset with same features as the model
    test_dataset = AudioDataSet(root_dir=args.data_dir, use_mel=use_mel, use_mfcc=use_mfcc)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Testing on {len(test_dataset)} validation samples")
    
    # Check if the validation dataset has samples
    if len(test_dataset) == 0:
        print("Warning: Validation dataset is empty. Please check the path.")
        return
    
    # Evaluate model
    accuracy, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device)
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES_MUSIC)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, TARGET_CLASSES_MUSIC, args.output_dir)
    
    # Plot and calculate per-class accuracy
    class_accuracy = plot_class_accuracy(all_labels, all_preds, TARGET_CLASSES_MUSIC, args.output_dir)
    
    # Plot confidence distribution
    plot_confidence_distribution(all_probs, all_preds, all_labels, TARGET_CLASSES_MUSIC, args.output_dir)
    
    # Save metrics
    save_metrics(accuracy, class_report, class_accuracy, all_probs, args.output_dir)
    
    # Print summary
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(class_report)
    print(f"\nResults saved to {args.output_dir}")

    # Check if any class has no samples in validation set
    class_counts = {TARGET_CLASSES_MUSIC[i]: 0 for i in range(len(TARGET_CLASSES_MUSIC))}
    for label in all_labels:
        class_counts[TARGET_CLASSES_MUSIC[label]] += 1
    
    empty_classes = [class_name for class_name, count in class_counts.items() if count == 0]
    if empty_classes:
        print("\nWarning: The following classes have no samples in the validation set:")
        for class_name in empty_classes:
            print(f"   - {class_name}")

if __name__ == "__main__":
    main()