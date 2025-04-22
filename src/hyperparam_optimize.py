import subprocess
import numpy as np
import pandas as pd
import os
import re
import time
import argparse
from datetime import datetime

class HyperparameterOptimizer:
    def __init__(self, base_cmd="python ./src/train.py", results_file="hyperparameter_results.csv"):
        self.base_cmd = base_cmd
        self.results_file = results_file
        
        # Initialize results DataFrame if it doesn't exist
        if not os.path.exists(results_file):
            self.results_df = pd.DataFrame(columns=[
                'trial', 'mel', 'mfcc', 'epochs', 'learning_rate', 'weight_decay', 
                'batch_size', 'patience', 'training_accuracy', 'validation_accuracy', 
                'early_stop_epoch', 'timestamp'
            ])
            self.results_df.to_csv(results_file, index=False)
        else:
            self.results_df = pd.read_csv(results_file)
        
        # Set initial parameter ranges
        self.param_ranges = {
            'learning_rate': [0.00015, 0.0002, 0.00025],
            'weight_decay': [1.75e-4, 2e-4, 2.25e-4],
            'epochs': [100],  # Fixed at 100 epochs
            'batch_size': [16, 32],
            'patience': [7, 10]  # Early stopping patience
        }
        
        # Features to use
        self.use_mel = True
        self.use_mfcc = True
    
    def generate_next_params(self, strategy='random'):
        """Generate next set of hyperparameters to try based on strategy."""
        if strategy == 'random':
            # Random selection from parameter ranges
            params = {
                'learning_rate': np.random.choice(self.param_ranges['learning_rate']),
                'weight_decay': np.random.choice(self.param_ranges['weight_decay']),
                'epochs': np.random.choice(self.param_ranges['epochs']),
                'batch_size': np.random.choice(self.param_ranges['batch_size']),
                'patience': np.random.choice(self.param_ranges['patience'])
            }
        elif strategy == 'grid':
            # Grid search - systematically try combinations
            params = self._next_grid_search_params()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return params
    
    def _next_grid_search_params(self):
        """Generate next grid search parameters based on previous results."""
        # Get all previously tried parameter combinations
        if len(self.results_df) == 0:
            # First run, just pick the first set of parameters
            return {
                'learning_rate': self.param_ranges['learning_rate'][0],
                'weight_decay': self.param_ranges['weight_decay'][0],
                'epochs': self.param_ranges['epochs'][0],
                'batch_size': self.param_ranges['batch_size'][0],
                'patience': self.param_ranges['patience'][0]
            }
        
        # Simple implementation: check which combinations haven't been tried
        all_combinations = []
        for lr in self.param_ranges['learning_rate']:
            for wd in self.param_ranges['weight_decay']:
                for ep in self.param_ranges['epochs']:
                    for bs in self.param_ranges['batch_size']:
                        for pat in self.param_ranges['patience']:
                            all_combinations.append((lr, wd, ep, bs, pat))
        
        # Find combinations that haven't been tried
        tried_combinations = []
        for _, row in self.results_df.iterrows():
            try:
                combo = (
                    row['learning_rate'], 
                    row['weight_decay'], 
                    row['epochs'], 
                    row['batch_size'],
                    row.get('patience', 7)  # Default to 7 if not in older results
                )
                tried_combinations.append(combo)
            except:
                pass  # Skip rows with missing data
        
        untried = [c for c in all_combinations if c not in tried_combinations]
        
        if not untried:
            print("All parameter combinations have been tried!")
            return self.generate_next_params(strategy='random')
        
        # Take the first untried combination
        next_combo = untried[0]
        return {
            'learning_rate': next_combo[0],
            'weight_decay': next_combo[1],
            'epochs': next_combo[2],
            'batch_size': next_combo[3],
            'patience': next_combo[4]
        }
    
    def execute_training(self, params):
        """Execute training with given parameters and return results."""
        # Construct command
        cmd = f"{self.base_cmd} --epochs {params['epochs']} --learning_rate {params['learning_rate']} --weight_decay {params['weight_decay']} --batch_size {params['batch_size']} --patience {params['patience']}"
        
        # Add feature flags
        if self.use_mel:
            cmd += " --use_mel"
        if self.use_mfcc:
            cmd += " --use_mfcc"
        
        print(f"Executing: {cmd}")
        
        # Run the command without capturing output to avoid freezing
        # This will just display the output in the console normally
        process = subprocess.run(cmd, shell=True, text=True)
        
        # Get results from the saved model file
        # We know the model is saved to models/best_model.pth
        # Let's try to extract the validation accuracy from it
        try:
            import torch
            checkpoint = torch.load("models/best_model.pth", map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'val_acc' in checkpoint:
                validation_acc = checkpoint['val_acc']
                early_stop_epoch = checkpoint.get('epoch', params['epochs'])
                print(f"Extracted from model: validation accuracy = {validation_acc}%, epoch = {early_stop_epoch}")
                return {
                    'training_accuracy': 0.0,  # We don't have this information
                    'validation_accuracy': validation_acc,
                    'early_stop_epoch': early_stop_epoch + 1  # +1 because epoch is 0-indexed
                }
        except Exception as e:
            print(f"Failed to extract results from saved model: {e}")
        
        # If we couldn't get results from the model file, try to parse train.log if it exists
        try:
            if os.path.exists("train.log"):
                with open("train.log", "r") as f:
                    content = f.read()
                    
                    # Look for best validation accuracy
                    match = re.search(r"Best validation accuracy: (\d+\.\d+)%", content)
                    if match:
                        validation_acc = float(match.group(1))
                        
                        # Look for early stopping epoch
                        early_stop_match = re.search(r"Early stopping after (\d+) epochs", content)
                        early_stop_epoch = int(early_stop_match.group(1)) if early_stop_match else params['epochs']
                        
                        print(f"Extracted from log: validation accuracy = {validation_acc}%, epoch = {early_stop_epoch}")
                        return {
                            'training_accuracy': 0.0,  # We don't have this information
                            'validation_accuracy': validation_acc,
                            'early_stop_epoch': early_stop_epoch
                        }
        except Exception as e:
            print(f"Failed to extract results from log file: {e}")
        
        # If all else fails, use default values
        print("Warning: Could not determine results automatically. Using defaults.")
        return {
            'training_accuracy': 0.0,
            'validation_accuracy': 0.0,
            'early_stop_epoch': params['epochs']
        }
    
    def save_results(self, params, results):
        """Save the results to CSV file."""
        # Create new row
        new_row = {
            'trial': len(self.results_df) + 1,
            'mel': int(self.use_mel),
            'mfcc': int(self.use_mfcc),
            'epochs': params['epochs'],
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'batch_size': params['batch_size'],
            'patience': params['patience'],
            'training_accuracy': results['training_accuracy'],
            'validation_accuracy': results['validation_accuracy'],
            'early_stop_epoch': results['early_stop_epoch'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Append to DataFrame and save
        new_df = pd.DataFrame([new_row])
        self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)
        self.results_df.to_csv(self.results_file, index=False)
        
        print(f"Results saved to {self.results_file}")
    
    def run_optimization(self, num_trials=10, strategy='grid'):
        """Run hyperparameter optimization for specified number of trials."""
        print(f"Starting hyperparameter optimization with {num_trials} trials using {strategy} strategy")
        
        for i in range(num_trials):
            print(f"\n=== Trial {i+1}/{num_trials} ===")
            
            # Generate next hyperparameters to try
            params = self.generate_next_params(strategy=strategy)
            print(f"Parameters: {params}")
            
            # Execute training
            results = self.execute_training(params)
            
            # Save results
            self.save_results(params, results)
            
            # Print current best
            if not self.results_df.empty and not self.results_df['validation_accuracy'].isna().all():
                # Find the row with the highest validation accuracy
                valid_df = self.results_df[self.results_df['validation_accuracy'] > 0]
                if not valid_df.empty:
                    best_row = valid_df.loc[valid_df['validation_accuracy'].idxmax()]
                    print(f"\nCurrent best: {best_row['validation_accuracy']}% validation accuracy with:")
                    print(f"  Learning rate: {best_row['learning_rate']}")
                    print(f"  Weight decay: {best_row['weight_decay']}")
                    print(f"  Batch size: {best_row['batch_size']}")
                    print(f"  Patience: {best_row['patience']}")
                else:
                    print("\nNo valid best result found yet.")
            else:
                print("\nNo valid results available yet.")
            
            # Optional: wait between trials
            if i < num_trials - 1:
                print("Waiting 10 seconds before next trial...")
                time.sleep(10)
        
        print("\nOptimization completed!")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of results."""
        if len(self.results_df) == 0 or self.results_df['validation_accuracy'].isna().all():
            print("No valid results available.")
            return
        
        print("\n=== Hyperparameter Optimization Summary ===")
        
        # Only consider rows with positive validation accuracy
        valid_df = self.results_df[self.results_df['validation_accuracy'] > 0]
        if valid_df.empty:
            print("No valid results with positive accuracy available.")
            return
            
        # Sort by validation accuracy
        sorted_df = valid_df.sort_values('validation_accuracy', ascending=False)
        
        # Print top 5 configurations (or fewer if we don't have 5)
        top_count = min(5, len(sorted_df))
        print(f"\nTop {top_count} configurations:")
        for i, (_, row) in enumerate(sorted_df.head(top_count).iterrows()):
            print(f"{i+1}. Validation Accuracy: {row['validation_accuracy']}%")
            print(f"   Learning Rate: {row['learning_rate']}")
            print(f"   Weight Decay: {row['weight_decay']}")
            print(f"   Batch Size: {row['batch_size']}")
            print(f"   Patience: {row['patience']}")
            print(f"   Early Stop Epoch: {row['early_stop_epoch']}")
            print()
        
        # Parameter importance analysis
        print("\nParameter importance analysis:")
        
        # Analyze learning rate
        lr_grouped = valid_df.groupby('learning_rate')['validation_accuracy'].mean().sort_values(ascending=False)
        print("\nLearning rate effectiveness (average validation accuracy):")
        for lr, acc in lr_grouped.items():
            print(f"  {lr}: {acc:.2f}%")
        
        # Analyze weight decay
        wd_grouped = valid_df.groupby('weight_decay')['validation_accuracy'].mean().sort_values(ascending=False)
        print("\nWeight decay effectiveness (average validation accuracy):")
        for wd, acc in wd_grouped.items():
            print(f"  {wd}: {acc:.2f}%")
        
        # Analyze batch size
        bs_grouped = valid_df.groupby('batch_size')['validation_accuracy'].mean().sort_values(ascending=False)
        print("\nBatch size effectiveness (average validation accuracy):")
        for bs, acc in bs_grouped.items():
            print(f"  {bs}: {acc:.2f}%")
        
        # Analyze patience
        pat_grouped = valid_df.groupby('patience')['validation_accuracy'].mean().sort_values(ascending=False)
        print("\nPatience effectiveness (average validation accuracy):")
        for pat, acc in pat_grouped.items():
            print(f"  {pat}: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for audio classification')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--strategy', type=str, default='grid', choices=['random', 'grid'], 
                        help='Optimization strategy')
    parser.add_argument('--results_file', type=str, default='hyperparameter_results.csv',
                        help='File to save results')
    
    args = parser.parse_args()
    
    optimizer = HyperparameterOptimizer(results_file=args.results_file)
    optimizer.run_optimization(num_trials=args.trials, strategy=args.strategy)