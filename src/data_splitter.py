import os
import shutil
import random
from collections import defaultdict
import argparse
from datetime import datetime

def count_and_redistribute_samples(base_dir, train_ratio=0.8, seed=42, update_readme=True):
    """
    Count samples in each class in train and val folders,
    then redistribute them with the specified train/val split.
    Also updates the README.md file with new statistics.
    
    Args:
        base_dir (str): Base directory containing train and val folders
        train_ratio (float): Ratio of samples to use for training (0.0-1.0)
        seed (int): Random seed for reproducibility
        update_readme (bool): Whether to update the README.md file
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define paths
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    readme_path = os.path.join(base_dir, 'README.md')
    
    # Get all class names
    all_classes = set()
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            all_classes.update([name for name in os.listdir(d) 
                               if os.path.isdir(os.path.join(d, name))])
    
    # Sort class names alphabetically for consistent output
    all_classes = sorted(list(all_classes))
    
    # Initialize counters
    counts = defaultdict(lambda: {'train': 0, 'val': 0, 'total': 0})
    file_paths = defaultdict(list)
    
    # Count existing files and collect file paths
    for class_name in all_classes:
        # Check training directory
        train_class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(train_class_dir):
            train_files = [f for f in os.listdir(train_class_dir) 
                           if os.path.isfile(os.path.join(train_class_dir, f))]
            counts[class_name]['train'] = len(train_files)
            file_paths[class_name].extend([os.path.join(train_class_dir, f) for f in train_files])
        
        # Check validation directory
        val_class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(val_class_dir):
            val_files = [f for f in os.listdir(val_class_dir) 
                         if os.path.isfile(os.path.join(val_class_dir, f))]
            counts[class_name]['val'] = len(val_files)
            file_paths[class_name].extend([os.path.join(val_class_dir, f) for f in val_files])
        
        counts[class_name]['total'] = counts[class_name]['train'] + counts[class_name]['val']
    
    # Print current statistics
    print("Current dataset statistics:")
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8}")
    print("-" * 45)
    for class_name in all_classes:
        count = counts[class_name]
        print(f"{class_name:<20} {count['train']:<8} {count['val']:<8} {count['total']:<8}")
    
    # Create temporary directory for redistribution
    temp_dir = os.path.join(base_dir, 'temp_redistribution')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Redistribute files
    new_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'total': 0})
    
    for class_name in all_classes:
        files = file_paths[class_name]
        
        # Shuffle files
        random.shuffle(files)
        
        # Calculate split
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        
        # Create new directories if they don't exist
        new_train_dir = os.path.join(train_dir, class_name)
        new_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(new_train_dir, exist_ok=True)
        os.makedirs(new_val_dir, exist_ok=True)
        
        # Move files to temporary location first
        temp_class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(temp_class_dir, exist_ok=True)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(temp_class_dir, filename))
        
        # Get all files from temp directory
        temp_files = [f for f in os.listdir(temp_class_dir) 
                     if os.path.isfile(os.path.join(temp_class_dir, f))]
        random.shuffle(temp_files)
        
        # Split into train and val
        train_files = temp_files[:train_count]
        val_files = temp_files[train_count:]
        
        # Clear existing directories
        for dir_path in [new_train_dir, new_val_dir]:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Move to train directory
        for file_name in train_files:
            src = os.path.join(temp_class_dir, file_name)
            dst = os.path.join(new_train_dir, file_name)
            shutil.copy(src, dst)
            new_counts[class_name]['train'] += 1
        
        # Move to val directory
        for file_name in val_files:
            src = os.path.join(temp_class_dir, file_name)
            dst = os.path.join(new_val_dir, file_name)
            shutil.copy(src, dst)
            new_counts[class_name]['val'] += 1
        
        new_counts[class_name]['total'] = new_counts[class_name]['train'] + new_counts[class_name]['val']
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)
    
    # Print new statistics
    print("\nNew dataset statistics:")
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Total':<8} {'Train %':<8}")
    print("-" * 55)
    for class_name in all_classes:
        count = new_counts[class_name]
        train_percent = (count['train'] / count['total']) * 100 if count['total'] > 0 else 0
        print(f"{class_name:<20} {count['train']:<8} {count['val']:<8} {count['total']:<8} {train_percent:.1f}%")
    
    # Update README.md file
    if update_readme:
        update_readme_file(readme_path, all_classes, new_counts, train_ratio)

def update_readme_file(readme_path, classes, counts, train_ratio):
    """
    Updates the README.md file with new dataset statistics.
    
    Args:
        readme_path (str): Path to the README.md file
        classes (list): List of class names
        counts (dict): Dictionary containing counts for each class
        train_ratio (float): Ratio used for train/val split
    """
    # Create README content if it doesn't exist
    if not os.path.exists(readme_path):
        readme_content = "# AudioSet Dataset\n\n"
    else:
        # Read existing README content
        with open(readme_path, 'r') as f:
            readme_content = f.read()
    
    # Create the dataset statistics section
    current_date = datetime.now().strftime("%Y-%m-%d")
    stats_header = f"## Dataset Statistics (Updated: {current_date})\n\n"
    stats_content = f"Train/Validation Split Ratio: {train_ratio:.2f}/{1-train_ratio:.2f}\n\n"
    stats_content += f"| Class | Train | Validation | Total | Train % |\n"
    stats_content += f"|-------|-------|------------|-------|--------|\n"
    
    # Add statistics for each class
    total_train = 0
    total_val = 0
    total_samples = 0
    
    for class_name in classes:
        count = counts[class_name]
        train_percent = (count['train'] / count['total']) * 100 if count['total'] > 0 else 0
        stats_content += f"| {class_name} | {count['train']} | {count['val']} | {count['total']} | {train_percent:.1f}% |\n"
        
        total_train += count['train']
        total_val += count['val']
        total_samples += count['total']
    
    # Add totals row
    total_train_percent = (total_train / total_samples) * 100 if total_samples > 0 else 0
    stats_content += f"| **Total** | **{total_train}** | **{total_val}** | **{total_samples}** | **{total_train_percent:.1f}%** |\n"
    
    # Find and replace existing statistics section or append new one
    stats_section_marker = "## Dataset Statistics"
    if stats_section_marker in readme_content:
        # Split by stats section header
        parts = readme_content.split(stats_section_marker)
        
        # Find the end of the stats section (next ## header or end of file)
        next_section_start = parts[1].find("##")
        if next_section_start != -1:
            # Keep content after the next section
            remainder = parts[1][next_section_start:]
            # Replace stats section
            readme_content = parts[0] + stats_header + stats_content + remainder
        else:
            # No next section, replace till end
            readme_content = parts[0] + stats_header + stats_content
    else:
        # Append to the end
        readme_content += "\n" + stats_header + stats_content
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME.md updated at {readme_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redistribute dataset with proper train/val split')
    parser.add_argument('--data_dir', type=str, default='./data/Music', 
                        help='Base directory containing train and val folders')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of samples to use for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no_readme_update', action='store_true',
                        help='Skip updating the README.md file')
    
    args = parser.parse_args()
    
    count_and_redistribute_samples(
        args.data_dir, 
        args.train_ratio, 
        args.seed, 
        update_readme=not args.no_readme_update
    )
    print("\nRedistribution complete!")