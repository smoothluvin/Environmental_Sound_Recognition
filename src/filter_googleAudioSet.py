import os
import pandas as pd
import subprocess
import shutil
import time
import random
import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

# Base directories
base_dir = './data/AudioSet'
metadata_dir = os.path.join(base_dir, 'metadata')
temp_dir = os.path.join(base_dir, 'temp')
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Define target classes - updated with your full list
ALL_CLASSES = [
    "Screaming",
    "Slam",
    "Smoke_detector_smoke_alarm",
    "crying_baby",
    "glass_breaking",
    "gun_shot",
    "siren"
]

# Target sample numbers - increased to 1000
TARGET_SAMPLES = 400
TRAIN_RATIO = 0.8  # 80% train, 20% validation

# Browser to use for cookies (must be installed on your system)
BROWSER = "firefox"  # Options: firefox, edge, opera (not chrome due to issues)

# Log file for tracking download attempts
DOWNLOAD_LOG = os.path.join(log_dir, 'download_attempts.json')

# Initialize or load download log
def get_download_log():
    if os.path.exists(DOWNLOAD_LOG):
        try:
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading download log: {e}. Creating new log.")
    
    # Initialize new log if not exists
    return {
        "last_run": None,
        "attempts": {},
        "successful": [],
        "failed": {}
    }

# Save download log
def save_download_log(log_data):
    with open(DOWNLOAD_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)

# Function to count existing files
def count_existing_files(target_classes=None):
    if target_classes is None:
        target_classes = ALL_CLASSES
        
    file_counts = {cls: {"train": 0, "val": 0} for cls in target_classes}
    
    for split in ['train', 'val']:
        for target_class in target_classes:
            safe_class_name = target_class.replace(',', '')
            class_dir = os.path.join(base_dir, split, safe_class_name)
            
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                file_counts[target_class][split] = len(files)
    
    return file_counts

# Function to extract YouTube IDs for specified classes from AudioSet CSVs
def extract_class_ids(target_classes):
    print(f"Extracting YouTube IDs for {', '.join(target_classes)} from AudioSet CSVs...")
    
    # Known class IDs for our target classes (from AudioSet ontology)
    class_ids = {
        "Screaming": ["/m/03qc9zr"], # Confirmed in Ontology | Screaming in fear
        "Slam": ["/m/07rjzl8"], # Confirmed in Ontology | Slam (door slam)
        "Smoke_detector_smoke_alarm": ["/m/01y3hg"], # Confirmed in Ontology | Smoke detector alarm
        "Thump_thud": ["/m/07qnq_y"], # Confirmed in Ontology | Thud (thump)
        "Yell": ["/m/07sr1lc"], # Confirmed in Ontology | Yelling in loud voice
        "crying_baby": ["/t/dd00002"], # Confirmed in Ontology | Crying baby
        "glass_breaking": ["/m/07rn7sz"], # Confirmed in Ontology | Glass breaking
        "gun_shot": ["/m/032s66"], # Confirmed in Ontology | Gunshot
        "siren": ["/m/03j1ly"] # Confirmed in Ontology | Siren (emergency vehicle)
    }
    
    # Original CSV files
    csv_files = [
        os.path.join(metadata_dir, "balanced_train_segments.csv"),
        os.path.join(metadata_dir, "eval_segments.csv"),
        os.path.join(metadata_dir, "unbalanced_train_segments.csv")
    ]
    
    # Extract YouTube IDs for each class
    class_entries = {cls: [] for cls in target_classes}
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue
        
        print(f"Processing {os.path.basename(csv_file)}...")
        
        # Determine which dataset split this is
        file_basename = os.path.basename(csv_file)
        if "eval" in file_basename:
            data_set = "val"
        else:
            data_set = "train"
        
        # Process CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Skip header lines (usually 3 for AudioSet CSVs)
            for _ in range(3):
                try:
                    next(reader)
                except StopIteration:
                    break
            
            # Process rows
            for row in reader:
                if len(row) < 4:  # Ensure row has enough columns
                    continue
                
                youtube_id = row[0]
                start_time = float(row[1])
                end_time = float(row[2])
                classes = row[3]
                
                # Check if any of our target class IDs are in this entry
                for cls in target_classes:
                    if cls in class_ids:
                        for class_id in class_ids[cls]:
                            if class_id in classes:
                                class_entries[cls].append({
                                    'youtube_id': youtube_id,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'set': data_set,
                                    'target_class': cls
                                })
                                break  # Only add once per class
    
    # Summarize findings
    print("\nEntries found in AudioSet CSVs:")
    for cls, entries in class_entries.items():
        print(f"  - {cls}: {len(entries)} entries")
    
    return class_entries

# Function to check what we still need
def calculate_needed_samples(target_classes):
    file_counts = count_existing_files(target_classes)
    needed_samples = {}
    
    print("\nCurrent file counts:")
    for cls in target_classes:
        total = file_counts[cls]["train"] + file_counts[cls]["val"]
        print(f"  - {cls}: {total} samples (Train: {file_counts[cls]['train']}, Val: {file_counts[cls]['val']})")
        
        if total >= TARGET_SAMPLES:
            needed_samples[cls] = 0
        else:
            needed_samples[cls] = TARGET_SAMPLES - total
    
    print("\nAdditional samples needed:")
    for cls, count in needed_samples.items():
        print(f"  - {cls}: {count} more files")
        
    return needed_samples, file_counts

# Function to check if a file already exists without downloading it
def file_already_exists(youtube_id, start_time, end_time, target_class):
    # Check both train and val directories
    for split in ['train', 'val']:
        safe_class_name = target_class.replace(',', '')
        class_dir = os.path.join(base_dir, split, safe_class_name)
        
        if os.path.exists(class_dir):
            filename = f"{youtube_id}_{int(start_time)}_{int(end_time)}.wav"
            if os.path.exists(os.path.join(class_dir, filename)):
                return True
    
    return False

# Function to download from YouTube with cookies
def download_from_youtube(youtube_id, start_time, end_time, target_class, data_set, download_log):
    # Track this attempt
    attempt_key = f"{youtube_id}_{int(start_time)}_{int(end_time)}"
    current_time = datetime.now().isoformat()
    
    if attempt_key not in download_log["attempts"]:
        download_log["attempts"][attempt_key] = []
    
    download_log["attempts"][attempt_key].append(current_time)
    save_download_log(download_log)
    
    # Check if this ID has previously failed multiple times
    failure_threshold = 3
    if youtube_id in download_log["failed"] and download_log["failed"][youtube_id] >= failure_threshold:
        print(f"Skipping {youtube_id} - failed {download_log['failed'][youtube_id]} times previously")
        return False
    
    # First check if file already exists in either train or val
    if file_already_exists(youtube_id, start_time, end_time, target_class):
        print(f"File already exists for {youtube_id}_{int(start_time)}_{int(end_time)}")
        if attempt_key not in download_log["successful"]:
            download_log["successful"].append(attempt_key)
            save_download_log(download_log)
        return True
    
    safe_class_name = target_class.replace(',', '')
    output_dir = os.path.join(base_dir, data_set, safe_class_name)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique filename
    filename = f"{youtube_id}_{int(start_time)}_{int(end_time)}.wav"
    output_path = os.path.join(output_dir, filename)
    
    # Temp file for initial download
    temp_file = os.path.join(temp_dir, f'temp_{youtube_id}.wav')
    
    # Download with yt-dlp using browser cookies
    try:
        cmd = [
            'yt-dlp',
            '--cookies-from-browser', BROWSER,
            '--no-playlist',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--output', temp_file,
            f'https://www.youtube.com/watch?v={youtube_id}'
        ]
        
        print(f"Downloading {youtube_id} using {BROWSER} cookies...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if download succeeded
        if os.path.exists(temp_file):
            # Extract segment using ffmpeg
            duration = end_time - start_time
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_file,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:a', 'pcm_s16le',
                '-ar', '44100',
                output_path,
                '-y'
            ]
            
            print(f"Extracting segment {start_time}s to {end_time}s...")
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True)
            
            # Check if segment extraction succeeded
            if os.path.exists(output_path):
                print(f"✅ Successfully downloaded: {output_path}")
                # Remove temp file
                os.remove(temp_file)
                # Record success
                download_log["successful"].append(attempt_key)
                save_download_log(download_log)
                return True
            else:
                print(f"❌ Failed to extract segment: {ffmpeg_result.stderr}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                # Record failure
                if youtube_id not in download_log["failed"]:
                    download_log["failed"][youtube_id] = 1
                else:
                    download_log["failed"][youtube_id] += 1
                save_download_log(download_log)
                return False
        else:
            print(f"❌ Download failed: {result.stderr}")
            # Record failure
            if youtube_id not in download_log["failed"]:
                download_log["failed"][youtube_id] = 1
            else:
                download_log["failed"][youtube_id] += 1
            save_download_log(download_log)
            return False
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        # Record failure
        if youtube_id not in download_log["failed"]:
            download_log["failed"][youtube_id] = 1
        else:
            download_log["failed"][youtube_id] += 1
        save_download_log(download_log)
        return False

# Function to download samples for specific classes
def download_class_samples(target_classes):
    # Load or initialize download log
    download_log = get_download_log()
    download_log["last_run"] = datetime.now().isoformat()
    save_download_log(download_log)
    
    # Check what we need
    needed_samples, existing_counts = calculate_needed_samples(target_classes)
    
    # Extract entries from original CSVs
    class_entries = extract_class_ids(target_classes)
    
    # Process each target class
    for cls in target_classes:
        if needed_samples[cls] <= 0:
            print(f"\nSkipping {cls} - already have enough samples")
            continue
        
        entries = class_entries[cls]
        if not entries:
            print(f"\nNo entries found for {cls}")
            continue
        
        print(f"\nDownloading for {cls} - need {needed_samples[cls]} more samples")
        
        # Filter out entries that have already been successfully downloaded
        filtered_entries = []
        for entry in entries:
            attempt_key = f"{entry['youtube_id']}_{int(entry['start_time'])}_{int(entry['end_time'])}"
            
            # Skip if already successfully downloaded
            if attempt_key in download_log["successful"]:
                continue
                
            # Skip if already exists on disk (double check)
            if file_already_exists(entry['youtube_id'], entry['start_time'], entry['end_time'], entry['target_class']):
                if attempt_key not in download_log["successful"]:
                    download_log["successful"].append(attempt_key)
                continue
                
            # Skip videos that have failed too many times
            if entry['youtube_id'] in download_log["failed"] and download_log["failed"][entry['youtube_id']] >= 3:
                continue
                
            filtered_entries.append(entry)
        
        print(f"Found {len(filtered_entries)} entries to try for {cls} after filtering out previous successes and repeated failures")
        
        if not filtered_entries:
            print(f"No remaining entries to try for {cls}")
            continue
        
        # Shuffle entries to get a random selection
        random.shuffle(filtered_entries)
        
        # Set a limit to avoid trying too many - increased to handle higher target
        max_attempts = min(needed_samples[cls] * 5, len(filtered_entries))
        filtered_entries = filtered_entries[:max_attempts]
        
        # Track success count
        success_count = 0
        consecutive_failures = 0
        
        # Download each entry
        for i, entry in enumerate(filtered_entries):
            # Stop if we've reached our target
            current_counts = count_existing_files([cls])
            total_current = current_counts[cls]["train"] + current_counts[cls]["val"]
            if total_current >= TARGET_SAMPLES:
                print(f"Reached target for {cls}")
                break
            
            print(f"\nAttempt {i+1}/{max_attempts} for {cls}")
            success = download_from_youtube(
                entry['youtube_id'],
                entry['start_time'],
                entry['end_time'],
                entry['target_class'],
                entry['set'],
                download_log
            )
            
            if success:
                success_count += 1
                consecutive_failures = 0
                print(f"Progress: {success_count}/{needed_samples[cls]} for {cls}")
            else:
                consecutive_failures += 1
            
            # Calculate wait time with exponential backoff for consecutive failures
            wait_time = 0.5 + random.random() * 5  # 5-10 seconds base
            #if consecutive_failures > 0:
                # Exponential backoff: 2^failures multiplier
            #    backoff_multiplier = min(2 ** consecutive_failures, 12)  # Cap at 12x
            #    wait_time = base_wait * backoff_multiplier
            #else:
            #    wait_time = base_wait
            
            print(f"Waiting {wait_time:.1f} seconds before next attempt...")
            time.sleep(wait_time)
            
            # Take a longer break every 10 attempts regardless of success/failure
            if (i + 1) % 10 == 0:
                pause = 10 + random.random() * 30  # 30-60 seconds
                print(f"Taking a {pause:.1f} second break to avoid rate limiting...")
                time.sleep(pause)
                
            # Take an even longer break every 50 attempts to avoid YouTube bans
            if (i + 1) % 50 == 0:
                long_pause = 180 + random.random() * 300  # 5-10 minutes
                print(f"Taking a longer {long_pause/60:.1f} minute break to avoid YouTube rate limiting...")
                time.sleep(long_pause)
    
    # Clean up temp directory
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    # Show final counts
    print("\nDownload complete. Final counts:")
    final_counts = count_existing_files(target_classes)
    for cls in target_classes:
        total = final_counts[cls]["train"] + final_counts[cls]["val"]
        print(f"  - {cls}: {total} samples (Train: {final_counts[cls]['train']}, Val: {final_counts[cls]['val']})")
    
    return final_counts

# Function to update the README file with current counts
def update_readme():
    # Get all class counts
    all_classes = []
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            for dir_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, dir_name)
                if os.path.isdir(class_dir):
                    if dir_name not in all_classes:
                        all_classes.append(dir_name)
    
    all_classes = sorted(all_classes)
    all_counts = {cls: {"train": 0, "val": 0} for cls in all_classes}
    
    for cls in all_classes:
        for split in ['train', 'val']:
            class_dir = os.path.join(base_dir, split, cls)
            
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                all_counts[cls][split] = len(files)
    
    # Create README content
    readme_path = os.path.join(base_dir, 'README.md')
    readme_content = """# AudioSet Dataset for Emergency Sound Detection

This dataset contains audio samples for emergency sound detection, extracted from Google AudioSet.

## Class Distribution

"""
    
    # Add class counts to README
    for cls in all_classes:
        total = all_counts[cls]["train"] + all_counts[cls]["val"]
        readme_content += f"- {cls}: {total} samples (Train: {all_counts[cls]['train']}, Val: {all_counts[cls]['val']})\n"
    
    # Add information about data sources
    readme_content += """
## Notes

- Audio extracted from YouTube videos in Google's AudioSet
- All files are in WAV format, 16-bit PCM, 44.1kHz
- The dataset is split into training and validation sets
- Target: 1000 samples per class
- Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME updated at {readme_path}")

# Main function with command line arguments
def main():
    parser = argparse.ArgumentParser(description='Download samples from AudioSet for specific classes')
    parser.add_argument('--class', dest='target_class', help='Specific class to download (e.g., "glass_breaking")')
    parser.add_argument('--all', action='store_true', help='Download all classes')
    args = parser.parse_args()
    
    print(f"Starting AudioSet downloader with target of {TARGET_SAMPLES} samples per class...")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Determine which classes to download
    if args.all:
        target_classes = ALL_CLASSES
        print(f"Processing all {len(target_classes)} classes")
    elif args.target_class:
        if args.target_class in ALL_CLASSES:
            target_classes = [args.target_class]
            print(f"Processing single class: {args.target_class}")
        else:
            print(f"Error: Class '{args.target_class}' not found in known classes")
            print(f"Available classes: {', '.join(ALL_CLASSES)}")
            return
    else:
        print("Please specify either a specific class with --class or use --all for all classes")
        print(f"Available classes: {', '.join(ALL_CLASSES)}")
        return
    
    # Download samples
    final_counts = download_class_samples(target_classes)
    
    # Update README
    update_readme()
    
    print(f"AudioSet download completed at: {datetime.now().isoformat()}")
    
    # Print final summary
    print("\nFinal class distributions:")
    for cls in sorted(ALL_CLASSES):
        counts = count_existing_files([cls])[cls]
        total = counts["train"] + counts["val"]
        print(f"{cls}: Train={counts['train']}, Val={counts['val']}, Total={total}")

if __name__ == "__main__":
    main()