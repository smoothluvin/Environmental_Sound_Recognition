import os
import shutil
import pandas as pd

URBANSOUND_DIR = "./data/UrbanSound8K/audio"
METADATA_FILE = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
TARGET_DIR = "./data/Filtered_Dataset"
TARGET_CLASSES = ["gun_shot", "siren"]

# Loading the metadata csv file in the UrbanSound8K Folder that Phillip has on his local machine
df = pd.read_csv(METADATA_FILE)

# Filter only the selected classes
df = df[df["class"].isin(TARGET_CLASSES)]

for _, row in df.iterrows():
    class_name = row["class"]
    source_file = os.path.join(URBANSOUND_DIR, f"fold{row['fold']}", row["slice_file_name"])
    target_class_dir = os.path.join(TARGET_DIR, class_name)

    if not os.path.exists(target_class_dir):
        os.makedirs(target_class_dir)

    shutil.copy(source_file, target_class_dir)

print("UrbanSound8K dataset filtering is complete!")

