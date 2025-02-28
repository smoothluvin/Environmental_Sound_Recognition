import os
import shutil
import pandas as pd

ESC50_DIR = "./data/ESC-50-master/audio"
METADATA_FILE = "./data/ESC-50-master/meta/esc50.csv"
TARGET_DIR = "./data/Filtered_Dataset"
TARGET_CLASSES = ["crying_baby", "glass_breaking"]

# Loading the metadata csv file in the UrbanSound8K Folder that Phillip has on his local machine
df = pd.read_csv(METADATA_FILE)

# Filter only the selected classes
df = df[df["category"].isin(TARGET_CLASSES)]

for _, row in df.iterrows():
    class_name = row["category"]
    source_file = os.path.join(ESC50_DIR, row["filename"])
    target_class_dir = os.path.join(TARGET_DIR, class_name)

    if not os.path.exists(target_class_dir):
        os.makedirs(target_class_dir)

    shutil.copy(source_file, target_class_dir)

print("ESC-50 dataset filtering is complete!")

