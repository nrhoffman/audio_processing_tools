import pandas as pd
from pathlib import Path
import os
import random
import yaml

from math import floor

from tools.find_first_dir import find_first_dir
from tools.load_save import load_save
from tools.logger import logger

HOURS_NEEDED = 50
PERCENTAGE = HOURS_NEEDED / 300
FILE_LENGTH = 6 #Seconds

logger = logger('dataset_metadata_creation_logger', 'dataset_metadata_creation.log')

data_path = r"E:\Data\noise_filter\raw_data"
clean_data = data_path + r"\clean_speech"
mixed_data = data_path + r"\mixed"

dataset_path = r"E:\Data\noise_filter\dataset"
yaml_path = "./models/data.yaml"

def distribute_files_randomly(directory, percentages):

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    random.shuffle(files)

    total_files = len(files)

    counts = [floor(total_files * (p / 100)) for p in percentages]

    list1, list2, list3 = [], [], []

    list1 = files[:counts[0]]
    list2 = files[counts[0]:counts[0] + counts[1]]
    list3 = files[counts[0] + counts[1]:counts[0] + counts[1] + counts[2]]
    
    return list1, list2, list3

def match_clean_files(directory, new_path, train_mixed_list, val_mixed_list, test_mixed_list):
    # Create a function to find the corresponding clean file
    def get_clean_filename(mixed_filename):
        parts = mixed_filename.split('_')
        return parts[0] + '_' + parts[1]

    # Initialize the clean file lists
    train_pairs = []
    val_pairs = []
    test_pairs = []

    # Get all clean files in the directory
    clean_files = {f[:-5] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.flac')}

    # Match mixed files with their clean counterparts
    for mixed_file in train_mixed_list:
        clean_file = get_clean_filename(mixed_file)
        if clean_file in clean_files:
            mixed_file = mixed_file[:-5]
            mixed_dataset_path = os.path.join("mixed", new_path, mixed_file + '.npy')
            clean_dataset_path = os.path.join("clean", new_path, clean_file + '.npy')
            train_pairs.append((mixed_dataset_path, clean_dataset_path))

    for mixed_file in val_mixed_list:
        clean_file = get_clean_filename(mixed_file)
        if clean_file in clean_files:
            mixed_file = mixed_file[:-5]
            mixed_dataset_path = os.path.join("mixed", new_path, mixed_file + '.npy')
            clean_dataset_path = os.path.join("clean", new_path, clean_file + '.npy')
            val_pairs.append((mixed_dataset_path, clean_dataset_path))

    for mixed_file in test_mixed_list:
        clean_file = get_clean_filename(mixed_file)
        if clean_file in clean_files:
            mixed_file = mixed_file[:-5]
            mixed_dataset_path = os.path.join("mixed", new_path, mixed_file + '.npy')
            clean_dataset_path = os.path.join("clean", new_path, clean_file + '.npy')
            test_pairs.append((mixed_dataset_path, clean_dataset_path))

    return train_pairs, val_pairs, test_pairs

first_level, second_level = load_save("./scripts/save/dataset_creation.txt")
first_level_dirs = find_first_dir(clean_data, first_level)
second_level_dirs = find_first_dir(clean_data, second_level, first_level_dirs)

files_needed = int((HOURS_NEEDED * 60 * 60) / 6)

training_files_needed = int(files_needed * 0.75)
validation_files_needed = int(files_needed * 0.15)
testing_files_needed = int(files_needed * 0.1)

logger.info(f"Training Files Needed: {training_files_needed}")
logger.info(f"Validation Files Needed: {validation_files_needed}")
logger.info(f"Testing Files Needed: {testing_files_needed}")

distribution = [75, 15, 10]

data = {
    'train':os.path.join(dataset_path,"train"),
    'val':os.path.join(dataset_path,"val"),
    'test':os.path.join(dataset_path,"test")
}

with open("./models/data.yaml", 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

file_index = 0
for first_dir in first_level_dirs:
    first_dir_path = os.path.join(clean_data, first_dir)
    if first_dir != "original":
        if os.path.isdir(first_dir_path):
            logger.info(f"First Level: {first_dir}")
            for second_dir in second_level_dirs[first_dir]:
                logger.info(f"Second Level: {second_dir}")

                clean_sec_path = os.path.join(first_dir_path, second_dir)
                mixed_sec_path = os.path.join(mixed_data, first_dir, second_dir)
                total_files = len([f for f in Path(mixed_sec_path).iterdir() if f.is_file()])
                train_mixed_list, val_mixed_list, test_mixed_list = distribute_files_randomly(mixed_sec_path,
                                                                                              distribution)
                file_index = file_index + total_files/4
                train_pairs, val_pairs, test_pairs = match_clean_files(
                                                                    clean_sec_path, os.path.join(first_dir, second_dir),
                                                                    train_mixed_list, val_mixed_list, test_mixed_list)
                
                os.makedirs(os.path.join(dataset_path,"train","mixed",first_dir,second_dir), exist_ok=True)
                os.makedirs(os.path.join(dataset_path,"val","mixed",first_dir,second_dir), exist_ok=True)
                os.makedirs(os.path.join(dataset_path,"test","mixed",first_dir,second_dir), exist_ok=True)
                os.makedirs(os.path.join(dataset_path,"train","clean",first_dir,second_dir), exist_ok=True)
                os.makedirs(os.path.join(dataset_path,"val","clean",first_dir,second_dir), exist_ok=True)
                os.makedirs(os.path.join(dataset_path,"test","clean",first_dir,second_dir), exist_ok=True)

                # Create DataFrames for each list of pairs
                train_df = pd.DataFrame(train_pairs, columns=["Mixed", "Clean"])
                val_df = pd.DataFrame(val_pairs, columns=["Mixed", "Clean"])
                test_df = pd.DataFrame(test_pairs, columns=["Mixed", "Clean"])

                # Save to a single CSV file for each list
                train_df.to_csv(os.path.join(dataset_path, "train", 'train_pairs.csv'), mode='a', index=False)
                val_df.to_csv(os.path.join(dataset_path, "val", 'val_pairs.csv'), mode='a', index=False)
                test_df.to_csv(os.path.join(dataset_path, "test", 'test_pairs.csv'), mode='a', index=False)

                if file_index >= files_needed:
                    break
    if file_index >= files_needed:
        break
print(file_index)