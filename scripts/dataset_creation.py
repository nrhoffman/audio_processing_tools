import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import os
import random
import yaml

from tools.logger import logger

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

logger = logger('dataset_creation_logger', 'dataset_creation.log')

data_path = r"E:\Data\noise_filter\raw_data"
clean_data = data_path + r"\clean_speech"
mixed_data = data_path + r"\mixed"

yaml_path = "./models/data.yaml"

with open(yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)

train_path = data_yaml['train']
val_path = data_yaml['val']
test_path = data_yaml['test']

train_df = pd.read_csv(os.path.join(train_path, 'train_pairs.csv'))
val_df = pd.read_csv(os.path.join(val_path, 'val_pairs.csv'))
test_df = pd.read_csv(os.path.join(test_path, 'test_pairs.csv'))

total_rows = len(train_df)
for index, row in enumerate(train_df.iterrows()):
    mixed_file = row[1]['Mixed']
    clean_file = row[1]['Clean']
    parts = mixed_file.split('\\')

    if len(parts) >= 3:

        per_complete = (index +1 / total_rows) * 100
        logger.info(f"Percentage Complete (Train): {per_complete:.2f}%")
        
        flac_file = parts[3][:-3] + 'flac'
        clean_flac = clean_file.split('\\')[3][:-3] + 'flac'
        mixed_path_sec_dir = os.path.join(mixed_data, parts[1], parts[2], flac_file)
        clean_path_sec_dir = os.path.join(clean_data, parts[1], parts[2], clean_flac)
        mixed_dataset_path = os.path.join(train_path, mixed_file)
        clean_dataset_path = os.path.join(train_path, clean_file)

        y, sr = librosa.load(mixed_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{mixed_dataset_path}', mel_spec_db)

        y, sr = librosa.load(clean_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{clean_dataset_path}', mel_spec_db)
    else:
        continue

total_rows = len(val_df)
for index, row in enumerate(val_df.iterrows()):
    mixed_file = row[1]['Mixed']
    clean_file = row[1]['Clean']
    parts = mixed_file.split('\\')
    if len(parts) >= 3:

        per_complete = (index +1 / total_rows) * 100
        logger.info(f"Percentage Complete (Val): {per_complete:.2f}%")

        flac_file = parts[3][:-3] + 'flac'
        clean_flac = clean_file.split('\\')[3][:-3] + 'flac'
        mixed_path_sec_dir = os.path.join(mixed_data, parts[1], parts[2], flac_file)
        clean_path_sec_dir = os.path.join(clean_data, parts[1], parts[2], clean_flac)
        mixed_dataset_path = os.path.join(val_path, mixed_file)
        clean_dataset_path = os.path.join(val_path, clean_file)

        y, sr = librosa.load(mixed_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{mixed_dataset_path}', mel_spec_db)

        y, sr = librosa.load(clean_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{clean_dataset_path}', mel_spec_db)
    else:
        continue

total_rows = len(test_df)
for index, row in enumerate(test_df.iterrows()):
    mixed_file = row[1]['Mixed']
    clean_file = row[1]['Clean']
    parts = mixed_file.split('\\')
    if len(parts) >= 3:
        
        per_complete = (index +1 / total_rows) * 100
        logger.info(f"Percentage Complete (Test): {per_complete:.2f}%")
        
        flac_file = parts[3][:-3] + 'flac'
        clean_flac = clean_file.split('\\')[3][:-3] + 'flac'
        mixed_path_sec_dir = os.path.join(mixed_data, parts[1], parts[2], flac_file)
        clean_path_sec_dir = os.path.join(clean_data, parts[1], parts[2], clean_flac)
        mixed_dataset_path = os.path.join(test_path, mixed_file)
        clean_dataset_path = os.path.join(test_path, clean_file)

        y, sr = librosa.load(mixed_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{mixed_dataset_path}', mel_spec_db)

        y, sr = librosa.load(clean_path_sec_dir, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(f'{clean_dataset_path}', mel_spec_db)
    else:
        continue

