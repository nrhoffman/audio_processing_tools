import configparser
import csv
import gc
import logging
import numpy as np
import os
import random
import tempfile
from pathlib import Path
from pydub import AudioSegment

#Local Imports
from tools.find_first_dir import find_first_dir
from tools.load_save import load_save
from tools.logger import logger
from tools.reset import reset

tempfile.tempdir = "C:\\Users\\nhoff\\AppData\\Local\\Temp_ffmpeg"

#Read in the config
config = configparser.ConfigParser()
config.read('./config/config.cfg')

#Create Global Variables
MIX_AUDIO = config['MIX_AUDIO']
CLIP_LENGTH = int(MIX_AUDIO['ClipLength']) * 1000
TARGET_SAMPLE_RATE = int(MIX_AUDIO.get('SampleRate', 16000))
FILE_LOG_LEVEL = getattr(logging, MIX_AUDIO['fileLogLevel'].upper(), logging.WARNING)
CONSOLE_LOG_LEVEL = getattr(logging, MIX_AUDIO['ConsoleLogLevel'].upper(), logging.INFO)
RESET = MIX_AUDIO.getboolean('Reset', fallback=False)

#Initializations
logger = logger('mix_audio_logger', 'mix_audio.log', FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL)
data_path = r"E:\Data\noise_filter\raw_data"
clean_data = data_path + r"\clean_speech"
noise_path = data_path + r"\noise"
mixed_data = data_path + r"\mixed"
logger.info(f"Clean Path: {clean_data}")
logger.info(f"Noise: {noise_path}")
logger.info(f"Mixed Path: {mixed_data}")

noise = [os.path.join(noise_path, file) for file in os.listdir(noise_path) if file.endswith(".flac")]

#Reset if not True
if RESET:
    logger.info("Resetting the environment...")
    reset(clean_data)
    MIX_AUDIO['Reset'] = 'False'
    with open('./config/config.cfg', 'w') as configfile:
        config.write(configfile)

# Function to normalize audio
def normalize_audio(audio_segment):
    return audio_segment.apply_gain(-audio_segment.max_dBFS)

def resample_audio(audio_segment, target_sample_rate):
    if audio_segment.frame_rate != target_sample_rate:
        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
    return audio_segment

# Function to calculate noise gain based on SNR
def calculate_gain(signal, noise, snr_db):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    desired_noise_power = signal_power / (10**(snr_db / 10))
    noise_gain = np.sqrt(desired_noise_power / noise_power)
    return noise_gain

# Function to mix audio with a specific SNR
def mix_audio(clean_audio, noise_audio, snr_db):
    noise_audio = normalize_audio(noise_audio)

    # Convert audio segments to numpy arrays
    clean_array = np.array(clean_audio.get_array_of_samples(), dtype=np.float32)
    noise_array = np.array(noise_audio.get_array_of_samples(), dtype=np.float32)

    if len(clean_array) < len(noise_array):
        noise_array = noise_array[:len(clean_array)]

    noise_gain = calculate_gain(clean_array, noise_array, snr_db)
    noise_scaled = noise_gain * noise_array[:len(clean_array)]
    mixed_array = clean_array + noise_scaled

    mixed_array /= np.max(np.abs(mixed_array))
    mixed_array = (mixed_array * 32767).astype(np.int16)

    sample_width = 2
    channels = clean_audio.channels
    length_required = sample_width * channels

    if len(mixed_array) % length_required != 0:
        padding_length = length_required - (len(mixed_array) % length_required)
        mixed_array = np.pad(mixed_array, (0, padding_length), 'constant')

    return AudioSegment(
        mixed_array.tobytes(),
        frame_rate=clean_audio.frame_rate,
        sample_width=sample_width,
        channels=channels
    )

noise_count = {}
snr_count = {}
total_clean_duration = 0
snr_values = [0, 5, 10, 15]

first_level, second_level = load_save("./scripts/save/mix_audio.txt")
first_level_dirs = find_first_dir(clean_data, first_level)
second_level_dirs = find_first_dir(clean_data, second_level, first_level_dirs)

# Get total files
total_files = 0
for first_dir in first_level_dirs:
    for second_dir in second_level_dirs[first_dir]:
        total_files += 1

processed_dirs = 0
for first_dir in first_level_dirs:
    first_dir_path = os.path.join(clean_data, first_dir)
    if first_dir != "original":
        if os.path.isdir(first_dir_path):
            logger.info(f"First Level: {first_dir}")
            for second_dir in second_level_dirs[first_dir]:
                logger.info(f"Second Level: {second_dir}")
                processed_dirs += 1
                per_complete = (processed_dirs / total_files) * 100
                logger.info(f"Percentage Complete: {per_complete:.2f}%")

                #Save location in case of interuption
                with open("./scripts/save/mix_audio.txt", "w") as file:
                    file.write(f"First Level: {"/" + first_dir}\n")
                    file.write(f"Second Level: {"/" + second_dir}\n")
                
                second_dir_path = os.path.join(first_dir_path, second_dir)
                new_mixed_data = os.path.join(mixed_data, first_dir, second_dir)
                Path(new_mixed_data).mkdir(parents=True, exist_ok=True)

                for audio_flac in os.listdir(second_dir_path):
                    if audio_flac.endswith(".flac"):

                        clean_audio_path = os.path.join(second_dir_path, audio_flac)
                        clean_audio = AudioSegment.from_file(clean_audio_path, format="flac")
                        clean_audio = resample_audio(clean_audio, TARGET_SAMPLE_RATE)
                        clean_audio = clean_audio.normalize()

                        if len(clean_audio) < CLIP_LENGTH:
                            silence_needed = CLIP_LENGTH - len(clean_audio)
                            padded_clean_audio = clean_audio + AudioSegment.silent(duration=silence_needed)
                        else:
                            padded_clean_audio = clean_audio

                        padded_clean_audio.export(clean_audio_path, format="flac")

                        total_clean_duration += len(padded_clean_audio) / (1000 * 60 * 60)           

                        selected_noises = random.sample(noise, 4)
                        for i, noise_sample in enumerate(selected_noises):
                            if noise_sample not in noise_count:
                                noise_count[noise_sample] = 0
                            snr = random.choice(snr_values)
                            noise_audio = AudioSegment.from_file(noise_sample, format="flac")
                            noise_audio = resample_audio(noise_audio, TARGET_SAMPLE_RATE)

                            noise_count[noise_sample] += 1
                            if noise_sample not in snr_count:
                                snr_count[noise_sample] = {}
                            if snr not in snr_count[noise_sample]:
                                snr_count[noise_sample][snr] = 0
                            snr_count[noise_sample][snr] += 1

                            new_mix_audio_name = f"{audio_flac[:-5]}_{i}_{snr}.flac"
                            new_mix_path = os.path.join(new_mixed_data, new_mix_audio_name)

                            mixed_audio = mix_audio(clean_audio, noise_audio, snr)
                            if len(mixed_audio) < CLIP_LENGTH:
                                silence_needed = CLIP_LENGTH - len(mixed_audio)
                                mixed_audio = mixed_audio + AudioSegment.silent(duration=silence_needed)

                            mixed_audio.export(new_mix_path, format="flac")
                            mixed_audio = None
                            noise_audio = None
                        clean_audio = None
    gc.collect()

csv_file_path = os.path.join(mixed_data, 'mix_summary.csv')
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Noise Sample', 'Usage Count', 'SNR Usage Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for noise_sample, count in noise_count.items():
        snr_usages = snr_count.get(noise_sample, {})
        writer.writerow({
            'Noise Sample': noise_sample,
            'Usage Count': count,
            'SNR Usage Count': dict(snr_usages)
        })
    
    # Write the total clean audio hours at the end of the CSV
    writer.writerow({})
    writer.writerow({'Noise Sample': 'Total Clean Audio Hours', 'Usage Count': total_clean_duration, 'SNR Usage Count': ''})

reset(clean_data)