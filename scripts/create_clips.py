import librosa
import logging
import os
from pydub import AudioSegment

#Local Imports
from tools.find_first_dir import find_first_dir
from tools.load_save import load_save
from tools.logger import logger

logger = logger('create_clips.log')
data_path = r"E:\Data\noise_filter\raw_data\clean_speech"
original_audio = data_path + r"\1-LibriSpeech\train-clean-360"
logging.debug(f"Data Path: {data_path}")
logging.debug(f"Original Audio: {original_audio}")

#Loads the save file from ./create_clips.txt
first_level, second_level = load_save("./scripts/save/create_clips.txt")
first_level_dirs = find_first_dir(original_audio, first_level)
second_level_dirs = find_first_dir(original_audio, second_level, first_level_dirs)
logging.debug(f"Loading from First Level: {first_level}, Second Level: {second_level}")

# Get total files
total_files = 0
for first_dir in first_level_dirs:
    for second_dir in second_level_dirs[first_dir]:
        total_files += 1

#Start at the loaded directory
file_iter = 0
for first_dir in first_level_dirs:
    for second_dir in second_level_dirs[first_dir]:
        file_iter += 1
        per_complete = (file_iter / total_files) * 100
        logging.info(f"Percentage Complete: {per_complete:.2f}%")

        #Build path to audio files
        path = original_audio + "/" + first_dir + "/" + second_dir
        for audio_flac in os.listdir(path):
            if audio_flac.endswith(".flac"):
                file_name = os.path.basename(os.path.join(path, audio_flac))
                print(file_name)
                # audio = AudioSegment.from_file(audio_flac, format="flac")


        #Save location in case of interuption
        with open("./scripts/save/create_clips.txt", "w") as file:
            file.write(f"First Level: {"/" + first_dir}\n")
            file.write(f"Second Level: {"/" + second_dir}\n")



#Resets the save file from the beginning
first_level = "/" + find_first_dir(original_audio)
second_level = "/" + find_first_dir(original_audio + first_level)
logging.debug(f"Rewriting save file to First Level: {first_level}, Second Level: {second_level}")
with open("./scripts/save/create_clips.txt", "w") as file:
    file.write(f"First Level: {first_level}\n")
    file.write(f"Second Level: {second_level}\n")