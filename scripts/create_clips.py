import configparser
import logging
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

#Local Imports
from tools.find_first_dir import find_first_dir
from tools.load_save import load_save
from tools.logger import logger
from tools.reset import reset

#Read in the config
config = configparser.ConfigParser()
config.read('./config/config.cfg')

#Create Global Variables
CREATE_CLIPS = config['CREATE_CLIPS']
FILE_LOG_LEVEL = getattr(logging, CREATE_CLIPS['fileLogLevel'].upper(), logging.WARNING)
CONSOLE_LOG_LEVEL = getattr(logging, CREATE_CLIPS['ConsoleLogLevel'].upper(), logging.INFO)
CLIP_LENGTH = int(CREATE_CLIPS['ClipLength']) * 1000
RESET = CREATE_CLIPS.getboolean('Reset', fallback=False)

#Initializations
logger = logger('create_clips_logger', 'create_clips.log', FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL)
data_path = r"E:\Data\noise_filter\raw_data\clean_speech"
original_audio = data_path + r"\original\train-clean-360"
logger.debug(f"Data Path: {data_path}")
logger.debug(f"Original Audio: {original_audio}")

#Reset if not True
if RESET:
    logger.info("Resetting the environment...")
    reset(original_audio)
    CREATE_CLIPS['Reset'] = 'False'
    with open('./config/config.cfg', 'w') as configfile:
        config.write(configfile)

#Loads the save file from ./create_clips.txt
first_level, second_level = load_save("./scripts/save/create_clips.txt")
first_level_dirs = find_first_dir(original_audio, first_level)
second_level_dirs = find_first_dir(original_audio, second_level, first_level_dirs)
logger.debug(f"Loading from First Level: {first_level}, Second Level: {second_level}")

# Get total files
total_files = 0
for first_dir in first_level_dirs:
    for second_dir in second_level_dirs[first_dir]:
        total_files += 1

#Start at the loaded directory
file_iter = 0
for first_dir in first_level_dirs:
    logger.info(f"First Level: {first_dir}")
    for second_dir in second_level_dirs[first_dir]:
        logger.info(f"Second Level: {second_dir}")
        file_iter += 1
        per_complete = (file_iter / total_files) * 100
        logger.info(f"Percentage Complete: {per_complete:.2f}%")       

        #Save location in case of interuption
        with open("./scripts/save/create_clips.txt", "w") as file:
            file.write(f"First Level: {"/" + first_dir}\n")
            file.write(f"Second Level: {"/" + second_dir}\n")

        # #Build path to audio files
        path = os.path.join(original_audio, first_dir, second_dir)
        new_path = os.path.join(data_path, first_dir, second_dir)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        for audio_flac in os.listdir(path):
            if audio_flac.endswith(".flac"):
                file_path = os.path.join(path, audio_flac)
                file_name = os.path.basename(file_path)

                audio = AudioSegment.from_file(file_path, format="flac")

                #Split the audio on silence and combine them back for audio without silences
                audio_chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

                #If there are chunks after splitting
                if audio_chunks:
                    combined_audio = sum(audio_chunks)

                    #Split the audio into CLIP_LENGTH samples
                    clips = [combined_audio[i:i + CLIP_LENGTH] for i in range(0, len(combined_audio), CLIP_LENGTH)]

                    if len(clips[-1]) < 6000:
                        #Combine short clips with sliced previous clips
                        if len(clips) > 1:
                            needed_length = CLIP_LENGTH - len(clips[-1])
                            sliced_portion = clips[-2][-needed_length:]
                            clips[-1] = sliced_portion + clips[-1]
                        else:
                            #Throw out clips less than 4 seconds without preceeding clips
                            if len(clips[-1]) < 4000:
                                logger.warning(f"Discarding clip {len(clips)} from {file_name} in {path} from because it is less than 4 seconds.")
                                clips.pop()
                    for i, clip in enumerate(clips):
                        base, ext = os.path.splitext(file_name)
                        new_file_name = f"{base}_{i}{ext}"
                        clip.export(os.path.join(new_path, new_file_name), format="flac")
                else:
                    logger.warning(f"No audio chunks found for {file_name} in {path}")



#Resets the save file from the beginning
reset(original_audio)