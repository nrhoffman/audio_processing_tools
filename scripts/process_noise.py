import configparser
import librosa
import numpy as np
import os

from pydub import AudioSegment

#Read in the config
config = configparser.ConfigParser()
config.read('./config/config.cfg')

#Create Global Variables
CREATE_CLIPS = config['PROCESS_NOISE']
CLIP_LENGTH = int(CREATE_CLIPS['cliplength']) * 1000

noise_path = "E:/Data/noise_filter/raw_data/noise"

for audio_flac in os.listdir(noise_path):
    if audio_flac.endswith(".flac"):
        file_path = os.path.join(noise_path, audio_flac)

        signal, sr = librosa.load(file_path, sr=16000)

        trimmed_signal, _ = librosa.effects.trim(signal, top_db=20)

        target_samples = int((CLIP_LENGTH / 1000) * sr) 

        if len(trimmed_signal) < target_samples :
            padding_length = target_samples - len(trimmed_signal)
            trimmed_signal = np.pad(trimmed_signal, (0, padding_length), mode='constant')
        if len(trimmed_signal) > target_samples:
            trimmed_signal = trimmed_signal[:target_samples]
    
        signal_int16 = np.int16(trimmed_signal * 32767)
        output_signal = AudioSegment(
            data=signal_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

        output_signal.export(file_path, format="flac")

