import io
import librosa
import librosa.display
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import noisereduce as nr
import numpy as np
import soundfile as sf
import audio_processing_tools as apt
from scipy.fft import fft, fftfreq
import webrtcvad
from pydub import AudioSegment
from pytubefix import YouTube

# Set up AudioSegment
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg/bin/ffprobe.exe"

logging.basicConfig(level=logging.INFO)

# Download audio from YouTube
samples, sr = apt.download_from_youtube("https://www.youtube.com/watch?v=72waW_oOzPQ")

# Apply bandpass filter
samples = apt.bandpass_filter(samples, 100, 3500, sr)
sf.write('after_bandpass.wav', samples, sr)

# Define speech and noise samples
speech_sample = samples[int(5 * sr):int(9 * sr)]
noise_sample = samples[int(21 * sr):int(25 * sr)]

# STFT parameters
n_fft = 512
hop_length = n_fft // 2

# Compute STFT
full_stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)

plt.figure(figsize=(24, 8))

# Plot spectrogram for full_stft (before filtering)
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(full_stft), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Full (Before Filters)')

# Apply Wiener filter on combined STFT
full_stft_filtered = apt.adaptive_wiener_filter(full_stft)

# Plot spectrogram for final filtered full STFT
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(full_stft_filtered), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Full (After Wiener Filter)')

# Inverse STFT to get back to time domain
full_filtered = librosa.istft(full_stft_filtered, n_fft=n_fft, hop_length=hop_length)

# Save the filtered audio
sf.write('after_weiner_full.wav', full_filtered, sr)

plt.tight_layout()
plt.show()