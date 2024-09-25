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
from plots import spectrogram_plot

# Set up AudioSegment
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg/bin/ffprobe.exe"

logging.basicConfig(level=logging.INFO)

def mix_audio(signal, noise, snr_db):

    signal_power = np.sum(signal**2) / len(signal)
    noise_power = np.sum(noise**2) / len(noise)

    desired_noise_power = signal_power / (10**(snr_db / 10))
    
    noise_scaling_factor = np.sqrt(desired_noise_power / noise_power)
    mixed = signal + noise_scaling_factor * noise[:len(signal)]

    mixed = mixed / np.max(np.abs(mixed))

    return mixed

signal, sr = librosa.load('./tutorial_guy.wav', sr=16000)
noise, sr = librosa.load('./cafe_noise.wav', sr=16000)

plt.figure(figsize=(24, 8))
signal = signal / np.max(np.abs(signal))
spectrogram_plot(signal, sr, 'Signal', 4, 4)
noise = noise / np.max(np.abs(noise))

mixed = mix_audio(signal, noise, snr_db=5)
mixed = mixed / np.max(np.abs(mixed))
spectrogram_plot(mixed, sr, 'Mixed', 4, 1)
sf.write('mixed_audio.wav', mixed, sr)

filtered = apt.bandpass_filter(mixed, 300, 3400, sr, order = 4)
spectrogram_plot(filtered, sr, 'After Bandpass', 4, 2)
# sf.write('bandpass_audio.wav', filtered, sr)

n_fft = 512
hop_length = n_fft // 2
stft = librosa.stft(filtered, n_fft=n_fft, hop_length=hop_length)

filtered_stft = apt.estimated_spectral_subtraction(stft)
filtered = librosa.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length)
filtered = filtered / np.max(np.abs(filtered))
spectrogram_plot(filtered, sr, 'After Spec_Sub', 4, 3)

# filtered_stft = apt.adaptive_wiener_filter(filtered_stft, alpha=0.97, noise_overestimate_factor=2.0)
# filtered = librosa.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length)
# filtered = filtered / np.max(np.abs(filtered))
# spectrogram_plot(filtered, sr, 'After Wiener', 4, 3)

# S_full, phase = librosa.magphase(filtered_stft)
# S_full, background = apt.soft_mask(S_full, sr, margin_v = 3, margin_i = 2, power = 3)
# filtered_stft = S_full*phase
# filtered = librosa.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length)
# filtered = filtered / np.max(np.abs(filtered))
# spectrogram_plot(filtered, sr, 'After Soft Mask', 4, 2)

# filtered_stft = apt.soft_threshold(filtered_stft, threshold_db = -15)
# filtered = librosa.istft(filtered_stft, n_fft=n_fft, hop_length=hop_length)
# filtered = filtered / np.max(np.abs(filtered))
# spectrogram_plot(filtered, sr, 'After Threshholding', 4, 3)
sf.write('filtered_audio.wav', filtered, sr)

plt.tight_layout()
plt.show()





# logging.info(f"Normalizing .wav")
# normalized_audio = audio_segment.normalize()

# samples = np.array(normalized_audio.get_array_of_samples())

# sr = normalized_audio.frame_rate

# # Apply bandpass filter
# samples = apt.bandpass_filter(samples, 100, 3500, sr)
# sf.write('after_bandpass.wav', samples, sr)

# # STFT parameters
# n_fft = 512
# hop_length = n_fft // 2

# # Compute STFT
# noise_sample = samples[int(0 * sr):int(30 * sr)]
# noise_stft = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)
# full_stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)

# noise_stft = apt.adaptive_wiener_filter(noise_stft)

# spectrogram_plot(noise_stft, sr=sr)


