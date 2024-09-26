import librosa
import numpy as np

class audio_signal:
    """
    A class representing audio in the form of a NumPy array.

    Attributes:
        self.n_fft = Number of Fast Fourier Transform points
        self.hop_length = 
        self.signal = Signal as a NumPy array
        self.sr = Number of audio signals between frames
    """
    n_fft = 512
    hop_length = n_fft // 2

    def __init__(self, signal, sr):
        self.signal = signal
        self.sr = sr
    
    def getSampleRate(self):
        """
        Returns sample rate of the signal

        Return:
        sr: Sample Rate
        """
        return self.sr

    def resample(self, sr):
        """
        Resamples the signal with a target sample rate.
        
        Parameters:
        sr: Target sample rate
        """
        self.signal = librosa.resample(self.signal, orig_sr=self.sr, target_sr=sr)
        self.sr = sr

    def npArrayToStft(self, n_fft=None, hop_length=None):
        """
        Resamples the signal with a target sample rate.
        
        Parameters:
        n_fft: Number of Fast Fourier Transform points
        hop_length: Number of audio signals between frames

        Return:
        stft: Short-Time Fourier Transform
        """
        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = self.hop_length
            
        stft = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)
        return stft
    
    def sliceAudio(self, start_time_s, end_time_s):
        """
        Splits the audio from start time to end time in seconds
        
        Parameters:
        start_time_s: Start time of the sliced audio
        end_time_s: End time of the sliced audio

        Return:
        sliceOfAudio: The slice of the audio split from the main signal
        """

        sliceOfAudio = self.signal[int(start_time_s * self.sr):int(end_time_s * self.sr)]

        return sliceOfAudio