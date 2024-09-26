import librosa
import numpy as np
from scipy.signal import butter, lfilter

class audio_signal:
    """
    A class representing audio in the form of a NumPy array.

    Attributes:
        self.signal = Signal as a NumPy array
        self.sr = Sample rate
    """
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

    def npArrayToStft(self, n_fft, hop_length):
        """
        Resamples the signal with a target sample rate.
        
        Parameters:
        n_fft: Number of Fast Fourier Transform points
        hop_length: Number of audio signals between frames

        Return:
        stft: Short-Time Fourier Transform
        """
        stft = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)
        return stft

# Adaptive Wiener filter for noise suppression
def adaptive_wiener_filter(noisy_stft, alpha=0.90, noise_overestimate_factor=2.0, eps=1e-8):
    """
    Applies an adaptive Wiener filter to a noisy STFT (Short-Time Fourier Transform) 
    to reduce noise while preserving speech.
    
    Parameters:
    noisy_stft (np.array): The noisy STFT of the audio signal.
    alpha (float): Smoothing factor for noise estimation, between 0 and 1.
    noise_overestimate_factor (float): Overestimate factor to avoid underestimating noise.
    eps (float): A small constant to avoid division by zero.

    Returns:
    filtered_stft (np.array): The denoised STFT.
    """
    # Compute the magnitude of the noisy STFT
    mag_noisy = np.abs(noisy_stft)

    # Initialize the noise estimate with the first time frame of the noisy STFT
    noise_estimate = np.zeros_like(mag_noisy)
    noise_estimate[:, 0] = mag_noisy[:, 0]

    # Estimate noise over time using an exponential moving average
    for t in range(1, mag_noisy.shape[1]):
        noise_estimate[:, t] = alpha * noise_estimate[:, t - 1] + (1 - alpha) * mag_noisy[:, t]

    # Compute noise power and noisy power
    noise_power = (noise_overestimate_factor * noise_estimate) ** 2
    noisy_power = np.abs(noisy_stft) ** 2

    # Compute Wiener gain (ratio of signal to noise)
    wiener_gain = noisy_power / (noisy_power + noise_power + eps)
    wiener_gain = np.clip(wiener_gain, 0, 1)  # Ensure gain is between 0 and 1

    # Apply the Wiener gain to the noisy STFT
    filtered_stft = wiener_gain * noisy_stft

    return filtered_stft


# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, sr, order=5):
    """
    Designs a Butterworth bandpass filter for a given frequency range.

    Parameters:
    lowcut (float): Lower cutoff frequency for the bandpass filter.
    highcut (float): Upper cutoff frequency for the bandpass filter.
    sr (int): The sampling rate of the audio signal.
    order (int): The order of the filter, higher values make the filter steeper.

    Returns:
    b, a: Filter coefficients for the bandpass filter.
    """
    nyq = 0.5 * sr  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Applies the Butterworth bandpass filter to audio data
def bandpass_filter(data, lowcut, highcut, sr, order=5):
    """
    Applies a Butterworth bandpass filter to audio data.

    Parameters:
    data (np.array): The audio data to be filtered.
    lowcut (float): Lower cutoff frequency for the bandpass filter.
    highcut (float): Upper cutoff frequency for the bandpass filter.
    sr (int): The sampling rate of the audio signal.
    order (int): The order of the filter.

    Returns:
    y (np.array): The bandpass filtered audio data.
    """

    # Calculates filter coefficients, b and a
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = lfilter(b, a, data)
    return y


# Dynamic Spectral Subtraction for noise reduction
def estimated_spectral_subtraction(noisy_stft):
    """
    Performs spectral subtraction with estimated noise to remove noise from an STFT.
    
    Parameters:
    noisy_stft (np.array): The noisy STFT of the audio signal.
    
    Returns:
    denoised_stft (np.array): The denoised STFT after spectral subtraction.
    """
    # Estimate the noise from the first few frames (assumed to be noise-only)
    noise_estimate = np.mean(np.abs(noisy_stft[:, :20]), axis=1, keepdims=True)
    noise_power = noise_estimate ** 2

    # Compute the magnitude of the noisy STFT
    mag_noisy = np.abs(noisy_stft)

    # Subtract noise power from noisy magnitude (ensuring non-negative values)
    mag_denoised = np.sqrt(np.maximum(mag_noisy ** 2 - noise_power, 0))

    # Combine with the phase of the original signal
    phase_noisy = np.angle(noisy_stft)
    denoised_stft = mag_denoised * np.exp(1j * phase_noisy)
    
    return denoised_stft

# Soft mask function for separating foreground (speech) and background (noise)
def soft_mask(S_full, sr, margin_v=10, margin_i=3, power=2):
    """
    Applies a soft mask to separate the foreground (speech) from background (noise).

    Parameters:
    S_full (np.array): The full STFT of the audio signal.
    sr (int): The sampling rate of the audio signal.
    margin_v (float): Margin for separating the vocal (foreground) components.
    margin_i (float): Margin for separating the instrumental (background) components.
    power (int): Power exponent for the softmask.

    Returns:
    foreground (np.array): The STFT of the foreground (speech).
    background (np.array): The STFT of the background (noise).
    """
    # Apply median filtering to estimate the background
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(1, sr=sr)))

    # Limit the background estimate to be no larger than the original signal
    S_filter = np.minimum(S_full, S_filter)

    # Create soft masks for background and foreground separation
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

    return mask_v * S_full, mask_i * S_full  # Foreground (speech) and background (noise)


# Soft thresholding for suppressing low-energy frequencies
def soft_threshold(stft, threshold_db):
    """
    Applies soft thresholding to an STFT to suppress low-energy components below a certain dB threshold.

    Parameters:
    stft (np.array): The STFT of the audio signal.
    threshold_db (float): The dB threshold for soft thresholding.

    Returns:
    filtered_stft (np.array): The STFT after applying soft thresholding.
    """
    # Convert threshold from dB to amplitude
    threshold_amplitude = librosa.db_to_amplitude(threshold_db)

    # Apply mask based on the threshold
    magnitude = np.abs(stft)
    mask = magnitude > threshold_amplitude

    # Zero out values below the threshold
    filtered_stft = stft * mask

    return filtered_stft


# Spectral subtraction for noise reduction
def spectral_subtraction(samples, noise_sample, n_fft, hop_length):
    """
    Performs spectral subtraction to remove noise from an audio signal by subtracting the noise STFT from the signal STFT.

    Parameters:
    samples (np.array): The audio samples of the noisy signal.
    noise_sample (np.array): The audio samples of the noise to be subtracted.
    n_fft (int): The FFT window size.
    hop_length (int): The hop length between successive frames.

    Returns:
    cleaned_stft (np.array): The STFT of the cleaned signal after noise subtraction.
    """
    # Compute STFTs of both the noisy signal and noise sample
    sample_stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
    noise_stft = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)

    # Compute the magnitude of both STFTs
    sample_magnitude = np.abs(sample_stft)
    noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    # Subtract the noise magnitude from the signal magnitude
    cleaned_magnitude = sample_magnitude - noise_magnitude
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0)  # Ensure no negative values

    # Reconstruct the cleaned STFT using the original phase information
    sample_phase = np.angle(sample_stft)
    cleaned_stft = cleaned_magnitude * np.exp(1j * sample_phase)

    return cleaned_stft