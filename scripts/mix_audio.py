def mix_audio(signal, noise, snr_db):

    signal_power = np.sum(signal**2) / len(signal)
    noise_power = np.sum(noise**2) / len(noise)

    desired_noise_power = signal_power / (10**(snr_db / 10))
    
    noise_scaling_factor = np.sqrt(desired_noise_power / noise_power)
    mixed = signal + noise_scaling_factor * noise[:len(signal)]

    mixed = mixed / np.max(np.abs(mixed))

    return mixed