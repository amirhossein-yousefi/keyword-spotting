from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from .utils import db_to_amplitude

def random_time_shift(wave: np.ndarray, sample_rate: int, max_shift_ms: int) -> np.ndarray:
    if max_shift_ms <= 0:
        return wave
    max_shift = int(sample_rate * max_shift_ms / 1000.0)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return wave
    if shift > 0:
        return np.r_[np.zeros(shift, dtype=wave.dtype), wave[:-shift]]
    else:
        return np.r_[wave[-shift:], np.zeros(-shift, dtype=wave.dtype)]

def add_noise_snr(wave: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise at a target SNR (dB) relative to signal power."""
    if snr_db is None:
        return wave
    sig_power = np.mean(wave ** 2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=wave.shape).astype(wave.dtype)
    return (wave + noise).astype(wave.dtype)

def random_gain(wave: np.ndarray, max_gain_db: float) -> np.ndarray:
    if max_gain_db <= 0:
        return wave
    gain_db = np.random.uniform(-max_gain_db, max_gain_db)
    return (wave * db_to_amplitude(gain_db)).astype(wave.dtype)

def apply_training_augs(
    wave: np.ndarray,
    sample_rate: int,
    time_shift_ms: int = 100,
    noise_snr_db_min: Optional[float] = 10.0,
    noise_snr_db_max: Optional[float] = 30.0,
    random_gain_db: float = 6.0,
) -> np.ndarray:
    """Compose augmentations; used only for training split."""
    wave = random_time_shift(wave, sample_rate, time_shift_ms)
    if noise_snr_db_min is not None and noise_snr_db_max is not None:
        snr_db = np.random.uniform(noise_snr_db_min, noise_snr_db_max)
        wave = add_noise_snr(wave, snr_db)
    wave = random_gain(wave, random_gain_db)
    # Clip to [-1, 1] to avoid saturation (assuming float32 waveform)
    wave = np.clip(wave, -1.0, 1.0)
    return wave
