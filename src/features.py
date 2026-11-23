"""Acoustic feature extraction used by Task 4.

The DDSP synthesizer is supervised by spectral losses computed against the
target audio, but at inference time we drive it directly from MIDI. For
training we also need an A-weighted loudness curve (the model's prediction
target), and at evaluation we need an F0 estimate of the synthesized output to
report pitch accuracy. Both live here so the rest of the pipeline only depends
on numpy / librosa primitives.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import librosa  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("librosa is required; pip install librosa") from e


SAMPLE_RATE = 16000
FRAME_RATE = 250  # Hz, used by DDSP for control signals
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE  # 64
WIN_LENGTH = 1024


# --------- A-weighting --------------------------------------------------------

def _a_weighting_db(freq_hz: np.ndarray) -> np.ndarray:
    """Standard A-weighting filter response in dB."""
    f = np.maximum(freq_hz, 1e-6)
    ra_num = (12200.0 ** 2) * (f ** 4)
    ra_den = (
        (f ** 2 + 20.6 ** 2)
        * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
        * (f ** 2 + 12200.0 ** 2)
    )
    ra = ra_num / ra_den
    return 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.0


def a_weighted_loudness(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
    n_fft: int = WIN_LENGTH,
) -> np.ndarray:
    """Per-frame loudness in dB, A-weighted, normalized so 0 dB ~= silence.

    Returns a (T,) float32 array, where ``T = ceil(len(audio) * frame_rate / sample_rate)``.
    """
    hop = max(1, int(sample_rate / frame_rate))
    audio = audio.astype(np.float32)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
    power = (np.abs(stft) ** 2).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_db = _a_weighting_db(freqs)
    a_lin = (10.0 ** (a_db / 20.0)).astype(np.float32)
    weighted = power * (a_lin[:, None] ** 2)
    energy = np.sum(weighted, axis=0) + 1e-12
    loudness_db = 10.0 * np.log10(energy)
    # normalize so the dataset has loudness in a comparable [-80, 0] range
    loudness_db = np.clip(loudness_db, -120.0, 40.0)
    return loudness_db.astype(np.float32)


def normalize_loudness(loudness_db: np.ndarray) -> np.ndarray:
    """Map raw loudness in dB to a roughly [0, 1] range for the model."""
    return ((loudness_db + 80.0) / 80.0).clip(0.0, 1.5).astype(np.float32)


def denormalize_loudness(loudness_norm: np.ndarray) -> np.ndarray:
    return (loudness_norm * 80.0 - 80.0).astype(np.float32)


# --------- F0 estimation via librosa.pyin (no extra deps) --------------------

def estimate_f0(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Probabilistic YIN F0 estimate. Returns (f0_hz, voiced_prob)."""
    hop = max(1, int(sample_rate / frame_rate))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop,
        fill_na=0.0,
    )
    return f0.astype(np.float32), voiced_prob.astype(np.float32)


# --------- Note-roll -> per-frame control signals (used at inference) -------

def midi_notes_to_frame_controls(
    notes: list[tuple[float, float, int, int]],
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
    max_voices: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of (start_sec, end_sec, pitch, velocity) notes into
    parallel per-voice (F0_hz, loudness_norm, active_mask) tensors.

    The polyphonic synthesizer treats each note as one voice. When polyphony
    exceeds ``max_voices`` at a given frame, lower-velocity notes are dropped.

    Returns three arrays each of shape (max_voices, T).
    """
    n_frames = int(np.ceil(duration_sec * frame_rate))
    f0_voices = np.zeros((max_voices, n_frames), dtype=np.float32)
    loud_voices = np.zeros((max_voices, n_frames), dtype=np.float32)
    active = np.zeros((max_voices, n_frames), dtype=np.float32)

    # Per frame, find the active notes ordered by velocity desc, assign to free slots.
    notes_sorted = sorted(notes, key=lambda n: -n[3])  # velocity desc
    # For each note, find a voice slot that is free for its whole duration.
    slot_used_until = [-1.0] * max_voices
    note_slot: list[int] = []
    for note in notes_sorted:
        start, end, _pitch, _vel = note
        chosen = -1
        for s in range(max_voices):
            if slot_used_until[s] <= start + 1e-6:
                chosen = s
                slot_used_until[s] = end
                break
        note_slot.append(chosen)

    for note, slot in zip(notes_sorted, note_slot, strict=False):
        if slot < 0:
            continue
        start, end, pitch, vel = note
        s_idx = int(np.floor(start * frame_rate))
        e_idx = int(np.ceil(end * frame_rate))
        s_idx = max(0, min(n_frames, s_idx))
        e_idx = max(0, min(n_frames, e_idx))
        if e_idx <= s_idx:
            continue
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        loud = (vel / 127.0) ** 0.5  # perceptual-ish mapping
        active[slot, s_idx:e_idx] = 1.0
        f0_voices[slot, s_idx:e_idx] = f0
        # Triangular envelope: quick attack, longer decay
        L = e_idx - s_idx
        env = np.ones(L, dtype=np.float32) * loud
        atk = min(8, L)
        rel = min(int(0.25 * frame_rate), L)
        if atk > 0:
            env[:atk] *= np.linspace(0.0, 1.0, atk, dtype=np.float32)
        if rel > 0:
            env[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
        loud_voices[slot, s_idx:e_idx] = env

    return f0_voices, loud_voices, active
