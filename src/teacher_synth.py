"""Physically-motivated piano-ish "teacher" synthesizer.

Used as the ground-truth audio source for Task 4 training because the full
MAESTRO audio archive is only distributed as a single 108 GB zip and the per-
file CDN does not allow individual downloads.

The teacher synth is a hand-written additive synthesizer with three components
chosen so the resulting audio is recognizable as a struck-string instrument:

  1. Harmonic oscillator bank, where each harmonic ``k`` has an exponentially
     decaying amplitude envelope with k-dependent decay rate (higher partials
     decay faster — a defining feature of real piano tones).
  2. Mild inharmonicity (partial k sits at f0 * k * sqrt(1 + B * k^2), with
     B ~ 0.0004 for typical mid-range piano strings).
  3. A short broadband "hammer" noise burst at note onset (bandpass-filtered
     white noise, ~5 ms).

The output is not photo-realistic, but it has structure that a learned
synthesizer can in principle reproduce, and it is fully reproducible
(no external dataset needed).

The teacher is intentionally separate from the DDSP model in
``model_task4.py`` so the two cannot cheat by sharing parameters.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_HARMONICS = 48
DEFAULT_INHARMONICITY_B = 4e-4  # mid-range piano


def midi_to_hz(pitch: float) -> float:
    return float(440.0 * (2.0 ** ((pitch - 69) / 12.0)))


def render_notes(
    notes: Sequence[Tuple[float, float, int, int]],
    duration_sec: float,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    inharmonicity_B: float = DEFAULT_INHARMONICITY_B,
    hammer_noise_ms: float = 6.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a list of (start, end, pitch, velocity) MIDI events into a mono
    waveform of length ``int(duration_sec * sample_rate)``.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)
    if not notes:
        return audio

    t_axis = np.arange(n_samples, dtype=np.float32) / sample_rate
    harm_ix = np.arange(1, n_harmonics + 1, dtype=np.float32)  # 1..K
    # k-dependent exponential decay time constants (seconds): partial k
    # has tau_k = tau_1 / sqrt(k) so higher partials die out faster.
    tau1 = 2.5
    tau_k = (tau1 / np.sqrt(harm_ix)).astype(np.float32)

    nyquist = sample_rate / 2.0

    for (start, end, pitch, velocity) in notes:
        if end <= start:
            continue
        s_idx = int(np.clip(start * sample_rate, 0, n_samples - 1))
        e_idx = int(np.clip(end * sample_rate, 0, n_samples))
        if e_idx <= s_idx:
            continue
        n = e_idx - s_idx
        f0 = midi_to_hz(pitch)
        amp = (velocity / 127.0) ** 0.7  # perceptual loudness
        local_t = np.arange(n, dtype=np.float32) / sample_rate

        # Partial frequencies with mild inharmonicity (piano string stiffness)
        f_k = f0 * harm_ix * np.sqrt(1.0 + inharmonicity_B * (harm_ix ** 2))
        mask = f_k < nyquist  # anti-alias
        f_k = f_k[mask]
        tau_k_eff = tau_k[mask]
        # Base harmonic distribution -- empirically piano-ish: 1/k^1.4
        base_amp = 1.0 / (harm_ix[mask] ** 1.4)
        # Decay envelope per harmonic
        env = np.exp(-local_t[:, None] / tau_k_eff[None, :])  # (n, K_eff)
        # Random phases per harmonic
        phases = rng.uniform(0, 2 * np.pi, size=f_k.shape[0]).astype(np.float32)
        # Synthesize: sum sin(2 pi f_k t + phi) * base_amp_k * env_k
        # Compute in chunks to avoid large memory if n is huge
        chunk = 4096
        out_voice = np.zeros(n, dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            phi = 2.0 * np.pi * f_k[None, :] * local_t[s:e, None] + phases[None, :]
            wave = np.sin(phi) * base_amp[None, :] * env[s:e, :]
            out_voice[s:e] = wave.sum(axis=-1)
        out_voice *= amp

        # Hammer noise burst (5-6 ms at attack, bandpass to ~ 200 Hz - 6 kHz)
        if hammer_noise_ms > 0:
            n_noise = int(min(hammer_noise_ms * 1e-3 * sample_rate, n))
            if n_noise > 4:
                noise = rng.standard_normal(n_noise).astype(np.float32)
                # one-pole bandpass via simple moving-average minus longer moving-average
                noise = _crude_bandpass(noise, sample_rate, 200.0, 5000.0)
                # short triangular envelope
                env_n = np.bartlett(n_noise).astype(np.float32)
                out_voice[:n_noise] = out_voice[:n_noise] + 0.25 * amp * noise * env_n

        audio[s_idx:e_idx] = audio[s_idx:e_idx] + out_voice

    # Soft compression / normalization
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 1.0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def _crude_bandpass(
    x: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> np.ndarray:
    """Cheap bandpass: highpass via subtracting a short MA, lowpass via a longer MA."""
    L_low = max(1, int(sample_rate / max(low_hz, 1e-3) / 4))
    L_high = max(1, int(sample_rate / max(high_hz, 1e-3) / 4))
    L_low = min(L_low, len(x))
    L_high = min(L_high, len(x))
    if L_low < 2 or L_high < 2:
        return x
    # lowpass at high_hz
    k_low = np.ones(L_high, dtype=np.float32) / L_high
    y = np.convolve(x, k_low, mode="same")
    # highpass at low_hz: subtract the longer moving average
    k_high = np.ones(L_low, dtype=np.float32) / L_low
    low_component = np.convolve(y, k_high, mode="same")
    return (y - low_component).astype(np.float32)


def render_pretty_midi(pm, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Convenience wrapper: render a pretty_midi.PrettyMIDI to mono audio."""
    notes: list[Tuple[float, float, int, int]] = []
    end_t = 0.0
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((n.start, n.end, int(n.pitch), int(n.velocity)))
            end_t = max(end_t, n.end)
    return render_notes(notes, end_t + 0.5, sample_rate=sample_rate)
