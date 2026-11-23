"""DDSP-style harmonic + filtered-noise synthesizer for Task 4.

Follows Engel et al. (DDSP, 2020) for the synthesizer core and extends it for
polyphony via sum-of-voices a la MIDI-DDSP (Wu et al. 2022). Each voice is
synthesized independently from its (F0, loudness, active) control signal, then
summed.

The trainable per-voice encoder maps (F0, loudness, active) at the control rate
to (harmonic_amps, harmonic_distribution, noise_filter) which parametrize the
DDSP modules. The synthesizer modules themselves are entirely differentiable
DSP -- they have no trainable parameters.

Loss: multi-scale STFT against the target waveform (computed in train_task4.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Differentiable DSP modules
# ---------------------------------------------------------------------------

def harmonic_synth(
    f0_hz: torch.Tensor,
    amplitudes: torch.Tensor,
    harmonic_distribution: torch.Tensor,
    sample_rate: int,
    frame_rate: int,
) -> torch.Tensor:
    """Harmonic-oscillator-bank synthesizer.

    Inputs:
        f0_hz: (B, T)
        amplitudes: (B, T) overall gain >= 0
        harmonic_distribution: (B, T, K) softmaxed over K harmonics; controls
            the relative amplitudes of harmonics 1..K.
    Returns:
        waveform: (B, n_samples)
    """
    B, T = f0_hz.shape
    K = harmonic_distribution.shape[-1]
    n_samples = int(T * sample_rate / frame_rate)
    # Upsample control signals to audio rate
    f0_smp = _upsample_linear(f0_hz, n_samples)             # (B, n)
    amp_smp = _upsample_linear(amplitudes, n_samples)       # (B, n)
    hd_smp = _upsample_linear(
        harmonic_distribution.permute(0, 2, 1), n_samples
    ).permute(0, 2, 1)                                       # (B, n, K)

    # Anti-alias: zero out harmonics whose freq > Nyquist
    harm_ix = torch.arange(1, K + 1, device=f0_hz.device, dtype=f0_hz.dtype)  # (K,)
    f_harm = f0_smp.unsqueeze(-1) * harm_ix  # (B, n, K)
    nyquist = sample_rate / 2.0
    aa_mask = (f_harm < nyquist).to(f0_hz.dtype)
    hd_smp = hd_smp * aa_mask
    # Renormalize so the distribution sums to 1 per frame (post-mask)
    hd_smp = hd_smp / (hd_smp.sum(dim=-1, keepdim=True) + 1e-8)

    # Phase accumulators per harmonic share a single F0 cumsum
    phase = 2.0 * math.pi * torch.cumsum(f0_smp / sample_rate, dim=-1)  # (B, n)
    # Sum sinusoids
    waves = torch.sin(phase.unsqueeze(-1) * harm_ix)  # (B, n, K)
    summed = (waves * hd_smp).sum(dim=-1)             # (B, n)
    return amp_smp * summed


def filtered_noise_synth(
    filter_mag: torch.Tensor,
    sample_rate: int,
    frame_rate: int,
) -> torch.Tensor:
    """Filtered-white-noise synthesizer.

    Given per-frame magnitude responses ``filter_mag`` of shape (B, T, n_bins)
    (linear gain, non-negative), generate (B, n_samples) of filtered white
    noise via inverse STFT with the predicted magnitude applied to a random-phase
    complex spectrogram.
    """
    B, T, n_bins = filter_mag.shape
    n_fft = (n_bins - 1) * 2
    hop = max(1, sample_rate // frame_rate)
    n_samples = T * hop

    # Random-phase complex spectrogram with the predicted magnitude
    phase = (
        2.0 * math.pi * torch.rand(B, n_bins, T, device=filter_mag.device, dtype=filter_mag.dtype)
    )
    mag = filter_mag.transpose(1, 2)  # (B, n_bins, T)
    spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    window = torch.hann_window(n_fft, device=filter_mag.device, dtype=filter_mag.dtype)
    audio = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        length=n_samples,
        center=True,
    )
    return audio


def _upsample_linear(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Linear upsampling along the last axis. ``x`` may be (B, T) or (B, T, K)."""
    if x.shape[-1] == target_length:
        return x
    if x.dim() == 2:
        return F.interpolate(
            x.unsqueeze(1), size=target_length, mode="linear", align_corners=True
        ).squeeze(1)
    # (B, K, T) for multichannel
    return F.interpolate(x, size=target_length, mode="linear", align_corners=True)


# ---------------------------------------------------------------------------
# Trainable per-voice encoder
# ---------------------------------------------------------------------------

@dataclass
class DDSPConfig:
    n_harmonics: int = 64
    noise_filter_taps: int = 65   # n_fft for noise = 2*(taps-1) = 128
    hidden_size: int = 256
    n_gru_layers: int = 1
    sample_rate: int = 16000
    frame_rate: int = 250
    max_voices: int = 8


class VoiceController(nn.Module):
    """Per-voice control signal -> DDSP parameter mapper.

    Input per frame: [log(f0+1), loudness, active]   (3 channels)
    Output per frame:
        - overall harmonic amplitude (1)
        - harmonic distribution logits (n_harmonics)
        - noise filter magnitude (noise_filter_taps)
    """

    def __init__(self, cfg: DDSPConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = 3
        H = cfg.hidden_size
        self.input_proj = nn.Linear(in_ch, H)
        self.gru = nn.GRU(H, H, num_layers=cfg.n_gru_layers, batch_first=True)
        self.amp_head = nn.Linear(H, 1)
        self.harm_head = nn.Linear(H, cfg.n_harmonics)
        self.noise_head = nn.Linear(H, cfg.noise_filter_taps)

    def forward(self, f0_hz: torch.Tensor, loudness: torch.Tensor, active: torch.Tensor):
        # Combine into per-frame input
        f0_norm = torch.log(f0_hz.clamp(min=1.0) + 1.0) / 9.0  # ~ [0,1] for piano range
        x = torch.stack([f0_norm, loudness, active], dim=-1)   # (B, T, 3)
        h = self.input_proj(x)
        h, _ = self.gru(h)
        amp = _scaled_sigmoid(self.amp_head(h)).squeeze(-1)                # (B, T) in [0,2]
        harm_dist = torch.softmax(self.harm_head(h), dim=-1)               # (B, T, K) sums to 1
        noise_mag = _scaled_sigmoid(self.noise_head(h))                    # (B, T, F) in [0,2]
        # Active gate
        amp = amp * active
        # Don't gate noise magnitude entirely — pianos have small release noise
        return amp, harm_dist, noise_mag


def _scaled_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Modified sigmoid used in DDSP: 2 * sigmoid(x)^log10(e) + 1e-7. Output range (0, 2)."""
    return 2.0 * torch.sigmoid(x) ** math.log10(math.e) + 1e-7


# ---------------------------------------------------------------------------
# Top-level polyphonic synthesizer
# ---------------------------------------------------------------------------

class PolyphonicDDSP(nn.Module):
    def __init__(self, cfg: DDSPConfig | None = None):
        super().__init__()
        cfg = cfg or DDSPConfig()
        self.cfg = cfg
        # All voices share one controller (weight sharing keeps the model small
        # and prevents voice-specific overfitting)
        self.controller = VoiceController(cfg)

    def synthesize_voice(
        self,
        f0_hz: torch.Tensor,
        loudness: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        amp, harm_dist, noise_mag = self.controller(f0_hz, loudness, active)
        harm = harmonic_synth(
            f0_hz=f0_hz,
            amplitudes=amp,
            harmonic_distribution=harm_dist,
            sample_rate=self.cfg.sample_rate,
            frame_rate=self.cfg.frame_rate,
        )
        noise = filtered_noise_synth(
            filter_mag=noise_mag,
            sample_rate=self.cfg.sample_rate,
            frame_rate=self.cfg.frame_rate,
        )
        # Bound noise loudness by the active mask + a small floor so silence stays silent
        n_samples = harm.shape[-1]
        active_smp = _upsample_linear(active, n_samples)
        noise = noise * active_smp
        return harm + noise

    def forward(
        self,
        f0_voices: torch.Tensor,
        loud_voices: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Inputs: (B, V, T). Returns: (B, n_samples)."""
        B, V, T = f0_voices.shape
        out: torch.Tensor | None = None
        for v in range(V):
            wav = self.synthesize_voice(
                f0_hz=f0_voices[:, v],
                loudness=loud_voices[:, v],
                active=active[:, v],
            )
            out = wav if out is None else (out + wav)
        assert out is not None
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Multi-scale STFT loss (trainable, used at training time)
# ---------------------------------------------------------------------------

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128), log_eps: float = 1e-5):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.log_eps = log_eps
        self._windows: dict[int, torch.Tensor] = {}

    def _window(self, n_fft: int, device, dtype) -> torch.Tensor:
        key = (n_fft, str(device), str(dtype))
        if key not in self._windows:
            self._windows[key] = torch.hann_window(n_fft, device=device, dtype=dtype)
        return self._windows[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-1] != target.shape[-1]:
            n = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :n]
            target = target[..., :n]
        loss = pred.new_zeros(())
        for n_fft in self.fft_sizes:
            hop = max(1, n_fft // 4)
            window = self._window(n_fft, pred.device, pred.dtype)
            Sp = torch.stft(
                pred, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True, center=True,
            )
            St = torch.stft(
                target, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True, center=True,
            )
            Mp = Sp.abs()
            Mt = St.abs()
            loss = loss + (Mp - Mt).abs().mean()
            loss = loss + (torch.log(Mp + self.log_eps) - torch.log(Mt + self.log_eps)).abs().mean()
        return loss
