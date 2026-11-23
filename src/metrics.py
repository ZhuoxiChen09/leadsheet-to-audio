"""Evaluation metrics for both tasks.

Task 2:
    - top-k chord accuracy
    - note-in-chord rate
    - root-motion smoothness
    - chord-transition KL vs. training distribution

Task 4:
    - multi-scale STFT distance (the training loss itself)
    - mel-cepstral distortion (MCD)
    - F0 accuracy of resynthesis (extracted F0 vs. MIDI F0)
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore

from .midi_io import chord_pitch_set
from .tokenizer import (
    CHORD_VOCAB,
    GRID_PER_BEAT,
    MELODY_ID_TO_TOK,
    chord_from_token_id,
)

# ---------------------------------------------------------------------------
# Task 2 metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(
    pred_logits_or_ids: np.ndarray,
    targets: np.ndarray,
    *,
    k: int = 1,
    ignore_index: int = 0,
) -> float:
    """Compute top-k accuracy.

    ``pred_logits_or_ids`` may be either logits (shape (N, V)) or hard ids (shape (N,)).
    """
    targets = np.asarray(targets)
    if pred_logits_or_ids.ndim == 1:
        if k != 1:
            raise ValueError("top-k>1 requires logits, not hard ids")
        mask = targets != ignore_index
        return float((pred_logits_or_ids[mask] == targets[mask]).mean()) if mask.any() else 0.0
    logits = pred_logits_or_ids
    topk = np.argsort(-logits, axis=-1)[:, :k]
    correct = (topk == targets[:, None]).any(axis=-1)
    mask = targets != ignore_index
    return float(correct[mask].mean()) if mask.any() else 0.0


def note_in_chord_rate(
    melody_ids: Sequence[int],
    chord_ids: Sequence[int],
    *,
    beats_per_bar: float = 4.0,
) -> float:
    """Fraction of melody note onsets that fall inside the bar's predicted chord set."""
    grid_per_bar = int(round(beats_per_bar * GRID_PER_BEAT))
    total = 0
    inside = 0
    for idx, tid in enumerate(melody_ids):
        tok = MELODY_ID_TO_TOK.get(int(tid))
        if tok is None or not tok.startswith("MEL_PITCH_"):
            continue
        pitch = int(tok.split("_")[-1])
        bar = idx // grid_per_bar
        if bar >= len(chord_ids):
            break
        sym = chord_from_token_id(int(chord_ids[bar]))
        pcs = chord_pitch_set(sym)
        total += 1
        if (pitch % 12) in pcs:
            inside += 1
    if total == 0:
        return 0.0
    return inside / total


def root_motion_stats(chord_ids: Sequence[int]) -> dict:
    """Average and stddev of absolute root motion in semitones (mod 12 minimum)."""
    motions = []
    prev_root = None
    for cid in chord_ids:
        sym = chord_from_token_id(int(cid))
        if sym.is_no_chord:
            continue
        if prev_root is not None:
            d = (sym.root - prev_root) % 12
            motions.append(min(d, 12 - d))
        prev_root = sym.root
    if not motions:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {"mean": float(np.mean(motions)), "std": float(np.std(motions)), "n": len(motions)}


def chord_distribution(chord_id_lists: Iterable[Sequence[int]]) -> np.ndarray:
    """Normalized histogram over chord vocab indices."""
    counts = Counter()
    for seq in chord_id_lists:
        for cid in seq:
            counts[int(cid)] += 1
    v = np.zeros(len(CHORD_VOCAB), dtype=np.float64)
    total = sum(counts.values()) + 1e-9
    for cid, cnt in counts.items():
        v[cid] = cnt / total
    return v


def chord_transition_kl(
    pred_sequences: Iterable[Sequence[int]],
    ref_sequences: Iterable[Sequence[int]],
    *,
    eps: float = 1e-6,
) -> float:
    """KL(pred || ref) over chord-to-chord transition distributions."""
    def bigram(seqs):
        m = np.zeros((len(CHORD_VOCAB), len(CHORD_VOCAB)), dtype=np.float64)
        for seq in seqs:
            for a, b in zip(seq[:-1], seq[1:], strict=False):
                m[int(a), int(b)] += 1
        m += eps
        m /= m.sum(axis=1, keepdims=True)
        return m
    p = bigram(pred_sequences)
    q = bigram(ref_sequences)
    kl = (p * (np.log(p) - np.log(q))).sum(axis=1)
    # weight by source-state frequency
    p_state = p.sum(axis=1)
    p_state = p_state / (p_state.sum() + eps)
    return float((p_state * kl).sum())


# ---------------------------------------------------------------------------
# Task 4 metrics
# ---------------------------------------------------------------------------

def multi_scale_stft_distance(
    pred: np.ndarray,
    target: np.ndarray,
    fft_sizes: Sequence[int] = (2048, 1024, 512, 256, 128),
    log_eps: float = 1e-5,
) -> float:
    """Sum over scales of L1(|STFT pred| - |STFT target|) + L1(log magnitudes)."""
    if librosa is None:
        raise RuntimeError("librosa required for STFT metrics")
    n = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :n].astype(np.float32)
    target = target[..., :n].astype(np.float32)
    loss = 0.0
    for n_fft in fft_sizes:
        hop = max(1, n_fft // 4)
        Sp = np.abs(librosa.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True))
        St = np.abs(
            librosa.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True)
        )
        loss += float(np.mean(np.abs(Sp - St)))
        loss += float(np.mean(np.abs(np.log(Sp + log_eps) - np.log(St + log_eps))))
    return loss


def mel_cepstral_distortion(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Average MCD across frames (excluding 0th coefficient)."""
    if librosa is None:
        raise RuntimeError("librosa required for MCD")
    n = min(pred.shape[-1], target.shape[-1])
    mfcc_p = librosa.feature.mfcc(
        y=pred[..., :n].astype(np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mfcc_t = librosa.feature.mfcc(
        y=target[..., :n].astype(np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    n_frames = min(mfcc_p.shape[-1], mfcc_t.shape[-1])
    diff = mfcc_p[1:, :n_frames] - mfcc_t[1:, :n_frames]
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    return float(mcd)


def f0_accuracy(
    pred: np.ndarray,
    target_f0_hz: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    tol_cents: float = 50.0,
) -> float:
    """Fraction of frames where pyin(pred) matches the target F0 within ``tol_cents``.

    ``target_f0_hz`` may be the mixed/maximum-velocity reference F0 (e.g. the
    highest voice from the MIDI control signals). Frames with zero target F0
    are excluded from the average.
    """
    from .features import estimate_f0

    pred_f0, _ = estimate_f0(pred, sample_rate=sample_rate, frame_rate=frame_rate)
    n = min(pred_f0.shape[0], target_f0_hz.shape[0])
    pred_f0 = pred_f0[:n]
    tgt = target_f0_hz[:n]
    mask = (tgt > 1.0) & (pred_f0 > 1.0)
    if mask.sum() == 0:
        return 0.0
    cents = 1200.0 * np.log2(pred_f0[mask] / tgt[mask])
    return float(np.mean(np.abs(cents) < tol_cents))
