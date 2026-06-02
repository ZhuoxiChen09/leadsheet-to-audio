"""Trivial / non-learned baselines for Tasks 2 and 4.

Every baseline has the same interface as the learned model so we can swap them
in and out at evaluation time. For Task 2 each baseline maps an encoded melody
to a list of chord token ids (one per bar). For Task 4 each baseline maps the
per-voice (F0, loudness, active) tensors to a synthesized waveform.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .midi_io import ChordSymbol, chord_pitch_set
from .tokenizer import (
    CHORD_VOCAB,
    GRID_PER_BEAT,
    MELODY_ID_TO_TOK,
    MELODY_VOCAB,
    chord_from_token_id,
    chord_token_id,
)


# ---------------------------------------------------------------------------
# Task 2: chord harmonization baselines
# ---------------------------------------------------------------------------

def _melody_pcs_per_bar(melody_ids: Sequence[int], beats_per_bar: float = 4.0) -> List[Counter]:
    grid_per_bar = int(round(beats_per_bar * GRID_PER_BEAT))
    bars: List[Counter] = []
    cur_pitch: int | None = None
    for idx, tid in enumerate(melody_ids):
        bar = idx // grid_per_bar
        while len(bars) <= bar:
            bars.append(Counter())
        tok = MELODY_ID_TO_TOK.get(int(tid))
        if tok is None:
            continue
        if tok == "MEL_REST":
            cur_pitch = None
        elif tok == "MEL_HOLD":
            if cur_pitch is not None:
                bars[bar][cur_pitch % 12] += 1
        elif tok.startswith("MEL_PITCH_"):
            cur_pitch = int(tok.split("_")[-1])
            bars[bar][cur_pitch % 12] += 1
    return bars


def baseline_always_C(n_bars: int) -> List[int]:
    cid = CHORD_VOCAB["CHORD_0_maj"]
    return [cid] * n_bars


def baseline_I_IV_V_I(n_bars: int, key_root_pc: int = 0) -> List[int]:
    seq = []
    pattern = [
        ("maj", 0),       # I
        ("maj", 5),       # IV
        ("maj", 7),       # V
        ("maj", 0),       # I
    ]
    for i in range(n_bars):
        quality, offset = pattern[i % 4]
        root = (key_root_pc + offset) % 12
        seq.append(CHORD_VOCAB[f"CHORD_{root}_{quality}"])
    return seq


def baseline_most_common(n_bars: int, train_chord_counts: Counter) -> List[int]:
    most = train_chord_counts.most_common(1)
    if not most:
        return [CHORD_VOCAB["CHORD_0_maj"]] * n_bars
    return [most[0][0]] * n_bars


class BigramChordBaseline:
    """Conditional chord_t | chord_{t-1} bigram model, melody-independent."""

    def __init__(self):
        self.counts: dict[int, Counter] = {}
        self.unigram: Counter = Counter()

    def fit(self, chord_sequences: Iterable[Sequence[int]]):
        for seq in chord_sequences:
            prev = None
            for c in seq:
                self.unigram[c] += 1
                if prev is not None:
                    self.counts.setdefault(prev, Counter())[c] += 1
                prev = c

    def predict(self, n_bars: int, start: int | None = None) -> List[int]:
        if not self.unigram:
            return [CHORD_VOCAB["CHORD_0_maj"]] * n_bars
        if start is None:
            start = self.unigram.most_common(1)[0][0]
        out = [start]
        for _ in range(1, n_bars):
            ctx = self.counts.get(out[-1])
            if ctx and len(ctx) > 0:
                out.append(ctx.most_common(1)[0][0])
            else:
                out.append(self.unigram.most_common(1)[0][0])
        return out


def baseline_rule_based(melody_ids: Sequence[int], beats_per_bar: float = 4.0) -> List[int]:
    """For each bar, pick the chord whose pitch-class set best matches the bar's
    melody pitch histogram. The candidate vocabulary covers maj/min/dom7 across
    all 12 roots — that is the most musically useful subset of our tokens.
    """
    bar_pcs = _melody_pcs_per_bar(melody_ids, beats_per_bar=beats_per_bar)
    candidates: List[Tuple[int, set]] = []
    for root in range(12):
        for q in ("maj", "min", "dom7"):
            cid = CHORD_VOCAB[f"CHORD_{root}_{q}"]
            sym = ChordSymbol(root=root, quality=q, start=0, duration=0)
            candidates.append((cid, chord_pitch_set(sym)))
    out: List[int] = []
    for hist in bar_pcs:
        if not hist:
            out.append(CHORD_VOCAB["CHORD_NC"])
            continue
        total = sum(hist.values()) + 1e-9
        best_cid = candidates[0][0]
        best_score = -1.0
        for cid, pcs in candidates:
            in_set = sum(c for pc, c in hist.items() if pc in pcs)
            outside = sum(c for pc, c in hist.items() if pc not in pcs)
            score = in_set / total - 0.25 * (outside / total)
            if score > best_score:
                best_score = score
                best_cid = cid
        out.append(best_cid)
    return out


# ---------------------------------------------------------------------------
# Task 4: synthesis baselines
# ---------------------------------------------------------------------------

def synth_sine_voices(
    f0_voices: np.ndarray,
    loud_voices: np.ndarray,
    sample_rate: int,
    frame_rate: int,
) -> np.ndarray:
    """Pure-sine baseline: each voice = sin(2 pi f t) * envelope."""
    return _voice_synth(f0_voices, loud_voices, sample_rate, frame_rate, kind="sine")


def synth_triangle_voices(
    f0_voices: np.ndarray,
    loud_voices: np.ndarray,
    sample_rate: int,
    frame_rate: int,
) -> np.ndarray:
    return _voice_synth(f0_voices, loud_voices, sample_rate, frame_rate, kind="triangle")


def _voice_synth(
    f0_voices: np.ndarray,
    loud_voices: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    kind: str = "sine",
) -> np.ndarray:
    n_voices, n_frames = f0_voices.shape
    n_samples = int(n_frames * sample_rate / frame_rate)
    out = np.zeros(n_samples, dtype=np.float32)
    samples_per_frame = sample_rate / frame_rate
    for v in range(n_voices):
        f0_s = _upsample_linear(f0_voices[v], n_samples)
        amp_s = _upsample_linear(loud_voices[v], n_samples)
        # numerical-stable phase integration
        phase = 2.0 * np.pi * np.cumsum(f0_s / sample_rate)
        if kind == "sine":
            wf = np.sin(phase)
        elif kind == "triangle":
            wf = (2.0 / np.pi) * np.arcsin(np.sin(phase))
        else:
            raise ValueError(kind)
        out += amp_s * wf.astype(np.float32)
    peak = np.max(np.abs(out)) + 1e-9
    out = (out / peak * 0.9).astype(np.float32)
    return out


def _upsample_linear(x: np.ndarray, target_length: int) -> np.ndarray:
    if x.shape[0] == target_length:
        return x.astype(np.float32)
    src_idx = np.linspace(0, x.shape[0] - 1, target_length, dtype=np.float32)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, x.shape[0] - 1)
    w = (src_idx - lo).astype(np.float32)
    return ((1.0 - w) * x[lo] + w * x[hi]).astype(np.float32)
