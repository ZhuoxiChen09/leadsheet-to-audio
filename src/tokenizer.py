"""Token vocabularies for Task 2 (melody -> chord harmonization).

The melody is encoded as one token per 16th-note grid cell:
  * MEL_HOLD            - sustain the previous pitch
  * MEL_REST            - this 16th is a rest
  * MEL_PITCH_{p}       - new note onset at midi pitch p (offset clipped to 36..96)

The chord stream is encoded as one token per bar (a 4/4 bar = 16 grid cells):
  * CHORD_{root}_{qual}     - explicit chord
  * CHORD_NC                - no chord
  * BOS_C / EOS_C           - sequence delimiters

We also keep a parallel ``bar_index`` array per training example so the decoder
emits one chord per bar; this avoids the need for a separate cross-attention
positional alignment loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .midi_io import (
    CHORD_QUALITIES,
    ChordSymbol,
    Leadsheet,
    MelodyNote,
)

GRID_PER_BEAT = 4  # 16th notes
MELODY_PITCH_LOW = 48  # C3
MELODY_PITCH_HIGH = 84  # C6 inclusive => 84-48+1=37 pitches


# ---------- Build vocabularies ----------------------------------------------

def _build_melody_vocab() -> Dict[str, int]:
    v: Dict[str, int] = {}
    v["<PAD>"] = 0
    v["<BOS>"] = 1
    v["<EOS>"] = 2
    v["MEL_HOLD"] = 3
    v["MEL_REST"] = 4
    for p in range(MELODY_PITCH_LOW, MELODY_PITCH_HIGH + 1):
        v[f"MEL_PITCH_{p}"] = len(v)
    return v


def _build_chord_vocab() -> Dict[str, int]:
    v: Dict[str, int] = {}
    v["<PAD>"] = 0
    v["<BOS>"] = 1
    v["<EOS>"] = 2
    v["CHORD_NC"] = 3
    for root in range(12):
        for q in CHORD_QUALITIES:
            v[f"CHORD_{root}_{q}"] = len(v)
    return v


MELODY_VOCAB: Dict[str, int] = _build_melody_vocab()
CHORD_VOCAB: Dict[str, int] = _build_chord_vocab()

MEL_PAD_ID = MELODY_VOCAB["<PAD>"]
MEL_BOS_ID = MELODY_VOCAB["<BOS>"]
MEL_EOS_ID = MELODY_VOCAB["<EOS>"]

CHORD_PAD_ID = CHORD_VOCAB["<PAD>"]
CHORD_BOS_ID = CHORD_VOCAB["<BOS>"]
CHORD_EOS_ID = CHORD_VOCAB["<EOS>"]


def melody_vocab_size() -> int:
    return len(MELODY_VOCAB)


def chord_vocab_size() -> int:
    return len(CHORD_VOCAB)


# Pre-compute reverse maps
MELODY_ID_TO_TOK = {i: t for t, i in MELODY_VOCAB.items()}
CHORD_ID_TO_TOK = {i: t for t, i in CHORD_VOCAB.items()}


def chord_token_id(chord: ChordSymbol) -> int:
    if chord.is_no_chord:
        return CHORD_VOCAB["CHORD_NC"]
    return CHORD_VOCAB[f"CHORD_{chord.root}_{chord.quality}"]


def chord_from_token_id(tok_id: int) -> ChordSymbol:
    tok = CHORD_ID_TO_TOK[tok_id]
    if tok == "CHORD_NC":
        return ChordSymbol(root=-1, quality="NC", start=0.0, duration=0.0)
    parts = tok.split("_")
    root = int(parts[1])
    quality = parts[2]
    return ChordSymbol(root=root, quality=quality, start=0.0, duration=0.0)


# ---------- Encoding a Leadsheet ------------------------------------------

@dataclass
class EncodedExample:
    melody_ids: List[int]
    chord_ids: List[int]  # one per bar, NOT including BOS/EOS
    n_bars: int


def _quantize(beat: float) -> int:
    return int(round(beat * GRID_PER_BEAT))


def encode_leadsheet(
    sheet: Leadsheet,
    *,
    max_bars: int = 32,
    transpose_to_c: bool = False,
) -> Optional[EncodedExample]:
    """Encode a Leadsheet into parallel melody-grid and chord-per-bar id streams.

    Returns ``None`` if the sheet is too short or has no chord coverage.
    """
    beats_per_bar = sheet.beats_per_bar
    grid_per_bar = int(round(beats_per_bar * GRID_PER_BEAT))
    if grid_per_bar <= 0:
        return None

    n_bars = sheet.n_bars
    if n_bars <= 0:
        return None
    n_bars = min(n_bars, max_bars)
    total_grid = n_bars * grid_per_bar

    # ---- Transposition for data augmentation / normalization ----
    shift = 0
    if transpose_to_c and sheet.key:
        key_root = _parse_key_root(sheet.key)
        if key_root is not None:
            # Move melody so tonic is C (pc 0); modulo octave
            shift = -key_root % 12
            if shift > 6:
                shift -= 12

    # ---- Build melody grid ----
    mel_grid: List[int] = [MELODY_VOCAB["MEL_REST"]] * total_grid
    for n in sheet.melody:
        if n.is_rest:
            continue
        pitch = int(n.pitch) + shift
        # snap pitch into the allowed range by transposing octaves
        while pitch < MELODY_PITCH_LOW:
            pitch += 12
        while pitch > MELODY_PITCH_HIGH:
            pitch -= 12
        start_idx = _quantize(n.start)
        end_idx = _quantize(n.start + n.duration)
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        if start_idx >= total_grid:
            break
        start_idx = max(0, start_idx)
        end_idx = min(total_grid, end_idx)
        mel_grid[start_idx] = MELODY_VOCAB[f"MEL_PITCH_{pitch}"]
        for k in range(start_idx + 1, end_idx):
            mel_grid[k] = MELODY_VOCAB["MEL_HOLD"]

    # ---- Build chord-per-bar list ----
    chord_per_bar: List[int] = [CHORD_VOCAB["CHORD_NC"]] * n_bars
    for c in sheet.chords:
        if c.is_no_chord:
            continue
        bar_idx = int(c.start // beats_per_bar)
        if bar_idx < 0 or bar_idx >= n_bars:
            continue
        root_pc = (c.root + shift) % 12 if c.root >= 0 else -1
        if root_pc < 0:
            chord_per_bar[bar_idx] = CHORD_VOCAB["CHORD_NC"]
        else:
            chord_per_bar[bar_idx] = CHORD_VOCAB[f"CHORD_{root_pc}_{c.quality}"]

    return EncodedExample(
        melody_ids=mel_grid,
        chord_ids=chord_per_bar,
        n_bars=n_bars,
    )


def _parse_key_root(key_text: str) -> Optional[int]:
    s = key_text.strip()
    if not s:
        return None
    head = s[0].upper()
    if head not in "ABCDEFG":
        return None
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[head]
    if len(s) > 1 and s[1] in ("#", "b"):
        pc = (pc + (1 if s[1] == "#" else -1)) % 12
    return pc


# ---------- Decoding back to a Leadsheet ----------------------------------

def decode_to_leadsheet(
    melody_ids: Sequence[int],
    chord_ids: Sequence[int],
    *,
    tempo: float = 120.0,
    beats_per_bar: float = 4.0,
) -> Leadsheet:
    melody: List[MelodyNote] = []
    cur_pitch: Optional[int] = None
    cur_start = 0.0
    cur_dur = 0.0
    for idx, tid in enumerate(melody_ids):
        tok = MELODY_ID_TO_TOK.get(int(tid))
        if tok is None:
            continue
        beat = idx / GRID_PER_BEAT
        if tok == "MEL_HOLD":
            if cur_pitch is not None:
                cur_dur += 1.0 / GRID_PER_BEAT
            continue
        # flush previous note
        if cur_pitch is not None:
            melody.append(MelodyNote(pitch=cur_pitch, start=cur_start, duration=cur_dur))
            cur_pitch = None
        if tok == "MEL_REST":
            melody.append(MelodyNote(pitch=-1, start=beat, duration=1.0 / GRID_PER_BEAT))
            continue
        if tok.startswith("MEL_PITCH_"):
            cur_pitch = int(tok.split("_")[-1])
            cur_start = beat
            cur_dur = 1.0 / GRID_PER_BEAT
    if cur_pitch is not None:
        melody.append(MelodyNote(pitch=cur_pitch, start=cur_start, duration=cur_dur))

    chords: List[ChordSymbol] = []
    for bar_idx, cid in enumerate(chord_ids):
        c = chord_from_token_id(int(cid))
        chords.append(
            ChordSymbol(
                root=c.root,
                quality=c.quality,
                start=bar_idx * beats_per_bar,
                duration=beats_per_bar,
            )
        )

    return Leadsheet(
        melody=melody,
        chords=chords,
        tempo=tempo,
        time_signature=(int(beats_per_bar), 4),
    )
