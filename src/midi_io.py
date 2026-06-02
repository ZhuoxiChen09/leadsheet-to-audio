"""MIDI / leadsheet I/O helpers.

We keep the leadsheet representation as a lightweight dataclass so the rest of the
pipeline (tokenizer, dataset, sampling, rendering) is decoupled from the file format.

Two key formats are supported:
  - Nottingham ABC files (folk tunes with chord symbols) for Task 2 training.
  - General MIDI files (e.g. MAESTRO recordings) for Task 4 training and for the
    output rendering of Task 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pretty_midi  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pretty_midi is required; pip install pretty_midi") from e


# ---------- Lightweight leadsheet representation ----------------------------

CHORD_QUALITIES = ("maj", "min", "dim", "aug", "dom7", "maj7", "min7", "sus")
PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class MelodyNote:
    """A single melody note in beats."""

    pitch: int  # midi pitch (0-127), -1 means rest
    start: float  # in beats (quarter note = 1.0)
    duration: float  # in beats

    @property
    def is_rest(self) -> bool:
        return self.pitch < 0


@dataclass(frozen=True)
class ChordSymbol:
    """A chord on a bar, encoded as (root_pc, quality)."""

    root: int  # pitch class 0..11, -1 = N.C. (no chord)
    quality: str  # one of CHORD_QUALITIES, or "NC"
    start: float  # in beats
    duration: float  # in beats

    @property
    def is_no_chord(self) -> bool:
        return self.root < 0 or self.quality == "NC"


@dataclass
class Leadsheet:
    """Melody + per-bar chord symbols."""

    melody: List[MelodyNote] = field(default_factory=list)
    chords: List[ChordSymbol] = field(default_factory=list)
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key: Optional[str] = None  # e.g. "C", "Gm"
    title: Optional[str] = None

    @property
    def beats_per_bar(self) -> float:
        num, den = self.time_signature
        return 4.0 * num / den

    @property
    def n_bars(self) -> int:
        if not self.melody and not self.chords:
            return 0
        end = 0.0
        for n in self.melody:
            end = max(end, n.start + n.duration)
        for c in self.chords:
            end = max(end, c.start + c.duration)
        return int(np.ceil(end / self.beats_per_bar))


# ---------- Chord parsing utilities -----------------------------------------

# Map common chord symbol suffixes to our canonical quality vocabulary.
_QUALITY_MAP = {
    "": "maj",
    "M": "maj",
    "maj": "maj",
    "Maj": "maj",
    "m": "min",
    "min": "min",
    "-": "min",
    "dim": "dim",
    "o": "dim",
    "aug": "aug",
    "+": "aug",
    "7": "dom7",
    "M7": "maj7",
    "maj7": "maj7",
    "Maj7": "maj7",
    "m7": "min7",
    "min7": "min7",
    "-7": "min7",
    "sus": "sus",
    "sus4": "sus",
    "sus2": "sus",
}

_ROOT_TO_PC = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def parse_chord_symbol(text: str) -> Optional[ChordSymbol]:
    """Parse a raw chord string like ``Gm7`` or ``F#`` or ``Bb`` or ``N.C.``.

    Returns ``None`` for the empty string and a ``ChordSymbol`` with start=0 dur=0
    otherwise (callers fill in timing).
    """
    s = text.strip()
    if not s or s.lower() in {"n.c.", "nc", "none", "rest"}:
        return ChordSymbol(root=-1, quality="NC", start=0.0, duration=0.0)
    root_char = s[0].upper()
    if root_char not in _ROOT_TO_PC:
        return None
    pc = _ROOT_TO_PC[root_char]
    i = 1
    if i < len(s) and s[i] in ("#", "b"):
        pc = (pc + (1 if s[i] == "#" else -1)) % 12
        i += 1
    quality_str = s[i:].split("/")[0]  # drop slash bass for now
    quality = _QUALITY_MAP.get(quality_str)
    if quality is None:
        # try a couple of forgiving fallbacks
        if quality_str.startswith("maj"):
            quality = "maj7" if "7" in quality_str else "maj"
        elif quality_str.startswith("min") or quality_str.startswith("m"):
            quality = "min7" if "7" in quality_str else "min"
        elif "7" in quality_str:
            quality = "dom7"
        else:
            return None
    return ChordSymbol(root=pc, quality=quality, start=0.0, duration=0.0)


def chord_pitch_set(chord: ChordSymbol) -> set:
    """Pitch classes that belong to a chord. Used by rule-based baseline & metrics."""
    if chord.is_no_chord:
        return set()
    intervals = {
        "maj": (0, 4, 7),
        "min": (0, 3, 7),
        "dim": (0, 3, 6),
        "aug": (0, 4, 8),
        "dom7": (0, 4, 7, 10),
        "maj7": (0, 4, 7, 11),
        "min7": (0, 3, 7, 10),
        "sus": (0, 5, 7),
    }[chord.quality]
    return {(chord.root + i) % 12 for i in intervals}


def chord_to_text(chord: ChordSymbol) -> str:
    """Render a ChordSymbol back to a short human-readable string."""
    if chord.is_no_chord:
        return "N.C."
    suffix = {
        "maj": "", "min": "m", "dim": "dim", "aug": "aug",
        "dom7": "7", "maj7": "maj7", "min7": "m7", "sus": "sus",
    }[chord.quality]
    return f"{PITCH_CLASS_NAMES[chord.root]}{suffix}"


# ---------- Nottingham ABC parsing -----------------------------------------

_ABC_PITCH_TABLE = {
    "C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71,
    "c": 72, "d": 74, "e": 76, "f": 77, "g": 79, "a": 81, "b": 83,
}


def _parse_abc_unit_length(header_value: str) -> float:
    try:
        num, den = header_value.strip().split("/")
        return float(num) / float(den)
    except Exception:
        return 1 / 8


def _parse_abc_meter(header_value: str) -> Tuple[int, int]:
    v = header_value.strip()
    if v.upper() == "C":
        return (4, 4)
    if v.upper() == "C|":
        return (2, 2)
    try:
        num, den = v.split("/")
        return (int(num), int(den))
    except Exception:
        return (4, 4)


def parse_nottingham_abc(text: str) -> List[Leadsheet]:
    """Parse a Nottingham ABC file (which contains many tunes separated by blank lines).

    The parser is intentionally permissive — it skips any tune it can't fully
    parse instead of raising. We only extract melody (one note at a time) and
    bar-level chord symbols. This is enough for the harmonization task.
    """
    tunes: List[Leadsheet] = []
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("X:") and current_lines:
            tune = _parse_single_abc_tune(current_lines)
            if tune is not None:
                tunes.append(tune)
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        tune = _parse_single_abc_tune(current_lines)
        if tune is not None:
            tunes.append(tune)
    return tunes


def _parse_single_abc_tune(lines: Sequence[str]) -> Optional[Leadsheet]:
    headers = {}
    body_lines: List[str] = []
    in_body = False
    for line in lines:
        if not in_body:
            if len(line) >= 2 and line[1] == ":":
                key = line[0]
                value = line[2:].strip()
                headers[key] = value
                if key == "K":
                    in_body = True
                continue
            # skip pre-K stray lines
            continue
        body_lines.append(line)

    if "K" not in headers:
        return None
    unit_length = _parse_abc_unit_length(headers.get("L", "1/8"))
    meter = _parse_abc_meter(headers.get("M", "4/4"))
    tempo = 120.0
    if "Q" in headers:
        q = headers["Q"]
        try:
            if "=" in q:
                tempo = float(q.split("=")[-1])
            else:
                tempo = float(q)
        except Exception:
            tempo = 120.0
    title = headers.get("T")

    beats_per_bar = 4.0 * meter[0] / meter[1]

    melody: List[MelodyNote] = []
    chords: List[ChordSymbol] = []
    cur_beat = 0.0
    bar_start = 0.0
    bar_chord: Optional[ChordSymbol] = None

    body = " ".join(body_lines)
    body = _strip_abc_decorations(body)

    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '"':
            # chord symbol
            end = body.find('"', i + 1)
            if end < 0:
                break
            sym_text = body[i + 1: end]
            sym = parse_chord_symbol(sym_text)
            if sym is not None:
                bar_chord = ChordSymbol(
                    root=sym.root,
                    quality=sym.quality,
                    start=bar_start,
                    duration=beats_per_bar,
                )
                chords.append(bar_chord)
            i = end + 1
            continue
        if ch in ("|", ":", "[", "]"):
            # bar line / repeat
            if cur_beat - bar_start > 1e-6:
                bar_start += beats_per_bar
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if ch == "z" or ch == "Z":
            length, nxt = _read_length(body, i + 1)
            beats = length * unit_length * 4  # length in beats (quarter = 1)
            melody.append(MelodyNote(pitch=-1, start=cur_beat, duration=beats))
            cur_beat += beats
            i = nxt
            continue
        if ch in _ABC_PITCH_TABLE or ch == "^" or ch == "_" or ch == "=":
            accidental = 0
            while body[i] in ("^", "_", "="):
                if body[i] == "^":
                    accidental += 1
                elif body[i] == "_":
                    accidental -= 1
                i += 1
                if i >= len(body):
                    break
            if i >= len(body):
                break
            pitch_char = body[i]
            if pitch_char not in _ABC_PITCH_TABLE:
                i += 1
                continue
            pitch = _ABC_PITCH_TABLE[pitch_char] + accidental
            i += 1
            # octave modifiers ' (up) and , (down)
            while i < len(body) and body[i] in ("'", ","):
                if body[i] == "'":
                    pitch += 12
                else:
                    pitch -= 12
                i += 1
            length, nxt = _read_length(body, i)
            beats = length * unit_length * 4
            melody.append(MelodyNote(pitch=pitch, start=cur_beat, duration=beats))
            cur_beat += beats
            i = nxt
            continue
        # anything else, skip
        i += 1

    if not melody or not chords:
        return None

    return Leadsheet(
        melody=melody,
        chords=chords,
        tempo=tempo,
        time_signature=meter,
        title=title,
        key=headers.get("K"),
    )


def _strip_abc_decorations(s: str) -> str:
    out_chars = []
    skip = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "%":  # comment to end of line; but we joined lines into one
            break
        if c == "!":
            end = s.find("!", i + 1)
            if end < 0:
                break
            i = end + 1
            continue
        if c == "(" and i + 1 < len(s) and s[i + 1].isdigit():
            # tuplet specifier like (3 — skip but DO NOT alter timing precisely;
            # the parser uses nominal lengths which is fine for our coarse use.
            i += 2
            continue
        if c == "{" or c == "}":  # grace notes
            end = s.find("}", i + 1)
            if end > 0:
                i = end + 1
            else:
                i += 1
            continue
        out_chars.append(c)
        i += 1
    return "".join(out_chars)


def _read_length(s: str, idx: int) -> Tuple[float, int]:
    """Read an ABC note length suffix starting at idx; returns (length, next_index)."""
    num_str = ""
    while idx < len(s) and s[idx].isdigit():
        num_str += s[idx]
        idx += 1
    num = int(num_str) if num_str else 1
    den = 1
    if idx < len(s) and s[idx] == "/":
        idx += 1
        den_str = ""
        while idx < len(s) and s[idx].isdigit():
            den_str += s[idx]
            idx += 1
        den = int(den_str) if den_str else 2
    return num / den, idx


# ---------- Leadsheet -> MIDI rendering -------------------------------------

def leadsheet_to_pretty_midi(
    sheet: Leadsheet,
    *,
    melody_program: int = 0,  # acoustic grand piano
    chord_program: int = 0,
    chord_octave: int = 4,
    melody_velocity: int = 96,
    chord_velocity: int = 72,
    arpeggiate: bool = False,
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=sheet.tempo)
    sec_per_beat = 60.0 / sheet.tempo

    melody_inst = pretty_midi.Instrument(program=melody_program, name="melody")
    chord_inst = pretty_midi.Instrument(program=chord_program, name="chords")

    for n in sheet.melody:
        if n.is_rest:
            continue
        start = n.start * sec_per_beat
        end = (n.start + n.duration) * sec_per_beat
        melody_inst.notes.append(
            pretty_midi.Note(velocity=melody_velocity, pitch=int(n.pitch), start=start, end=end)
        )

    for c in sheet.chords:
        if c.is_no_chord:
            continue
        pcs = sorted(chord_pitch_set(c))
        # voice them around chord_octave
        base_midi = 12 * chord_octave  # C{octave}
        chord_notes = [base_midi + pc for pc in pcs]
        start_b = c.start
        dur_b = c.duration
        if arpeggiate:
            step = dur_b / max(len(chord_notes), 1)
            for i, p in enumerate(chord_notes):
                s = (start_b + i * step) * sec_per_beat
                e = (start_b + dur_b) * sec_per_beat
                chord_inst.notes.append(
                    pretty_midi.Note(velocity=chord_velocity, pitch=p, start=s, end=e)
                )
        else:
            s = start_b * sec_per_beat
            e = (start_b + dur_b) * sec_per_beat
            for p in chord_notes:
                chord_inst.notes.append(
                    pretty_midi.Note(velocity=chord_velocity, pitch=p, start=s, end=e)
                )

    pm.instruments.append(melody_inst)
    pm.instruments.append(chord_inst)
    return pm


def save_leadsheet_midi(sheet: Leadsheet, path: str | Path, **kwargs) -> None:
    pm = leadsheet_to_pretty_midi(sheet, **kwargs)
    pm.write(str(path))


# ---------- MIDI -> note grid (used by Task 4) ------------------------------

def midi_to_note_events(path: str | Path) -> List[Tuple[float, float, int, int]]:
    """Read a MIDI file into a flat list of (start_sec, end_sec, pitch, velocity)
    note events. Pitched percussion is excluded.
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    out: List[Tuple[float, float, int, int]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            out.append((n.start, n.end, int(n.pitch), int(n.velocity)))
    out.sort()
    return out


def pretty_midi_to_note_events(pm: "pretty_midi.PrettyMIDI") -> List[Tuple[float, float, int, int]]:
    out: List[Tuple[float, float, int, int]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            out.append((n.start, n.end, int(n.pitch), int(n.velocity)))
    out.sort()
    return out


def midi_pitch_to_hz(pitch: int) -> float:
    """Equal-temperament frequency for a MIDI pitch."""
    return float(440.0 * (2.0 ** ((pitch - 69) / 12.0)))
