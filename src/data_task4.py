"""Dataset for Task 4 (MIDI -> audio).

For each training clip we sample a fixed-length window of MIDI notes, render
the ground-truth audio with our teacher synth, and compute the per-voice
(F0, loudness, active) control signals fed to the DDSP model.

Two data sources are supported:
    * **synthesized-from-MIDI**: take an arbitrary directory of .mid / .midi /
      .abc files (e.g. the Nottingham dataset) and render their audio on the
      fly with ``src.teacher_synth``. Default, no external audio needed.
    * **paired** (legacy): if you happen to have a directory of paired .wav and
      .midi files (e.g. a manually-downloaded MAESTRO clip), the original
      paired loader is still available via ``discover_local_pairs`` and the
      ``MaestroClip`` dataclass.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("librosa + soundfile are required") from e

from .features import FRAME_RATE, SAMPLE_RATE, midi_notes_to_frame_controls
from .midi_io import (
    leadsheet_to_pretty_midi,
    midi_to_note_events,
    parse_nottingham_abc,
    pretty_midi_to_note_events,
)
from .teacher_synth import render_notes

# ---------------------------------------------------------------------------

@dataclass
class MaestroClip:
    audio_path: Path
    midi_path: Path
    duration: float


def parse_maestro_csv(csv_path: str | Path) -> List[MaestroClip]:
    """Parse a MAESTRO v3.0.0 metadata CSV into a list of clips."""
    csv_path = Path(csv_path)
    base = csv_path.parent
    clips: List[MaestroClip] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio = base / row.get("audio_filename", "")
            midi = base / row.get("midi_filename", "")
            try:
                dur = float(row.get("duration", 0.0))
            except ValueError:
                dur = 0.0
            if not audio.exists() or not midi.exists():
                continue
            clips.append(MaestroClip(audio_path=audio, midi_path=midi, duration=dur))
    return clips


def discover_local_pairs(root: str | Path) -> List[MaestroClip]:
    """Fallback if no CSV is available: pair MIDI and wav/flac files by basename."""
    root = Path(root)
    audio_files = {p.stem: p for p in list(root.rglob("*.wav")) + list(root.rglob("*.flac"))}
    midi_files = {p.stem: p for p in root.rglob("*.mid*")}
    clips: List[MaestroClip] = []
    for stem in sorted(set(audio_files) & set(midi_files)):
        audio = audio_files[stem]
        midi = midi_files[stem]
        try:
            info = sf.info(str(audio))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        clips.append(MaestroClip(audio_path=audio, midi_path=midi, duration=dur))
    return clips


# ---------------------------------------------------------------------------

@dataclass
class DDSPTrainExample:
    audio: np.ndarray  # (n_samples,) float32, target waveform
    f0_voices: np.ndarray  # (max_voices, T) float32
    loud_voices: np.ndarray  # (max_voices, T) float32
    active: np.ndarray  # (max_voices, T) float32


def load_clip_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def slice_example(
    audio: np.ndarray,
    notes: Sequence[Tuple[float, float, int, int]],
    *,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
    max_voices: int = 8,
) -> DDSPTrainExample:
    s_smp = int(start_sec * sample_rate)
    n_smp = int(duration_sec * sample_rate)
    y = audio[s_smp: s_smp + n_smp]
    if y.shape[0] < n_smp:
        y = np.pad(y, (0, n_smp - y.shape[0]))
    # Shift overlapping notes to the local timeline
    local_notes = []
    end_sec = start_sec + duration_sec
    for (s, e, p, v) in notes:
        if e <= start_sec or s >= end_sec:
            continue
        local_notes.append(
            (
                max(0.0, s - start_sec),
                min(duration_sec, e - start_sec),
                int(p),
                int(v),
            )
        )
    f0v, loudv, act = midi_notes_to_frame_controls(
        local_notes,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        max_voices=max_voices,
    )
    return DDSPTrainExample(
        audio=y.astype(np.float32), f0_voices=f0v, loud_voices=loudv, active=act
    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synthesized-from-MIDI source (default for Task 4)
# ---------------------------------------------------------------------------

@dataclass
class MidiSource:
    """A MIDI-only training source. Audio is generated on the fly by the teacher synth."""

    midi_path: Path
    notes: list  # list[(start_sec, end_sec, pitch, velocity)]
    duration: float


def load_midi_sources_from_abc(
    abc_root: str | Path, max_files: int | None = None
) -> list[MidiSource]:
    """Convert every ABC tune under ``abc_root`` into MidiSources.

    Each tune is rendered as a melody-only pretty_midi.PrettyMIDI (no chords)
    because the DDSP synth has to learn a per-note instrument timbre, and the
    melody alone gives plenty of pitch / dynamics variety. This keeps the data
    pipeline lightweight (no need to pre-render audio to disk).
    """
    abc_root = Path(abc_root)
    sources: list[MidiSource] = []
    for path in sorted(abc_root.rglob("*.abc")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for sheet in parse_nottingham_abc(text):
            # Use just the melody for Task 4 training (clean monophonic-ish source).
            pm = leadsheet_to_pretty_midi(
                type(sheet)(
                    melody=sheet.melody,
                    chords=[],  # drop chords so the source is monophonic
                    tempo=sheet.tempo,
                    time_signature=sheet.time_signature,
                ),
            )
            notes = pretty_midi_to_note_events(pm)
            if not notes:
                continue
            dur = max(n[1] for n in notes) + 0.25
            sources.append(MidiSource(midi_path=path, notes=notes, duration=float(dur)))
            if max_files is not None and len(sources) >= max_files:
                return sources
    return sources


def load_midi_sources_from_dir(
    midi_root: str | Path, max_files: int | None = None
) -> list[MidiSource]:
    """Load .mid / .midi files from a directory."""
    midi_root = Path(midi_root)
    sources: list[MidiSource] = []
    for path in sorted(list(midi_root.rglob("*.mid")) + list(midi_root.rglob("*.midi"))):
        notes = midi_to_note_events(path)
        if not notes:
            continue
        dur = max(n[1] for n in notes) + 0.25
        sources.append(MidiSource(midi_path=path, notes=notes, duration=float(dur)))
        if max_files is not None and len(sources) >= max_files:
            return sources
    return sources


try:
    import torch
    from torch.utils.data import Dataset

    class DDSPSynthDataset(Dataset):
        """Random fixed-length clips from MIDI sources, audio rendered on the fly.

        For each crop we slice the matching MIDI notes, render them via the
        teacher synth, and compute per-voice (F0, loudness, active) controls.
        """

        def __init__(
            self,
            sources: Sequence[MidiSource],
            *,
            clip_seconds: float = 3.0,
            crops_per_source: int = 4,
            max_voices: int = 8,
            sample_rate: int = SAMPLE_RATE,
            frame_rate: int = FRAME_RATE,
            seed: int = 153,
        ):
            self.sources = list(sources)
            self.clip_seconds = float(clip_seconds)
            self.crops_per_source = int(crops_per_source)
            self.max_voices = max_voices
            self.sample_rate = sample_rate
            self.frame_rate = frame_rate
            self._rng = random.Random(seed)

        def __len__(self) -> int:
            return max(1, len(self.sources)) * self.crops_per_source

        def __getitem__(self, idx: int):
            src = self.sources[idx % len(self.sources)]
            max_start = max(0.0, src.duration - self.clip_seconds - 0.05)
            start = self._rng.uniform(0.0, max_start) if max_start > 0 else 0.0
            end = start + self.clip_seconds
            # Pull notes overlapping [start, end] and shift to local timeline
            local_notes = []
            for (s, e, p, v) in src.notes:
                if e <= start or s >= end:
                    continue
                local_notes.append(
                    (max(0.0, s - start), min(self.clip_seconds, e - start), int(p), int(v))
                )
            audio = render_notes(
                local_notes, duration_sec=self.clip_seconds, sample_rate=self.sample_rate, seed=idx
            )
            f0, loud, act = midi_notes_to_frame_controls(
                local_notes,
                duration_sec=self.clip_seconds,
                sample_rate=self.sample_rate,
                frame_rate=self.frame_rate,
                max_voices=self.max_voices,
            )
            return {
                "audio": torch.from_numpy(audio),
                "f0": torch.from_numpy(f0),
                "loud": torch.from_numpy(loud),
                "active": torch.from_numpy(act),
            }

    class DDSPClipDataset(Dataset):
        """Random fixed-length clips from a list of (audio, midi) pairs.

        ``num_epochs_per_clip`` controls how many random crops we draw per
        clip per dataset epoch (so each epoch sees ``len(clips) * num_epochs_per_clip``
        examples).
        """

        def __init__(
            self,
            clips: Sequence[MaestroClip],
            *,
            clip_seconds: float = 4.0,
            crops_per_clip: int = 4,
            max_voices: int = 8,
            sample_rate: int = SAMPLE_RATE,
            frame_rate: int = FRAME_RATE,
            seed: int = 153,
        ):
            self.clips = list(clips)
            self.clip_seconds = float(clip_seconds)
            self.crops_per_clip = int(crops_per_clip)
            self.max_voices = max_voices
            self.sample_rate = sample_rate
            self.frame_rate = frame_rate
            self._rng = random.Random(seed)
            self._audio_cache: dict[Path, np.ndarray] = {}
            self._notes_cache: dict[Path, list] = {}

        def __len__(self) -> int:
            return len(self.clips) * self.crops_per_clip

        def _ensure_loaded(self, clip: MaestroClip) -> Tuple[np.ndarray, list]:
            if clip.audio_path not in self._audio_cache:
                self._audio_cache[clip.audio_path] = load_clip_audio(
                    clip.audio_path, sr=self.sample_rate
                )
            if clip.midi_path not in self._notes_cache:
                self._notes_cache[clip.midi_path] = midi_to_note_events(clip.midi_path)
            return self._audio_cache[clip.audio_path], self._notes_cache[clip.midi_path]

        def __getitem__(self, idx: int):
            clip = self.clips[idx % len(self.clips)]
            audio, notes = self._ensure_loaded(clip)
            max_start = max(0.0, clip.duration - self.clip_seconds - 0.5)
            start = self._rng.uniform(0.0, max_start) if max_start > 0 else 0.0
            ex = slice_example(
                audio,
                notes,
                start_sec=start,
                duration_sec=self.clip_seconds,
                sample_rate=self.sample_rate,
                frame_rate=self.frame_rate,
                max_voices=self.max_voices,
            )
            return {
                "audio": torch.from_numpy(ex.audio),
                "f0": torch.from_numpy(ex.f0_voices),
                "loud": torch.from_numpy(ex.loud_voices),
                "active": torch.from_numpy(ex.active),
            }

    def ddsp_collate(batch):
        return {
            "audio": torch.stack([b["audio"] for b in batch]),
            "f0": torch.stack([b["f0"] for b in batch]),
            "loud": torch.stack([b["loud"] for b in batch]),
            "active": torch.stack([b["active"] for b in batch]),
        }

except ImportError:  # pragma: no cover
    DDSPClipDataset = None  # type: ignore
    ddsp_collate = None  # type: ignore
