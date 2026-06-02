"""Dataset / dataloader for Task 2 (Nottingham harmonization).

The Nottingham Music Database is a public collection of ~1200 British and
American folk tunes in ABC notation, originally compiled by Eric Foxley and
later cleaned by Seymour Shlien. It is the de-facto small-scale benchmark for
melody+chord modelling (e.g. Boulanger-Lewandowski 2012).

We download the cleaned ABC tarball published by Seymour Shlien's mirror
and parse each tune into a Leadsheet, then encode with ``tokenizer.encode_leadsheet``.
"""
from __future__ import annotations

import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .midi_io import Leadsheet, parse_nottingham_abc
from .tokenizer import (
    CHORD_BOS_ID,
    CHORD_EOS_ID,
    CHORD_PAD_ID,
    EncodedExample,
    MEL_BOS_ID,
    MEL_EOS_ID,
    MEL_PAD_ID,
    encode_leadsheet,
)


NOTTINGHAM_DEFAULT_URL = (
    "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.tar.gz"
)


def download_nottingham(dest_dir: str | Path, url: str = NOTTINGHAM_DEFAULT_URL) -> Path:
    """Download and extract the Jukedeck mirror of the Nottingham ABC dataset.

    Returns the path to the directory containing ``.abc`` files.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "nottingham.tar.gz"
    if not tar_path.exists():
        urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    # Locate the abc/ subfolder
    for candidate in dest_dir.rglob("*.abc"):
        return candidate.parent
    raise FileNotFoundError(f"No .abc files found under {dest_dir} after extraction.")


def find_abc_files(root: str | Path) -> List[Path]:
    return sorted(Path(root).rglob("*.abc"))


def load_nottingham_leadsheets(abc_root: str | Path) -> List[Leadsheet]:
    """Parse every ABC file under ``abc_root`` into a list of Leadsheets."""
    out: List[Leadsheet] = []
    for path in find_abc_files(abc_root):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tunes = parse_nottingham_abc(text)
        out.extend(tunes)
    return out


# ---------------------------------------------------------------------------

@dataclass
class HarmonizationExample:
    """Encoded (melody, chord-sequence) pair ready for the Transformer."""

    melody_ids: np.ndarray  # int64, length = n_bars * grid_per_bar
    chord_target: np.ndarray  # int64, length = n_bars + 2 (BOS .. EOS)
    n_bars: int


def encode_dataset(
    sheets: Iterable[Leadsheet],
    *,
    max_bars: int = 16,
    transpose_to_c: bool = False,
    augmentations: int = 0,
) -> List[HarmonizationExample]:
    """Encode a list of Leadsheets into model-ready examples.

    Optionally apply key transposition augmentation (``augmentations`` extra
    copies per tune, each shifted by a random number of semitones in [-5..6]).
    """
    out: List[HarmonizationExample] = []
    sheets_list = list(sheets)
    for sheet in sheets_list:
        enc = encode_leadsheet(sheet, max_bars=max_bars, transpose_to_c=transpose_to_c)
        if enc is None:
            continue
        out.append(_to_example(enc))
        for _ in range(augmentations):
            shift = random.randint(-5, 6)
            aug_sheet = _transpose_leadsheet(sheet, shift)
            enc_a = encode_leadsheet(aug_sheet, max_bars=max_bars, transpose_to_c=False)
            if enc_a is None:
                continue
            out.append(_to_example(enc_a))
    return out


def _transpose_leadsheet(sheet: Leadsheet, semitones: int) -> Leadsheet:
    from .midi_io import ChordSymbol, MelodyNote
    new_melody = [
        MelodyNote(pitch=(n.pitch + semitones) if not n.is_rest else -1, start=n.start, duration=n.duration)
        for n in sheet.melody
    ]
    new_chords = [
        ChordSymbol(
            root=(c.root + semitones) % 12 if c.root >= 0 else -1,
            quality=c.quality,
            start=c.start,
            duration=c.duration,
        )
        for c in sheet.chords
    ]
    return Leadsheet(
        melody=new_melody,
        chords=new_chords,
        tempo=sheet.tempo,
        time_signature=sheet.time_signature,
        key=sheet.key,
        title=sheet.title,
    )


def _to_example(enc: EncodedExample) -> HarmonizationExample:
    chord_seq = np.array([CHORD_BOS_ID, *enc.chord_ids, CHORD_EOS_ID], dtype=np.int64)
    melody = np.array(enc.melody_ids, dtype=np.int64)
    return HarmonizationExample(
        melody_ids=melody,
        chord_target=chord_seq,
        n_bars=enc.n_bars,
    )


# ---------------------------------------------------------------------------
# Torch dataset / collate
# ---------------------------------------------------------------------------

try:  # torch is an optional import here so module is testable without GPU env
    import torch
    from torch.utils.data import Dataset

    class HarmonizationDataset(Dataset):
        def __init__(self, examples: Sequence[HarmonizationExample]):
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
            ex = self.examples[idx]
            return (
                torch.from_numpy(ex.melody_ids),
                torch.from_numpy(ex.chord_target),
                ex.n_bars,
            )

    def harmonization_collate(batch):
        mels, chords, n_bars = zip(*batch)
        max_m = max(t.shape[0] for t in mels)
        max_c = max(t.shape[0] for t in chords)
        mel_padded = torch.full((len(batch), max_m), MEL_PAD_ID, dtype=torch.long)
        chord_padded = torch.full((len(batch), max_c), CHORD_PAD_ID, dtype=torch.long)
        mel_mask = torch.zeros((len(batch), max_m), dtype=torch.bool)
        chord_mask = torch.zeros((len(batch), max_c), dtype=torch.bool)
        for i, (m, c) in enumerate(zip(mels, chords)):
            mel_padded[i, : m.shape[0]] = m
            chord_padded[i, : c.shape[0]] = c
            mel_mask[i, : m.shape[0]] = True
            chord_mask[i, : c.shape[0]] = True
        return {
            "melody": mel_padded,
            "chord": chord_padded,
            "melody_mask": mel_mask,
            "chord_mask": chord_mask,
            "n_bars": torch.tensor(n_bars, dtype=torch.long),
        }

except ImportError:  # pragma: no cover
    HarmonizationDataset = None  # type: ignore
    harmonization_collate = None  # type: ignore


def train_val_split(
    examples: Sequence[HarmonizationExample],
    val_frac: float = 0.1,
    seed: int = 153,
) -> Tuple[List[HarmonizationExample], List[HarmonizationExample]]:
    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_frac))
    val = [examples[i] for i in idxs[:n_val]]
    train = [examples[i] for i in idxs[n_val:]]
    return train, val
