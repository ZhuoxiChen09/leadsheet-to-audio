"""Use a trained harmonizer to harmonize a melody.

Provides two convenience entry points:
    - ``harmonize_leadsheet(model, sheet, ...)`` -> Leadsheet with predicted chords
    - ``harmonize_melody_ids(model, melody_ids, n_bars, ...)`` -> chord id list

The CLI loads a checkpoint, harmonizes a default test melody (Twinkle Twinkle),
and writes the result to ``samples/twinkle.mid``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import torch

from .midi_io import (
    ChordSymbol,
    Leadsheet,
    MelodyNote,
    leadsheet_to_pretty_midi,
)
from .model_task2 import Harmonizer, HarmonizerConfig
from .seed import get_device, seed_everything
from .tokenizer import (
    CHORD_EOS_ID,
    chord_from_token_id,
    encode_leadsheet,
)


def load_harmonizer(ckpt_path: str | Path, device: str | None = None) -> Harmonizer:
    device = device or get_device()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = HarmonizerConfig(**ckpt["cfg"])
    model = Harmonizer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def harmonize_melody_ids(
    model: Harmonizer,
    melody_ids: Sequence[int],
    n_bars: int,
    *,
    temperature: float = 0.9,
    top_p: float = 0.95,
    greedy: bool = False,
    device: str | None = None,
) -> List[int]:
    device = device or next(model.parameters()).device
    mel = torch.tensor([list(melody_ids)], dtype=torch.long, device=device)
    out = model.harmonize(mel, n_bars=n_bars, temperature=temperature, top_p=top_p, greedy=greedy)
    seq = out[0].tolist()
    # Strip any EOS token if present, and ensure length n_bars
    if CHORD_EOS_ID in seq:
        seq = seq[: seq.index(CHORD_EOS_ID)]
    if len(seq) < n_bars:
        seq = seq + [seq[-1] if seq else 0] * (n_bars - len(seq))
    return seq[:n_bars]


def harmonize_leadsheet(
    model: Harmonizer,
    sheet: Leadsheet,
    *,
    max_bars: int = 16,
    temperature: float = 0.9,
    top_p: float = 0.95,
    greedy: bool = False,
) -> Leadsheet:
    enc = encode_leadsheet(sheet, max_bars=max_bars)
    if enc is None:
        raise ValueError("Could not encode leadsheet (empty or too short).")
    chord_ids = harmonize_melody_ids(
        model,
        enc.melody_ids,
        n_bars=enc.n_bars,
        temperature=temperature,
        top_p=top_p,
        greedy=greedy,
    )
    chords: list[ChordSymbol] = []
    bpb = sheet.beats_per_bar
    for bar_idx, cid in enumerate(chord_ids):
        base = chord_from_token_id(int(cid))
        chords.append(
            ChordSymbol(
                root=base.root,
                quality=base.quality,
                start=bar_idx * bpb,
                duration=bpb,
            )
        )
    return Leadsheet(
        melody=list(sheet.melody),
        chords=chords,
        tempo=sheet.tempo,
        time_signature=sheet.time_signature,
        key=sheet.key,
        title=sheet.title,
    )


# ---- Demo melodies --------------------------------------------------------

def twinkle_twinkle() -> Leadsheet:
    """A simple Twinkle Twinkle Little Star melody in C major, 8 bars."""
    pitches = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60,
               67, 67, 65, 65, 64, 64, 62, 67, 67, 65, 65, 64, 64, 62,
               60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
    durations = [1.0] * len(pitches)
    melody: List[MelodyNote] = []
    t = 0.0
    for p, d in zip(pitches, durations, strict=False):
        melody.append(MelodyNote(pitch=p, start=t, duration=d))
        t += d
    return Leadsheet(
        melody=melody,
        chords=[],  # empty — to be filled
        tempo=110.0,
        time_signature=(4, 4),
        key="C",
        title="Twinkle Twinkle",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/task2/best.pt")
    p.add_argument("--out", default="samples/twinkle.mid")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=153)
    args = p.parse_args()

    seed_everything(args.seed)
    model = load_harmonizer(args.ckpt)
    sheet = twinkle_twinkle()
    harmonized = harmonize_leadsheet(
        model, sheet, temperature=args.temperature, top_p=args.top_p, greedy=args.greedy
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pm = leadsheet_to_pretty_midi(harmonized, arpeggiate=False)
    pm.write(args.out)
    print(f"[sample] Wrote {args.out}")


if __name__ == "__main__":
    main()
