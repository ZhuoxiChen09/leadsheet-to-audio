"""Use a trained DDSP synth to render a MIDI file as audio."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .features import FRAME_RATE, SAMPLE_RATE, midi_notes_to_frame_controls
from .midi_io import midi_to_note_events
from .model_task4 import DDSPConfig, PolyphonicDDSP
from .seed import get_device


def load_ddsp(ckpt_path: str | Path, device: str | None = None) -> PolyphonicDDSP:
    device = device or get_device()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = DDSPConfig(**ckpt["cfg"])
    model = PolyphonicDDSP(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def synthesize_midi_file(
    model: PolyphonicDDSP,
    midi_path: str | Path,
    out_path: str | Path,
    *,
    chunk_seconds: float = 5.0,
    device: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
    max_voices: int | None = None,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    notes = midi_to_note_events(midi_path)
    if not notes:
        raise ValueError(f"No notes found in {midi_path}")
    total = max(n[1] for n in notes) + 0.5
    if max_voices is None:
        max_voices = model.cfg.max_voices

    out_chunks: list[np.ndarray] = []
    t = 0.0
    while t < total:
        seg_dur = min(chunk_seconds, total - t)
        f0, loud, act = midi_notes_to_frame_controls(
            [(s - t, e - t, p, v) for (s, e, p, v) in notes if e > t and s < t + seg_dur],
            duration_sec=seg_dur,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            max_voices=max_voices,
        )
        f0_t = torch.from_numpy(f0).unsqueeze(0).to(device)
        loud_t = torch.from_numpy(loud).unsqueeze(0).to(device)
        act_t = torch.from_numpy(act).unsqueeze(0).to(device)
        with torch.no_grad():
            wav = model(f0_t, loud_t, act_t).squeeze(0).cpu().numpy()
        out_chunks.append(wav)
        t += seg_dur
    audio = np.concatenate(out_chunks)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    audio = (audio / peak * 0.95).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate)
    return audio


def encode_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "128k") -> None:
    """Use imageio_ffmpeg's bundled binary to encode wav -> mp3."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/task4/best.pt")
    p.add_argument("--midi", required=True)
    p.add_argument("--out", default="samples/synthesized.wav")
    p.add_argument("--mp3", default=None)
    p.add_argument("--chunk-seconds", type=float, default=5.0)
    args = p.parse_args()

    model = load_ddsp(args.ckpt)
    synthesize_midi_file(model, args.midi, args.out, chunk_seconds=args.chunk_seconds)
    print(f"[sample_task4] Wrote {args.out}")
    if args.mp3:
        encode_mp3(args.out, args.mp3)
        print(f"[sample_task4] Wrote {args.mp3}")


if __name__ == "__main__":
    main()
