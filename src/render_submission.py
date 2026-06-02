"""Render the final ``symbolic_conditioned.mid`` and ``continuous_conditioned.mp3``.

Pipeline:
1. Pick a target melody (Twinkle Twinkle by default).
2. Run the trained harmonizer to produce a chord sequence.
3. Save the (melody + chords) leadsheet as a MIDI file.
4. Run the trained DDSP synth on that MIDI to produce a wav file.
5. Encode the wav to a compressed mp3.

Run with:
    python -m src.render_submission \
        --task2-ckpt checkpoints/task2/best.pt \
        --task4-ckpt checkpoints/task4/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .midi_io import leadsheet_to_pretty_midi
from .sample_task2 import harmonize_leadsheet, load_harmonizer, twinkle_twinkle
from .sample_task4 import encode_mp3, load_ddsp, synthesize_midi_file
from .seed import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task2-ckpt", default="checkpoints/task2/best.pt")
    p.add_argument("--task4-ckpt", default="checkpoints/task4/best.pt")
    p.add_argument("--out-mid", default="symbolic_conditioned.mid")
    p.add_argument("--out-mp3", default="continuous_conditioned.mp3")
    p.add_argument("--out-wav", default="samples/final.wav")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--seed", type=int, default=153)
    args = p.parse_args()

    seed_everything(args.seed)

    # ---- Task 2: harmonize ----
    model2 = load_harmonizer(args.task2_ckpt)
    sheet = twinkle_twinkle()
    harmonized = harmonize_leadsheet(
        model2, sheet, temperature=args.temperature, top_p=args.top_p
    )
    pm = leadsheet_to_pretty_midi(harmonized, arpeggiate=False)
    pm.write(args.out_mid)
    print(f"[render] Task 2: wrote {args.out_mid}")

    # ---- Task 4: synthesize ----
    model4 = load_ddsp(args.task4_ckpt)
    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    synthesize_midi_file(model4, args.out_mid, args.out_wav)
    print(f"[render] Task 4: wrote {args.out_wav}")
    encode_mp3(args.out_wav, args.out_mp3)
    print(f"[render] Task 4: wrote {args.out_mp3}")


if __name__ == "__main__":
    main()
