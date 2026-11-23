"""Train the polyphonic DDSP synthesizer.

The default data source is MIDI files rendered on the fly with our teacher
synth (``src.teacher_synth``). This avoids depending on the ~108 GB MAESTRO
audio archive (only the metadata CSV is publicly retrievable per-file).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_task4 import (
    DDSPClipDataset,
    DDSPSynthDataset,
    ddsp_collate,
    discover_local_pairs,
    load_midi_sources_from_abc,
    load_midi_sources_from_dir,
    parse_maestro_csv,
)
from .features import FRAME_RATE, SAMPLE_RATE
from .model_task4 import DDSPConfig, MultiScaleSTFTLoss, PolyphonicDDSP
from .seed import get_device, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["abc", "midi-dir", "paired"], default="abc",
                   help="Where to get training MIDI / audio from.")
    p.add_argument("--abc-root", default="data/nottingham/nottingham-dataset-master/ABC_cleaned",
                   help="Used when --source=abc.")
    p.add_argument("--midi-root", default="data/maestro",
                   help="Used when --source=midi-dir.")
    p.add_argument("--maestro-csv", default="data/maestro/maestro-v3.0.0.csv",
                   help="Used when --source=paired.")
    p.add_argument("--audio-root", default="data/maestro",
                   help="Used when --source=paired and no CSV.")
    p.add_argument("--out-dir", default="checkpoints/task4")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-seconds", type=float, default=3.0)
    p.add_argument("--crops-per-clip", type=int, default=8)
    p.add_argument("--max-voices", type=int, default=4)
    p.add_argument("--n-harmonics", type=int, default=48)
    p.add_argument("--noise-bins", type=int, default=65)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=153)
    p.add_argument("--device", default=None)
    p.add_argument("--max-sources", type=int, default=64,
                   help="Cap on number of MIDI files used.")
    args = p.parse_args()

    seed_everything(args.seed)
    device = args.device or get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "paired":
        if Path(args.maestro_csv).exists():
            clips = parse_maestro_csv(args.maestro_csv)
            print(f"[task4] Loaded {len(clips)} clips from {args.maestro_csv}")
        else:
            clips = discover_local_pairs(args.audio_root)
            print(f"[task4] Discovered {len(clips)} pairs under {args.audio_root}")
        if not clips:
            raise RuntimeError(
                "No audio/MIDI pairs found for --source=paired. "
                "Switch to --source=abc to use the on-the-fly teacher-synth pipeline."
            )
        if args.max_sources and len(clips) > args.max_sources:
            clips = clips[: args.max_sources]
        n_val = max(1, int(len(clips) * args.val_frac))
        val_clips = clips[:n_val]
        train_clips = clips[n_val:]
        train_ds = DDSPClipDataset(
            train_clips,
            clip_seconds=args.clip_seconds,
            crops_per_clip=args.crops_per_clip,
            max_voices=args.max_voices,
            seed=args.seed,
        )
        val_ds = DDSPClipDataset(
            val_clips,
            clip_seconds=args.clip_seconds,
            crops_per_clip=2,
            max_voices=args.max_voices,
            seed=args.seed + 1,
        )
    else:
        if args.source == "abc":
            sources = load_midi_sources_from_abc(args.abc_root, max_files=args.max_sources)
        else:
            sources = load_midi_sources_from_dir(args.midi_root, max_files=args.max_sources)
        print(
            f"[task4] Loaded {len(sources)} MIDI sources from "
            f"{args.abc_root if args.source == 'abc' else args.midi_root}"
        )
        if not sources:
            raise RuntimeError(
                "No MIDI sources found. Did you run "
                "`python scripts/download_data.py --task task2` first?"
            )
        n_val = max(1, int(len(sources) * args.val_frac))
        val_sources = sources[:n_val]
        train_sources = sources[n_val:]
        train_ds = DDSPSynthDataset(
            train_sources,
            clip_seconds=args.clip_seconds,
            crops_per_source=args.crops_per_clip,
            max_voices=args.max_voices,
            seed=args.seed,
        )
        val_ds = DDSPSynthDataset(
            val_sources,
            clip_seconds=args.clip_seconds,
            crops_per_source=2,
            max_voices=args.max_voices,
            seed=args.seed + 1,
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ddsp_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ddsp_collate, num_workers=0
    )

    cfg = DDSPConfig(
        n_harmonics=args.n_harmonics,
        noise_filter_taps=args.noise_bins,
        hidden_size=args.hidden,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        max_voices=args.max_voices,
    )
    model = PolyphonicDDSP(cfg).to(device)
    print(f"[task4] Model has {model.num_parameters() / 1e6:.3f}M parameters.")

    loss_fn = MultiScaleSTFTLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"task4 e{epoch}", leave=False)
        for batch in bar:
            f0 = batch["f0"].to(device)
            loud = batch["loud"].to(device)
            act = batch["active"].to(device)
            target = batch["audio"].to(device)
            pred = model(f0, loud, act)
            loss = loss_fn(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
            bar.set_postfix(loss=running / max(1, n_batches))
        sched.step()
        train_loss = running / max(1, n_batches)

        # validation
        model.eval()
        with torch.no_grad():
            v_run = 0.0
            v_n = 0
            for batch in val_loader:
                f0 = batch["f0"].to(device)
                loud = batch["loud"].to(device)
                act = batch["active"].to(device)
                target = batch["audio"].to(device)
                pred = model(f0, loud, act)
                v_run += float(loss_fn(pred, target).item())
                v_n += 1
        val_loss = v_run / max(1, v_n)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_sec": time.time() - t0,
            }
        )
        print(f"[task4 e{epoch}] train={train_loss:.4f} val={val_loss:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                out_dir / "best.pt",
            )
        torch.save(
            {"model_state_dict": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
            out_dir / "last.pt",
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[task4] Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
