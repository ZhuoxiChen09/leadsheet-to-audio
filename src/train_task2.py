"""Train the melody -> chord harmonizer on Nottingham."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_task2 import (
    HarmonizationDataset,
    encode_dataset,
    harmonization_collate,
    load_nottingham_leadsheets,
    train_val_split,
)
from .model_task2 import Harmonizer, HarmonizerConfig
from .seed import get_device, seed_everything
from .tokenizer import CHORD_PAD_ID


def evaluate(model: Harmonizer, loader: DataLoader, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0
    for batch in loader:
        melody = batch["melody"].to(device)
        chord = batch["chord"].to(device)
        mel_mask = batch["melody_mask"].to(device)
        chord_mask = batch["chord_mask"].to(device)
        chord_in = chord[:, :-1]
        chord_out = chord[:, 1:]
        chord_in_mask = chord_mask[:, :-1]
        logits = model(melody, chord_in, mel_mask=mel_mask, chord_mask=chord_in_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chord_out.reshape(-1),
            ignore_index=CHORD_PAD_ID,
            reduction="sum",
        )
        n_tokens = (chord_out != CHORD_PAD_ID).sum().item()
        total_loss += float(loss.item())
        total_tokens += int(n_tokens)
        # accuracy
        valid_mask = chord_out != CHORD_PAD_ID
        if valid_mask.any():
            top5 = logits.topk(5, dim=-1).indices
            tgt = chord_out.unsqueeze(-1)
            correct_top1 += int(((top5[..., :1] == tgt) & valid_mask.unsqueeze(-1)).sum().item())
            correct_top5 += int(((top5 == tgt) & valid_mask.unsqueeze(-1)).sum().item())
    avg_loss = total_loss / max(1, total_tokens)
    ppl = float(np.exp(min(avg_loss, 20)))
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "top1": correct_top1 / max(1, total_tokens),
        "top5": correct_top5 / max(1, total_tokens),
        "tokens": total_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--abc-root", default="data/nottingham/ABC_cleaned")
    p.add_argument("--out-dir", default="checkpoints/task2")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-bars", type=int, default=16)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--augmentations", type=int, default=3, help="Random transpositions per tune.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=153)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    seed_everything(args.seed)
    device = args.device or get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task2] Loading leadsheets from {args.abc_root} ...")
    sheets = load_nottingham_leadsheets(args.abc_root)
    print(f"[task2] Parsed {len(sheets)} leadsheets.")
    if len(sheets) < 20:
        raise RuntimeError(
            f"Only {len(sheets)} leadsheets parsed — check the ABC root path and the parser."
        )

    examples = encode_dataset(sheets, max_bars=args.max_bars, augmentations=args.augmentations)
    print(f"[task2] Encoded {len(examples)} examples (incl. augmentations).")

    train_ex, val_ex = train_val_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[task2] Train / val split: {len(train_ex)} / {len(val_ex)}")

    train_loader = DataLoader(
        HarmonizationDataset(train_ex),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=harmonization_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        HarmonizationDataset(val_ex),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=harmonization_collate,
        num_workers=0,
    )

    cfg = HarmonizerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_layers,
        n_dec_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = Harmonizer(cfg).to(device)
    print(f"[task2] Model has {model.num_parameters() / 1e6:.2f}M trainable parameters.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        tokens = 0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"task2 e{epoch}", leave=False)
        for batch in bar:
            melody = batch["melody"].to(device)
            chord = batch["chord"].to(device)
            mel_mask = batch["melody_mask"].to(device)
            chord_mask = batch["chord_mask"].to(device)
            chord_in = chord[:, :-1]
            chord_out = chord[:, 1:]
            chord_in_mask = chord_mask[:, :-1]
            logits = model(melody, chord_in, mel_mask=mel_mask, chord_mask=chord_in_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chord_out.reshape(-1),
                ignore_index=CHORD_PAD_ID,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = (chord_out != CHORD_PAD_ID).sum().item()
            running += float(loss.item()) * n
            tokens += n
            bar.set_postfix(loss=running / max(1, tokens))
        sched.step()
        train_loss = running / max(1, tokens)
        val_stats = evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, **val_stats, "wall_sec": time.time() - t0}
        )
        print(
            f"[task2 e{epoch}] train={train_loss:.4f} "
            f"val_loss={val_stats['loss']:.4f} ppl={val_stats['perplexity']:.2f} "
            f"top1={val_stats['top1']:.3f} top5={val_stats['top5']:.3f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_stats": val_stats,
                },
                out_dir / "best.pt",
            )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "val_stats": val_stats,
            },
            out_dir / "last.pt",
        )

    # Also save a bigram chord baseline and the chord unigram counts
    chord_seqs = [ex.chord_target[1:-1].tolist() for ex in train_ex]
    unigram = Counter()
    for seq in chord_seqs:
        unigram.update(seq)
    (out_dir / "train_chord_unigram.json").write_text(json.dumps(dict(unigram)))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"[task2] Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
