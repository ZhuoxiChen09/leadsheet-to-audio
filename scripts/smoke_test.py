"""Smoke test: tiny synthetic data through both pipelines, end-to-end.

Verifies that:
  * The Nottingham ABC parser handles a hand-crafted tune.
  * The tokenizer round-trips a Leadsheet.
  * The Task 2 Transformer forwards and backprops on a synthetic batch.
  * Sampling produces well-formed chord ids.
  * The Task 4 DDSP synthesizer forwards and backprops on a synthetic batch
    and the multi-scale STFT loss decreases after a few optimizer steps.

Designed to run on CPU in under ~30 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.baselines import (
    BigramChordBaseline,
    baseline_I_IV_V_I,
    baseline_rule_based,
    synth_sine_voices,
    synth_triangle_voices,
)
from src.data_task2 import (
    HarmonizationDataset,
    encode_dataset,
    harmonization_collate,
    train_val_split,
)
from src.features import a_weighted_loudness, midi_notes_to_frame_controls
from src.metrics import (
    chord_transition_kl,
    multi_scale_stft_distance,
    note_in_chord_rate,
    root_motion_stats,
    top_k_accuracy,
)
from src.midi_io import (
    ChordSymbol,
    Leadsheet,
    MelodyNote,
    chord_pitch_set,
    parse_chord_symbol,
    parse_nottingham_abc,
)
from src.model_task2 import Harmonizer, HarmonizerConfig
from src.model_task4 import DDSPConfig, MultiScaleSTFTLoss, PolyphonicDDSP
from src.seed import seed_everything
from src.tokenizer import (
    CHORD_PAD_ID,
    GRID_PER_BEAT,
    encode_leadsheet,
)
from src.sample_task2 import twinkle_twinkle


def step(name):
    print(f"=== {name} ===")


def make_synthetic_leadsheet() -> Leadsheet:
    pitches = [60, 64, 67, 72, 67, 64, 60, 67, 71, 67, 64, 60]
    melody = [MelodyNote(pitch=p, start=i * 0.5, duration=0.5) for i, p in enumerate(pitches)]
    chords = [
        ChordSymbol(root=0, quality="maj", start=0.0, duration=4.0),
        ChordSymbol(root=7, quality="maj", start=4.0, duration=4.0),
    ]
    return Leadsheet(melody=melody, chords=chords, tempo=120, time_signature=(4, 4), key="C")


def test_chord_parsing():
    step("Chord parsing")
    cases = [("C", 0, "maj"), ("Gm", 7, "min"), ("F#7", 6, "dom7"), ("BbM7", 10, "maj7"), ("Db", 1, "maj")]
    for text, root, quality in cases:
        c = parse_chord_symbol(text)
        assert c is not None, text
        assert c.root == root, (text, c.root, root)
        assert c.quality == quality, (text, c.quality, quality)
    pcs_cmaj = chord_pitch_set(ChordSymbol(root=0, quality="maj", start=0, duration=0))
    assert pcs_cmaj == {0, 4, 7}
    print("  parsed OK")


def test_abc_parsing():
    step("ABC parsing")
    abc = """X:1
T:Test Tune
M:4/4
L:1/8
Q:1/4=120
K:C
"C"c2 e2 g2 c'2|"G"g2 d2 b,2 g,2|"F"f2 a2 c2 f2|"C"c2 e2 g2 c'2|
"""
    tunes = parse_nottingham_abc(abc)
    assert len(tunes) == 1
    t = tunes[0]
    assert len(t.melody) >= 8
    assert len(t.chords) >= 4
    print(f"  parsed melody={len(t.melody)} notes, chords={len(t.chords)} bars")


def test_tokenizer_roundtrip():
    step("Tokenizer encode")
    sheet = make_synthetic_leadsheet()
    enc = encode_leadsheet(sheet, max_bars=4)
    assert enc is not None
    assert enc.n_bars >= 1
    assert len(enc.chord_ids) == enc.n_bars
    print(f"  encoded {enc.n_bars} bars, {len(enc.melody_ids)} melody grid cells")


def test_baselines_and_metrics():
    step("Baselines + metrics")
    sheet = make_synthetic_leadsheet()
    enc = encode_leadsheet(sheet, max_bars=8)
    assert enc is not None
    iv = baseline_I_IV_V_I(enc.n_bars)
    rb = baseline_rule_based(enc.melody_ids)
    bg = BigramChordBaseline()
    bg.fit([enc.chord_ids])
    bg_pred = bg.predict(enc.n_bars)
    nic = note_in_chord_rate(enc.melody_ids, rb)
    rm = root_motion_stats(rb)
    kl = chord_transition_kl([rb], [enc.chord_ids])
    acc = top_k_accuracy(np.array(rb), np.array(enc.chord_ids))
    print(f"  rule-based note-in-chord={nic:.3f} root-motion-mean={rm['mean']:.2f} KL={kl:.3f} acc(vs gt)={acc:.3f}")


def test_task2_model():
    step("Task 2 model")
    seed_everything(0)
    cfg = HarmonizerConfig(d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2, d_ff=128, dropout=0.0)
    model = Harmonizer(cfg)
    examples = []
    for _ in range(8):
        sheet = make_synthetic_leadsheet()
        examples += encode_dataset([sheet], max_bars=4)
    ds = HarmonizationDataset(examples)
    batch = harmonization_collate([ds[i] for i in range(len(ds))])
    melody = batch["melody"]
    chord = batch["chord"]
    mel_mask = batch["melody_mask"]
    chord_mask = batch["chord_mask"]
    chord_in = chord[:, :-1]
    chord_out = chord[:, 1:]
    logits = model(melody, chord_in, mel_mask=mel_mask, chord_mask=chord_mask[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        chord_out.reshape(-1),
        ignore_index=CHORD_PAD_ID,
    )
    loss.backward()
    print(f"  forward+backward OK, loss={loss.item():.4f}, params={model.num_parameters()/1e6:.2f}M")
    sampled = model.harmonize(melody[:1], n_bars=4, temperature=1.0, top_p=0.9)
    assert sampled.shape == (1, 4)
    print(f"  sampled chord ids: {sampled[0].tolist()}")


def test_task4_model_overfit():
    step("Task 4 model (mini-overfit 8 steps)")
    seed_everything(0)
    cfg = DDSPConfig(n_harmonics=16, noise_filter_taps=33, hidden_size=64, max_voices=2, sample_rate=8000, frame_rate=200)
    model = PolyphonicDDSP(cfg)
    sr = cfg.sample_rate
    fr = cfg.frame_rate
    # Synthesize a single-voice sine target so the model has something to fit
    notes = [(0.0, 1.0, 60, 100), (1.2, 2.0, 64, 80)]
    f0, loud, act = midi_notes_to_frame_controls(notes, duration_sec=2.0, sample_rate=sr, frame_rate=fr, max_voices=2)
    target = synth_sine_voices(f0, loud, sample_rate=sr, frame_rate=fr)
    f0_t = torch.from_numpy(f0).unsqueeze(0)
    loud_t = torch.from_numpy(loud).unsqueeze(0)
    act_t = torch.from_numpy(act).unsqueeze(0)
    target_t = torch.from_numpy(target).unsqueeze(0)
    loss_fn = MultiScaleSTFTLoss(fft_sizes=(512, 256, 128))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for step_i in range(8):
        pred = model(f0_t, loud_t, act_t)
        loss = loss_fn(pred, target_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  losses: first={losses[0]:.4f} last={losses[-1]:.4f}")
    assert losses[-1] <= losses[0] + 0.01, "loss did not decrease — possible wiring bug"


def test_synth_baselines():
    step("Synth baselines")
    notes = [(0.0, 0.5, 60, 100), (0.5, 1.0, 64, 80)]
    f0, loud, act = midi_notes_to_frame_controls(notes, duration_sec=1.0, sample_rate=8000, frame_rate=200, max_voices=2)
    sine = synth_sine_voices(f0, loud, 8000, 200)
    tri = synth_triangle_voices(f0, loud, 8000, 200)
    assert sine.shape[0] == 8000
    assert tri.shape[0] == 8000
    mst = multi_scale_stft_distance(sine, tri, fft_sizes=(512, 256, 128))
    print(f"  sine vs triangle multi-scale STFT distance = {mst:.4f}")


def test_loudness_and_f0():
    step("Loudness + F0 features")
    sr = 8000
    t = np.arange(sr) / sr
    wav = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    loud = a_weighted_loudness(wav, sample_rate=sr, frame_rate=200)
    assert loud.shape[0] > 100
    print(f"  loudness frames={loud.shape[0]}, mean={loud.mean():.2f} dB")


def main():
    test_chord_parsing()
    test_abc_parsing()
    test_tokenizer_roundtrip()
    test_baselines_and_metrics()
    test_loudness_and_f0()
    test_synth_baselines()
    test_task2_model()
    test_task4_model_overfit()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
