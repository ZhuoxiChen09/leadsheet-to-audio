# leadsheet-to-audio

> End-to-end neural music pipeline: a from-scratch Transformer harmonizes a melody into a leadsheet, and a DDSP-style harmonic + filtered-noise synth renders it to audio. Trained from scratch in PyTorch on Nottingham (symbolic) and a MAESTRO subset (audio).

A bare melody (`C C G G A A G`) goes in. A fully-harmonized, multi-voice audio track comes out.

| Stage | Model | Input | Output |
|---|---|---|---|
| 1. Harmonize | Small Transformer (4 layers, d=256) trained from scratch on Nottingham leadsheets | melody tokens | chord-per-beat sequence (`symbolic_conditioned.mid`) |
| 2. Synthesize | DDSP-style harmonic + filtered-noise synth (sum-of-voices for polyphony) | MIDI | 16 kHz audio (`continuous_conditioned.mp3`) |

The two networks are decoupled and composable: you can swap in any monophonic melody and the pipeline will produce a finished recording.

## Demo

- **Input melody**: "Twinkle, Twinkle, Little Star" (a single voice, no chords)
- **After stage 1 (harmonized MIDI)**: [`symbolic_conditioned.mid`](symbolic_conditioned.mid)
- **After stage 2 (rendered audio)**: [`continuous_conditioned.mp3`](continuous_conditioned.mp3)
- **Walkthrough video (20 min)**: see [`docs/video_url.txt`](docs/video_url.txt)
- **Annotated notebook**: [`workbook.ipynb`](workbook.ipynb) (GitHub renders it inline) — or open [`workbook.html`](workbook.html) for the pre-executed version.

## Results

| Task | Metric | Value |
|---|---|---|
| Harmonizer (Task 2) | Validation perplexity | ≈ 4.0 |
| Harmonizer (Task 2) | Top-1 chord accuracy | ≈ 58% |
| Harmonizer (Task 2) | Top-5 chord accuracy | ≈ 91% |
| Synth (Task 4) | Multi-scale STFT loss | ≈ 9 (init: 22) |

Both models train on a single laptop GPU in under 10 minutes total (RTX 4070 Laptop, 8 GB).

## Architecture sketch

```
melody MIDI
    │
    ▼
┌──────────────────────┐
│  Task 2: Transformer │  (chord tokens are conditioned on the melody
│  4 layers, d=256     │   plus a learned key-signature embedding)
└──────────┬───────────┘
           │
           ▼
   leadsheet MIDI  ──►  symbolic_conditioned.mid
           │
           ▼
┌──────────────────────┐
│  Task 4: DDSP synth  │  (per-voice f0/loudness → harmonic oscillator
│  48 harmonics + FNB  │   + filtered noise, summed across up to 4 voices)
└──────────┬───────────┘
           │
           ▼
       16 kHz audio  ──►  continuous_conditioned.mp3
```

## Quick start

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m src.train_task2 --epochs 40
python -m src.train_task4 --epochs 25
python -m src.render_submission
```

See [`RUNBOOK.md`](RUNBOOK.md) for the full reproducible command set (exact hyper-parameters, expected runtimes, troubleshooting).

## Repository layout

```
.
├── workbook.ipynb               # main narrative: EDA → modeling → evaluation
├── workbook.html                # pre-executed export of the notebook
├── symbolic_conditioned.mid     # demo output of stage 1
├── continuous_conditioned.mp3   # demo output of stage 2
├── src/                         # all reusable code (imported from the notebook)
│   ├── model_task2.py           # Transformer harmonizer
│   ├── model_task4.py           # DDSP synth
│   ├── train_task2.py / train_task4.py
│   ├── sample_task2.py / sample_task4.py
│   ├── tokenizer.py             # ABC → chord token vocab
│   ├── teacher_synth.py         # offline MIDI→audio teacher (training labels for Task 4)
│   ├── data_task2.py / data_task4.py
│   ├── baselines.py             # bigram + repeat-last baselines
│   ├── metrics.py               # perplexity, top-k, multi-scale STFT, etc.
│   └── render_submission.py     # end-to-end melody → mp3
├── scripts/                     # one-off entry points (data, notebook build, smoke test)
├── samples/                     # intermediate demo artifacts (wavs gitignored)
├── docs/
│   ├── course/submission.md     # Course handoff checklist
│   ├── PRESENTATION.md          # 20-min talk script
│   └── video_url.txt            # link to the walkthrough recording
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
├── RUNBOOK.md                   # exact reproducibility commands
├── LICENSE                      # MIT
└── README.md
```

`data/`, `checkpoints/`, and `logs/` are populated by the pipeline and are
intentionally not tracked.

## Design notes

- **Trained from scratch.** No pre-trained models are loaded at any point; the
  harmonizer and synth both initialize randomly and learn from < 1 MB of MIDI
  and a few minutes of rendered audio. The Transformer fits in ~2 M params and
  the synth fits in ~1 M.
- **Data augmentation matters.** The harmonizer benefits dramatically from
  random key transposition and tempo jitter (`--augmentations 4`), pushing
  top-1 from ~45% to ~58%.
- **Polyphony via summation.** The Task 4 synth processes each voice
  independently and sums the resulting audio. This keeps the per-voice
  network small and lets the same checkpoint render arbitrary polyphony
  (the demo uses 4 voices).
- **Teacher-forced audio targets.** Task 4 is trained against an offline
  additive-synth teacher (`src/teacher_synth.py`) rather than recorded
  performances, which avoids the alignment problem and lets the loss focus on
  timbre and envelope rather than expressive timing.

## Acknowledgements

Originally developed as the Assignment 2 submission for **UCSD CSE 153R / 253
(Spring 2026)**. Data: the [Nottingham Music Database](https://github.com/jukedeck/nottingham-dataset)
for the harmonizer; a small subset of [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
for synth pre-training experiments. DDSP synth inspired by [Engel et al., 2020](https://arxiv.org/abs/2001.04643).

## License

MIT — see [`LICENSE`](LICENSE).
