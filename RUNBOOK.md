# Reproduction runbook

Run these commands from the repository root. PowerShell and Bash versions are
included.

## 0. Environment

```powershell
# Python 3.11 with CUDA-enabled PyTorch already installed (verified with RTX 4070 / 8 GB).
python -m pip install -r requirements.txt
```

```bash
# Python 3.11 with CUDA-enabled PyTorch already installed (verified with RTX 4070 / 8 GB).
python -m pip install -r requirements.txt
```

Verify the GPU is visible to PyTorch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For the pinned Python 3.11 CPU environment, install `requirements-lock.txt` using
the PyTorch CPU index as shown below. For GPU training, install a CUDA-enabled
PyTorch build and use `requirements.txt`.

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

## 1. Download the dataset (one-time)

The pipeline uses [Nottingham ABC](https://github.com/jukedeck/nottingham-dataset)
(~1 MB) for both tasks. Task 4 renders audio from these MIDIs on the fly using
the teacher synth in `src/teacher_synth.py`, so no audio download is needed.

```powershell
python scripts/download_data.py
```

```bash
python scripts/download_data.py
```

This populates `data/nottingham/nottingham-dataset-master/ABC_cleaned/` with
~14 ABC files containing ~1000 folk tunes.

## 2. Smoke test (optional but recommended)

```powershell
python scripts/smoke_test.py
```

```bash
python scripts/smoke_test.py
```

Verifies that the chord parser, ABC parser, tokenizer, Task 2 model, Task 4
DDSP synth, baselines and metrics all work on tiny synthetic data. Runs in
~10 seconds on CPU.

## 3. Train Task 2 (harmonization Transformer)

```powershell
python -m src.train_task2 `
    --abc-root data/nottingham/nottingham-dataset-master/ABC_cleaned `
    --epochs 40 --batch-size 64 --d-model 256 --n-layers 4 --augmentations 4
```

```bash
python -m src.train_task2 \
    --abc-root data/nottingham/nottingham-dataset-master/ABC_cleaned \
    --epochs 40 --batch-size 64 --d-model 256 --n-layers 4 --augmentations 4
```

Expected runtime: ~6-7 minutes on an RTX 4070 Laptop (8 GB).
Expected outcome:

* `checkpoints/task2/best.pt`  ← best val checkpoint (~epoch 19)
* `checkpoints/task2/last.pt`  ← last checkpoint
* `checkpoints/task2/history.json`
* Val perplexity ≈ 4.0, top-1 ≈ 58%, top-5 ≈ 91%.

## 4. Train Task 4 (polyphonic DDSP synth)

```powershell
python -m src.train_task4 `
    --source abc --epochs 25 --batch-size 6 --max-sources 40 `
    --clip-seconds 3.0 --crops-per-clip 6 --max-voices 4 `
    --n-harmonics 48 --hidden 256 --lr 4e-4
```

```bash
python -m src.train_task4 \
    --source abc --epochs 25 --batch-size 6 --max-sources 40 \
    --clip-seconds 3.0 --crops-per-clip 6 --max-voices 4 \
    --n-harmonics 48 --hidden 256 --lr 4e-4
```

Expected runtime: ~4 minutes on an RTX 4070 Laptop.
Expected outcome:

* `checkpoints/task4/best.pt`
* `checkpoints/task4/history.json`
* Multi-scale STFT loss ≈ 9 (down from 22 at init).

## 5. Render the MIDI and audio artifacts

```powershell
python -m src.render_submission
```

```bash
python -m src.render_submission
```

Produces:

* `symbolic_conditioned.mid` — Task 2 output, harmonized "Twinkle Twinkle".
* `samples/final.wav` — intermediate wav.
* `continuous_conditioned.mp3` — Task 4 output, 128 kbps mp3.

To use a different input melody, edit `src/sample_task2.py` (function
`twinkle_twinkle`) or pass a different leadsheet to `harmonize_leadsheet`.

## 6. Execute the notebook + export HTML

Install `requirements-dev.txt` to get Jupyter and nbconvert before running these commands.

```powershell
python scripts/execute_notebook.py
python -m nbconvert --to html workbook.ipynb
```

```bash
python scripts/execute_notebook.py
python -m nbconvert --to html workbook.ipynb
```

The first command runs the notebook in-process (the kernel subprocess has a
torch DLL load issue on this Windows machine — see `scripts/execute_notebook.py`
for the workaround). The second produces `workbook.html`.

See [Course presentation and submission](docs/course/submission.md) for the assignment checklist.

## Troubleshooting

* **OSError loading torch DLL in Jupyter** — this is a Windows DLL search
  path issue with `ipykernel`. Workaround: use `scripts/execute_notebook.py`
  instead of `jupyter nbconvert --execute`. The script imports torch in the
  parent Python process where it works correctly.
* **`pretty_midi` not found** — `pip install pretty_midi` (already in
  `requirements.txt`).
* **ffmpeg not on PATH** — `pip install imageio-ffmpeg`; the bundled binary is
  used by `src/sample_task4.encode_mp3`.
* **CUDA OOM during Task 4 training** — drop `--batch-size` to 4 or
  `--clip-seconds` to 2.0; the model fits comfortably in 6 GB at those
  settings.
* **Loss is NaN early in training** — almost always means a bad LR. Drop
  `--lr` by 2× and retry.
