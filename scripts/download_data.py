"""Download datasets.

We only need one external dataset (Nottingham ABC tunes, ~1 MB) — both
Task 2 (harmonization) and Task 4 (MIDI-to-audio) use it. Task 4 renders
the Nottingham MIDIs to audio on the fly with our teacher synth, so no
audio download is required.

MAESTRO would have been the natural Task 4 dataset, but its audio is only
distributed as a single 108 GB zip with no per-file CDN access. We document
this dead end in the workbook.

Usage:
    python scripts/download_data.py            # downloads Nottingham
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

# Ensure ``import src`` works when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NOTTINGHAM_URL = (
    "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.tar.gz"
)
MAESTRO_BASE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0"
MAESTRO_CSV = f"{MAESTRO_BASE}/maestro-v3.0.0.csv"


def _download(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ already present: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  ↓ {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
    print(f"    {dest.stat().st_size/1e6:.1f} MB")


def download_nottingham(dest_dir: Path) -> Path:
    print(f"[task2] Downloading Nottingham ABC dataset -> {dest_dir}")
    tar_path = dest_dir / "nottingham.tar.gz"
    _download(NOTTINGHAM_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    # Find a directory containing .abc files
    for p in dest_dir.rglob("*.abc"):
        abc_root = p.parent
        print(f"[task2] ABC files in: {abc_root}")
        return abc_root
    raise RuntimeError("No .abc files found after extraction.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nottingham-dir", default="data/nottingham")
    args = p.parse_args()
    download_nottingham(Path(args.nottingham_dir))
    print("\nAll done. The same dataset is used by both tasks.")


if __name__ == "__main__":
    main()
