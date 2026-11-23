import numpy as np
import torch

from scripts import smoke_test
from src.metrics import top_k_accuracy
from src.midi_io import parse_chord_symbol
from src.tokenizer import chord_from_token_id, chord_token_id


def test_synthetic_pipeline_smoke():
    smoke_test.main()


def test_chord_tokenizer_round_trip():
    chord = parse_chord_symbol("Dm7")
    decoded = chord_from_token_id(chord_token_id(chord))
    assert decoded == chord


def test_top_k_accuracy_ignores_padding():
    logits = torch.tensor([[0.0, 5.0, 2.0], [0.0, 4.0, 3.0], [0.0, 2.0, 4.0]])
    targets = np.array([1, 2, 0])
    assert top_k_accuracy(logits.numpy(), targets, k=1) == 0.5
    assert top_k_accuracy(logits.numpy(), targets, k=2) == 1.0
