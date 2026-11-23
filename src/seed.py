"""Global seed control so plots / metrics / generated samples are reproducible."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 153, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True) -> "str":
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"
