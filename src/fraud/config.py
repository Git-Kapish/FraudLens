"""Load config.yaml and seed every RNG from it."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("FRAUD_CONFIG", ROOT / "config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read config.yaml once and return it as a dict."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(key: str) -> Path:
    """Return the absolute path for an entry under `paths:` in the config."""
    return ROOT / load_config()["paths"][key]


def seed_everything(seed: int | None = None) -> int:
    """Seed random, numpy and torch (if installed). Returns the seed used."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
