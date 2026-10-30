"""Download the ULB credit card fraud dataset from Kaggle and validate it."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fraud.config import resolve_path
from fraud.data import load_raw, validate_schema

DATASET = "mlg-ulb/creditcardfraud"
MANUAL_STEPS = """
Kaggle credentials not found. Download the data manually:
  1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and download the zip.
  2. Unzip creditcard.csv into data/raw/creditcard.csv
Or set up the API: create a token at https://www.kaggle.com/settings and save it as
~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY), then rerun `make data`.
"""

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("download_data")


def has_credentials() -> bool:
    """True if a Kaggle token file or env vars are present."""
    env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return bool(env) or (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> int:
    target = resolve_path("raw_data")
    if not target.exists():
        if not has_credentials():
            print(MANUAL_STEPS)
            return 1
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", DATASET, target.parent)
        api.dataset_download_files(DATASET, path=str(target.parent), unzip=True)
    validate_schema(load_raw(target))
    logger.info("Data OK at %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
