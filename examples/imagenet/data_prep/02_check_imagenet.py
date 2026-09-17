"""
Verify the ImageNet-1k validation set and extract true_labels.npy.

Reads parquet files from IMAGENET_VAL_DIR (set in config.py) and
saves the ground-truth labels as data/probs/true_labels.npy.

Usage
-----
    python data_prep/02_check_imagenet.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # so config.py can be found regardless of working directory
from config import IMAGENET_VAL_DIR

PROBS_DIR = Path(__file__).parent.parent / "data" / "probs"
PROBS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROBS_DIR / "true_labels.npy"
VAL_DIR  = Path(IMAGENET_VAL_DIR)


if __name__ == "__main__":
    if OUT_PATH.exists():
        print("true_labels.npy already exists, skipping.")
    else:
        print("Loading validation labels...")
        dfs = [pd.read_parquet(f, columns=["label"])
               for f in sorted(VAL_DIR.glob("validation-*.parquet"))]
        labels = np.concatenate([df["label"].values for df in dfs]).astype(np.int64)

        assert len(labels) == 50_000
        assert labels.min() >= 0 and labels.max() <= 999
        assert len(np.unique(labels)) == 1000

        np.save(OUT_PATH, labels)
        print(f"Saved {len(labels)} labels to {OUT_PATH}")