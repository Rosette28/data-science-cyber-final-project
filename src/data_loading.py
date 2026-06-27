"""
Data loading utilities for CIC-IDS2017 and CSE-CIC-IDS2018.
Implemented in Phase 2 (task 2.2).
"""

import os
import glob
import pandas as pd
import numpy as np


# CIC-IDS2017 has 78 columns; CSE-CIC-IDS2018 has 80 columns.
# Two columns present only in 2017 are dropped so schemas align.
_2017_ONLY_COLS = ['Fwd Header Length.1']

LABEL_COL_2017 = ' Label'
LABEL_COL_2018 = 'Label'


def load_cic2017(data_dir: str, subsample_frac: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Load and concatenate all CIC-IDS2017 per-day CSVs, keeping subsample_frac rows."""
    raise NotImplementedError("Implement in Phase 2 — task 2.2")


def load_cic2018(data_dir: str, subsample_frac: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Load and concatenate CSE-CIC-IDS2018 CSVs, harmonise schema to match 2017."""
    raise NotImplementedError("Implement in Phase 2 — task 2.2")


def _read_csv_subsample(path: str, frac: float, seed: int) -> pd.DataFrame:
    """Read a single CSV file, keeping only frac of rows. Uses chunked reading to avoid OOM."""
    chunks = []
    rng = np.random.default_rng(seed)
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        mask = rng.random(len(chunk)) < frac
        chunks.append(chunk.loc[mask])
    return pd.concat(chunks, ignore_index=True)
