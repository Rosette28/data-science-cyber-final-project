"""
Data loading for CIC-IDS2017 (train) and CSE-CIC-IDS2018 (progressive test).

Expected directory layout under DATA_DIR:
    DATA_DIR/
        cic2017/   ← all per-day CIC-IDS2017 CSVs
        cic2018/   ← all per-day CSE-CIC-IDS2018 CSVs

If those subdirs don't exist the functions fall back to scanning DATA_DIR directly.
"""

import os
import glob
import numpy as np
import pandas as pd

LABEL_COL = 'Label'
SEED = 42

# Column present only in CIC-IDS2017 (duplicate of 'Fwd Header Length')
_DROP_2017 = {'Fwd Header Length.1'}


def load_cic2017(data_dir: str, subsample_frac: float = 0.10, seed: int = SEED) -> pd.DataFrame:
    """Load all CIC-IDS2017 per-day CSVs, keeping ~subsample_frac of rows."""
    csv_dir = _resolve_dir(data_dir, 'cic2017')
    frames = [_read_csv_subsampled(p, subsample_frac, seed) for p in _find_csvs(csv_dir)]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = _clean(df, drop_cols=_DROP_2017, label_col=LABEL_COL)
    print(f"CIC-IDS2017 loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_cic2018(data_dir: str, subsample_frac: float = 0.10, seed: int = SEED) -> pd.DataFrame:
    """Load all CSE-CIC-IDS2018 per-day CSVs, keeping ~subsample_frac of rows."""
    csv_dir = _resolve_dir(data_dir, 'cic2018')
    frames = [_read_csv_subsampled(p, subsample_frac, seed) for p in _find_csvs(csv_dir)]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = _clean(df, drop_cols=set(), label_col=LABEL_COL)
    print(f"CSE-CIC-IDS2018 loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def align_schemas(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only columns present in both DataFrames (label always kept).
    Call this after loading both datasets to ensure identical feature sets.
    """
    feature_cols = [c for c in train.columns if c != LABEL_COL]
    test_feature_cols = [c for c in test.columns if c != LABEL_COL]
    shared = sorted(set(feature_cols) & set(test_feature_cols))
    cols = shared + [LABEL_COL]
    dropped_train = set(feature_cols) - set(shared)
    dropped_test = set(test_feature_cols) - set(shared)
    if dropped_train:
        print(f"Dropped from train (not in test): {dropped_train}")
    if dropped_test:
        print(f"Dropped from test (not in train): {dropped_test}")
    return train[cols], test[cols]


# ── internals ────────────────────────────────────────────────────────────────

def _resolve_dir(base: str, subdir: str) -> str:
    candidate = os.path.join(base, subdir)
    return candidate if os.path.isdir(candidate) else base


def _find_csvs(directory: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(directory, '*.csv')))
    print(f"Found {len(paths)} CSV file(s) in {directory}")
    return paths


def _read_csv_subsampled(path: str, frac: float, seed: int) -> pd.DataFrame:
    """Read one CSV in chunks and keep ~frac of rows to avoid OOM on free Colab."""
    rng = np.random.default_rng(seed)
    chunks = []
    for chunk in pd.read_csv(
        path,
        chunksize=50_000,
        low_memory=False,
        encoding='latin-1',    # CIC datasets use Windows-1252 / latin-1
    ):
        mask = rng.random(len(chunk)) < frac
        chunks.append(chunk.loc[mask])
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {os.path.basename(path)}: {df.shape[0]:,} rows sampled")
    return df


def _clean(df: pd.DataFrame, drop_cols: set, label_col: str) -> pd.DataFrame:
    # Normalise column names: strip surrounding whitespace
    df.columns = df.columns.str.strip()

    # Drop dataset-specific duplicate columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure label column exists
    if label_col not in df.columns:
        raise KeyError(f"Expected label column '{label_col}' not found. "
                       f"Available: {list(df.columns)}")

    # Separate features and label before numeric coercion
    labels = df[label_col].copy()
    df = df.drop(columns=[label_col])

    # Coerce all feature columns to numeric, turn non-numeric → NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace ±inf with NaN, then drop any row with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    labels = labels.loc[df.index]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with inf/NaN ({dropped / before:.2%})")

    # Downcast float64 → float32 to halve memory usage
    float_cols = df.select_dtypes(include='float64').columns
    df[float_cols] = df[float_cols].astype(np.float32)

    df[label_col] = labels.values
    df = df.reset_index(drop=True)
    return df
