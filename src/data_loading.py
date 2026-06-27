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

# CSE-CIC-IDS2018 uses abbreviated column names; map them to 2017's full names
# so align_schemas can match on identical strings instead of dropping everything.
_CIC2018_RENAME = {
    'Dst Port':           'Destination Port',
    'Flow Byts/s':        'Flow Bytes/s',
    'Flow Pkts/s':        'Flow Packets/s',
    'Tot Fwd Pkts':       'Total Fwd Packets',
    'Tot Bwd Pkts':       'Total Backward Packets',
    'TotLen Fwd Pkts':    'Total Length of Fwd Packets',
    'TotLen Bwd Pkts':    'Total Length of Bwd Packets',
    'Fwd Pkt Len Max':    'Fwd Packet Length Max',
    'Fwd Pkt Len Min':    'Fwd Packet Length Min',
    'Fwd Pkt Len Mean':   'Fwd Packet Length Mean',
    'Fwd Pkt Len Std':    'Fwd Packet Length Std',
    'Bwd Pkt Len Max':    'Bwd Packet Length Max',
    'Bwd Pkt Len Min':    'Bwd Packet Length Min',
    'Bwd Pkt Len Mean':   'Bwd Packet Length Mean',
    'Bwd Pkt Len Std':    'Bwd Packet Length Std',
    'Fwd IAT Tot':        'Fwd IAT Total',
    'Bwd IAT Tot':        'Bwd IAT Total',
    'Fwd Header Len':     'Fwd Header Length',
    'Bwd Header Len':     'Bwd Header Length',
    'Fwd Pkts/s':         'Fwd Packets/s',
    'Bwd Pkts/s':         'Bwd Packets/s',
    'Pkt Len Min':        'Min Packet Length',
    'Pkt Len Max':        'Max Packet Length',
    'Pkt Len Mean':       'Packet Length Mean',
    'Pkt Len Std':        'Packet Length Std',
    'Pkt Len Var':        'Packet Length Variance',
    'Pkt Size Avg':       'Average Packet Size',
    'Fwd Seg Size Avg':   'Avg Fwd Segment Size',
    'Bwd Seg Size Avg':   'Avg Bwd Segment Size',
    'Fwd Byts/b Avg':     'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg':     'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg':   'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg':     'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg':     'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg':   'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts':   'Subflow Fwd Packets',
    'Subflow Fwd Byts':   'Subflow Fwd Bytes',
    'Subflow Bwd Pkts':   'Subflow Bwd Packets',
    'Subflow Bwd Byts':   'Subflow Bwd Bytes',
    'Init Fwd Win Byts':  'Init_Win_bytes_forward',
    'Init Bwd Win Byts':  'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':  'act_data_pkt_fwd',
    'Fwd Seg Size Min':   'min_seg_size_forward',
    'FIN Flag Cnt':       'FIN Flag Count',
    'SYN Flag Cnt':       'SYN Flag Count',
    'RST Flag Cnt':       'RST Flag Count',
    'PSH Flag Cnt':       'PSH Flag Count',
    'ACK Flag Cnt':       'ACK Flag Count',
    'URG Flag Cnt':       'URG Flag Count',
    'ECE Flag Cnt':       'ECE Flag Count',
    # index artifact in some pre-processed 2018 files
    'Unnamed: 0':         '__drop__',
}

# 2018-only columns with no 2017 equivalent — drop them
_DROP_2018 = {'Protocol', '__drop__'}


def load_cic2017(data_dir: str, subsample_frac: float = 0.10, seed: int = SEED) -> pd.DataFrame:
    """Load all CIC-IDS2017 per-day CSVs, keeping ~subsample_frac of rows."""
    csv_dir = _resolve_dir(data_dir, 'cic2017')
    frames = [_read_csv_subsampled(p, subsample_frac, seed) for p in _find_csvs(csv_dir)]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = _clean(df, drop_cols=_DROP_2017)
    print(f"CIC-IDS2017 loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_cic2018(data_dir: str, subsample_frac: float = 0.10, seed: int = SEED) -> pd.DataFrame:
    """Load all CSE-CIC-IDS2018 per-day CSVs, keeping ~subsample_frac of rows."""
    csv_dir = _resolve_dir(data_dir, 'cic2018')
    frames = [_read_csv_subsampled(p, subsample_frac, seed) for p in _find_csvs(csv_dir)]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat(frames, ignore_index=True)
    # Normalise column names first, then rename to match 2017 schema
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_CIC2018_RENAME)
    df = _clean(df, drop_cols=_DROP_2018)
    print(f"CSE-CIC-IDS2018 loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def align_schemas(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only columns present in both DataFrames (Label always kept).
    After rename, this should drop at most 1–2 columns (e.g. Down/Up Ratio).
    """
    feature_cols      = [c for c in train.columns if c != LABEL_COL]
    test_feature_cols = [c for c in test.columns  if c != LABEL_COL]
    shared = sorted(set(feature_cols) & set(test_feature_cols))
    cols = shared + [LABEL_COL]

    dropped_train = set(feature_cols) - set(shared)
    dropped_test  = set(test_feature_cols) - set(shared)
    if dropped_train:
        print(f"Columns only in train (dropped): {sorted(dropped_train)}")
    if dropped_test:
        print(f"Columns only in test  (dropped): {sorted(dropped_test)}")

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
    """Read one CSV in chunks, keeping ~frac of rows to avoid OOM on free Colab."""
    rng = np.random.default_rng(seed)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False, encoding='latin-1'):
        mask = rng.random(len(chunk)) < frac
        chunks.append(chunk.loc[mask])
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {os.path.basename(path)}: {df.shape[0]:,} rows sampled")
    return df


def _clean(df: pd.DataFrame, drop_cols: set) -> pd.DataFrame:
    # Strip column name whitespace (idempotent if already done)
    df.columns = df.columns.str.strip()

    # Drop unwanted columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure label column exists
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Expected label column '{LABEL_COL}' not found. "
            f"Available: {list(df.columns)}"
        )

    labels = df[LABEL_COL].copy()
    df = df.drop(columns=[LABEL_COL])

    # Coerce all features to numeric, replace ±inf → NaN, drop NaN rows
    df = df.apply(pd.to_numeric, errors='coerce')
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

    df[LABEL_COL] = labels.values
    return df.reset_index(drop=True)
