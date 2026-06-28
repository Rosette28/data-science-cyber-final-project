"""
Preprocessing pipeline for IDS progressive evaluation.

Provides modular functions covering:
    binary_relabel        – collapse multi-class attack labels → binary
    balance_1to1          – random downsample to 1:1 class ratio
    add_derived_features  – four engineered flow features with cyber rationale
    get_feature_cols      – helper: all columns except Label
    fit_scaler            – StandardScaler fitted on training data
    apply_scaler          – apply pre-fitted scaler to any set
    select_features_rf    – RF importance ranking, returns top-N feature names
    brute_force_select    – add-one-feature CV accuracy curve (mirrors paper)
    compute_vif           – Variance Inflation Factor for redundancy analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
LABEL_COL = "Label"
BINARY_POS = "ATTACK"
BINARY_NEG = "BENIGN"


# ── §3.2  Binary relabelling ──────────────────────────────────────────────────

def binary_relabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all non-BENIGN labels to 'ATTACK'. Returns a copy.

    Mirrors the authors' Step 1 of 01_Dataset_Preprocessing.ipynb.
    Original multiclass labels are lost; keep a separate copy if needed
    for the optional multiclass extension (Phase 8.3).
    """
    df = df.copy()
    df[LABEL_COL] = np.where(
        df[LABEL_COL].str.strip().str.upper() == "BENIGN",
        BINARY_NEG,
        BINARY_POS,
    )
    return df


# ── §3.3  Class balancing ─────────────────────────────────────────────────────

def balance_1to1(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downsample the majority class so both classes have equal count.

    Returns:
        df_balanced  – 1:1 balanced copy (mirrors the paper's training data)
        df_original  – unchanged copy at real prevalence (for Phase 8.2)

    The benign class is always the majority in these datasets (~80–84%).
    Downsampling is random; seed ensures reproducibility.
    """
    n_min = min(
        (df[LABEL_COL] == BINARY_POS).sum(),
        (df[LABEL_COL] == BINARY_NEG).sum(),
    )
    df_balanced = (
        pd.concat(
            [
                df[df[LABEL_COL] == cls].sample(n=n_min, random_state=seed)
                for cls in [BINARY_NEG, BINARY_POS]
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return df_balanced, df.copy()


# ── §3.5  Derived (engineered) features ──────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append four engineered columns with documented network-security rationale.

    New columns:
        feat_bwd_fwd_ratio   – flow directionality; near 0 = unidirectional DoS
        feat_pkt_len_range   – size variance; 0 = fixed-size flood (DDoS)
        feat_bytes_per_pkt   – time-normalised payload density
        feat_win_ratio       – TCP window asymmetry; deviates from OS defaults in bots/scanners

    All four are computed from base features present after schema alignment,
    so the function is safe to call on train and test independently
    (no target leakage, no cross-set statistics).
    """
    df = df.copy()

    def _col(name: str, default: float = 0.0) -> pd.Series:
        return df[name].astype(float) if name in df.columns else pd.Series(default, index=df.index)

    fwd   = _col("Total Fwd Packets",      1.0)
    bwd   = _col("Total Backward Packets")
    p_max = _col("Max Packet Length")
    p_min = _col("Min Packet Length")
    b_s   = _col("Flow Bytes/s")
    p_s   = _col("Flow Packets/s",         1.0)
    w_fwd = _col("Init_Win_bytes_forward")
    w_bwd = _col("Init_Win_bytes_backward", 1.0)

    df["feat_bwd_fwd_ratio"] = (bwd / (fwd + 1e-6)).clip(-1e6, 1e6).astype(np.float32)
    df["feat_pkt_len_range"] = (p_max - p_min).clip(0).astype(np.float32)
    df["feat_bytes_per_pkt"] = (b_s / (p_s + 1e-6)).clip(-1e8, 1e8).astype(np.float32)
    df["feat_win_ratio"]     = (w_fwd / (w_bwd.abs() + 1e-6)).clip(-1e6, 1e6).astype(np.float32)

    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all column names except Label."""
    return [c for c in df.columns if c != LABEL_COL]


# ── §3.6  Feature scaling ─────────────────────────────────────────────────────

def fit_scaler(
    X_train: pd.DataFrame,
) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit StandardScaler on training features.
    Returns (fitted_scaler, X_train_scaled).
    SVM and neural nets require zero-mean, unit-variance inputs;
    tree-based models ignore scaling but it does no harm.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train).astype(np.float32),
        columns=X_train.columns,
        index=X_train.index,
    )
    return scaler, X_scaled


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a pre-fitted StandardScaler to X."""
    return pd.DataFrame(
        scaler.transform(X).astype(np.float32),
        columns=X.columns,
        index=X.index,
    )


# ── §3.7  Feature selection ───────────────────────────────────────────────────

def select_features_rf(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
    seed: int = SEED,
) -> tuple[list[str], pd.Series]:
    """
    Fit a RandomForest on X, y and return
        (top_n_feature_names_sorted_by_importance, full_importance_series).
    Mirrors the authors' Stage 1 of 02_Feature_Selection.ipynb.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(top_n).index.tolist(), imp


def brute_force_select(
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    max_features: int = 20,
    subsample_n: int = 10_000,
    cv: int = 3,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Stage 2 of the authors' feature selection: add one RF-ranked feature at a time
    and record k-fold CV accuracy for three representative models.

    Uses a subsample (default 10 k rows) for speed; mirrors the paper's approach.

    Returns a DataFrame with columns:
        ['n_features', 'LinearSVC', 'NaiveBayes', 'MLP']
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(subsample_n, len(X)), replace=False)
    X_s = X.iloc[idx].reset_index(drop=True)
    y_s = y.iloc[idx].reset_index(drop=True)

    models = {
        "LinearSVC":  LinearSVC(max_iter=2_000, random_state=seed),
        "NaiveBayes": GaussianNB(),
        "MLP":        MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=seed),
    }

    ranked = [f for f in ranked_features if f in X_s.columns][:max_features]
    rows = []
    for n in range(1, len(ranked) + 1):
        feats = ranked[:n]
        row: dict = {"n_features": n}
        for name, clf in models.items():
            scores = cross_val_score(clf, X_s[feats], y_s, cv=cv, scoring="accuracy", n_jobs=-1)
            row[name] = float(scores.mean())
        rows.append(row)
        print(
            f"  n={n:2d}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "n_features")
        )

    return pd.DataFrame(rows)


# ── §3.8  Redundancy analysis ─────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each column in X via sklearn R².
    VIF = 1 / (1 - R²) where R² is from regressing each feature on all others.
    VIF > 5 → notable multicollinearity; VIF > 10 → severe.
    """
    cols = X.columns.tolist()
    vif_vals = []
    X_arr = X.values.astype(float)
    for i, col in enumerate(cols):
        y_col = X_arr[:, i]
        others = np.delete(X_arr, i, axis=1)
        r2 = LinearRegression().fit(others, y_col).score(others, y_col)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_vals.append(vif)
    return (
        pd.DataFrame({"feature": cols, "VIF": vif_vals})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
