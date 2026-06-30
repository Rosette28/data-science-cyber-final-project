"""
Evaluation helpers for IDS progressive evaluation.

Metric choices beyond the paper (Accuracy / Precision / Recall / F1):
    F2      β=2 weights recall twice over precision; in IDS, a missed attack
            (FN) is more costly than a false alarm (FP), so the metric
            should penalise FN more than FP.
    MCC     Matthews Correlation Coefficient: a single balanced metric that
            accounts for all four confusion-matrix cells.  On an 84%-benign
            test set a naive "always BENIGN" classifier achieves 84% accuracy
            but MCC = 0 — MCC correctly signals no skill.
    ROC-AUC Area under the ROC curve: threshold-independent summary of the
            full precision-recall trade-off space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BETA        = 2          # F-beta parameter
POS_LABEL   = 'ATTACK'
NEG_LABEL   = 'BENIGN'
MODEL_NAMES = ['DT', 'RF', 'SVM', 'NB', 'ANN', 'DNN']


# ── per-model metrics ─────────────────────────────────────────────────────────

def evaluate_one(clf, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a dict of classification metrics for one fitted classifier."""
    y_pred = clf.predict(X)

    # Probability scores for ROC-AUC (ATTACK = positive class)
    if hasattr(clf, 'predict_proba'):
        classes = list(clf.classes_)
        pos_idx = classes.index(POS_LABEL) if POS_LABEL in classes else 1
        y_prob  = clf.predict_proba(X)[:, pos_idx]
    else:
        y_prob = None

    kw = dict(pos_label=POS_LABEL, zero_division=0)
    m = {
        'Accuracy':    round(float(accuracy_score(y, y_pred)),                 4),
        'Precision':   round(float(precision_score(y, y_pred, **kw)),           4),
        'Recall':      round(float(recall_score(y, y_pred, **kw)),              4),
        'F1':          round(float(f1_score(y, y_pred, **kw)),                  4),
        f'F{BETA}':    round(float(fbeta_score(y, y_pred, beta=BETA, **kw)),    4),
        'MCC':         round(float(matthews_corrcoef(y, y_pred)),                4),
    }
    if y_prob is not None:
        y_bin        = (np.asarray(y) == POS_LABEL).astype(int)
        m['ROC-AUC'] = round(float(roc_auc_score(y_bin, y_prob)), 4)
    else:
        m['ROC-AUC'] = float('nan')
    return m


# ── all models ────────────────────────────────────────────────────────────────

def evaluate_all(
    models:    dict,
    X:         pd.DataFrame,
    y:         pd.Series,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Evaluate every model in `models` on (X, y).
    Returns a DataFrame indexed by model name.
    If save_path is given, persists the result with joblib (resume-safe).
    """
    import joblib, os
    if save_path and os.path.exists(save_path):
        df = joblib.load(save_path)
        print(f"Results loaded from Drive ({len(df)} models). "
              f"Delete {os.path.basename(save_path)} to re-run.")
        print(df.to_string())
        return df

    rows = []
    for name in MODEL_NAMES:
        if name not in models:
            continue
        print(f"  Evaluating {name} ...", end=' ', flush=True)
        m         = evaluate_one(models[name], X, y)
        m['Model'] = name
        rows.append(m)
        print(f"acc={m['Accuracy']:.4f}  MCC={m['MCC']:.4f}  ROC-AUC={m.get('ROC-AUC', 'n/a')}")

    df = pd.DataFrame(rows).set_index('Model')
    if save_path:
        joblib.dump(df, save_path)
        print(f"\nResults saved → {save_path}")
    return df


# ── confusion matrix grid ─────────────────────────────────────────────────────

def confusion_grid(
    models:    dict,
    X:         pd.DataFrame,
    y:         pd.Series,
    title:     str   = '',
    save_path: str   = None,
    figsize:   tuple = (18, 10),
) -> None:
    """3-column grid of confusion matrices for all models in MODEL_NAMES order."""
    names  = [n for n in MODEL_NAMES if n in models]
    ncols  = 3
    nrows  = (len(names) + ncols - 1) // ncols
    labels = [NEG_LABEL, POS_LABEL]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        y_pred = models[name].predict(X)
        cm     = confusion_matrix(y, y_pred, labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, colorbar=False, cmap='Blues'
        )
        acc = accuracy_score(y, y_pred)
        ax.set_title(f'{name}  (acc={acc:.4f})', fontsize=11)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    plt.show()
