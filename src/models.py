"""
Model factory for IDS progressive evaluation.

Six scikit-learn classifiers mirroring Chua & Salam (2023):
    DT  — DecisionTreeClassifier
    RF  — RandomForestClassifier
    SVM — SVC (RBF kernel)
    NB  — GaussianNB
    ANN — MLPClassifier (1 hidden layer)
    DNN — MLPClassifier (3 hidden layers)

Key constants
    SEED            reproducibility seed (default 42)
    PARAM_GRIDS     GridSearchCV search spaces mirroring the paper
    SVM_TUNE_FRAC   fraction of training data used for SVM GridSearch (O(n²) mitigation)
    SVM_TRAIN_FRAC  fraction of training data used for SVM final fit
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score

SEED = 42
MODEL_NAMES = ['DT', 'RF', 'SVM', 'NB', 'ANN', 'DNN']

# SVC scales ~O(n²); limit training data to avoid multi-hour runs on free Colab.
SVM_TUNE_FRAC  = 0.10   # fraction used for GridSearchCV
SVM_TRAIN_FRAC = 0.20   # fraction used for final fit (more data → better model)


# ── model factory ─────────────────────────────────────────────────────────────

def make_model(name: str, params: dict = None, seed: int = SEED):
    """Return a fresh unfitted estimator, optionally with params applied."""
    _base = {
        'DT':  DecisionTreeClassifier(random_state=seed),
        'RF':  RandomForestClassifier(random_state=seed, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=seed, probability=True),
        'NB':  GaussianNB(),
        'ANN': MLPClassifier(
                   random_state=seed, max_iter=1000,
                   early_stopping=True, n_iter_no_change=20,
               ),
        'DNN': MLPClassifier(
                   random_state=seed, max_iter=1000,
                   early_stopping=True, n_iter_no_change=20,
               ),
    }
    if name not in _base:
        raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")
    clf = _base[name]
    if params:
        clf.set_params(**params)
    return clf


def make_all_models(seed: int = SEED) -> dict:
    """Return dict of all 6 unfitted estimators."""
    return {name: make_model(name, seed=seed) for name in MODEL_NAMES}


# ── hyperparameter grids ──────────────────────────────────────────────────────

# Search spaces mirror the paper's 03_Optimize_Hyperparameter.ipynb.
# ccp_alpha=1.44e-5 is the paper's reported best value for DT — included directly.
PARAM_GRIDS = {
    'DT': {
        'ccp_alpha': [0.0, 1.44e-5, 5e-5, 1e-4],
    },
    'RF': {
        'n_estimators':     [100, 350],
        'max_depth':        [10, 20, None],
        'min_samples_leaf': [1, 1e-5],
        'criterion':        ['gini'],
    },
    'SVM': {
        'C':      [1, 10, 100],
        'gamma':  [0.01, 0.1, 1],
        'kernel': ['rbf'],
    },
    'NB': {
        'var_smoothing': [1e-9, 1e-5, 1.0],
    },
    'ANN': {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation':         ['tanh', 'relu'],
        'solver':             ['adam'],
        'alpha':              [1e-4, 1e-3],
    },
    'DNN': {
        'hidden_layer_sizes': [(15, 15, 15), (50, 50, 50)],
        'activation':         ['tanh'],
        'solver':             ['adam'],
        'alpha':              [1e-5, 1e-4],
    },
}


# ── internal helpers ──────────────────────────────────────────────────────────

def _subsample(X: pd.DataFrame, y: pd.Series, frac: float, seed: int):
    """Random subsample at given fraction; returns aligned (X_sub, y_sub)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max(1, int(len(X) * frac)), replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


# ── GridSearchCV ──────────────────────────────────────────────────────────────

def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    seed: int = SEED,
) -> tuple:
    """
    GridSearchCV for one model.
    Returns (best_params_dict, best_cv_accuracy).
    SVM is tuned on SVM_TUNE_FRAC of X_train to keep runtime manageable.
    """
    X_fit, y_fit = X_train, y_train
    if name == 'SVM':
        X_fit, y_fit = _subsample(X_train, y_train, SVM_TUNE_FRAC, seed)
        print(f"  SVM: tuning on {len(X_fit):,} rows ({SVM_TUNE_FRAC:.0%} subsample — O(n²) mitigation)")

    gs = GridSearchCV(
        make_model(name, seed=seed),
        PARAM_GRIDS[name],
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_fit, y_fit)
    return gs.best_params_, float(gs.best_score_)


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: str = None,
    cv: int = 5,
    seed: int = SEED,
) -> dict:
    """
    Tune all 6 models via GridSearchCV.
    Skips any model whose params file already exists in save_dir (resume-safe).
    Returns {model_name: best_params_dict}.
    """
    results = {}
    for name in MODEL_NAMES:
        path = os.path.join(save_dir, f'best_params_{name}.joblib') if save_dir else None
        if path and os.path.exists(path):
            results[name] = joblib.load(path)
            print(f"  {name}: reloaded params → {results[name]}")
            continue

        print(f"\n{'─' * 52}\nGridSearchCV: {name}  (k={cv})\n{'─' * 52}")
        best_params, best_score = tune_model(name, X_train, y_train, cv=cv, seed=seed)
        results[name] = best_params
        print(f"  → best params:      {best_params}")
        print(f"  → best CV accuracy: {best_score:.4f}")
        if path:
            joblib.dump(best_params, path)
            print(f"  → saved to {path}")

    return results


# ── k-fold cross-validation ───────────────────────────────────────────────────

def cross_validate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    cv: int = 5,
    seed: int = SEED,
    save_dir: str = None,
) -> pd.DataFrame:
    """
    k-fold CV for all 6 models with best hyperparameters.
    SVM is evaluated on SVM_TUNE_FRAC subsample (same as GridSearch) to keep runtime manageable.
    If save_dir is given, results are saved to cv_results.joblib and reloaded on the next call.
    Returns DataFrame with mean/std accuracy per model.
    """
    path = os.path.join(save_dir, 'cv_results.joblib') if save_dir else None
    if path and os.path.exists(path):
        df = joblib.load(path)
        print(f"CV results loaded from Drive ({len(df)} models). Delete cv_results.joblib to re-run.")
        print(df.to_string())
        return df

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scorer   = make_scorer(accuracy_score)
    rows = []

    for name in MODEL_NAMES:
        if name not in best_params:
            continue
        X_cv, y_cv = X_train, y_train
        note = ''
        if name == 'SVM':
            X_cv, y_cv = _subsample(X_train, y_train, SVM_TUNE_FRAC, seed)
            note = f'subsample {SVM_TUNE_FRAC:.0%}'

        clf = make_model(name, params=best_params[name], seed=seed)
        result = cross_validate(clf, X_cv, y_cv, cv=splitter,
                                scoring={'acc': scorer}, n_jobs=-1)
        acc = result['test_acc']
        rows.append({
            'Model':       name,
            'CV Mean Acc': round(float(acc.mean()), 4),
            'CV Std':      round(float(acc.std()), 4),
            'CV Min':      round(float(acc.min()), 4),
            'CV Max':      round(float(acc.max()), 4),
            'Note':        note,
        })
        suffix = f'  ({note})' if note else ''
        print(f"  {name:4s}: {acc.mean():.4f} ± {acc.std():.4f}{suffix}")

    df = pd.DataFrame(rows).set_index('Model')
    if path:
        joblib.dump(df, path)
        print(f"CV results saved to Drive.")
    return df


# ── final training and persistence ────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    save_dir: str = None,
    seed: int = SEED,
) -> dict:
    """
    Fit each model on the training set using best hyperparameters.
    SVM is fitted on SVM_TRAIN_FRAC (mirrors the paper's approach for O(n²) scaling).
    Saves/reloads from save_dir to survive Colab disconnects.
    Returns {name: fitted_estimator}.
    """
    models = {}
    for name in MODEL_NAMES:
        if name not in best_params:
            continue
        path = os.path.join(save_dir, f'model_{name}.joblib') if save_dir else None
        if path and os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  {name}: loaded from Drive")
            continue

        X_fit, y_fit = X_train, y_train
        note = ''
        if name == 'SVM':
            X_fit, y_fit = _subsample(X_train, y_train, SVM_TRAIN_FRAC, seed)
            note = f' ({SVM_TRAIN_FRAC:.0%} subsample — O(n²))'

        print(f"  Training {name} on {len(X_fit):,} rows{note} ...")
        clf = make_model(name, params=best_params[name], seed=seed)
        clf.fit(X_fit, y_fit)
        models[name] = clf

        if path:
            joblib.dump(clf, path)
            print(f"  {name}: saved to Drive")

    return models


def load_models(save_dir: str) -> dict:
    """Reload all saved models from save_dir."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(save_dir, f'model_{name}.joblib')
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: not found at {path}")
    return models
