"""Metrics, cross-validation, timing and plots for occupancy models."""
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

import config


def get_metrics(y_true, y_pred, model_name):
    """Classification metrics for one model. Occupied (1) is the positive class.

    Recall matters most for ventilation: a missed occupied period means people sit in
    under-ventilated air, whereas a false alarm only wastes energy.
    """
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_all_models(y_test, predictions_dict):
    """Metrics table with one row per entry of ``{model_name: predictions}``."""
    metrics_summary = []
    for name, pred in predictions_dict.items():
        metrics_summary.append(get_metrics(y_test, pred, name))

    summary_df = pd.DataFrame(metrics_summary)
    return summary_df


def time_call(func, *args, repeats=3, **kwargs):
    """Run ``func`` ``repeats`` times; return (last result, best wall time in seconds)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        best = min(best, time.perf_counter() - start)
    return result, best


def time_series_cv(fit_predict, X, y, n_splits=config.CV_SPLITS):
    """Expanding-window cross-validation that never trains on future data.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(X_train, y_train, X_val)`` returning predictions for ``X_val``.

    Returns
    -------
    pandas.DataFrame
        Metrics for each fold.
    """
    rows = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        pred = fit_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx])
        row = get_metrics(y.iloc[val_idx], pred, "")
        row.pop('Model')
        row['Fold'] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_matrices(y_test, predictions_dict, save_path=None):
    """Plot one confusion matrix per model; save to ``save_path`` or show interactively."""
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(5 * len(predictions_dict), 4.5))
    if len(predictions_dict) == 1:
        axes = [axes]

    for i, (name, pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Vacant', 'Occupied'],
                    yticklabels=['Vacant', 'Occupied'])
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()
