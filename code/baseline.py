"""Simple rule-based baselines that any ML model must beat to be worth deploying."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import config


def run_baseline_model(X_test, threshold=500.0):
    """CO2 threshold rule: predict occupied when CO2 exceeds `threshold` ppm."""
    return (X_test["co2"] > threshold).astype(int)


def fit_co2_threshold(X_train, y_train, candidates=None):
    """Choose the CO2 threshold (ppm) that maximises balanced accuracy on the training data.

    Balanced accuracy is used rather than F1 because, with most bins occupied, F1 is
    maximised by the trivial threshold that labels every bin occupied.

    Returns
    -------
    float
        The best threshold, to be applied unchanged to the test data.
    """
    if candidates is None:
        candidates = np.arange(350.0, 1000.0, 5.0)
    scores = [balanced_accuracy_score(y_train, run_baseline_model(X_train, t)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def run_always_occupied(X):
    """Trivial reference: every bin is occupied (equivalent to running HVAC continuously)."""
    return pd.Series(1, index=X.index)


def run_schedule_baseline(X, start_hour=config.SCHEDULE_START_HOUR, end_hour=config.SCHEDULE_END_HOUR):
    """Fixed operating schedule: occupied on weekdays from `start_hour` to `end_hour` local time.

    This mirrors how many buildings run HVAC today, without any sensing.
    """
    in_hours = (X["hour"] >= start_hour) & (X["hour"] < end_hour)
    return (in_hours & (X["is_weekend"] == 0)).astype(int)
