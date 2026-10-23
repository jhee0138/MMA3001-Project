import numpy as np
import pandas as pd
import pytest

import baseline
import evaluate
import train_models


@pytest.fixture
def toy_data():
    """Synthetic bins where occupancy raises CO2, so the signal is learnable."""
    rng = np.random.default_rng(0)
    n = 400
    occupied = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "co2": 420 + 150 * occupied + rng.normal(0, 20, n),
        "temperature": 21 + rng.normal(0, 0.5, n),
        "humidity": 50 + rng.normal(0, 2, n),
        "co2_rate": rng.normal(0, 1, n),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "is_weekend": 0,
    }, index=pd.date_range("2024-03-01", periods=n, freq="15min", tz="UTC"))
    return X, pd.Series(occupied, index=X.index)


def test_co2_rule():
    X = pd.DataFrame({"co2": [450, 500, 501]})
    assert baseline.run_baseline_model(X, 500).tolist() == [0, 0, 1]


def test_fitted_co2_threshold_separates_classes(toy_data):
    X, y = toy_data
    threshold = baseline.fit_co2_threshold(X, y)
    assert 450 < threshold < 540


def test_schedule_baseline():
    X = pd.DataFrame({"hour": [7, 8, 17, 18, 12], "is_weekend": [0, 0, 0, 0, 1]})
    assert baseline.run_schedule_baseline(X).tolist() == [0, 1, 1, 0, 0]


@pytest.mark.parametrize("kind", list(train_models.MODEL_BUILDERS))
@pytest.mark.parametrize("feature_set", list(train_models.FEATURE_SETS))
def test_models_learn_and_respect_feature_set(toy_data, kind, feature_set):
    X, y = toy_data
    model = train_models.train_model(kind, feature_set, X.iloc[:300], y.iloc[:300])
    assert model.n_features_in_ == len(train_models.FEATURE_SETS[feature_set])
    pred = train_models.predict(model, feature_set, X.iloc[300:])
    assert (pred == y.iloc[300:]).mean() > 0.9


def test_metrics_known_values():
    m = evaluate.get_metrics([1, 1, 0, 0], [1, 0, 0, 1], "m")
    assert m["Accuracy"] == 0.5
    assert m["Precision"] == 0.5
    assert m["Recall"] == 0.5


def test_time_series_cv_never_trains_on_future(toy_data):
    X, y = toy_data

    def fit_predict(X_train, y_train, X_val):
        assert X_train.index.max() < X_val.index.min()
        return np.ones(len(X_val), dtype=int)

    folds = evaluate.time_series_cv(fit_predict, X, y, n_splits=4)
    assert len(folds) == 4
