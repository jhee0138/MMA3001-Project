"""Decision trees for building occupancy detection: a classification tree and a regression tree.


Each model is trained on a named feature set so the same code covers the
environment-only model and the environment + time model.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config

FEATURE_SETS = {
    "env": config.ENV_FEATURES,
    "env+time": config.ALL_FEATURES,
}


def train_decision_tree(X_train, y_train, max_depth=5, random_state=config.RANDOM_STATE):
    """Fit a single decision tree (interpretable, fast)."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model



MODEL_BUILDERS = {
    "DecisionTree": train_decision_tree,
}


def train_model(kind, feature_set, X_train, y_train, **params):
    """Train ``kind`` (key of ``MODEL_BUILDERS``) on the columns of ``feature_set``."""
    return MODEL_BUILDERS[kind](X_train[FEATURE_SETS[feature_set]], y_train, **params)


def predict(model, feature_set, X):
    """Predict with a model trained by ``train_model`` on the same feature set."""
    return model.predict(X[FEATURE_SETS[feature_set]])


def train_regression_tree(X_train, y_fraction, max_depth=5, random_state=config.RANDOM_STATE):
    """Fit a regression tree to the share of spaces occupied (0 to 1).

    This is the form of decision tree covered in the unit: splits minimise squared
    error and each leaf returns the mean share of its training intervals.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_fraction)
    return model


def classify_fraction(fraction, threshold=config.OCCUPIED_FRACTION_THRESHOLD):
    """Turn a predicted share of spaces occupied into the binary occupied label."""
    return (np.asarray(fraction) >= threshold).astype(int)
