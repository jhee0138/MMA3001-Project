"""Tree-based classifiers for building occupancy detection.

Each model is trained on a named feature set so the same code covers the
environment-only model and the environment + time model.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=5, random_state=config.RANDOM_STATE):
    """Fit a random forest (averages many trees to reduce variance)."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


MODEL_BUILDERS = {
    "DecisionTree": train_decision_tree,
    "RandomForest": train_random_forest,
}


def train_model(kind, feature_set, X_train, y_train, **params):
    """Train ``kind`` (key of ``MODEL_BUILDERS``) on the columns of ``feature_set``."""
    return MODEL_BUILDERS[kind](X_train[FEATURE_SETS[feature_set]], y_train, **params)


def predict(model, feature_set, X):
    """Predict with a model trained by ``train_model`` on the same feature set."""
    return model.predict(X[FEATURE_SETS[feature_set]])
