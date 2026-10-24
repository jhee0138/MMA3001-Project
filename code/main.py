"""Run the full building-occupancy detection workflow and write results to ``results/``.

Usage (from the repository root):
    python code/main.py                 # sensor 869, default settings
    python code/main.py --sensor 326    # comparison sensor with no occupancy signal
    python code/main.py --quick         # skip the slower sweeps
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import config
import data_loader
import baseline
import train_models
import evaluate

MODELS = [(kind, fs) for kind in train_models.MODEL_BUILDERS for fs in train_models.FEATURE_SETS]
REGRESSOR = "RegressionTree (env+time)"


def model_name(kind, feature_set):
    """Display name for a model, e.g. ``"DecisionTree (env+time)"``."""
    return f"{kind} ({feature_set})"


def make_fit_predict(kind, feature_set, **params):
    """Wrap a model as ``fit_predict(X_train, y_train, X_val)`` for cross-validation."""
    def fit_predict(X_train, y_train, X_val):
        model = train_models.train_model(kind, feature_set, X_train, y_train, **params)
        return train_models.predict(model, feature_set, X_val)
    return fit_predict


def make_regression_fit_predict(threshold=config.OCCUPIED_FRACTION_THRESHOLD, **params):
    """Regression tree on the share of spaces occupied, thresholded to occupied/vacant."""
    features = train_models.FEATURE_SETS["env+time"]

    def fit_predict(X_train, y_train, X_val):
        model = train_models.train_regression_tree(X_train[features], X_train["occupied_fraction"], **params)
        return train_models.classify_fraction(model.predict(X_val[features]), threshold)
    return fit_predict


def co2_threshold_fit_predict(X_train, y_train, X_val):
    """CO2 rule with its threshold tuned on the training data, applied to ``X_val``."""
    return baseline.run_baseline_model(X_val, baseline.fit_co2_threshold(X_train, y_train))


def schedule_fit_predict(X_train, y_train, X_val):
    """Fixed-schedule baseline in ``fit_predict`` form (nothing is learned from the training data)."""
    return baseline.run_schedule_baseline(X_val)


def compare_on_test(X_train, y_train, X_test, y_test, out_dir):
    """Train every model once, score it on the held-out period and time it."""
    co2_threshold = baseline.fit_co2_threshold(X_train, y_train)
    predictions = {
        "Always occupied": baseline.run_always_occupied(X_test),
        "Schedule (Mon-Fri 8-18)": baseline.run_schedule_baseline(X_test),
        "CO2 > 500 ppm": baseline.run_baseline_model(X_test, 500.0),
        f"CO2 > {co2_threshold:.0f} ppm (tuned)": baseline.run_baseline_model(X_test, co2_threshold),
    }
    timings = []
    fitted = {}
    for kind, fs in MODELS:
        name = model_name(kind, fs)
        model, fit_s = evaluate.time_call(train_models.train_model, kind, fs, X_train, y_train)
        pred, predict_s = evaluate.time_call(train_models.predict, model, fs, X_test)
        predictions[name] = pred
        fitted[name] = model
        timings.append({"Model": name, "Fit time (s)": fit_s, "Predict time (s)": predict_s})

    # Regression tree: predicts the share of spaces occupied, then applies the label threshold
    features = train_models.FEATURE_SETS["env+time"]
    regressor, fit_s = evaluate.time_call(train_models.train_regression_tree, X_train[features],
                                          X_train["occupied_fraction"])
    fraction_pred, predict_s = evaluate.time_call(regressor.predict, X_test[features])
    predictions[REGRESSOR] = train_models.classify_fraction(fraction_pred)
    fitted[REGRESSOR] = regressor
    timings.append({"Model": REGRESSOR, "Fit time (s)": fit_s, "Predict time (s)": predict_s})
    regression = pd.DataFrame([
        evaluate.get_regression_metrics(X_test["occupied_fraction"], fraction_pred, REGRESSOR),
        evaluate.get_regression_metrics(X_test["occupied_fraction"],
                                        [X_train["occupied_fraction"].mean()] * len(X_test),
                                        "Training mean (no model)"),
    ])
    regression.to_csv(os.path.join(out_dir, "regression_metrics.csv"), index=False)

    metrics = evaluate.evaluate_all_models(y_test, predictions)
    metrics = metrics.merge(pd.DataFrame(timings), on="Model", how="left")
    metrics.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    evaluate.plot_confusion_matrices(y_test, predictions, os.path.join(out_dir, "confusion_matrices.png"))

    timeline_models = {name: predictions[name] for name in
                       [REGRESSOR, model_name("DecisionTree", "env+time"), "Schedule (Mon-Fri 8-18)"]}
    evaluate.plot_test_timeline(X_test, timeline_models, os.path.join(out_dir, "test_timeline.png"))
    # Zoom into the first full Monday-to-Sunday week of the test period
    test_start = X_test.index.min().tz_convert(config.LOCAL_TZ).normalize()
    first_week = test_start + pd.Timedelta(days=(7 - test_start.dayofweek) % 7)
    evaluate.plot_test_timeline(X_test, timeline_models, os.path.join(out_dir, "test_timeline_week.png"),
                                start=first_week, end=first_week + pd.Timedelta("7D"))

    # The recommended model (regression tree) and the classification tree it is compared with
    tree_features = train_models.FEATURE_SETS["env+time"]
    trees = {"regression_tree": (REGRESSOR, "Regression tree (env+time)"),
             "decision_tree": (model_name("DecisionTree", "env+time"), "Classification tree (env+time)")}
    for slug, (name, title) in trees.items():
        evaluate.plot_confusion_matrix(y_test, predictions[name], f"{title}, test period",
                                       os.path.join(out_dir, f"confusion_matrix_{slug}.png"))
        evaluate.plot_decision_tree(fitted[name], tree_features, os.path.join(out_dir, f"{slug}_top.png"),
                                    max_depth=3, title=title)
        evaluate.plot_decision_tree(fitted[name], tree_features, os.path.join(out_dir, f"{slug}_full.png"),
                                    rules_path=os.path.join(out_dir, f"{slug}_rules.txt"), title=title)

    importances = pd.DataFrame({"RegressionTree": regressor.feature_importances_,
                                "DecisionTree": fitted[model_name("DecisionTree", "env+time")].feature_importances_},
                               index=tree_features)
    importances.sort_values("RegressionTree", ascending=False).to_csv(os.path.join(out_dir, "feature_importance.csv"))
    return metrics, regression


def cross_validate_all(X_train, y_train, out_dir):
    """Expanding-window CV on the training period: is the ranking of models stable?"""
    candidates = {"Schedule (Mon-Fri 8-18)": schedule_fit_predict,
                  "CO2 threshold (tuned)": co2_threshold_fit_predict}
    for kind, fs in MODELS:
        candidates[model_name(kind, fs)] = make_fit_predict(kind, fs)
    candidates[REGRESSOR] = make_regression_fit_predict()

    rows = []
    for name, fit_predict in candidates.items():
        folds = evaluate.time_series_cv(fit_predict, X_train, y_train)
        rows.append({"Model": name,
                     "F1 mean": folds["F1-Score"].mean(), "F1 std": folds["F1-Score"].std(),
                     "Recall mean": folds["Recall"].mean(),
                     "Balanced Accuracy mean": folds["Balanced Accuracy"].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)
    return summary


def depth_sweep(X_train, y_train, out_dir, depths=(2, 3, 5, 8, 12, None)):
    """Tune tree depth with CV on the training period only (the test set stays untouched)."""
    rows = []
    for kind in list(train_models.MODEL_BUILDERS) + ["RegressionTree"]:
        for depth in depths:
            if kind == "RegressionTree":
                fit_predict = make_regression_fit_predict(max_depth=depth)
            else:
                fit_predict = make_fit_predict(kind, "env+time", max_depth=depth)
            folds = evaluate.time_series_cv(fit_predict, X_train, y_train)
            rows.append({"Model": kind, "max_depth": "None" if depth is None else depth,
                         "F1 mean": folds["F1-Score"].mean(), "F1 std": folds["F1-Score"].std()})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out_dir, "depth_sweep.csv"), index=False)
    return sweep


def threshold_sensitivity(data, out_dir, thresholds=(0.2, 0.4, 0.5, 0.6, 0.8)):
    """How results change with the occupied-fraction threshold that defines the label."""
    rows = []
    for threshold in thresholds:
        labelled = data.assign(occupied=data_loader.label_occupied(data["occupied_fraction"], threshold))
        train, test = data_loader.chronological_split(labelled)
        X_train, y_train, X_test, y_test = train, train["occupied"], test, test["occupied"]
        preds = {
            "CO2 threshold (tuned)": co2_threshold_fit_predict(X_train, y_train, X_test),
            "DecisionTree (env+time)": make_fit_predict("DecisionTree", "env+time")(X_train, y_train, X_test),
            REGRESSOR: make_regression_fit_predict(threshold)(X_train, y_train, X_test),
        }
        for name, pred in preds.items():
            row = evaluate.get_metrics(y_test, pred, name)
            row["Label threshold"] = threshold
            row["Test occupied rate"] = y_test.mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "threshold_sensitivity.csv"), index=False)
    return table


def main():
    """Parse command-line options, run every analysis and write the outputs to ``results/sensor_<id>/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", choices=sorted(config.SENSOR_IDS), default="869",
                        help="environmental sensor to use as input")
    parser.add_argument("--threshold", type=float, default=config.OCCUPIED_FRACTION_THRESHOLD,
                        help="occupied fraction of spaces at which the building counts as occupied")
    parser.add_argument("--recent-as-occupied", action="store_true",
                        help="count RecentlyOccupied as occupied")
    parser.add_argument("--quick", action="store_true", help="skip CV, depth and threshold sweeps")
    args = parser.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, f"sensor_{args.sensor}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading and aligning data...")
    data, audit = data_loader.build_dataset(config.SENSOR_IDS[args.sensor], args.threshold,
                                            args.recent_as_occupied)
    if len(data) < 100:
        raise SystemExit(f"Only {len(data)} complete bins for sensor {args.sensor}; nothing to model.")
    train, test = data_loader.chronological_split(data)
    audit["train_bins"], audit["test_bins"] = len(train), len(test)
    audit["train_period"] = [str(train.index.min()), str(train.index.max())]
    audit["test_period"] = [str(test.index.min()), str(test.index.max())]
    with open(os.path.join(out_dir, "data_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))

    X_train, y_train = train, train["occupied"]
    X_test, y_test = test, test["occupied"]

    pd.set_option("display.width", 200)
    print("\nHeld-out test period:")
    metrics, regression = compare_on_test(X_train, y_train, X_test, y_test, out_dir)
    print(metrics.round(3).to_string(index=False))
    print("\nRegression tree, share of spaces occupied (test period):")
    print(regression.round(3).to_string(index=False))

    if not args.quick:
        print("\nTime-series cross-validation on the training period:")
        print(cross_validate_all(X_train, y_train, out_dir).round(3).to_string(index=False))
        print("\nTree depth sweep (CV F1, env+time):")
        print(depth_sweep(X_train, y_train, out_dir).round(3).to_string(index=False))
        print("\nLabel threshold sensitivity (test set):")
        print(threshold_sensitivity(data, out_dir)[["Label threshold", "Model", "Test occupied rate",
                                                    "Recall", "F1-Score"]].round(3).to_string(index=False))

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
