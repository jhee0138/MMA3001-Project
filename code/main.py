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


def model_name(kind, feature_set):
    return f"{kind} ({feature_set})"


def make_fit_predict(kind, feature_set, **params):
    """Wrap a model as ``fit_predict(X_train, y_train, X_val)`` for cross-validation."""
    def fit_predict(X_train, y_train, X_val):
        model = train_models.train_model(kind, feature_set, X_train, y_train, **params)
        return train_models.predict(model, feature_set, X_val)
    return fit_predict


def co2_threshold_fit_predict(X_train, y_train, X_val):
    return baseline.run_baseline_model(X_val, baseline.fit_co2_threshold(X_train, y_train))


def schedule_fit_predict(X_train, y_train, X_val):
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

    metrics = evaluate.evaluate_all_models(y_test, predictions)
    metrics = metrics.merge(pd.DataFrame(timings), on="Model", how="left")
    metrics.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)
    evaluate.plot_confusion_matrices(y_test, predictions, os.path.join(out_dir, "confusion_matrices.png"))

    rf = fitted[model_name("RandomForest", "env+time")]
    importances = pd.Series(rf.feature_importances_, index=train_models.FEATURE_SETS["env+time"])
    importances.sort_values(ascending=False).to_csv(os.path.join(out_dir, "feature_importance.csv"),
                                                    header=["importance"])
    return metrics


def cross_validate_all(X_train, y_train, out_dir):
    """Expanding-window CV on the training period: is the ranking of models stable?"""
    candidates = {"Schedule (Mon-Fri 8-18)": schedule_fit_predict,
                  "CO2 threshold (tuned)": co2_threshold_fit_predict}
    for kind, fs in MODELS:
        candidates[model_name(kind, fs)] = make_fit_predict(kind, fs)

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
    for kind in train_models.MODEL_BUILDERS:
        for depth in depths:
            folds = evaluate.time_series_cv(make_fit_predict(kind, "env+time", max_depth=depth),
                                            X_train, y_train)
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
            "RandomForest (env)": make_fit_predict("RandomForest", "env")(X_train, y_train, X_test),
            "RandomForest (env+time)": make_fit_predict("RandomForest", "env+time")(X_train, y_train, X_test),
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
    print(compare_on_test(X_train, y_train, X_test, y_test, out_dir).round(3).to_string(index=False))

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
