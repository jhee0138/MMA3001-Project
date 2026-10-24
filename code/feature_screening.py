"""Screen every recorded environmental variable as a candidate model input.

Three views, all computed on the training period only so the test set plays no part
in choosing inputs:
  1. correlation of each variable with building occupancy (Pearson and Spearman)
  2. the same correlation after removing the typical value for that hour of day and
     weekday/weekend, i.e. what a variable adds beyond what the clock already says
  3. ablation: cross-validated F1 when each variable is added to (or removed from) the
     current env+time feature set

Usage (from the repository root):
    python code/feature_screening.py               # sensor 869
    python code/feature_screening.py --sensor 326
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

import config
import data_loader
import evaluate
import train_models

CANDIDATES = list(config.CANDIDATE_ENV_VARIABLES) + ["co2_rate"]

# Diverging scale: red (negative) - neutral grey (zero) - blue (positive)
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])


def load_screening_data(sensor_id, threshold=config.OCCUPIED_FRACTION_THRESHOLD, min_coverage=0.5):
    """All candidate variables plus time features and occupancy, training period only.

    Variables present in fewer than ``min_coverage`` of the bins where the sensor was
    reporting anything are dropped
    (e.g. a sensor that stopped reporting one quantity) so they do not empty the table.

    Returns
    -------
    train : pandas.DataFrame
    candidates : list of str
        The candidate variables that had enough data.
    """
    env = data_loader.load_environment(sensor_id, variables=config.CANDIDATE_ENV_VARIABLES)
    env = data_loader.add_features(env)
    records = data_loader.load_occupancy_records()
    fraction = data_loader.building_occupied_fraction(data_loader.space_occupancy_bins(records))

    data = env.join(fraction, how="inner").dropna(subset=["occupied_fraction"])
    active = data[CANDIDATES].notna().any(axis=1)
    coverage = data.loc[active, CANDIDATES].notna().mean()
    candidates = [c for c in CANDIDATES if coverage[c] >= min_coverage]
    for c in CANDIDATES:
        if c not in candidates:
            print(f"Skipping {c}: only {coverage[c]:.0%} of bins have data")

    data = data.dropna(subset=candidates)
    data["occupied"] = data_loader.label_occupied(data["occupied_fraction"], threshold)
    train, _ = data_loader.chronological_split(data)
    return train, candidates


def remove_time_pattern(series, data):
    """Subtract the mean for each (hour, weekend) group: what is left is not explained by the clock."""
    return series - series.groupby([data["hour"], data["is_weekend"]]).transform("mean")


def correlation_table(data, candidates):
    """Correlation of each candidate with the occupied fraction, raw and with the time pattern removed."""
    target = data["occupied_fraction"]
    target_resid = remove_time_pattern(target, data)
    rows = []
    for var in candidates:
        rows.append({
            "variable": var,
            "pearson": data[var].corr(target),
            "spearman": data[var].corr(target, method="spearman"),
            "pearson_beyond_time": remove_time_pattern(data[var], data).corr(target_resid),
        })
    table = pd.DataFrame(rows)
    return table.reindex(table["pearson"].abs().sort_values(ascending=False).index)


def plot_correlation_matrix(data, candidates, save_path):
    """Heatmap of Pearson correlations between all variables and occupancy."""
    cols = candidates + config.TIME_FEATURES + ["occupied_fraction"]
    corr = data[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "#2b2b2b")
    # Outline the occupancy row so the answer to "what correlates with occupancy" stands out
    n = len(cols) - 1
    ax.add_patch(plt.Rectangle((-0.5, n - 0.5), len(cols), 1, fill=False, edgecolor="#2b2b2b", lw=1.5))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")
    ax.set_title("Correlation between sensor variables, time features and building occupancy\n"
                 "(training period)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def ablation(data, candidates):
    """Cross-validated F1 of the regression tree when one variable is added to or dropped from env+time.

    The tree predicts the share of spaces occupied, thresholded to occupied/vacant.
    """
    base = config.ALL_FEATURES

    def cv_f1(features):
        def fit_predict(X_train, y_train, X_val):
            model = train_models.train_regression_tree(X_train[features], X_train["occupied_fraction"])
            return train_models.classify_fraction(model.predict(X_val[features]))
        folds = evaluate.time_series_cv(fit_predict, data, data["occupied"])
        return folds["F1-Score"].mean(), folds["Balanced Accuracy"].mean()

    base_f1, base_bacc = cv_f1(base)
    rows = [{"change": "none (current env+time)", "F1 mean": base_f1, "Balanced Accuracy mean": base_bacc}]
    for var in candidates:
        features = [f for f in base if f != var] if var in base else base + [var]
        label = f"- {var}" if var in base else f"+ {var}"
        f1, bacc = cv_f1(features)
        rows.append({"change": label, "F1 mean": f1, "Balanced Accuracy mean": bacc})
    table = pd.DataFrame(rows)
    table["F1 change"] = table["F1 mean"] - base_f1
    table["Balanced Accuracy change"] = table["Balanced Accuracy mean"] - base_bacc
    return table.sort_values("F1 change", ascending=False)


def main():
    """Parse command-line options, screen all variables and write the outputs to ``results/sensor_<id>/feature_screening/``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", choices=sorted(config.SENSOR_IDS), default="869")
    args = parser.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, f"sensor_{args.sensor}", "feature_screening")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading all variables...")
    data, candidates = load_screening_data(config.SENSOR_IDS[args.sensor])
    print(f"{len(data)} training bins, {data.index.min()} to {data.index.max()}")

    pd.set_option("display.width", 200)
    corr = correlation_table(data, candidates)
    corr.to_csv(os.path.join(out_dir, "correlation_with_occupancy.csv"), index=False)
    print("\nCorrelation with building occupied fraction:")
    print(corr.round(3).to_string(index=False))

    plot_correlation_matrix(data, candidates, os.path.join(out_dir, "correlation_matrix.png"))

    abl = ablation(data, candidates)
    abl.to_csv(os.path.join(out_dir, "ablation.csv"), index=False)
    print("\nAblation (5-fold time-series CV on the training period, regression tree):")
    print(abl.round(3).to_string(index=False))

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
