"""Metrics, cross-validation, timing and plots for occupancy models."""
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import plot_tree, export_text

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


def get_regression_metrics(y_true, y_pred, model_name):
    """Regression metrics for a predicted share of spaces occupied (0 to 1)."""
    return {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'R2': r2_score(y_true, y_pred)
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


def plot_decision_tree(model, feature_names, save_path, max_depth=None, rules_path=None,
                       title="Decision tree (env+time)"):
    """Draw a fitted classification or regression tree; optionally write its rules as text.

    Each box shows the split rule and the share of training bins reaching it. For a
    classification tree it also shows the class shares and majority class (orange =
    vacant, blue = occupied); for a regression tree, ``value`` is the mean share of
    spaces occupied, with darker orange for higher values.
    """
    is_classifier = hasattr(model, "classes_")
    shown_depth = model.get_depth() if max_depth is None else max_depth
    fig, ax = plt.subplots(figsize=(max(12, 2.4 * 2 ** shown_depth), 2.6 * (shown_depth + 1)))
    plot_tree(model, feature_names=feature_names, class_names=['Vacant', 'Occupied'] if is_classifier else None,
              filled=True, rounded=True, proportion=True, impurity=False, precision=2,
              max_depth=max_depth, fontsize=9 if shown_depth <= 3 else 6, ax=ax)
    if max_depth is not None and max_depth < model.get_depth():
        title += f": top {max_depth} of {model.get_depth()} levels"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)

    if rules_path:
        with open(rules_path, "w") as f:
            f.write(export_text(model, feature_names=list(feature_names), show_weights=is_classifier))


# Outcome colours for the timeline strip (occupied is the positive class)
OUTCOME_STYLES = [
    ("Correct: occupied", "#2a78d6"),
    ("Correct: vacant", "#d9d8d4"),
    ("Missed occupancy", "#e34948"),
    ("False alarm", "#eda100"),
]


def classify_outcomes(y_true, y_pred):
    """Label each bin as one of the ``OUTCOME_STYLES`` names."""
    y_true = pd.Series(y_true).to_numpy()
    y_pred = pd.Series(y_pred).to_numpy()
    names = [name for name, _ in OUTCOME_STYLES]
    out = pd.Series(names[1], index=range(len(y_true)))
    out[(y_true == 1) & (y_pred == 1)] = names[0]
    out[(y_true == 1) & (y_pred == 0)] = names[2]
    out[(y_true == 0) & (y_pred == 1)] = names[3]
    return out.to_numpy()


def plot_test_timeline(test, predictions_dict, save_path, start=None, end=None,
                       threshold=config.OCCUPIED_FRACTION_THRESHOLD, local_tz=config.LOCAL_TZ):
    """Plot the test period over time: CO2, temperature, actual occupancy and each model's outcomes.

    Parameters
    ----------
    test : pandas.DataFrame
        Test rows from ``data_loader.build_dataset`` (UTC index).
    predictions_dict : dict
        ``{model name: predictions}`` aligned with ``test``; one outcome strip per model.
    start, end : str, optional
        Local-time window to zoom into, e.g. ``"2024-05-27"``.
    """
    data = test.copy()
    data.index = data.index.tz_convert(local_tz)
    outcomes = {name: pd.Series(classify_outcomes(test["occupied"], pred), index=data.index)
                for name, pred in predictions_dict.items()}
    if start is not None or end is not None:
        window = slice(start, end)
        data = data.loc[window]
        outcomes = {name: o.loc[window] for name, o in outcomes.items()}

    # Reindex to a regular grid so lines break across data gaps instead of bridging them
    grid = pd.date_range(data.index.min(), data.index.max(), freq=config.BIN)
    lines = data[["co2", "temperature", "occupied_fraction"]].reindex(grid)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.6, 0.4 + 0.5 * len(outcomes)]})
    ink, muted = "#2b2b2b", "#6b6b6b"

    axes[0].plot(grid, lines["co2"], color="#2a78d6", lw=1)
    axes[0].set_ylabel("CO2 (ppm)")
    axes[1].plot(grid, lines["temperature"], color="#2a78d6", lw=1)
    axes[1].set_ylabel("Temperature (°C)")
    axes[2].fill_between(grid, lines["occupied_fraction"], step="post", color="#2a78d6", alpha=0.35, lw=0)
    axes[2].axhline(threshold, color=ink, lw=1, ls="--")
    axes[2].text(grid[0], threshold + 0.04, f"occupied label threshold ({threshold:.0%})", color=ink, fontsize=8)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Share of spaces\noccupied (actual)")

    # Outcome strips: one row per model
    width = pd.Timedelta(config.BIN)
    ax = axes[3]
    for row, (name, outcome) in enumerate(outcomes.items()):
        for label, colour in OUTCOME_STYLES:
            times = outcome.index[outcome.to_numpy() == label]
            ax.bar(times, 0.8, width=width, bottom=row - 0.4, color=colour, align="edge", lw=0)
    ax.set_yticks(range(len(outcomes)), list(outcomes))
    ax.set_ylim(-0.6, len(outcomes) - 0.4)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in OUTCOME_STYLES]
    ax.legend(handles, [n for n, _ in OUTCOME_STYLES], loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=4, frameon=False, fontsize=9)

    # Shade weekends in every panel
    days = pd.date_range(grid[0].normalize(), grid[-1].normalize(), freq="D")
    for day in days[days.dayofweek >= 5]:
        for a in axes:
            a.axvspan(max(day, grid[0]), min(day + pd.Timedelta("1D"), grid[-1]), color="#f0efec", zorder=0)

    for a in axes:
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        a.tick_params(colors=muted, labelsize=8)
        a.yaxis.label.set_color(ink)
    for a in axes[:3]:
        a.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.tick_params(axis="y", colors=ink, labelsize=9)

    import matplotlib.dates as mdates
    span_days = (grid[-1] - grid[0]).days
    locator = mdates.HourLocator(byhour=[0, 12], tz=grid.tz) if span_days <= 8 else mdates.DayLocator(tz=grid.tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b\n%H:%M" if span_days <= 8 else "%d %b", tz=grid.tz))
    ax.set_xlim(grid[0], grid[-1])

    title = f"Test period, {grid[0]:%d %b} – {grid[-1]:%d %b %Y} (Melbourne time; weekends shaded)"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """One model's confusion matrix with counts, row percentages and outcome names.

    Cells use the timeline colours: correct occupied (blue), correct vacant (grey),
    missed occupancy (red) and false alarm (yellow). Percentages are of each actual class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    colours = dict(OUTCOME_STYLES)
    cells = [[("Correct: vacant", cm[0, 0]), ("False alarm", cm[0, 1])],
             [("Missed occupancy", cm[1, 0]), ("Correct: occupied", cm[1, 1])]]
    ink = "#2b2b2b"

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for i, row in enumerate(cells):
        row_total = sum(count for _, count in row)
        for j, (label, count) in enumerate(row):
            fill = colours[label]
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.97, 0.97, color=fill, alpha=0.9 if label != "Correct: vacant" else 1))
            text = "white" if label in ("Correct: occupied", "Missed occupancy") else ink
            ax.text(j + 0.485, 1 - i + 0.60, f"{count:,}", ha="center", va="center", fontsize=22, color=text, weight="bold")
            ax.text(j + 0.485, 1 - i + 0.38, f"{count / row_total:.0%} of actual", ha="center", va="center", fontsize=10, color=text)
            ax.text(j + 0.485, 1 - i + 0.18, label, ha="center", va="center", fontsize=10, color=text)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Vacant", "Occupied"])
    ax.set_yticks([1.485, 0.485], [f"Vacant\n(n = {cm[0].sum():,})", f"Occupied\n(n = {cm[1].sum():,})"])
    ax.set_xlabel("Predicted", color=ink)
    ax.set_ylabel("Actual", color=ink)
    ax.tick_params(length=0, labelsize=10, colors=ink)
    for spine in ax.spines.values():
        spine.set_visible(False)

    m = get_metrics(y_true, y_pred, model_name)
    ax.set_title(f"{model_name}\nrecall {m['Recall']:.2f}, precision {m['Precision']:.2f}, F1 {m['F1-Score']:.2f}",
                 color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
