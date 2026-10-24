# MMA3001-Project

Building-level occupancy detection from low-cost environmental sensors (CO2, temperature,
humidity) and time-of-day patterns, evaluated against occupancy-sensor ground truth.

## Repository structure

```
code/
  config.py             all tunable settings (sensor, time bin, label threshold, feature sets)
  data_loader.py        load, de-duplicate, range-check and align env + occupancy data
  baseline.py           rule-based baselines (always occupied, fixed schedule, CO2 threshold)
  train_models.py       regression tree (recommended) and classification tree
  evaluate.py           metrics, time-series cross-validation, timing, confusion matrices
  main.py               runs the full workflow and writes results/
  feature_screening.py  correlation matrix and ablation of every recorded variable
tests/                  pytest unit tests
docs/                   HTML documentation of the source code (generated with pdoc)
data/
  environment/          environmental sensor readings, one CSV per variable
  occupancy/            occupancy status events, one file per status
  raw/                  original environmental export the CSVs were extracted from
results/                generated outputs, one folder per input sensor; test_report.html = latest test run
```

## Data

Monash smart-building exports. Environmental sensors (`sensorid`) report CO2 (ppm),
temperature (°C), humidity (%RH), pressure (mb), VOCs (ppb), formaldehyde and particulate
matter (µg/m³) and battery voltage (V) roughly every 10 minutes; each reading is repeated
several times in the export and is de-duplicated on load. Occupancy events come from one
device covering five floor spaces (`floorspaceid`), with status `CurrentlyOccupied`,
`RecentlyOccupied` or `NotOccupied`. Only sensors 6012002000869 (Mar-Jun 2024) and
6012002000326 (Sep 2025-Apr 2026) contain usable environmental data.

## Running

Requires Python 3.10+.

```
pip install -r requirements.txt
python code/main.py                # main analysis, sensor 6012002000869 (Mar-Jun 2024)
python code/main.py --sensor 326   # comparison sensor 6012002000326 (Sep 2025-Apr 2026)
python code/main.py --quick        # skip cross-validation and sweeps
python code/feature_screening.py   # screen all recorded variables as candidate inputs
python -m pytest                   # run the tests
python -m pytest --html=results/test_report.html --self-contained-html   # save the test report
```

Options: `--threshold` sets the fraction of spaces that must be occupied for the building to
count as occupied (default 0.5); `--recent-as-occupied` counts `RecentlyOccupied` as occupied.

Each run writes to `results/sensor_<id>/`: `data_audit.json` (cleaning counts, periods),
`test_metrics.csv`, `confusion_matrices.png`, `feature_importance.csv`, `cv_summary.csv`,
`depth_sweep.csv`, `threshold_sensitivity.csv`, `regression_metrics.csv` (regression tree
MAE, RMSE and R² for the share of spaces occupied), `test_timeline.png` and
`test_timeline_week.png` (test period over time with each model's errors), and
`regression_tree_*` and `decision_tree_*` (`_top.png`, `_full.png`, `_rules.txt`) for the
regression and classification trees, with `confusion_matrix_regression_tree.png` and
`confusion_matrix_decision_tree.png`. `feature_screening.py` writes
`correlation_matrix.png`, `correlation_with_occupancy.csv` and `ablation.csv` to
`results/sensor_<id>/feature_screening/`.

## Documentation

HTML documentation of every module and function, generated from the docstrings with
[pdoc](https://pdoc.dev), is in `docs/`. Open `docs/index.html` in a browser. To regenerate it
after changing the code:

```
pip install pdoc
PYTHONPATH=code python -m pdoc code/config.py code/data_loader.py code/baseline.py \
    code/train_models.py code/evaluate.py code/feature_screening.py code/main.py -o docs
```

On Windows PowerShell, set the path first with `$env:PYTHONPATH="code"` and run the same
`python -m pdoc ...` command on one line.

## Project summary

**Problem.** Heating, ventilation and air conditioning often runs on a fixed timetable whether or
not a building is in use. Demand-controlled ventilation needs to know when people are present.
This project tests whether a building's existing low-cost environmental sensors, combined with
time of day, can detect occupancy well enough to support that decision.

**Inputs.** CO2 (ppm), temperature (°C) and relative humidity (%RH) from sensor 6012002000869,
plus CO2 rate of change, hour of day, day of week and a weekend flag, in 15-minute intervals
(3 March – 13 June 2024, 8,887 complete intervals).

**Outputs.** The predicted share of the five monitored floor spaces that are occupied (0–1), and
an occupied/vacant decision (occupied when the predicted share is at least 0.5).

**Method.** A depth-5 regression tree (recommended) is compared with a classification tree and
four rule-based baselines: always occupied, a fixed Mon–Fri 08:00–18:00 schedule, and CO2
thresholds of 500 ppm and 470 ppm (tuned on training data). Data are split chronologically:
first 80% for training, last 20% for testing.

**Results (test period, 24 May – 13 June 2024).**

| Method | F1 | Recall | Precision | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Fixed schedule | 0.604 | 0.450 | 0.916 | 0.694 |
| CO2 > 470 ppm (tuned) | 0.313 | 0.200 | 0.717 | 0.541 |
| Classification tree (env + time) | 0.838 | 0.902 | 0.781 | 0.762 |
| **Regression tree (env + time)** | **0.836** | **0.828** | **0.843** | **0.799** |

The regression tree estimates the share of spaces occupied with MAE 0.18, RMSE 0.23 and
R² 0.49, against R² −0.02 for always predicting the training average. Time-series
cross-validation on the training period gives F1 0.893 ± 0.011.

**Main limitations.** One usable sensor over about three months; sensor locations are unknown,
so results are at building level; most predictive power comes from time of day, with CO2 the
only sensor variable carrying occupancy information beyond the daily cycle.

The full analysis, validation and discussion are in the written project documentation submitted
with this repository.

## AI use

AI tools (Claude, Claude Code and Gemini in Google Colab) were used during this project for
scoping discussions, code development and proofreading, as disclosed in full in the AI-use
reflection of the written documentation.
