# MMA3001 Project: Building Occupancy Detection

Building-level occupancy detection from low-cost environmental sensors (CO2, temperature and humidity) and time-of-day patterns, evaluated against occupancy-sensor ground truth.

## Project summary

**Problem.** Heating, ventilation and air conditioning (HVAC) often runs on a fixed timetable whether or not a building is in use. Demand-controlled ventilation needs to know when people are present. This project tests whether a building's existing low-cost environmental sensors, combined with time of day, can detect occupancy well enough to support that decision.

**Inputs.** CO2 (ppm), temperature (°C) and relative humidity (%RH) from sensor 6012002000869, plus CO2 rate of change, hour of day, day of week and a weekend flag, in 15-minute intervals (3 March – 13 June 2024, 8,887 complete intervals).

**Outputs.** The predicted share of the five monitored floor spaces that are occupied (0 to 1), and an occupied/vacant decision (occupied when the predicted share is at least 0.5).

**Method.** A depth-5 regression tree (recommended) is compared with a classification tree and four rule-based baselines: always occupied, a fixed Monday–Friday 08:00–18:00 schedule, and CO2 thresholds of 500 ppm and 470 ppm (tuned on training data). Data are split chronologically: the first 80% for training and the last 20% for testing.

**Results (test period, 24 May – 13 June 2024).**

| Method | F1 | Recall | Precision | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Fixed schedule | 0.604 | 0.450 | 0.916 | 0.694 |
| CO2 > 470 ppm (tuned) | 0.313 | 0.200 | 0.717 | 0.541 |
| Classification tree (env + time) | 0.838 | 0.902 | 0.781 | 0.762 |
| **Regression tree (env + time)** | **0.836** | **0.828** | **0.843** | **0.799** |

The regression tree estimates the share of spaces occupied with MAE 0.18, RMSE 0.23 and R² 0.49, compared with R² −0.02 for always predicting the training average. Time-series cross-validation on the training period gives F1 0.893 ± 0.011.

**Main limitations.** Only one usable sensor over about three months; sensor locations are unknown, so results are at building level; most predictive power comes from time of day, with CO2 the only sensor variable carrying occupancy information beyond the daily cycle.

The full analysis, validation and discussion are in the written project documentation submitted with this repository.

## Repository structure

| Folder or file | Contents |
| --- | --- |
| code/config.py | All tunable settings (sensor, time interval, label threshold, feature sets) |
| code/data_loader.py | Loads, de-duplicates, range-checks and aligns the environmental and occupancy data |
| code/baseline.py | Rule-based baselines (always occupied, fixed schedule, CO2 threshold) |
| code/train_models.py | Regression tree (recommended) and classification tree |
| code/evaluate.py | Metrics, time-series cross-validation, timing and figures |
| code/main.py | Runs the full workflow and writes the results |
| code/feature_screening.py | Correlation matrix and ablation of every recorded variable |
| tests/ | Unit tests (pytest) |
| docs/ | HTML documentation of the source code (generated with pdoc) |
| data/environment/ | Environmental sensor readings, one CSV file per variable |
| data/occupancy/ | Occupancy status events, one file per status |
| data/raw/ | Original environmental export the CSV files were extracted from |
| results/ | Generated outputs, one folder per sensor, plus the latest test report (test_report.html) |

## Data

The data are Monash smart-building exports.

- **Environmental sensors** report CO2 (ppm), temperature (°C), humidity (%RH), pressure (mb), VOCs (ppb), formaldehyde and particulate matter (µg/m³) and battery voltage (V) roughly every 10 minutes. Each reading is repeated several times in the export and is de-duplicated when loaded.
- **Occupancy events** come from one device covering five floor spaces, with status CurrentlyOccupied, RecentlyOccupied or NotOccupied.
- Only sensors 6012002000869 (March–June 2024) and 6012002000326 (September 2025 – April 2026) contain usable environmental data.

## How to run

Requires Python 3.10 or later. Run these commands from the repository folder.

| Command | What it does |
| --- | --- |
| pip install -r requirements.txt | Installs the required Python packages |
| python code/main.py | Main analysis for sensor 6012002000869 (March–June 2024) |
| python code/main.py --sensor 326 | Same analysis for the comparison sensor 6012002000326 |
| python code/main.py --quick | Faster run that skips cross-validation and the sweeps |
| python code/feature_screening.py | Screens all recorded variables as candidate inputs |
| python -m pytest | Runs the unit tests |
| python -m pytest --html=results/test_report.html --self-contained-html | Runs the tests and saves the HTML test report |

Optional settings for main.py:

- **--threshold** sets the share of spaces that must be occupied for the building to count as occupied (default 0.5).
- **--recent-as-occupied** counts RecentlyOccupied as occupied.

## Outputs

Each run of main.py writes to results/sensor_869/ (or results/sensor_326/):

- **data_audit.json**: cleaning counts and the training and test periods
- **test_metrics.csv**: every method's scores and timings on the test period
- **regression_metrics.csv**: the regression tree's MAE, RMSE and R²
- **cv_summary.csv**: time-series cross-validation results
- **depth_sweep.csv**: cross-validated F1 for each tree depth
- **threshold_sensitivity.csv**: results for different definitions of "occupied"
- **feature_importance.csv**: how much each tree relies on each input
- **confusion_matrix_regression_tree.png** and **confusion_matrix_decision_tree.png**: confusion matrices of the two trees
- **confusion_matrices.png**: confusion matrices of all methods
- **regression_tree_top.png / _full.png / _rules.txt** and **decision_tree_top.png / _full.png / _rules.txt**: diagrams and rule lists of the two trees
- **test_timeline.png** and **test_timeline_week.png**: the test period over time with each model's errors

feature_screening.py writes correlation_matrix.png, correlation_with_occupancy.csv and ablation.csv to results/sensor_869/feature_screening/.

## Documentation

HTML documentation of every module and function, generated from the code's docstrings with [pdoc](https://pdoc.dev), is in the docs folder. Open docs/index.html in a web browser to view it.

To regenerate it after changing the code, install pdoc (pip install pdoc) and run this from the repository folder:

- **Mac or Linux:** PYTHONPATH=code python -m pdoc code/config.py code/data_loader.py code/baseline.py code/train_models.py code/evaluate.py code/feature_screening.py code/main.py -o docs
- **Windows PowerShell:** first run $env:PYTHONPATH="code", then the same python -m pdoc command

## AI use

AI tools (Claude, Claude Code and Gemini in Google Colab) were used during this project for scoping discussions, code development and proofreading, as disclosed in full in the AI-use reflection of the written documentation.
