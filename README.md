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

## 1. Engineering Problem

 
Your Project documentation must explain:
* the engineering context;
* the problem being addressed;
* why the problem matters;
* the intended use of the computational solution;
* the limitations of the selected scope.


## 2. Inputs and Outputs
Documentation - must clearly define:
* the input data accepted by the solution;
* the expected format, units and domain of the inputs;
* the outputs produced;
* the meaning and intended use of each output;
* how invalid, missing or unsupported inputs are handled.
The documentation should address top-level inputs and outputs of the project. Later part of
the documentation should address specific inputs and output of functions you have
developed. You can use Docstrings and Documentation generators to assist with this.
A computational method cannot be evaluated properly unless its inputs and outputs are
unambiguous.


## 3. Computational Solution
Students must develop a functioning and reproducible computational workflow.
The selected approach may involve any defensible computational method. Students must
explain:
* how the method works at an appropriate level;
* why it was selected;
* its important assumptions;
* its settings or parameters;
* how it was implemented;
* its known limitations.

## 4. Alternative Solutions
Students must identify and compare credible alternative approaches.
At least one alternative or baseline solution should be considered. The comparison may
address:
* accuracy;
* robustness;
* computational time;
* memory use;
* implementation complexity;
* interpretability;
* scalability;
* suitability for the intended engineering application.
The purpose is not to test every possible method. It is to demonstrate that the final approach
was selected using evidence rather than preference alone.

## 5. Validation
Every project must include a clear and justified validation method.
Depending on the project, validation may use:
* known ground truth;
* withheld test data;
* independent experimental data;
* repeated measurements;
* comparison with a trusted benchmark;
* comparison with an established computational method;
* cross-validation;
* physical constraints or conservation relationships;
* sensitivity and robustness testing;
* manual inspection supported by defined criteria.
Students must explain:
* what was validated;
* why the validation method is appropriate;
* which metrics or acceptance criteria were used;
* whether the validation information influenced model selection;
* what the validation cannot establish.
A numerical result or visually convincing output is not sufficient evidence of accuracy.


## 6. Optimisation and Performance
Students must analyse and, where appropriate, improve the performance of their solution.
Relevant considerations may include:
* solution accuracy;
* computational runtime;
* memory requirements;
* robustness;
* numerical stability;
* model or algorithm settings;
* input resolution;
* data requirements;
* repeatability;
* scalability;
* trade-offs between performance measures.
Optimisation should be relevant to the project. Students are not expected to optimise every
possible characteristic.
Students should make use of profilers, and timing routines, and where appropriate estimates
of arithmetic intensity and required number of FLOPS associated with a method to help
explain and justify the performance of their code.
The report must compare the original or baseline solution with the improved or selected
solution and justify the final design.


## 7. AI Use and Reflection
Students may use AI tools throughout the project.
All material AI use must be disclosed. The report must include an AI reflection that explains:
* which AI tools were used;
* what they were used for;
* the approximate level of AI contribution;
* why AI was used for those tasks;
* how AI-generated material was checked;
* which important decisions remained the student's responsibility;
* any errors, limitations or unhelpful suggestions produced by AI;
* how AI use affected the student's understanding or workflow.
Students remain responsible for all submitted code, results, claims, references and
engineering decisions. Disclosure does not replace verification.
