"""Central settings for the building-level occupancy detection pipeline.

Every tunable choice lives here so it can be reported and varied in one place.
"""
import os

# Repository layout
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(REPO_ROOT, "cleaned_data_full")
OCC_DIR = os.path.join(REPO_ROOT, "new_cleaned_data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Campus local time, used for hour-of-day and weekday features
LOCAL_TZ = "Australia/Melbourne"

# Environmental sensor used as model input. 6012002000869 is the only sensor whose
# CO2 rises with building occupancy (Mar-Jun 2024); 6012002000326 (Sep 2025 onwards)
# shows no occupancy signal and is kept as a comparison case.
DEFAULT_SENSOR_ID = 6012002000869
SENSOR_IDS = {"869": 6012002000869, "326": 6012002000326}

# Environmental variables: short name -> (file in ENV_DIR, unit, valid physical range)
ENV_VARIABLES = {
    "co2": ("carbon_dioxide_full.csv", "ppm", (300.0, 5000.0)),
    "temperature": ("temperature_full.csv", "degC", (0.0, 45.0)),
    "humidity": ("humidity_full.csv", "%RH", (0.0, 100.0)),
}

# Occupancy status files and which statuses count as "occupied"
OCC_STATUSES = ["CurrentlyOccupied", "RecentlyOccupied", "NotOccupied"]
OCC_FILE_PATTERN = "occupancy_status_{}.csv.gz"
# RecentlyOccupied means people have left, so it is not counted as occupied by default
RECENT_AS_OCCUPIED = False

# Time discretisation
BIN = "15min"
BIN_MINUTES = 15
# A space keeps its last reported status for at most this long without new records
MAX_STATUS_HOLD = "12h"
# Building occupancy is only defined when at least this many spaces have a known status
MIN_SPACES_KNOWN = 3
# Building is labelled occupied when at least this fraction of spaces is occupied
OCCUPIED_FRACTION_THRESHOLD = 0.5

# Feature sets
ENV_FEATURES = ["co2", "temperature", "humidity", "co2_rate"]
TIME_FEATURES = ["hour", "day_of_week", "is_weekend"]
ALL_FEATURES = ENV_FEATURES + TIME_FEATURES

# Fixed-schedule baseline: occupied on weekdays between these local hours
SCHEDULE_START_HOUR = 8
SCHEDULE_END_HOUR = 18

# Evaluation
TEST_FRACTION = 0.2
CV_SPLITS = 5
RANDOM_STATE = 42
