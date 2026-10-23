import numpy as np
import pandas as pd
import pytest

import config
import data_loader


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def records(events):
    """events: list of (space, time string, occupied)"""
    df = pd.DataFrame(events, columns=["floorspaceid", "time", "occupied"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time")


def test_status_held_until_next_event():
    bins = data_loader.space_occupancy_bins(records([
        ("a", "2024-03-01 00:02", 1),
        ("a", "2024-03-01 01:05", 0),
    ]))["a"]
    # Occupied from 00:02 until 01:05, so every bin up to and including 01:00 is occupied
    assert bins[utc("2024-03-01 00:00")] == 1
    assert bins[utc("2024-03-01 00:45")] == 1
    assert bins[utc("2024-03-01 01:00")] == 1
    assert len(bins) == 5


def test_status_expires_after_max_hold():
    bins = data_loader.space_occupancy_bins(records([
        ("a", "2024-03-01 00:00", 0),
        ("a", "2024-03-02 00:00", 1),
    ]), max_hold="1h")["a"]
    assert bins[utc("2024-03-01 00:45")] == 0
    assert np.isnan(bins[utc("2024-03-01 06:00")])
    assert bins[utc("2024-03-02 00:00")] == 1


def test_short_visit_inside_bin_counts_as_occupied():
    bins = data_loader.space_occupancy_bins(records([
        ("a", "2024-03-01 00:00", 0),
        ("a", "2024-03-01 00:20", 1),
        ("a", "2024-03-01 00:25", 0),
    ]))["a"]
    assert bins[utc("2024-03-01 00:00")] == 0
    assert bins[utc("2024-03-01 00:15")] == 1


def test_building_fraction_requires_enough_known_spaces():
    t = pd.date_range("2024-03-01", periods=2, freq="15min", tz="UTC")
    spaces = pd.DataFrame({"a": [1, 1], "b": [0, np.nan], "c": [1, np.nan]}, index=t)
    frac = data_loader.building_occupied_fraction(spaces, min_spaces_known=2)
    assert frac.iloc[0] == pytest.approx(2 / 3)
    assert np.isnan(frac.iloc[1])


def test_label_threshold_is_inclusive():
    frac = pd.Series([0.0, 0.4, 0.5, 1.0])
    assert data_loader.label_occupied(frac, 0.5).tolist() == [0, 0, 1, 1]


def test_features_use_local_time_and_skip_gaps():
    t = pd.date_range("2024-03-03 23:00", periods=4, freq="15min", tz="UTC")
    env = pd.DataFrame({"co2": [450.0, 480.0, np.nan, 500.0]}, index=t)
    out = data_loader.add_features(env)
    # 30 ppm over 15 minutes = 2 ppm/min; no rate across the missing bin
    assert out["co2_rate"].iloc[1] == pytest.approx(2.0)
    assert np.isnan(out["co2_rate"].iloc[3])
    # 23:00 UTC on Sunday 3 March 2024 is 10:00 Monday in Melbourne (AEDT, UTC+11)
    assert out["hour"].iloc[0] == 10
    assert out["day_of_week"].iloc[0] == 0
    assert out["is_weekend"].iloc[0] == 0


def test_env_loader_removes_duplicates_and_faults(tmp_path):
    rows = [
        (1, 400, "2024-03-01 00:01:00+00:00"),
        (1, 400, "2024-03-01 00:01:00+00:00"),   # duplicate
        (1, 420, "2024-03-01 00:11:00+00:00"),
        (1, 9999, "2024-03-01 00:20:00+00:00"),  # outside physical range
        (2, 800, "2024-03-01 00:05:00+00:00"),   # other sensor
    ]
    pd.DataFrame(rows, columns=["sensorid", "value", "readable_time"]).to_csv(
        tmp_path / config.ENV_VARIABLES["co2"][0], index=False)
    audit = {}
    series = data_loader.load_env_variable("co2", 1, env_dir=tmp_path, audit=audit)
    assert series[utc("2024-03-01 00:00")] == pytest.approx(410.0)
    assert len(series) == 1  # the faulty 00:20 reading leaves no bin behind
    assert audit["co2"]["duplicate_rows_removed"] == 1
    assert audit["co2"]["out_of_range_removed"] == 1


@pytest.mark.parametrize("recent, expected", [(False, [1, 0, 0]), (True, [1, 1, 0])])
def test_occupancy_status_mapping(tmp_path, recent, expected):
    for i, status in enumerate(config.OCC_STATUSES):
        pd.DataFrame({"floorspaceid": ["a"], "occupancystatus": [status],
                      "collecteddate": [f"2024-03-01 1{i}:00:00+11"]}).to_csv(
            tmp_path / config.OCC_FILE_PATTERN.format(status), index=False)
    df = data_loader.load_occupancy_records(occ_dir=tmp_path, recent_as_occupied=recent)
    assert df["occupied"].tolist() == expected
    assert str(df["time"].dt.tz) == "UTC"
    assert df["time"].iloc[0] == utc("2024-02-29 23:00")


def test_chronological_split_has_no_overlap():
    data = pd.DataFrame({"x": range(10)}, index=pd.date_range("2024-03-01", periods=10, freq="15min"))
    train, test = data_loader.chronological_split(data, test_fraction=0.2)
    assert len(train) == 8 and len(test) == 2
    assert train.index.max() < test.index.min()
