"""Load, audit and align environmental and occupancy data on a regular time grid.

Pipeline:
    raw env readings (one sensor)  -> dedupe, range-check, 15-min means
    raw occupancy status events    -> per-space occupied/vacant per 15-min bin
                                   -> building occupied fraction -> binary label
    join on time, add features, drop incomplete rows (no imputation of labels)
"""
import os
import pandas as pd

import config


def load_env_variable(name, sensor_id, env_dir=config.ENV_DIR, audit=None):
    """Load one environmental variable for one sensor as a 15-min mean series.

    Parameters
    ----------
    name : str
        Key of ``config.CANDIDATE_ENV_VARIABLES`` (e.g. ``"co2"``).
    sensor_id : int
        Environmental sensor id (``sensorid`` column).
    env_dir : str
        Folder holding the ``*_full.csv`` files.
    audit : dict, optional
        If given, row counts from each cleaning step are recorded under ``name``.

    Returns
    -------
    pandas.Series
        Mean value per bin on a regular UTC grid; NaN where there was no valid reading.
    """
    filename, unit, (lo, hi) = config.CANDIDATE_ENV_VARIABLES[name]
    df = pd.read_csv(os.path.join(env_dir, filename), usecols=["sensorid", "value", "readable_time"])
    df = df[df["sensorid"] == sensor_id]
    n_raw = len(df)

    df["readable_time"] = pd.to_datetime(df["readable_time"], utc=True)
    # The export repeats most readings several times; keep one per timestamp
    df = df.drop_duplicates(["readable_time"])
    n_unique = len(df)

    # Readings outside the physical range are treated as sensor faults
    valid = df["value"].between(lo, hi)
    df = df[valid]

    if audit is not None:
        audit[name] = {
            "unit": unit,
            "raw_rows": int(n_raw),
            "duplicate_rows_removed": int(n_raw - n_unique),
            "out_of_range_removed": int((~valid).sum()),
            "valid_rows": int(len(df)),
        }

    return df.set_index("readable_time")["value"].sort_index().resample(config.BIN).mean().rename(name)


def load_environment(sensor_id, env_dir=config.ENV_DIR, audit=None, variables=config.ENV_VARIABLES):
    """Load the given variables (default: the model inputs) for one sensor into one DataFrame."""
    series = [load_env_variable(name, sensor_id, env_dir, audit) for name in variables]
    return pd.concat(series, axis=1)


def load_occupancy_records(occ_dir=config.OCC_DIR, recent_as_occupied=config.RECENT_AS_OCCUPIED):
    """Load raw occupancy status events for all floor spaces.

    Returns
    -------
    pandas.DataFrame
        Columns ``floorspaceid``, ``time`` (UTC) and ``occupied`` (0/1), sorted by time.
    """
    occupied_statuses = {"CurrentlyOccupied"}
    if recent_as_occupied:
        occupied_statuses.add("RecentlyOccupied")

    frames = []
    for status in config.OCC_STATUSES:
        path = os.path.join(occ_dir, config.OCC_FILE_PATTERN.format(status))
        frames.append(pd.read_csv(path, usecols=["floorspaceid", "occupancystatus", "collecteddate"]))
    df = pd.concat(frames, ignore_index=True)

    df["time"] = pd.to_datetime(df["collecteddate"], utc=True, format="ISO8601")
    df["occupied"] = df["occupancystatus"].isin(occupied_statuses).astype(int)
    df = df.drop_duplicates(["floorspaceid", "time"])
    return df[["floorspaceid", "time", "occupied"]].sort_values("time").reset_index(drop=True)


def space_occupancy_bins(records, bin_freq=config.BIN, max_hold=config.MAX_STATUS_HOLD):
    """Convert status events into an occupied/vacant state per space per bin.

    The sensor only reports while something is happening, so a status is assumed to
    persist until the next event, for at most ``max_hold``. A bin is occupied if the
    space was occupied at any point within it: either an occupied event falls inside
    the bin or the status carried in from the previous bin was occupied.

    Returns
    -------
    pandas.DataFrame
        One column per floor space with values 1, 0 or NaN (state unknown).
    """
    hold_bins = int(pd.Timedelta(max_hold) / pd.Timedelta(bin_freq))
    columns = {}
    for space, group in records.groupby("floorspaceid"):
        status = group.set_index("time")["occupied"]
        any_occupied_in_bin = status.resample(bin_freq).max()
        carried_in = status.resample(bin_freq).last().ffill(limit=hold_bins).shift(1)
        columns[space] = pd.concat([any_occupied_in_bin, carried_in], axis=1).max(axis=1)
    return pd.DataFrame(columns)


def building_occupied_fraction(space_bins, min_spaces_known=config.MIN_SPACES_KNOWN):
    """Fraction of spaces occupied per bin, NaN when too few spaces have a known state."""
    known = space_bins.notna().sum(axis=1)
    return space_bins.mean(axis=1).where(known >= min_spaces_known).rename("occupied_fraction")


def label_occupied(occupied_fraction, threshold=config.OCCUPIED_FRACTION_THRESHOLD):
    """Binary building occupancy label: 1 when the occupied fraction reaches ``threshold``."""
    return (occupied_fraction >= threshold).astype(int).rename("occupied")


def add_features(env, local_tz=config.LOCAL_TZ, bin_minutes=config.BIN_MINUTES):
    """Add CO2 rate of change and local-time calendar features to binned env data.

    ``co2_rate`` is in ppm/min between consecutive bins; it is NaN when the previous
    bin is missing, so rates are never computed across data gaps.
    """
    out = env.copy()
    out["co2_rate"] = out["co2"].diff() / bin_minutes

    local = out.index.tz_convert(local_tz)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    return out


def build_dataset(sensor_id=config.DEFAULT_SENSOR_ID, threshold=config.OCCUPIED_FRACTION_THRESHOLD,
                  recent_as_occupied=config.RECENT_AS_OCCUPIED, occupancy_fraction=None):
    """Build the modelling table for one environmental sensor.

    Parameters
    ----------
    sensor_id : int
        Environmental sensor used as input.
    threshold : float
        Occupied-fraction threshold for the binary label.
    recent_as_occupied : bool
        Whether ``RecentlyOccupied`` counts as occupied.
    occupancy_fraction : pandas.Series, optional
        Precomputed building occupied fraction, to avoid reloading occupancy data.

    Returns
    -------
    data : pandas.DataFrame
        ``config.ALL_FEATURES`` plus ``occupied_fraction`` and ``occupied``, indexed by
        bin start time (UTC), containing only complete rows.
    audit : dict
        Row counts from each cleaning step, for reporting.
    """
    audit = {"sensor_id": int(sensor_id), "bin": config.BIN, "threshold": threshold}
    env = add_features(load_environment(sensor_id, audit=audit))

    if occupancy_fraction is None:
        records = load_occupancy_records(recent_as_occupied=recent_as_occupied)
        occupancy_fraction = building_occupied_fraction(space_occupancy_bins(records))

    data = env.join(occupancy_fraction, how="inner")
    audit["bins_with_env_or_occupancy"] = int(len(data))
    data = data.dropna(subset=config.ALL_FEATURES + ["occupied_fraction"])
    data["occupied"] = label_occupied(data["occupied_fraction"], threshold)

    audit["complete_bins"] = int(len(data))
    if len(data):
        audit["start"] = str(data.index.min())
        audit["end"] = str(data.index.max())
        audit["occupied_rate"] = float(data["occupied"].mean())
    return data, audit


def chronological_split(data, test_fraction=config.TEST_FRACTION):
    """Split rows in time order so the test set lies strictly after the training set."""
    n_train = int(round(len(data) * (1 - test_fraction)))
    return data.iloc[:n_train], data.iloc[n_train:]
