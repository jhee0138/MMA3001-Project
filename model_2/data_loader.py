import os
import pandas as pd
import numpy as np
import zlib
from sklearn.model_selection import train_test_split

def load_and_prepare_dataset(base_dir):
    cleaned_full_dir = os.path.join(base_dir, "cleaned_data_full")

    def load_and_prepare(filename, value_col):
        path = os.path.join(cleaned_full_dir, filename)
        df = pd.read_csv(path)
        df['readable_time'] = pd.to_datetime(df['readable_time'])
        df = df[['readable_time', 'value']].rename(columns={'value': value_col})
        df = df.set_index('readable_time')
        df = df.groupby(level=0).mean()
        return df

    # Load environmental features from their full cleaned directories
    co2_df = load_and_prepare("carbon_dioxide_full.csv", "co2")
    temp_df = load_and_prepare("temperature_full.csv", "temperature")
    hum_df = load_and_prepare("humidity_full.csv", "humidity")

    for df in [co2_df, temp_df, hum_df]:
        df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')

    occ_path = "/content/occupancy_extracted/recovered_occupancy.csv"
    if not os.path.exists(occ_path):
        occ_path = os.path.join(cleaned_full_dir, "occupancy_full.csv")

    # Read the full dataset
    occ_raw = pd.read_csv(occ_path)
    time_col = 'readable_time' if 'readable_time' in occ_raw.columns else 'collecteddate' if 'collecteddate' in occ_raw.columns else occ_raw.columns[0]
    occ_raw[time_col] = pd.to_datetime(occ_raw[time_col], utc=True)
    status_col = 'occupancystatus' if 'occupancystatus' in occ_raw.columns else 'occupancy' if 'occupancy' in occ_raw.columns else occ_raw.columns[1]

    # Precise class status mapping
    mapping = {
        'CurrentlyOccupied': 1,
        'RecentlyOccupied': 1,
        'NotOccupied': 0
    }
    occ_raw['occupancy_binary'] = occ_raw[status_col].map(mapping).fillna(0).astype(int)

    occ_df = occ_raw[[time_col, 'occupancy_binary']].rename(columns={time_col: 'readable_time'}).set_index('readable_time')
    occ_df = occ_df.groupby(level=0).max()
    occ_df.index = occ_df.index.tz_convert('UTC') if occ_df.index.tz is not None else occ_df.index.tz_localize('UTC')

    merged_df = co2_df.join([temp_df, hum_df, occ_df], how='outer').sort_index()
    merged_df = merged_df.ffill().fillna(merged_df.median())

    # Base environmental features
    merged_df['co2_rate_of_change'] = merged_df['co2'].diff().fillna(0.0)

    # Extract temporal features directly from DatetimeIndex
    merged_df['hour'] = merged_df.index.hour
    merged_df['day_of_week'] = merged_df.index.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)

    features = ['co2', 'temperature', 'humidity', 'co2_rate_of_change', 'hour', 'day_of_week', 'is_weekend']
    X = merged_df[features]
    y = merged_df['occupancy_binary'].astype(int)

    return train_test_split(X, y, test_size=0.20, shuffle=False)