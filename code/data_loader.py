import os
import pandas as pd
import numpy as np
import zlib
from sklearn.model_selection import train_test_split

def load_and_prepare_dataset(base_dir):
    # Enforce targeting the complete full datasets
    cleaned_full_dir = os.path.join(base_dir, "cleaned_data_full")

    def load_and_prepare(filename, value_col):
        path = os.path.join(cleaned_full_dir, filename)
        df = pd.read_csv(path)
        df['readable_time'] = pd.to_datetime(df['readable_time'])
        df = df[['readable_time', 'value']].rename(columns={'value': value_col})
        df = df.set_index('readable_time')
        df = df.groupby(level=0).mean()
        return df

    # Load full environmental features
    co2_df = load_and_prepare("carbon_dioxide_full.csv", "co2")
    temp_df = load_and_prepare("temperature_full.csv", "temperature")
    hum_df = load_and_prepare("humidity_full.csv", "humidity")

    # Resolve timezones
    for df in [co2_df, temp_df, hum_df]:
        df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')

    # Load full occupancy labels
    occ_path = "/content/occupancy_extracted/recovered_occupancy.csv"
    if not os.path.exists(occ_path):
        occ_path = os.path.join(base_dir, "cleaned_data_full", "occupancy_full.csv")
        
    # Load all records to avoid class truncation issues
    occ_raw = pd.read_csv(occ_path)
    time_col = 'readable_time' if 'readable_time' in occ_raw.columns else 'collecteddate' if 'collecteddate' in occ_raw.columns else occ_raw.columns[0]
    occ_raw[time_col] = pd.to_datetime(occ_raw[time_col], utc=True)
    status_col = 'occupancystatus' if 'occupancystatus' in occ_raw.columns else 'occupancy' if 'occupancy' in occ_raw.columns else occ_raw.columns[1]
    occ_raw['occupancy_binary'] = occ_raw[status_col].apply(lambda x: 1 if 'Occupied' in str(x) else 0)
    occ_df = occ_raw[[time_col, 'occupancy_binary']].rename(columns={time_col: 'readable_time'}).set_index('readable_time')
    occ_df = occ_df.groupby(level=0).max()
    occ_df.index = occ_df.index.tz_convert('UTC') if occ_df.index.tz is not None else occ_df.index.tz_localize('UTC')

    # Merge and fill missing values
    merged_df = co2_df.join([temp_df, hum_df, occ_df], how='outer').sort_index()
    merged_df = merged_df.ffill().fillna(merged_df.median())

    # Feature Engineering
    merged_df['co2_rate_of_change'] = merged_df['co2'].diff().fillna(0.0)

    features = ['co2', 'temperature', 'humidity', 'co2_rate_of_change']
    X = merged_df[features]
    y = merged_df['occupancy_binary'].astype(int)

    # Chronological Split
    return train_test_split(X, y, test_size=0.20, shuffle=False)