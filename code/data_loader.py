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

    # Point to the recovered path or use the relocation directory target
    occ_path = "/content/occupancy_extracted/recovered_occupancy.csv"
    if not os.path.exists(occ_path):
        occ_path = os.path.join(cleaned_full_dir, "occupancy_full.csv")

    # If neither exists, unpack dynamically from relocated ZIP archive
    if not os.path.exists(occ_path):
        occ_zip_path = os.path.join(cleaned_full_dir, "5occupancySensor_MayToDec2024_9MRows.zip")
        extract_dir = "/content/occupancy_extracted"
        os.makedirs(extract_dir, exist_ok=True)
        recovered_csv_path = os.path.join(extract_dir, "recovered_occupancy.csv")

        if os.path.exists(occ_zip_path):
            with open(occ_zip_path, 'rb') as f:
                data = f.read()
            fn = b"5occupancySensor_MayToDec2024_9MRows.csv"
            fn_idx = data.find(fn)
            if fn_idx != -1:
                header_start = fn_idx - 30
                extra_len = int.from_bytes(data[header_start+28:header_start+30], byteorder='little')
                payload_start = fn_idx + len(fn) + extra_len
                decompressor = zlib.decompressobj(-15)
                decompressed_data = decompressor.decompress(data[payload_start:])
                with open(recovered_csv_path, 'wb') as out_f:
                    out_f.write(decompressed_data)
                occ_path = recovered_csv_path

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
    merged_df['co2_rate_of_change'] = merged_df['co2'].diff().fillna(0.0)

    features = ['co2', 'temperature', 'humidity', 'co2_rate_of_change']
    X = merged_df[features]
    y = merged_df['occupancy_binary'].astype(int)

    return train_test_split(X, y, test_size=0.20, shuffle=False)