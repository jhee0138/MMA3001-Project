import pandas as pd

def run_baseline_model(X_test, threshold=500.0):
    return (X_test["co2"] > threshold).astype(int)