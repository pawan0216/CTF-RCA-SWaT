import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/normal_processed.csv")
OUTPUT = Path("data/processed/normal_windows_60s.npz")

WINDOW = 60
STEP = 1

print("=" * 70)
print("SWaT TEMPORAL WINDOW CONSTRUCTION")
print("=" * 70)

df = pd.read_csv(INPUT)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

exclude = ["Timestamp", "Label", "Dataset"]

features = [
    c for c in df.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(df[c])
]

X = df[features].to_numpy(dtype=np.float32)

print("Observations:", len(df))
print("Features:", len(features))
print("Window:", WINDOW, "seconds")
print("Step:", STEP, "second")

# Number of windows
n_windows = 1 + (len(X) - WINDOW) // STEP

print("Number of windows:", n_windows)

windows = np.empty(
    (n_windows, WINDOW, len(features)),
    dtype=np.float32
)

timestamps = []

for i in range(n_windows):

    start = i * STEP
    end = start + WINDOW

    windows[i] = X[start:end]

    timestamps.append(
        df["Timestamp"].iloc[end - 1]
    )

timestamps = np.array(
    timestamps,
    dtype="datetime64[s]"
)

np.savez_compressed(
    OUTPUT,
    X=windows,
    timestamps=timestamps,
    feature_names=np.array(features)
)

print("\nSaved:", OUTPUT)
print("Shape:", windows.shape)
print("Memory-equivalent size:",
      round(windows.nbytes / (1024**3), 2),
      "GB")

print("\nFeature names:")
print(features)
