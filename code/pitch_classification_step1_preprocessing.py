# Pitch Type Classification
# PART 1: Data Collection & Preprocessing

# pip install pybaseball pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import statcast
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 1. DATA COLLECTION
# Pulling the full 2023 MLB regular season from Statcast.
# Returns every pitch thrown ~700,000+ rows.

print("Fetching Statcast data...")
df_raw = statcast(start_dt='2023-03-30', end_dt='2023-10-01')
print(f"Raw dataset shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

# Save raw data to not re-download
# df_raw.to_csv("statcast_2023_raw.csv", index=False)
# print("Raw data saved to statcast_2023_raw.csv")



# 2. FEATURE SELECTION
# Selecting physics-based features that capture pitch identity
# i.e. release mechanics, movement, velocity components, and plate location.

FEATURES = [
    'release_speed',       # Pitch velocity (mph)
    'release_spin_rate',   # Spin rate (rpm)
    'pfx_x',               # Horizontal movement (inches, catcher's perspective)
    'pfx_z',               # Vertical movement (inches)
    'release_pos_x',       # Horizontal release point (feet)
    'release_pos_y',       # Distance from home plate at release (feet)
    'release_pos_z',       # Vertical release point (feet)
    'plate_x',             # Horizontal location at plate (feet)
    'plate_z',             # Vertical location at plate (feet)
    'vx0',                 # Velocity x-component at release
    'vy0',                 # Velocity y-component at release
    'vz0',                 # Velocity z-component at release
    'ax',                  # Acceleration x-component
    'ay',                  # Acceleration y-component
    'az',                  # Acceleration z-component
]

TARGET = 'pitch_type'

df = df_raw[FEATURES + [TARGET]].copy()
print(f"\nSelected feature set shape: {df.shape}")



# 3. HANDLING MISSING VALUES
print("\n--- Missing Values (before cleaning) ---")
print(df.isnull().sum())

# Dropping rows where the target label is missing or null
df = df[df[TARGET].notna()]


# converting column to a float
df['release_spin_rate'] = df['release_spin_rate'].astype(float)

# release_spin_rate has frequent nulls (especially for older pitchers' data).
# Imputing with the median spin rate per pitch type. A fair assumption
# since spin rate is highly characteristic of pitch type.
df['release_spin_rate'] = df.groupby(TARGET)['release_spin_rate'].transform(
    lambda x: x.fillna(x.median())
)

# For all remaining features, drop rows with any nulls.
before = len(df)
df = df.dropna(subset=FEATURES)
after = len(df)
print(f"\nDropped {before - after} rows with remaining nulls ({(before-after)/before*100:.1f}%)")



# 4. FILTER RARE PITCH TYPES
# Pitch types with very few examples hurt model training and
# inflate class imbalance. Keeping only types with 500+ occurrences.

print("\n--- Pitch Type Distribution (before filtering) ---")
print(df[TARGET].value_counts())

MIN_COUNT = 500
counts = df[TARGET].value_counts()
valid_types = counts[counts >= MIN_COUNT].index.tolist()
df = df[df[TARGET].isin(valid_types)]

print(f"\n--- Pitch Types Retained (>= {MIN_COUNT} occurrences) ---")
print(df[TARGET].value_counts())
print(f"\nFinal dataset shape: {df.shape}")



# 5. ENCODE TARGET LABELS
le = LabelEncoder()
df['pitch_label'] = le.fit_transform(df[TARGET])

print(f"\nLabel encoding map:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")



# 6. SCALE FEATURES
# Features operate on different scale

X = df[FEATURES].values
y = df['pitch_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeature means after scaling (should be ~0): {X_scaled.mean(axis=0).round(3)}")
print(f"Feature stds after scaling (should be ~1):  {X_scaled.std(axis=0).round(3)}")


# 7. TRAIN / TEST SPLIT
# Stratify by pitch type to preserve class proportions in splits.

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]:,}")
print(f"Test set size:  {X_test.shape[0]:,}")



# 8. SAVE PREPROCESSED DATA
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("label_classes.npy", le.classes_)

print("\nPreprocessed data saved:")
print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy, label_classes.npy")
print("\nStep 1 complete. Run step2_eda.py next.")
