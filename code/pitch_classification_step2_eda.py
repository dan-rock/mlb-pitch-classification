
# Pitch Type Classification 
# PART 2: Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# LOAD DATA

df = pd.read_csv("statcast_2023_raw.csv", low_memory=False)

FEATURES = [
    'release_speed', 'release_spin_rate',
    'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'plate_x', 'plate_z',
    'vx0', 'vy0', 'vz0',
    'ax', 'ay', 'az',
]
TARGET = 'pitch_type'

# Apply same cleaning as in PART 1
df = df[FEATURES + [TARGET]].copy()
df = df[df[TARGET].notna()]
df['release_spin_rate'] = df.groupby(TARGET)['release_spin_rate'].transform(
    lambda x: x.fillna(x.median())
)
df = df.dropna(subset=FEATURES)

# Only considering pitches thrown >= 500 times
MIN_COUNT = 500
counts = df[TARGET].value_counts()
valid_types = counts[counts >= MIN_COUNT].index.tolist()
df = df[df[TARGET].isin(valid_types)]

# Readable pitch type names for plot labels
PITCH_NAMES = {
    'FF': '4-Seam Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'SL': 'Slider',
    'SW': 'Sweeper',
    'ST': 'Sweeping Curve',
    'CU': 'Curveball',
    'CH': 'Changeup',
    'FS': 'Splitter',
    'KC': 'Knuckle Curve',
    'KN': 'Knuckleball',
    'EP': 'Eephus',
    'FO': 'Forkball',
    'SC': 'Screwball',
    'CS': 'Slow Curve',
}
df['pitch_name'] = df[TARGET].map(PITCH_NAMES).fillna(df[TARGET])

print(f"Dataset shape: {df.shape}")
print(f"Pitch types in dataset:\n{df[TARGET].value_counts()}\n")



# 1. CLASS DISTRIBUTION
fig, ax = plt.subplots(figsize=(10, 5))

pitch_counts = df['pitch_name'].value_counts()
colors = plt.cm.tab10(np.linspace(0, 1, len(pitch_counts)))

bars = ax.barh(pitch_counts.index, pitch_counts.values, color=colors)

# Annotate with counts and percentages
total = len(df)
for bar, count in zip(bars, pitch_counts.values):
    pct = count / total * 100
    ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
            f'{count:,}  ({pct:.1f}%)', va='center', fontsize=9)

ax.set_xlabel('Pitch Count', fontsize=11)
ax.set_title('2023 MLB Pitch Type Distribution (Statcast)', fontsize=13, fontweight='bold')
ax.set_xlim(0, pitch_counts.max() * 1.2)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
plt.show()
print("Saved: eda_class_distribution.png")



# 2. PITCH MOVEMENT PROFILE
# pfx_x = horizontal break (positive = arm side, negative = glove side)
# pfx_z = vertical break (positive = rise, negative = drop)
# All movement values are relative to a theoretical spin-less pitch.

fig, ax = plt.subplots(figsize=(10, 8))

pitch_types = df[TARGET].unique()
palette = dict(zip(pitch_types, plt.cm.tab10(np.linspace(0, 1, len(pitch_types)))))

# Sample for readability (plotting 700k points is slow and unreadable)
sample = df.groupby(TARGET, group_keys=False).apply(
    lambda x: x.sample(min(len(x), 1500), random_state=42)
)

for pt in pitch_types:
    subset = sample[sample[TARGET] == pt]
    name = PITCH_NAMES.get(pt, pt)
    ax.scatter(subset['pfx_x'], subset['pfx_z'],
               alpha=0.25, s=10, color=palette[pt], label=name)

# Reference lines
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

ax.set_xlabel('Horizontal Movement — pfx_x (inches)\n← Glove Side   |   Arm Side →', fontsize=11)
ax.set_ylabel('Vertical Movement — pfx_z (inches)\n↓ Drop   |   Rise ↑', fontsize=11)
ax.set_title('Pitch Movement Profile by Type — 2023 MLB Season\n(Catcher\'s Perspective, vs. Spin-Less Pitch)', 
             fontsize=13, fontweight='bold')

legend = ax.legend(loc='upper right', markerscale=2.5, fontsize=9,
                   framealpha=0.9, title='Pitch Type')
plt.tight_layout()
plt.savefig("eda_movement_profile.png", dpi=150)
plt.show()
print("Saved: eda_movement_profile.png")


# 3. VELOCITY DISTRIBUTION BY PITCH TYPE
fig, ax = plt.subplots(figsize=(11, 6))

pitch_order = df.groupby('pitch_name')['release_speed'].median().sort_values(ascending=False).index

sns.boxplot(
    data=df,
    x='release_speed',
    y='pitch_name',
    order=pitch_order,
    palette='tab10',
    width=0.6,
    fliersize=1.5,
    ax=ax
)

ax.set_xlabel('Release Speed (mph)', fontsize=11)
ax.set_ylabel('')
ax.set_title('Pitch Velocity Distribution by Type — 2023 MLB Season', fontsize=13, fontweight='bold')
ax.axvline(df['release_speed'].mean(), color='red', linestyle='--', linewidth=1, label='League Average')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("eda_velocity_distribution.png", dpi=150)
plt.show()
print("Saved: eda_velocity_distribution.png")


# 4. SPIN RATE DISTRIBUTION BY PITCH TYPE
fig, ax = plt.subplots(figsize=(11, 6))

spin_order = df.groupby('pitch_name')['release_spin_rate'].median().sort_values(ascending=False).index

sns.boxplot(
    data=df,
    x='release_spin_rate',
    y='pitch_name',
    order=spin_order,
    palette='tab10',
    width=0.6,
    fliersize=1.5,
    ax=ax
)

ax.set_xlabel('Spin Rate (rpm)', fontsize=11)
ax.set_ylabel('')
ax.set_title('Spin Rate Distribution by Pitch Type — 2023 MLB Season', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_spin_rate_distribution.png", dpi=150)
plt.show()
print("Saved: eda_spin_rate_distribution.png")



# 5. FEATURE CORRELATION HEATMAP
fig, ax = plt.subplots(figsize=(12, 9))

corr = df[FEATURES].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.4, ax=ax, annot_kws={'size': 7})

ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150)
plt.show()
print("Saved: eda_correlation_heatmap.png")



# 6. SUMMARY STATISTICS TABLE
print("\n--- Summary Statistics by Pitch Type (key features) ---")
summary = df.groupby('pitch_name')[['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']].agg(['mean', 'std']).round(2)
print(summary.to_string())

print("\nStep 2 complete. Review plots before moving to Step 3 (modeling).")
