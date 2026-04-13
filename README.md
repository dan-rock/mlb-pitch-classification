# MLB Pitch Type Classification

Pitch type classification using the full 2023 MLB regular season from Baseball Savant (Statcast). Three models are benchmarked — Logistic Regression, XGBoost, and a PyTorch MLP — on 717,000+ pitches across 13 pitch types, with analysis of feature importance and confusion patterns grounded in baseball biomechanics.

---

## Overview

Every pitch thrown has a physical signature: how fast it moves, how much it spins, and how it breaks horizontally and vertically on its way to the plate. This project builds a classifier that identifies pitch type from those physical measurements alone — without knowing the pitcher or the game situation.

The goal is not just to maximize accuracy, but to understand *why* certain pitches are harder to distinguish, and which physical features carry the most discriminating information.

---

## Data

- **Source:** [Baseball Savant](https://baseballsavant.mlb.com) via the `pybaseball` library
- **Coverage:** 2023 MLB regular season (March 30 – October 1)
- **Size:** 717,290 pitches after cleaning, 143,458 held out for testing
- **Target:** 13 pitch types (4-Seam Fastball, Sinker, Cutter, Slider, Sweeping Curve, Curveball, Changeup, Splitter, Knuckle Curve, and others)

**Features (15 physics-based inputs):**

| Feature | Description |
|---|---|
| `release_speed` | Pitch velocity (mph) |
| `release_spin_rate` | Spin rate (rpm) |
| `pfx_x`, `pfx_z` | Horizontal & vertical movement (inches) |
| `release_pos_x/y/z` | 3D release point (feet) |
| `plate_x`, `plate_z` | Location at home plate (feet) |
| `vx0`, `vy0`, `vz0` | Velocity components at release |
| `ax`, `ay`, `az` | Acceleration components |

---

## Methodology

### Preprocessing
- Aggregated and cleaned raw Statcast data; removed pitches with missing core features
- Imputed missing `release_spin_rate` values using per-pitch-type median — a defensible choice since spin rate is highly characteristic of pitch family
- Filtered pitch types with fewer than 500 occurrences to maintain class integrity
- Applied `StandardScaler` to normalize features across very different scales (velocity ~90, spin ~2400, movement ~10)
- Stratified 80/20 train/test split to preserve class proportions

### Exploratory Data Analysis
- Visualized pitch movement profiles (pfx_x vs pfx_z) to assess cluster separability by pitch type
- Analyzed velocity and spin rate distributions — confirmed strong discriminating power at the extremes (e.g., Eephus, Forkball) but overlap in the mid-range (Changeup, Slider, Cutter)
- Computed feature correlation matrix — identified near-perfect collinearity between `vy0` and `release_speed` (-1.00) and between `ax` and `pfx_x` (0.98)

### Modeling
Three models were evaluated in sequence, from simplest to most complex:

**1. Logistic Regression (baseline)**
- Multinomial logistic regression with L2 regularization
- Establishes a linear decision boundary baseline

**2. XGBoost (main model)**
- Gradient boosted trees: 400 estimators, max depth 8, learning rate 0.1
- Subsample and column sampling to reduce overfitting
- Evaluated with mlogloss on validation set across all 400 rounds

**3. PyTorch MLP (deep learning)**
- 3 hidden layers: 256 → 256 → 128 with BatchNorm and Dropout
- Adam optimizer with weight decay; ReduceLROnPlateau scheduler
- Trained for 40 epochs with batch size 512

---

## Results

| Model | Accuracy | Train Time |
|---|---|---|
| Logistic Regression | 76.2% | 31s |
| PyTorch MLP | 90.7% | 143s |
| **XGBoost** | **92.6%** | **21s** |

XGBoost was the best-performing model — achieving 92.6% accuracy while training 7x faster than the MLP. This is consistent with the well-documented advantage of gradient boosting on tabular data with mixed feature scales, where deep learning does not automatically provide an advantage.

### Feature Importance (XGBoost)
The top features by gain were:
1. `az` — vertical acceleration (23%)
2. `pfx_z` — vertical movement (15%)
3. `ax` — horizontal acceleration (13%)
4. `release_speed` — pitch velocity (13%)

Vertical acceleration dominates because spin-induced vertical force is the primary biomechanical signal distinguishing pitch families — a curveball's topspin produces strong downward acceleration, while a 4-seam fastball's backspin resists gravity.

### Confusion Analysis
The hardest classification boundaries were:
- **Cutter → Slider** (14% misclassification) — both occupy similar glove-side movement profiles
- **Splitter → Changeup** (16% misclassification) — both are arm-side, low-movement pitches with similar velocity separation from fastballs
- **Sweeping Curve ↔ Slider** (9% mutual confusion) — a genuinely ambiguous boundary even for human scouts

These confusions are physically meaningful and consistent with what analysts consider the most difficult pitch type distinctions in practice.


## Requirements

```
pybaseball
pandas
numpy
scikit-learn
xgboost
torch
matplotlib
seaborn
statsmodels
```

Install with:
```bash
pip install pybaseball pandas numpy scikit-learn xgboost torch matplotlib seaborn statsmodels
```

---

## Author

**Daniel Rocha**  
Master's in Applied Statistics and Data Science, UCLA  
[LinkedIn](https://linkedin.com/in/daniel-alejandro-rocha) · [GitHub](https://github.com/dan-rock)
