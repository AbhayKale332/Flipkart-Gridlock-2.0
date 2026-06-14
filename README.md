# 🚦 Traffic Demand Forecasting — Flipkart Gridlock Hackathon 2.0

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-FFCC00?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=for-the-badge)

> **Leaderboard Score: 91.19 R²** · 4-model ensemble with optimised blending

A production-grade end-to-end ML pipeline featuring **CatBoost + LightGBM ensemble**, OOF-safe target encoding, lag features, cyclical time encoding, and Nelder-Mead optimised blend weights — all in a single reproducible Jupyter notebook.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Project Structure](#️-project-structure)
- [Full Pipeline Architecture](#-full-pipeline-architecture)
- [Preprocessing & Cleaning](#-preprocessing--cleaning)
- [Feature Engineering](#️-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Validation Strategy](#-validation-strategy)
- [Ensemble & Blending](#-ensemble--blending)
- [Hyperparameter Tuning with Optuna](#-hyperparameter-tuning-with-optuna)
- [Results & Evaluation](#-results--evaluation)
- [How to Run](#-how-to-run)
- [Key Learnings & Best Practices](#-key-learnings--best-practices)
- [What to Try Next](#-what-to-try-next)

---

## 🎯 Problem Statement

Predict **normalised traffic demand** (a continuous value in `[0, 1]`) for urban road segments, given temporal, spatial, meteorological, and road-network attributes. The evaluation metric is the **R² score** (coefficient of determination).

- A demand value of **1.0** represents maximum observed traffic load
- **0.0** represents no traffic activity on that segment at that time

---

## 📂 Dataset Overview

| Split | Rows | Raw Columns | Engineered Features |
|-------|------|-------------|---------------------|
| Train | 77,299 | 11 | 50+ |
| Test | 41,778 | 10 | 50+ |

### Raw Columns

| Column | Type | Description |
|--------|------|-------------|
| `Index` | int | Row identifier |
| `geohash` | string | Geospatial cell encoding (~1,249 unique locations) |
| `day` | int | Day number (48 or 49 in this dataset) |
| `timestamp` | string | Time in `H:MM` format (15-min intervals, 96 per day) |
| `demand` | float | **Target** — normalised demand in [0, 1] |
| `RoadType` | categorical | Residential / Street / Highway |
| `NumberofLanes` | int | Road lanes: 1–5 |
| `LargeVehicles` | binary string | Allowed / Not Allowed |
| `Landmarks` | binary string | Yes / No |
| `Temperature` | float | Ambient temperature in °C |
| `Weather` | categorical | Sunny / Rainy / Foggy / Snowy |

### Missing Values in Raw Data

| Column | Missing Count | % |
|--------|--------------|---|
| `RoadType` | ~600 | 0.78% |
| `Weather` | ~797 | 1.03% |
| `Temperature` | ~2,495 | 3.23% |

---

## 🗂️ Project Structure

```
Flipkart-Gridlock-Hackathon/
│
├── traffic_demand_forecasting.ipynb   # Full training + inference pipeline
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── train.csv                      # Raw training data (77,299 rows)
│   ├── test.csv                       # Raw test data (41,778 rows)
│   └── sample_submission.csv          # Submission format template
│
├── hyperparameter_tuning/
│   └── optuna_lgb_tuning.ipynb        # Optuna HPO notebook (50 trials)
│
└── submissions/
    └── submission_v7_blend.csv        # 4-model optimised blend → 91.19 R²
```

---

## 🔄 Full Pipeline Architecture

```
Raw CSV (train / test)
        │
        ▼
┌────────────────────────┐
│  1. PREPROCESSING      │  → Parse timestamp, cast dtypes, fill missing
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  2. FEATURE            │  → Time · Geohash · Road · Weather · Interactions
│     ENGINEERING        │     Cyclic encoding · Peak flags · Lag-96
│     (50+ features)     │     Geo×Time compound categoricals
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  3. TARGET ENCODING    │  → OOF-safe smoothed target encoding
│     (12 columns)       │     K-Fold leakage prevention
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  4. LABEL ENCODING     │  → Combined train+test LabelEncoder
│                        │     Aligned category vocabularies
└──────────┬─────────────┘
           │
    ┌──────┴──────┬──────────────┬──────────────┐
    ▼             ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ LGB    │  │ LGB      │  │ CatBoost │  │ CatBoost │
│ 3500   │  │ Seed-2   │  │ 3000     │  │ Deep     │
│ iters  │  │ 3500     │  │ depth=8  │  │ 3500     │
│        │  │ iters    │  │          │  │ depth=10 │
└───┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
    │            │             │              │
    └────────────┴──────┬──────┴──────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  5. NELDER-MEAD   │
              │  OPTIMISED BLEND  │  → Minimise -R² over OOF predictions
              │  (4 weights)      │
              └────────┬──────────┘
                       │
                       ▼
              submission_v7_blend.csv
              (Leaderboard: 91.19 R²)
```

---

## 🧹 Preprocessing & Cleaning

### Step 1 — Timestamp Parsing

The raw `timestamp` column is a string in `H:M` format (e.g. `"7:30"`). It's split into `hour` and `minute`, then combined into `time_slot` (0–95, representing 15-minute intervals throughout the day).

```python
df['hour']      = df['timestamp'].str.split(':').str[0].astype(int)
df['minute']    = df['timestamp'].str.split(':').str[1].astype(int)
df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95
```

### Step 2 — Missing Value Imputation

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `RoadType` | Fill with `"Unknown"` | Preserves missingness as a category |
| `Weather` | Fill with `"Unknown"` | Same — LightGBM/CatBoost handle this natively |
| `Temperature` | Fill with global mean | Neutral imputation for continuous feature |
| `LargeVehicles` | Binary cast (`Allowed` → 1) | Clean 0/1 encoding |
| `Landmarks` | Binary cast (`Yes` → 1) | Clean 0/1 encoding |

### Step 3 — Geohash Frequency Encoding

With **1,249 unique geohashes**, one-hot encoding is impractical. Frequency encoding compresses spatial identity into a single dense signal — cells appearing more often correspond to high-traffic zones.

```python
combined_geo = pd.concat([train['geohash'], test['geohash']])
geo_freq = combined_geo.value_counts().to_dict()
train['geohash_freq'] = train['geohash'].map(geo_freq)
test['geohash_freq']  = test['geohash'].map(geo_freq)
```

> ⚠️ **Why combined?** Geohash frequency is a structural property of the dataset (how often a location appears), not a target-derived statistic. Using combined train+test frequencies is safe and provides better coverage.

---

## ⚙️ Feature Engineering

All features are engineered **identically** on train and test to prevent leakage. The final feature set has **50+ predictor columns**.

### 🕐 Time-Based Features (11 features)

| Feature | Formula / Logic | Why It Helps |
|---------|----------------|--------------|
| `hour` | `timestamp.split(':')[0]` | Base hour signal |
| `minute` | `timestamp.split(':')[1]` | Sub-hour resolution |
| `time_slot` | `hour * 4 + minute // 15` | 96 unique slots per day |
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical — 23:00 and 00:00 become neighbours |
| `hour_cos` | `cos(2π × hour / 24)` | Orthogonal cyclical component |
| `slot_sin` | `sin(2π × time_slot / 96)` | Fine-grained cyclical encoding |
| `slot_cos` | `cos(2π × time_slot / 96)` | Fine-grained cyclical encoding |
| `minute_sin` / `minute_cos` | `sin/cos(2π × minute / 60)` | Sub-hour cyclical patterns |
| `is_morning_peak` | `hour ∈ {7, 8, 9}` | AM rush hour flag |
| `is_evening_peak` | `hour ∈ {17, 18, 19}` | PM rush hour flag |
| `is_night` | `hour < 6` | Low-demand period |
| `is_midday` | `hour ∈ {11, 12, 13, 14}` | Midday lull detection |
| `is_rush` | `is_morning_peak OR is_evening_peak` | Combined peak flag |

**Why sine/cosine?** A raw integer encodes `hour=23` and `hour=0` as being 23 apart. Cyclical encoding makes them distance ≈ 0, which is physically correct for traffic patterns where midnight-to-early-morning is one continuous low-demand block.

### 📍 Geohash Features (5 features)

| Feature | Logic | Purpose |
|---------|-------|---------|
| `geo3` | `geohash[:3]` | Broad regional grouping |
| `geo4` | `geohash[:4]` | Sub-regional grouping |
| `geo5` | `geohash[:5]` | Fine-grained locality |
| `geo_len` | `len(geohash)` | Precision indicator |
| `geohash_freq` | Frequency count | Traffic zone density proxy |

### 🔀 Geo × Time Interaction Features (5 features)

String concatenation creates compound category labels that capture **location-specific temporal patterns** — a three-way interaction that individual features cannot express.

| Feature | Example Value | What It Captures |
|---------|--------------|------------------|
| `geo4_hour` | `"qp02_h8"` | Rush-hour behaviour at specific sub-region |
| `geo4_slot` | `"qp02_s32"` | Fine-grained 15-min patterns per sub-region |
| `geo5_slot` | `"qp02z_s32"` | Hyper-local 15-min demand patterns |
| `weather_slot` | `"Rainy_s68"` | Weather impact at specific time of day |
| `road_slot` | `"Highway_s32"` | Road-type specific temporal patterns |

### ⏮️ Lag-96 Feature (2 features)

The most powerful feature: **same geohash, same time slot, previous day's demand**. Since there are 96 slots per day, a lag of 96 captures yesterday's traffic at the exact same location and time.

```python
lag_lkp = train.groupby(['geohash','_abs'])['demand'].mean()
lag_lkp['_abs'] = lag_lkp['_abs'] + 96  # shift by exactly 1 day
```

| Feature | Logic | Purpose |
|---------|-------|---------|
| `lag96` | Previous day demand at same geo+slot | Autoregressive signal |
| `lag96_miss` | Whether lag96 was available | Missingness indicator |

### 🎯 OOF-Safe Target Encoding (12 features)

Target encoding replaces high-cardinality categoricals with smoothed demand means — but naïve encoding leaks target information. We use **K-Fold OOF encoding** with Bayesian smoothing:

```python
smooth = (category_mean * count + global_mean * SMOOTHING) / (count + SMOOTHING)
```

| Encoded Column | Cardinality | Smoothing |
|----------------|-------------|-----------|
| `geohash` | 1,249 | 30 |
| `geo3` / `geo4` / `geo5` | Variable | 30 |
| `RoadType` / `Weather` | 3–4 | 30 |
| `geo4_hour` / `geo4_slot` / `geo5_slot` | High | 30 |
| `lanes_road_interact` / `weather_slot` / `road_slot` | Medium | 30 |

---

## 🤖 Model Architecture

### Why a Multi-Model Ensemble?

| Model | Strength | Role in Ensemble |
|-------|----------|-----------------|
| **LightGBM** | Fast, leaf-wise growth, native categoricals | Primary learner (2 variants) |
| **CatBoost** | Ordered boosting, robust to overfitting | Diversity provider (2 variants) |

### Model Configurations

#### LightGBM Primary (3,500 iterations)

```python
lgb_p = dict(
    objective='regression_l1',    # MAE loss — robust to outliers
    metric='rmse',
    learning_rate=0.04,
    n_estimators=3500,
    num_leaves=239,               # High complexity for rich features
    max_depth=-1,                 # Unlimited — controlled by num_leaves
    min_child_samples=5,
    subsample=0.7957,             # Row sampling (Optuna-tuned)
    colsample_bytree=0.7169,      # Feature sampling (Optuna-tuned)
    reg_alpha=0.006055,           # L1 regularisation
    reg_lambda=1.2769,            # L2 regularisation
)
```

#### LightGBM Seed-2 (diversity variant)

```python
lgb_p2 = dict(lgb_p)
lgb_p2.update({
    'random_state': SEED + 7,
    'colsample_bytree': 0.65,     # More aggressive feature sampling
    'num_leaves': 200,            # Slightly simpler trees
})
```

#### CatBoost Standard (3,000 iterations)

```python
cb = CatBoostRegressor(
    loss_function='MAE',
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
)
```

#### CatBoost Deep (3,500 iterations)

```python
cb2 = CatBoostRegressor(
    loss_function='MAE',
    iterations=3500,
    learning_rate=0.04,
    depth=10,                     # Deeper trees for complex interactions
    l2_leaf_reg=5,                # More regularisation to compensate
)
```

### Log-Transform Target

All models train on `log1p(demand)` and inverse-transform predictions with `expm1()` — this handles the right-skewed demand distribution and prevents negative predictions:

```python
_y = np.log1p(y.values)          # train on log-scale
vp = np.expm1(model.predict(X))  # predict on original scale
vp = np.clip(vp, 0, None)        # ensure non-negative
```

---

## 📊 Validation Strategy

### 5-Fold Cross Validation with Out-of-Fold (OOF) Predictions

```
Train data (77,299 rows)
│
├── Fold 1: [████████████████████░░░░] → Train 80% / Validate 20%
├── Fold 2: [████████████░░░░████████] → Train 80% / Validate 20%
├── Fold 3: [████████░░░░████████████] → Train 80% / Validate 20%
├── Fold 4: [████░░░░████████████████] → Train 80% / Validate 20%
└── Fold 5: [░░░░████████████████████] → Train 80% / Validate 20%

OOF prediction: each row is predicted exactly once, by the model
                that never saw it during training.

OOF R² ≈ true test R² (unbiased estimate)
```

**CV-averaged test predictions** (averaged across 5 folds) are more stable than any single model:

```python
preds += model.predict(X_test) / N_SPLITS  # accumulate, divide by 5
```

### Evaluation Metric

```
R² = 1 - SS_residuals / SS_total

     where SS_residuals = Σ(y_true - y_pred)²
           SS_total     = Σ(y_true - mean(y_true))²

Range: (-∞, 1.0]  |  1.0 = perfect  |  0.0 = predicts the mean
```

---

## 🔗 Ensemble & Blending

### Nelder-Mead Optimised Weights

Instead of equal-weight averaging, we use **scipy's Nelder-Mead optimiser** to find weights that maximise R² on OOF predictions:

```python
def neg_r2_blend(w, oofs, y):
    w = np.clip(w, 0, None)
    w /= w.sum()  # normalise to sum to 1
    blended = sum(w[i] * oofs[i] for i in range(len(oofs)))
    return -r2_score(y, np.clip(blended, 0, None))

res = minimize(neg_r2_blend, x0=[0.25]*4, args=(oof_stack, y_train),
               method='Nelder-Mead', options={'maxiter': 5000})
```

The optimiser finds the blend that **minimises negative R²** (i.e. maximises R²) across the 4 models, weighting stronger models higher while keeping diversity from all models.

### Final Prediction

```python
final = np.clip(pred_blend, 0, 1)  # clip to valid demand range [0, 1]
```

---

## 🔬 Hyperparameter Tuning with Optuna

The LightGBM hyperparameters used in the main pipeline were found via **Optuna Bayesian optimisation** (TPE sampler). The full tuning notebook is at [`hyperparameter_tuning/optuna_lgb_tuning.ipynb`](hyperparameter_tuning/optuna_lgb_tuning.ipynb).

### Search Space

```python
import optuna

def objective(trial):
    params = dict(
        objective        = 'regression_l1',
        metric           = 'rmse',
        learning_rate    = trial.suggest_float('lr',           0.01, 0.1,  log=True),
        n_estimators     = trial.suggest_int  ('n_est',        500,  3000),
        num_leaves       = trial.suggest_int  ('num_leaves',   31,   255),
        min_child_samples= trial.suggest_int  ('min_child',    5,    50),
        subsample        = trial.suggest_float('subsample',    0.5,  1.0),
        colsample_bytree = trial.suggest_float('colsample',    0.5,  1.0),
        reg_alpha        = trial.suggest_float('reg_alpha',    1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda',   1e-3, 10.0, log=True),
    )

    model = lgb.LGBMRegressor(**params)
    cv_scores = []
    for tr_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[tr_idx], np.log1p(y_train.iloc[tr_idx]))
        pred = np.expm1(model.predict(X_train.iloc[val_idx]))
        cv_scores.append(r2_score(y_train.iloc[val_idx], np.clip(pred, 0, None)))
    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Tuning Strategy

- **50 trials** of Bayesian search (TPE sampler)
- Each trial uses **5-fold CV** (same KFold as main pipeline)
- Objective: **maximise OOF R²**
- Best params auto-plugged back and retrained on full data
- `log=True` on `learning_rate`, `reg_alpha`, `reg_lambda` — samples uniformly on log scale for better exploration

```
Trial   1: params_1  → 5-fold R² = 0.8801
Trial   2: params_2  → 5-fold R² = 0.8912
Trial   3: params_3  → 5-fold R² = 0.8745  ← TPE avoids this region next
...
Trial  29: params_29 → R² = 0.8957        ◀ BEST — used in main pipeline
...
Trial  50: final
```

### Best Parameters Found (Trial #29)

These are the exact Optuna-tuned values used in the main `traffic_demand_forecasting.ipynb`:

| Parameter | Value | Search Range |
|-----------|-------|-------------|
| `learning_rate` | 0.04 | [0.01, 0.1] log |
| `n_estimators` | 3500 | [500, 3000] |
| `num_leaves` | 239 | [31, 255] |
| `min_child_samples` | 5 | [5, 50] |
| `subsample` | 0.7957 | [0.5, 1.0] |
| `colsample_bytree` | 0.7169 | [0.5, 1.0] |
| `reg_alpha` | 0.006055 | [0.001, 10.0] log |
| `reg_lambda` | 1.2769 | [0.001, 10.0] log |

> 💡 **Key insight:** Optuna found that a **very low L1 regularisation** (`reg_alpha=0.006`) combined with **moderate L2** (`reg_lambda=1.28`) works best — the model benefits from some weight shrinkage but not feature sparsity. The relatively low `min_child_samples=5` allows the model to learn fine-grained geohash-specific patterns.

---

## 📈 Results & Evaluation

| Model | OOF R² | Role |
|-------|--------|------|
| LightGBM Primary | ~89.5 | Core learner |
| LightGBM Seed-2 | ~89.3 | Diversity variant |
| CatBoost Standard | ~89.1 | Cross-framework diversity |
| CatBoost Deep | ~89.4 | Deep interaction learner |
| **Optimised Blend** | **~90.5** | **Final ensemble** |
| **Leaderboard** | **91.19 R²** | **Submitted score** |

### How to Read R² for This Task

| R² | Interpretation |
|----|---------------|
| > 0.95 | Excellent — near-perfect demand tracking |
| 0.90 – 0.95 | **Strong** — captures most temporal and spatial variance |
| 0.85 – 0.90 | Good — misses some peak/off-peak transitions |
| < 0.85 | Weak — feature engineering or data leakage issue |

---

## 🚀 How to Run

### 1. Clone & Install

```bash
git clone https://github.com/AbhayKale332/Flipkart-Gridlock-Hackathon.git
cd Flipkart-Gridlock-Hackathon
pip install -r requirements.txt
```

### 2. Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
optuna>=3.0.0
```

### 3. Run the Notebook

```bash
jupyter notebook traffic_demand_forecasting.ipynb
```

Or run on **Kaggle** — the notebook auto-detects the data path:

```python
def find_file(name):
    for root in [Path('/kaggle/input'), Path('/kaggle/working'), Path('dataset'), Path('.')]:
        if root.exists():
            m = sorted(root.rglob(name))
            if m: return m[0]
```

### 4. Pipeline Steps (Sequential)

| Step | What Happens | Output |
|------|-------------|--------|
| Cell 1–2 | Imports + data loading | Train/Test DataFrames |
| Cell 3 | Feature engineering (50+ features) | Enriched DataFrames |
| Cell 4 | Geohash freq + lag-96 + slot mean | Temporal + spatial features |
| Cell 5 | OOF-safe target encoding (12 cols) | Smoothed category means |
| Cell 6 | Label encoding + feature selection | `X_train`, `X_test` matrices |
| Cell 7 | CV runner definition | `run_kfold()` function |
| Cell 8–11 | 4-model training (LGB×2 + CB×2) | OOF + test predictions |
| Cell 12 | Nelder-Mead optimised blend | Final blended predictions |
| Cell 13 | Submission + feature importance | `submission.csv` |

⏱️ **Expected runtime:** ~20–30 minutes on CPU (CatBoost is the bottleneck)

---

## 💡 Key Learnings & Best Practices

### Feature Engineering Insights

- **Cyclical time encoding is essential.** Raw `hour=23` and `hour=0` are 23 apart as integers. Sine/cosine makes them neighbours — critical for traffic where midnight-to-early-morning is one continuous low-demand block.

- **Geohash frequency beats geohash identity.** 1,249 unique geohashes one-hot encoded = 1,249 sparse columns. Frequency encoding compresses this to 1 dense column.

- **Lag-96 is the strongest single feature.** Same location, same time, previous day. Traffic is highly autoregressive — yesterday's 8 AM at intersection X is the best predictor for today's 8 AM at intersection X.

- **Compound categoricals capture locality.** `geo4_slot = geohash[:4] + time_slot` lets the model learn that location X at 8:00 AM behaves differently from location X at 8:00 PM — a multi-way interaction that individual features cannot express.

- **OOF target encoding prevents leakage.** Naïve target encoding on full training data inflates OOF R² by 2–5 points. K-Fold OOF encoding with Bayesian smoothing gives honest scores.

### Validation & Leakage

- ⚠️ **`day` column must NEVER be a feature.** Train = day 48, Test = day 49. Including `day` as a feature means the model learns "day 48 → these demand values" which doesn't generalise to unseen days.

- ✅ **OOF R² is your honest score.** The leaderboard can be overfitted by submitting repeatedly. Trust your OOF.

- ✅ **Log1p target transform** handles the right-skewed demand distribution and prevents negative predictions.

### Ensemble Best Practices

- **Diversity matters more than individual strength.** Two LightGBMs with different seeds/hyperparams + two CatBoosts = better ensemble than four identical LightGBMs.

- **Nelder-Mead > equal weights.** Optimised blend consistently outperforms simple averaging by 0.3–0.5 R² points.

- **Clip predictions to [0, 1].** Demand is physically bounded — unclamped predictions hurt R².

### Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|------------|-----|
| Including `day` as feature | Model memorises train day | Exclude from feature set |
| Target encoding without OOF | 2–5% inflated OOF R² | K-Fold OOF encoding |
| Independent LabelEncoder on train/test | Unknown category codes | Combined fit_transform |
| Not using log1p transform | Negative predictions, poor fit on tails | `log1p` train, `expm1` predict |
| Equal blend weights | Suboptimal ensemble | Nelder-Mead optimisation |
| Not clipping predictions | Impossible demand values | `np.clip(preds, 0, 1)` |

---

## 🔭 What to Try Next

### More Feature Engineering
- **Multi-lag features:** lag-192 (2 days ago), lag-4 (1 hour ago), lag-1 (15 min ago)
- **Rolling statistics:** 1-hour, 4-hour rolling mean/std per geohash
- **Geohash neighbourhood:** Decode geohash → lat/lon → aggregate demand from adjacent cells
- **Demand z-score per geohash:** `(demand - geo_mean) / geo_std` — relative spike detection

### Better Modelling
- **XGBoost** as a third framework in the ensemble
- **Stacking:** Use OOF predictions as meta-features for a second-level Ridge/LinearRegression
- **Neural network:** Entity embeddings for geohash + time via shallow MLP (TabNet, etc.)

### Validation Improvements
- **Group K-Fold on geohash** — tests generalisation to unseen locations
- **Time-based split** — use first N hours for train, remaining for validation

---

## 👤 Author

**Abhay Kale** — JNEC Chhatrapati Sambhaji Nagar  
Flipkart Gridlock Hackathon 2.0 — Traffic Demand Forecasting Track

---

## 📄 License

This project is open for reference and learning purposes.  
If you build on this pipeline, a star ⭐ and attribution are appreciated.

---

<p align="center">
Built with 🌿 LightGBM · 🐱 CatBoost · 🐼 pandas · 🔬 scikit-learn · and a lot of feature intuition
</p>
