import { useState, useEffect } from "react";
import Header from "./Header";
// ─── PHASE DATA ────────────────────────────────────────────────────────────
const PHASES = [
  {
    id: 0,
    phase: "00",
    title: "Problem Understanding",
    subtitle: "Before touching any data",
    time: "30 min",
    color: "#94a3b8",
    icon: "🧭",
    mode: "both",
    summary: "The single most skipped and most costly step. Every downstream decision — CV strategy, metric choice, model family, feature engineering — is dictated by the problem structure. Get this wrong and nothing else matters.",
    steps: [
      {
        title: "Read everything twice",
        detail: "For competitions: read the problem description, data description, evaluation metric, and discussion forum. For projects: understand the business objective, cost of false positives vs false negatives, SLA requirements, and who consumes the model output.",
        code: null,
        warnings: [],
        tips: ["In competitions, the evaluation metric page is the most important document. Read it before looking at any data.", "Ask: what does the model predict? What data does it have access to at prediction time? This defines your CV strategy."],
      },
      {
        title: "Define the prediction task precisely",
        detail: "Binary classification? Multi-class? Regression? Multi-label? Ranking? Sequence? Time series forecasting? The answer determines your entire approach. Misidentifying this leads to wrong loss functions, wrong metrics, and wrong baselines.",
        code: `# Questions to answer before coding:
# 1. Target type: binary / multiclass / regression / multilabel / ordinal?
# 2. Prediction unit: row-level / entity-level / sequence?
# 3. Time structure: i.i.d. or ordered?
# 4. Evaluation: what metric? symmetric cost or asymmetric?
# 5. Deployment: batch / real-time? latency budget?`,
        warnings: ["If the task has any time ordering at all, random CV is wrong — note this now before building anything."],
        tips: ["Write out the 5 questions above in a markdown cell before loading data. This discipline saves hours."],
      },
      {
        title: "Identify the correct evaluation metric",
        detail: "Match training loss to evaluation metric from day one. For imbalanced classification: ROC-AUC, PR-AUC, or F-beta — not accuracy. For regression with skewed targets: RMSLE often beats RMSE. For ranking tasks: NDCG or MAP. Set this in stone before any modelling.",
        code: `# Metric quick reference:
# Balanced classification     → accuracy, F1
# Imbalanced classification   → ROC-AUC, PR-AUC, F-beta
# Probability output          → log-loss (requires calibration)
# Regression                  → RMSE (symmetric), MAE (robust), RMSLE (skewed target)
# Ranking                     → NDCG, MAP
# Object detection            → mAP
# Time series                 → MASE, SMAPE`,
        warnings: ["Optimising accuracy on imbalanced data will produce a model that always predicts the majority class. Check target balance BEFORE choosing metric."],
        tips: ["In competitions, the public leaderboard metric is ground truth. Implement it locally and use it as your CV score from day one."],
      },
    ],
    feRefs: [],
    edaRefs: ["Target Distribution Plot"],
    problemRefs: ["Wrong Evaluation Metric", "Metric Mismatch: Train Loss ≠ Eval Metric"],
  },
  {
    id: 1,
    phase: "01",
    title: "Environment & Reproducibility",
    subtitle: "Set seeds, structure, track experiments",
    time: "15 min",
    color: "#64748b",
    icon: "⚙️",
    mode: "both",
    summary: "Non-reproducible experiments are undebuggable experiments. Setting seeds and structuring experiment tracking costs 15 minutes and saves days of confusion.",
    steps: [
      {
        title: "Set all random seeds globally",
        detail: "Every stochastic component — data splitters, model initialisers, bootstrap samplers, shuffles — needs its seed fixed. A single unseeded operation anywhere in the pipeline makes results non-reproducible.",
        code: `import random, os
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# For sklearn: pass random_state=SEED to every estimator, splitter, and sampler
# For PyTorch:
# import torch
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True`,
        warnings: ["A model trained twice without a fixed seed that gives different CV scores makes it impossible to know if an improvement is real or just a lucky seed."],
        tips: [],
      },
      {
        title: "Set up experiment tracking",
        detail: "Track every experiment: hyperparameters, CV folds used, features included, CV score, and key notes. Without this, you will forget which model gave which score. Weights & Biases and MLflow are standard tools; a simple CSV log works fine too.",
        code: `# Minimal experiment log — paste into notebook:
experiment_log = []

def log_experiment(name, params, cv_score, notes=""):
    experiment_log.append({
        "name": name,
        "params": params,
        "cv_score": round(cv_score, 5),
        "notes": notes
    })
    print(f"[{name}] CV: {cv_score:.5f} | {notes}")

# Example:
# log_experiment("lgbm_baseline", {"n_est": 500, "lr": 0.05}, oof_score, "no FE yet")`,
        warnings: [],
        tips: ["In competitions, maintain a local spreadsheet: experiment name | CV score | public LB score | notes. This is how winners track the correlation between their local score and LB."],
      },
      {
        title: "Structure your project directory",
        detail: "A clean directory structure prevents the chaos of 'notebook_v2_final_FINAL_USE_THIS.ipynb'. Separate raw data (never modified), processed data, feature-engineered data, models, and submission files.",
        code: `project/
├── data/
│   ├── raw/          # NEVER touch these files
│   ├── processed/    # After cleaning
│   └── features/     # After feature engineering
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── features.py   # Feature engineering functions
│   ├── models.py     # Training loops
│   └── utils.py      # CV, metrics, helpers
└── submissions/      # Never overwrite, always version`,
        warnings: [],
        tips: ["Save every submission file with timestamp + CV score in the filename. This prevents accidentally submitting an old file."],
      },
    ],
    feRefs: [],
    edaRefs: [],
    problemRefs: ["Model Not Reproducible Across Runs", "Overfitting to Public Leaderboard (LB Probing)"],
  },
  {
    id: 2,
    phase: "02",
    title: "Data Loading & First Glance",
    subtitle: "Shape, dtypes, sample rows, basic stats",
    time: "15 min",
    color: "#38bdf8",
    icon: "📂",
    mode: "both",
    summary: "Before any EDA, establish the raw facts: shape, types, cardinalities, and a sanity check that the data loaded correctly. Many issues are visible in df.head() if you look carefully.",
    steps: [
      {
        title: "Load and inspect structure",
        detail: "Check shape, dtypes, memory usage, and the first few rows. Confirm the join/merge result has the expected number of rows if multiple files were combined. A row count after a merge that's different from expected is a silent bug that corrupts everything downstream.",
        code: `import pandas as pd

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"\\nDtypes:\\n{train.dtypes.value_counts()}")
print(f"\\nMemory: {train.memory_usage(deep=True).sum()/1e6:.1f} MB")

# Downcast immediately — saves memory and speeds up everything
train[train.select_dtypes('float64').columns] = train.select_dtypes('float64').astype('float32')
train[train.select_dtypes('int64').columns]   = train.select_dtypes('int64').astype('int32')

train.head(5)  # Always read these rows carefully`,
        warnings: ["If train and test have different column sets (besides the target), that asymmetry needs investigation. Missing columns in test may be post-event features = potential leakage."],
        tips: ["Cast to float32/int32 immediately. A 10M row float64 dataset becomes 2× smaller as float32 — often the difference between OOM and working fine."],
      },
      {
        title: "Check cardinality, unique values, and constants",
        detail: "Identify column types correctly: numeric-looking string columns, ID columns accidentally included, constant columns, near-constant columns, and high-cardinality categoricals that need special encoding.",
        code: `# Quick profiling table — paste and run:
def quick_profile(df):
    return pd.DataFrame({
        'dtype':    df.dtypes,
        'n_unique': df.nunique(),
        'null_pct': (df.isnull().mean() * 100).round(1),
        'sample':   df.iloc[0],
    }).sort_values('n_unique', ascending=False)

quick_profile(train)

# Flag suspicious columns:
# n_unique == 1          → constant, drop it
# n_unique == len(df)    → likely an ID column, drop or encode carefully
# dtype==object but looks numeric → convert with pd.to_numeric(errors='coerce')`,
        warnings: ["ID columns in training data cause severe overfitting — the model memorises IDs. Drop them before any model training."],
        tips: ["Save the output of quick_profile() to a variable and refer to it throughout your EDA. It's your compass."],
      },
    ],
    feRefs: ["Rare Label Encoding", "Variance Threshold", "Date/Time Feature Extraction"],
    edaRefs: ["ydata-profiling (Pandas Profiling)", "AutoViz"],
    problemRefs: ["Incorrect Data Types / Schema Drift", "Duplicated Rows / Near-Duplicates", "Missing Value Mishandling"],
  },
  {
    id: 3,
    phase: "03",
    title: "Automated EDA",
    subtitle: "Run profiling tools — read every alert",
    time: "30 min",
    color: "#22d3ee",
    icon: "🔭",
    mode: "both",
    summary: "Automated EDA tools produce a triage list. The Alerts section of ydata-profiling and the train/test comparison in SweetViz do in 2 lines of code what would take hours manually. Read every flag they raise.",
    steps: [
      {
        title: "Run ydata-profiling and read the Alerts",
        detail: "The Alerts tab is a prioritised list of data quality issues: high cardinality, high correlation, high missingness, skewness, near-constant columns, and duplicate rows. Treat each alert as a task item to resolve before modelling.",
        code: `from ydata_profiling import ProfileReport

profile = ProfileReport(train, explorative=True, title="Train EDA")
profile.to_file("eda_train.html")
# Open in browser — spend 20 min reading EVERY alert

# For large datasets use minimal=True to save time:
# ProfileReport(train, minimal=True)`,
        warnings: ["Do not skip the Alerts tab. Every alert is a potential source of model failure. A '99% zeros' alert means that column is near-constant and will waste model capacity."],
        tips: ["The Correlations tab shows Pearson, Spearman, Kendall, and Cramér's V in one place. Check it for near-1.0 correlations — they are either redundancy or leakage."],
      },
      {
        title: "Run SweetViz train vs test comparison",
        detail: "sv.compare(train, test) generates side-by-side distributions for every feature. This is your first and fastest way to detect covariate shift — features that look different between train and test will cause your model to underperform on the leaderboard.",
        code: `import sweetviz as sv

report = sv.compare([train, "Train"], [test, "Test"], target_feat="target")
report.show_html("sweetviz_compare.html")
# Open in browser — scan every feature for distribution differences
# Red flags: very different means, ranges, or shapes between train and test`,
        warnings: ["Any feature where train and test look visually different is a liability — your model trained on that distribution will extrapolate on test. Flag it for the adversarial validation step."],
        tips: ["In competitions, this single plot has identified leaky features, ID-like columns, and distribution-shifted features that explained public/private LB gaps."],
      },
    ],
    feRefs: ["Rare Label Encoding", "Correlation-Based Elimination", "Missing Value Indicator"],
    edaRefs: ["ydata-profiling (Pandas Profiling)", "SweetViz", "D-Tale"],
    problemRefs: ["Train/Validation Distribution Mismatch", "Covariate Shift", "Duplicated Rows / Near-Duplicates", "Missing Value Mishandling"],
  },
  {
    id: 4,
    phase: "04",
    title: "Target Analysis",
    subtitle: "Understand what you're predicting",
    time: "20 min",
    color: "#4ade80",
    icon: "🎯",
    mode: "both",
    summary: "The target variable dictates your entire strategy. Its distribution determines loss function, metric, resampling needs, and whether a log-transform is necessary. This is the most important single plot you will make.",
    steps: [
      {
        title: "Plot target distribution",
        detail: "For classification: check the class balance ratio. For regression: check for right skew, bimodality, and impossible values. The target distribution tells you whether to use class_weight, SMOTE, log-transform, or a special loss function.",
        code: `import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Classification
train['target'].value_counts(normalize=True).plot(kind='bar', ax=axes[0], color='#4ade80')
axes[0].set_title("Class Balance")

# Regression — check both original and log-transformed
axes[1].hist(train['target'], bins=50, color='#38bdf8', alpha=0.7, label='raw')
axes[1].hist(np.log1p(train['target']), bins=50, color='#f472b6', alpha=0.7, label='log1p')
axes[1].legend(); axes[1].set_title("Target Distribution")

# Print key stats
print(f"Imbalance ratio: {train['target'].value_counts(normalize=True).to_dict()}")
print(f"Skewness: {train['target'].skew():.3f} (>1 = significant right skew)")`,
        warnings: ["If imbalance ratio > 10:1, accuracy is a misleading metric and default threshold (0.5) will produce poor recall on the minority class. Address with class_weight='balanced' as first fix."],
        tips: ["If target skewness > 0.75, log1p transform typically improves RMSE by 5-15% for linear/neural models without any other change. This is the single highest-leverage regression trick."],
      },
      {
        title: "Check target mean by every categorical feature",
        detail: "For each categorical column, plot the mean target value per category. Features where the mean varies significantly across categories are highly predictive — these are your highest-priority features for target encoding.",
        code: `# Target mean per category — run for every categorical column
cat_cols = train.select_dtypes('object').columns

fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 4*len(cat_cols)))

for ax, col in zip(axes if len(cat_cols)>1 else [axes], cat_cols):
    means = (train.groupby(col)['target'].mean()
                   .sort_values().tail(20))  # top 20 categories
    means.plot(kind='barh', ax=ax, color='#c084fc')
    ax.axvline(train['target'].mean(), color='red', linestyle='--', label='global mean')
    ax.set_title(f"{col}: target mean by category")
    ax.legend()

plt.tight_layout()`,
        warnings: ["Features where a single category has very high target mean should be investigated for leakage — especially if that category is rare."],
        tips: ["The variance of category-level target means tells you how much information a categorical feature carries. High variance = strong feature = prioritise for target encoding."],
      },
      {
        title: "Scan for leaky features (near-perfect target correlation)",
        detail: "Plot every numeric feature against the target as a scatter. Any feature with |Pearson r| > 0.9 with the target is either a phenomenally good feature or a leaky one — both require immediate investigation.",
        code: `# Correlation with target — leakage audit
target_corr = (train.corr(numeric_only=True)['target']
                    .drop('target').abs().sort_values(ascending=False))

print("Top 15 features by |target correlation|:")
print(target_corr.head(15))

# Any feature with correlation > 0.9 → INVESTIGATE IMMEDIATELY
leakage_suspects = target_corr[target_corr > 0.9].index.tolist()
if leakage_suspects:
    print(f"\\n⚠️  LEAKAGE SUSPECTS: {leakage_suspects}")
    print("→ Check: could this feature be derived from the target?")
    print("→ Check: is this feature computed using post-event information?")`,
        warnings: ["A feature with 0.99 correlation with the target that 'makes sense' still needs investigation. Ask: 'At prediction time, would this value actually be available?' If not, it's leakage."],
        tips: ["In competitions, check feature names carefully. Columns like 'loan_status', 'paid_amount', or 'resolution_date' are classic leaky features — they only exist after the target event."],
      },
    ],
    feRefs: ["Target / Mean Encoding", "Log Transform", "CatBoost Encoding", "Weight of Evidence (WoE)"],
    edaRefs: ["Target Distribution Plot", "Target Mean by Categorical Feature", "Target Leakage Scatter"],
    problemRefs: ["Target Leakage", "Class Imbalance", "Wrong Evaluation Metric", "Threshold Set at Default 0.5"],
  },
  {
    id: 5,
    phase: "05",
    title: "Deep Univariate EDA",
    subtitle: "Every feature's distribution, outliers, and shape",
    time: "45 min",
    color: "#60a5fa",
    icon: "🔬",
    mode: "both",
    summary: "You need to know what every individual feature looks like before deciding how to treat it. Skewness determines transforms, outliers determine scaling strategy, cardinality determines encoding choice. This step generates your feature preprocessing plan.",
    steps: [
      {
        title: "Histogram + KDE sweep for all numeric features",
        detail: "Batch-plot histograms for all numeric columns. Mark which ones are right-skewed (need log/power transform), which have outliers (need RobustScaler or Winsorise), and which appear normal (StandardScaler is fine).",
        code: `# Batch histogram sweep
num_cols = train.select_dtypes(['int32','int64','float32','float64']).columns.drop('target', errors='ignore')

n_cols = 4
n_rows = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    skew = train[col].skew()
    color = '#f87171' if abs(skew) > 1 else '#60a5fa'
    train[col].hist(bins=40, ax=ax, color=color, alpha=0.8)
    ax.set_title(f"{col}\\nskew={skew:.2f}", fontsize=9)

for ax in axes[len(num_cols):]: ax.set_visible(False)
plt.tight_layout()
# Red bars = skewed (|skew| > 1) → needs transform for linear/distance models`,
        warnings: ["Bimodal distribution (two peaks) in a numeric feature often signals a hidden categorical variable. Investigate before treating it as a continuous feature."],
        tips: ["Create a preprocessing_plan dict as you go: {'col_name': {'transform': 'log1p', 'scale': 'robust', 'outlier': 'winsorise'}}. This becomes your ColumnTransformer specification."],
      },
      {
        title: "Outlier detection sweep",
        detail: "Compute Z-scores and IQR violations across all numeric features. Plot the outlier fraction per column sorted by severity. Investigate the top offenders — are they recording errors or legitimate extremes?",
        code: `from scipy.stats import zscore

# Z-score outlier fraction per column
z_outlier_pct = (np.abs(zscore(train[num_cols].fillna(train[num_cols].median()))) > 3).mean().sort_values(ascending=False)

print("Outlier fraction by column (|Z| > 3):")
print(z_outlier_pct[z_outlier_pct > 0.01].round(3))

# IQR method for each column
for col in z_outlier_pct.head(5).index:
    Q1, Q3 = train[col].quantile([0.01, 0.99])
    iqr_mask = (train[col] < Q1) | (train[col] > Q3)
    print(f"\\n{col}: {iqr_mask.sum()} potential outliers")
    print(train.loc[iqr_mask, col].describe())  # Inspect actual values`,
        warnings: [],
        tips: ["For each high-outlier column, print the actual outlier rows. Age=999 or price=-1 are recording errors to Winsorise. A $10M transaction is legitimate and should be kept — just use RobustScaler."],
      },
      {
        title: "Categorical cardinality audit",
        detail: "For each categorical feature, print unique value counts and flag high-cardinality columns that need special encoding, and near-constant columns that may be droppable.",
        code: `cat_cols = train.select_dtypes('object').columns

for col in cat_cols:
    n_unique = train[col].nunique()
    top_freq = train[col].value_counts(normalize=True).iloc[0]
    print(f"{col:30s} | n_unique={n_unique:5d} | top_freq={top_freq:.1%}")

# Decision matrix:
# n_unique <= 10          → OneHotEncoder
# 10 < n_unique <= 50     → OHE or OrdinalEncoder for trees
# 50 < n_unique <= 500    → TargetEncoder or FrequencyEncoder
# n_unique > 500          → TargetEncoder + RareLabelEncoder, or HashingEncoder
# top_freq > 0.99         → near-constant → strong candidate for dropping`,
        warnings: ["A categorical column with 500+ unique values One-Hot Encoded will create 500+ sparse columns — likely more than your total row count. This almost always causes overfitting."],
        tips: ["Frequency encoding is always safe as a first-pass for high-cardinality features. It requires no cross-validation, carries meaningful signal (rare categories get low values), and never leaks."],
      },
    ],
    feRefs: ["Log Transform", "Yeo-Johnson / Box-Cox Transform", "Robust Scaler", "Rare Label Encoding", "One-Hot Encoding", "Target / Mean Encoding", "Frequency / Count Encoding"],
    edaRefs: ["Histogram + KDE", "Box Plot", "QQ Plot", "Bar Chart (Value Counts)", "Z-Score / IQR Outlier Flag Plot"],
    problemRefs: ["Outlier Contamination", "Feature Scale Sensitivity for Distance Models", "Incorrect Data Types / Schema Drift"],
  },
  {
    id: 6,
    phase: "06",
    title: "Bivariate & Correlation EDA",
    subtitle: "Feature–feature and feature–target relationships",
    time: "45 min",
    color: "#34d399",
    icon: "🔗",
    mode: "both",
    summary: "Pairwise relationships reveal redundancy (features you can drop), interactions (features you should create), and non-linear signals (features that need transformations before linear models can use them).",
    steps: [
      {
        title: "Pearson + Spearman correlation heatmaps",
        detail: "Plot both heatmaps. Pearson reveals linear correlations; Spearman reveals monotone (non-linear) correlations. Differences between them flag features that need log-transforms to linearise their relationship with the target.",
        code: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

mask = np.triu(np.ones_like(train.corr(numeric_only=True), dtype=bool))
sns.heatmap(train.corr(numeric_only=True, method='pearson'),
            mask=mask, annot=False, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=ax1, square=True)
ax1.set_title("Pearson (linear)")

sns.heatmap(train.corr(numeric_only=True, method='spearman'),
            mask=mask, annot=False, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, ax=ax2, square=True)
ax2.set_title("Spearman (monotone)")

# HIGH Spearman + LOW Pearson → non-linear monotone relationship
# Consider log-transform to linearise it for linear models`,
        warnings: ["Features with |Pearson r| > 0.95 between them are near-duplicates. Remove one before training any linear model, or use Ridge to handle it automatically."],
        tips: ["Build a 'redundancy list' from the heatmap: any pair with |r| > 0.9. During feature selection, you'll remove one from each redundant pair."],
      },
      {
        title: "Scatter plots for top features vs target",
        detail: "For your top 10 features by mutual information, plot each against the target coloured by a second feature. Non-linear patterns suggest polynomial features; clustered patterns suggest interaction features.",
        code: `from sklearn.feature_selection import mutual_info_classif

# Get top features by MI
mi = mutual_info_classif(train[num_cols].fillna(0), train['target'], random_state=42)
top_mi_features = pd.Series(mi, index=num_cols).sort_values(ascending=False).head(10).index

# Scatter each against target
n = len(top_mi_features)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for ax, col in zip(axes, top_mi_features):
    ax.scatter(train[col], train['target'], alpha=0.2, s=5, c='#60a5fa')
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel('target', fontsize=8)
    ax.set_title(f"MI={mi[num_cols.tolist().index(col)]:.3f}", fontsize=9)

plt.tight_layout()`,
        warnings: [],
        tips: ["A curved (parabolic) scatter suggests a squared term. A step-function pattern suggests binning. An L-shaped pattern suggests a log transform. Each pattern is a direct engineering instruction."],
      },
      {
        title: "Mutual information bar chart",
        detail: "Compare MI rankings vs Pearson correlation rankings. Features high in MI but low in Pearson have non-linear target relationships — these are invisible to linear models but exploitable by GBMs and MLPs.",
        code: `from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# For classification:
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
# For regression:
# mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
mi_series.head(20).plot(kind='barh', ax=ax1, color='#22d3ee')
ax1.set_title("Mutual Information (top 20)")
ax1.invert_yaxis()

# Compare to Pearson
pearson_series = train.corr(numeric_only=True)['target'].drop('target').abs()
(mi_series - pearson_series.reindex(mi_series.index).fillna(0)).sort_values(ascending=False).head(20).plot(kind='barh', ax=ax2, color='#c084fc')
ax2.set_title("MI minus |Pearson| — non-linear extras")
ax2.invert_yaxis()
plt.tight_layout()`,
        warnings: [],
        tips: ["Features at the top of the 'MI minus Pearson' chart carry non-linear signal that linear models completely miss. These are the features that tree models and neural nets exploit that linear models cannot."],
      },
    ],
    feRefs: ["Pearson Correlation", "Spearman / Kendall Rank Correlation", "Mutual Information (MI)", "Polynomial Features", "Interaction / Product Features", "Ratio Features"],
    edaRefs: ["Correlation Heatmap (Pearson)", "Spearman Correlation Heatmap", "Scatter Plot", "Pair Plot", "Mutual Information Bar Chart", "Correlation Funnel"],
    problemRefs: ["Multicollinear Features Breaking Linear Models", "Target Leakage", "Misleading Feature Importance (MDI Bias)"],
  },
  {
    id: 7,
    phase: "07",
    title: "Missing Data & Outlier Strategy",
    subtitle: "Classify, visualise, plan imputation",
    time: "30 min",
    color: "#f87171",
    icon: "🕳️",
    mode: "both",
    summary: "Missingness is information. Before imputing anything, understand whether values are Missing At Random (MAR), Missing Not At Random (MNAR), or Completely At Random (MCAR). The answer determines whether the missingness itself should become a feature.",
    steps: [
      {
        title: "Visualise missingness patterns with missingno",
        detail: "The missingno matrix reveals co-occurrence: do the same rows always have multiple columns missing together? That's a structural MNAR pattern. The heatmap shows which column pairs always go missing together. The dendrogram groups columns by their missingness similarity.",
        code: `import missingno as msno

# Three essential plots in sequence:
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

msno.matrix(train, ax=axes[0])     # Visual pattern — vertical bands = co-occurrence
msno.heatmap(train, ax=axes[1])    # Correlation of missingness between column pairs  
msno.dendrogram(train, ax=axes[2]) # Hierarchical clustering by missingness similarity

print("\\nMissingness rate per column (>5% flagged):")
miss_rate = train.isnull().mean().sort_values(ascending=False)
print(miss_rate[miss_rate > 0.05].round(3))`,
        warnings: ["Imputing inside cross-validation folds requires a Pipeline. If you call imputer.fit_transform(full_data) before splitting, test-fold statistics leak into training — especially dangerous with KNNImputer and IterativeImputer."],
        tips: ["For HistGradientBoosting, LightGBM, or XGBoost as your downstream model, you may not need to impute at all — they handle NaN natively. Focus imputation effort on linear models, SVMs, and MLP."],
      },
      {
        title: "Classify missingness type and create indicators",
        detail: "Check if missingness rate differs by target class (MNAR). If it does, the fact that a value is missing is itself a signal — create a binary indicator column before imputing.",
        code: `# Check if missingness is MNAR (differs by target class)
miss_by_target = (train.isnull()
                      .groupby(train['target'])
                      .mean()
                      .T
                      .assign(delta=lambda x: x.iloc[:,1] - x.iloc[:,0])
                      .sort_values('delta', key=abs, ascending=False))

print("Features where missingness rate differs by class:")
print(miss_by_target[miss_by_target['delta'].abs() > 0.05].round(3))

# For MNAR columns, always add an indicator:
mnar_cols = miss_by_target[miss_by_target['delta'].abs() > 0.05].index.tolist()
for col in mnar_cols:
    train[f'{col}_was_missing'] = train[col].isnull().astype(int)
    test[f'{col}_was_missing']  = test[col].isnull().astype(int)
    print(f"Created indicator: {col}_was_missing")`,
        warnings: [],
        tips: ["MNAR indicators are among the most powerful engineered features in medical, financial, and survey data. The fact that a doctor didn't measure something, or a loan applicant didn't disclose something, is highly predictive."],
      },
      {
        title: "Choose imputation strategy per column",
        detail: "The right imputation strategy depends on distribution and missingness type. Mean for symmetric, median for skewed, most_frequent for categorical, IterativeImputer for complex multivariate patterns. All must be wrapped in a Pipeline.",
        code: `from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imputation strategy decision:
# MAR/MCAR + symmetric distribution    → mean
# MAR/MCAR + skewed distribution       → median
# MAR/MCAR + categorical               → most_frequent or 'Unknown' constant
# MNAR                                 → add indicator first, then median impute
# Complex multivariate MAR             → IterativeImputer(estimator=BayesianRidge())

# Example pipeline (put imputers INSIDE the pipeline):
preprocessor = ColumnTransformer([
    ('num_impute',  SimpleImputer(strategy='median'), num_cols),
    ('cat_impute',  SimpleImputer(strategy='most_frequent'), cat_cols),
])
# Never call preprocessor.fit_transform(train) directly — always through cross_validate()`,
        warnings: ["KNNImputer with large k on large datasets is extremely slow. Use it only for small datasets (< 10,000 rows). For large datasets, IterativeImputer(max_iter=3) is a good compromise."],
        tips: ["For competition datasets where you don't have time for IterativeImputer, median imputation + a missing indicator for MNAR columns is a robust default that works well with GBMs."],
      },
    ],
    feRefs: ["Missing Value Indicator", "Missing Value Mishandling"],
    edaRefs: ["Missingno Matrix", "Missingno Heatmap + Dendrogram", "Missing Value Heatmap by Target Class"],
    problemRefs: ["Missing Value Mishandling", "Preprocessing Leakage (Fit on Full Data)"],
  },
  {
    id: 8,
    phase: "08",
    title: "Adversarial Validation",
    subtitle: "Competition-critical: detect train/test shift",
    time: "20 min",
    color: "#f472b6",
    icon: "⚔️",
    mode: "competition",
    summary: "Train a binary classifier to distinguish train rows (label=0) from test rows (label=1). If it succeeds (AUC > 0.6), your distributions differ and your CV score will be optimistic. The features that drive the adversarial model are the ones causing the shift.",
    steps: [
      {
        title: "Run adversarial validation",
        detail: "Combine train and test, label them 0 and 1, train a LightGBM classifier, and check the ROC-AUC. AUC ≈ 0.5 = no shift. AUC > 0.6 = detectable shift. AUC > 0.8 = severe shift — your CV is unreliable.",
        code: `import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Combine train and test
train_adv = train.drop('target', axis=1).copy()
test_adv  = test.copy()

train_adv['is_test'] = 0
test_adv['is_test']  = 1

combined = pd.concat([train_adv, test_adv], ignore_index=True)
X_adv = combined.drop('is_test', axis=1)
y_adv = combined['is_test']

# Fill NaN for the adversarial model
X_adv = X_adv.fillna(-9999)

# Encode categoricals
from sklearn.preprocessing import OrdinalEncoder
for col in X_adv.select_dtypes('object').columns:
    X_adv[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X_adv[[col]])

adv_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
adv_auc = cross_val_score(adv_model, X_adv, y_adv, cv=StratifiedKFold(5),
                           scoring='roc_auc').mean()

print(f"Adversarial AUC: {adv_auc:.4f}")
print("Interpretation:")
print(f"  ~0.50  → No detectable shift. CV is trustworthy.")
print(f"  0.6–0.7 → Moderate shift. Remove or transform shifted features.")
print(f"  >0.8   → Severe shift. CV will be very optimistic. Prioritise fixing.")`,
        warnings: ["If adversarial AUC > 0.7 and you don't act on it, your public LB score will be higher than private LB. This is one of the primary causes of Kaggle 'shake-up'."],
        tips: ["Run adversarial validation after adding every batch of new features. A new feature that increases adversarial AUC is introducing distribution shift."],
      },
      {
        title: "Identify and remove shifted features",
        detail: "Train the adversarial model, extract feature importances, and investigate the top features. They are either ID-like columns, time-correlated features, or leaky features that should be removed.",
        code: `# Identify which features drive the adversarial classifier
adv_model.fit(X_adv, y_adv)
adv_importances = pd.Series(adv_model.feature_importances_, index=X_adv.columns)
                    .sort_values(ascending=False)

print("Top 15 adversarial features (features that differ train vs test):")
print(adv_importances.head(15))

# For each top feature, overlay train vs test distributions
for col in adv_importances.head(5).index:
    plt.figure(figsize=(8,3))
    sns.kdeplot(train[col].dropna(), label='train', color='#60a5fa')
    sns.kdeplot(test[col].dropna(),  label='test',  color='#f87171')
    plt.title(f"{col} — train vs test distribution")
    plt.legend(); plt.show()

# Decision: drop or transform features with high adversarial importance
# Re-run adversarial validation after each removal until AUC < 0.55`,
        warnings: [],
        tips: ["Frequency encoding high-cardinality categoricals is more stable across train/test splits than OHE or ordinal encoding — the frequency rank is similar in both sets."],
      },
    ],
    feRefs: ["Frequency / Count Encoding", "Correlation-Based Elimination", "Log Transform"],
    edaRefs: ["Adversarial Validation Plot", "Train vs Test Distribution Overlay", "ECDF Plot"],
    problemRefs: ["Covariate Shift (Train/Test Feature Drift)", "Train/Validation Distribution Mismatch", "Public/Private LB Split Gap"],
  },
  {
    id: 9,
    phase: "09",
    title: "Define CV Strategy",
    subtitle: "The most important modelling decision",
    time: "20 min",
    color: "#818cf8",
    icon: "📐",
    mode: "both",
    summary: "Your cross-validation strategy determines whether every subsequent decision is trustworthy. The wrong CV makes model A look better than model B when the reverse is true. Set it once, correctly, based on the data structure — not convenience.",
    steps: [
      {
        title: "Choose the correct CV splitter for your data structure",
        detail: "There are four data structure archetypes, each with the correct splitter. Using the wrong one produces an optimistic CV that doesn't predict test performance.",
        code: `from sklearn.model_selection import (
    StratifiedKFold,       # Classification, balanced folds
    KFold,                 # Regression (no stratification needed)
    TimeSeriesSplit,       # Time-ordered data
    GroupKFold,            # Entity-grouped data (patients, users, stores)
    StratifiedGroupKFold,  # Grouped + classification (most complex, most robust)
)

# Decision tree:
# ─ Is data time-ordered?           → TimeSeriesSplit(n_splits=5)
# ─ Do rows share entities?         → GroupKFold(groups=entity_id_col)
# ─ Both time + entities?           → custom walk-forward per entity
# ─ Pure i.i.d. classification?     → StratifiedKFold(n_splits=5, shuffle=True)
# ─ Pure i.i.d. regression?         → KFold(n_splits=5, shuffle=True)
# ─ Very small dataset?             → RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

# DEFINE YOUR SPLITTER HERE — NEVER CHANGE IT DURING THE COMPETITION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
GROUPS = None  # or df['entity_id'] for GroupKFold`,
        warnings: ["Using StratifiedKFold on time-series data leaks future information into training folds. Using KFold on grouped entity data leaks entity-level information. Both produce inflated CV scores that collapse on the private LB."],
        tips: ["Check whether the competition leaderboard split mirrors a temporal split. If public LB = most recent months, your local CV must also split temporally."],
      },
      {
        title: "Build an OOF (out-of-fold) prediction framework",
        detail: "Always generate out-of-fold predictions for the training set, not just a fold-level score. OOF predictions let you compute the exact evaluation metric on the full training set, analyse errors, and perform meta-learning (stacking).",
        code: `import numpy as np
from sklearn.metrics import roc_auc_score

def run_oof(model, X, y, cv, groups=None):
    """Generate OOF predictions and compute CV score."""
    oof_preds = np.zeros(len(y))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_score = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.5f}")

    overall = roc_auc_score(y, oof_preds)
    print(f"\\nOOF AUC: {overall:.5f} ± {np.std(fold_scores):.5f}")
    return oof_preds, overall`,
        warnings: ["Never compute the final CV score on fold means — always on the concatenated OOF predictions. Fold-mean AUC ≠ overall AUC, especially for imbalanced data."],
        tips: ["Save oof_preds to disk. These become the input for any stacking meta-learner and let you analyse exactly which samples your model gets wrong."],
      },
    ],
    feRefs: [],
    edaRefs: [],
    problemRefs: ["Wrong Cross-Validation Strategy", "Group Leakage in Cross-Validation", "Temporal / Future Leakage", "Nested CV Neglect", "Overfitting to Public Leaderboard (LB Probing)"],
  },
  {
    id: 10,
    phase: "10",
    title: "Feature Engineering",
    subtitle: "Transform, encode, create — in the right order",
    time: "Variable",
    color: "#a3e635",
    icon: "⚗️",
    mode: "both",
    summary: "Feature engineering is where competitions are won. Apply transformations in the correct sequence: clean → encode → scale → create new features → select. Every step must live inside a Pipeline or be applied identically to train and test.",
    steps: [
      {
        title: "Apply the correct encoding strategy per column",
        detail: "Use the cardinality audit from Phase 5 to assign each categorical to the right encoder. Build a ColumnTransformer that applies the correct encoding per column type.",
        code: `from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

# Encoding decision by cardinality (from Phase 05 audit):
low_card_cat  = [col for col in cat_cols if train[col].nunique() <= 10]   # OHE
mid_card_cat  = [col for col in cat_cols if 10 < train[col].nunique() <= 100]  # Ordinal for trees
high_card_cat = [col for col in cat_cols if train[col].nunique() > 100]   # Target encoding

skewed_num   = [col for col in num_cols if abs(train[col].skew()) > 1]   # RobustScaler + log
normal_num   = [col for col in num_cols if abs(train[col].skew()) <= 1]  # StandardScaler

preprocessor = ColumnTransformer([
    ('ohe',    Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), low_card_cat),
    ('ord',    Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), mid_card_cat),
    # Note: TargetEncoder must be used inside CV loop to prevent leakage
    ('scaler', Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())]), normal_num),
    ('rob',    Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('sc',  RobustScaler())]), skewed_num),
])`,
        warnings: ["Target encoding MUST be computed inside CV folds. Put TargetEncoder inside the pipeline and pass it to cross_validate() — never fit it on the full training set outside a fold."],
        tips: ["For tree-based models (GBM, Random Forest), you often don't need StandardScaler at all. Build two preprocessors: one for linear/distance models and one for tree models."],
      },
      {
        title: "Engineer new features from domain knowledge and EDA insights",
        detail: "Apply all the engineering insights discovered in EDA: interactions spotted in scatter plots, SHAP dependence relationships, group aggregations from categorical-target analysis, and date/time decompositions.",
        code: `def engineer_features(df, train_stats=None, is_train=True):
    """Apply feature engineering. train_stats carries aggregates computed on train only."""
    df = df.copy()

    # 1. Ratio features (from domain knowledge)
    df['debt_to_income'] = df['debt'] / (df['income'] + 1e-9)
    df['price_per_sqft'] = df['price'] / (df['sqft'] + 1e-9)

    # 2. Datetime decomposition
    if 'date_col' in df.columns:
        df['date_col'] = pd.to_datetime(df['date_col'])
        df['year']       = df['date_col'].dt.year
        df['month']      = df['date_col'].dt.month
        df['dayofweek']  = df['date_col'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)  # cyclical
        df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

    # 3. Group aggregation features (train_stats computed on train fold only)
    if train_stats is not None:
        df = df.merge(train_stats['cat_mean_target'],  on='cat_col', how='left')
        df = df.merge(train_stats['cat_count'], on='cat_col', how='left')
    elif is_train:
        # Compute stats for merging (call inside CV folds for clean leakage prevention)
        cat_mean = df.groupby('cat_col')['target'].mean().rename('cat_mean_target').reset_index()
        cat_count = df.groupby('cat_col').size().rename('cat_count').reset_index()
        return df, {'cat_mean_target': cat_mean, 'cat_count': cat_count}

    return df`,
        warnings: ["Group aggregation features computed on the full training set BEFORE the CV split are a form of target leakage. Always compute them within each training fold."],
        tips: ["Engineering features that combine domain knowledge (like debt-to-income or price-per-sqft) is one of the few things that works for ALL model types simultaneously."],
      },
      {
        title: "Apply feature selection to remove noise",
        detail: "After engineering, you likely have more features than you started with. Filter down using variance threshold, MI, and SHAP-based selection. Remove correlated redundancies before training your final models.",
        code: `from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif

# Step 1: Remove constants and near-constants (free, do this first)
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X_train)
print(f"After variance filter: {X_train.shape[1]} → {X_var.shape[1]} features")

# Step 2: Remove highly correlated pairs (redundancy removal)
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
print(f"Dropping {len(to_drop)} correlated features")

# Step 3: After first model, use SHAP to remove the bottom quartile
# import shap
# shap_values = shap.TreeExplainer(lgbm_model).shap_values(X_val)
# mean_shap   = np.abs(shap_values).mean(axis=0)
# keep_mask   = mean_shap > np.percentile(mean_shap, 25)
# X_selected  = X_train[:, keep_mask]`,
        warnings: ["Do not use Boruta or RFE on your first pass — they take hours. Use variance + MI + correlation removal as your fast first-pass selection, then use SHAP after your first GBM run."],
        tips: ["SHAP-based selection after your first GBM is the highest-quality feature selection available. Run it once you have a working baseline, not before."],
      },
    ],
    feRefs: ["One-Hot Encoding", "Target / Mean Encoding", "Log Transform", "Interaction / Product Features", "Date/Time Feature Extraction", "Cyclical Encoding", "Group Aggregation Features", "SHAP Values", "Correlation-Based Elimination", "Variance Threshold"],
    edaRefs: ["SHAP Summary Plot (Beeswarm)", "SHAP Dependence Plot", "Mutual Information Bar Chart", "Residual Plot"],
    problemRefs: ["Preprocessing Leakage (Fit on Full Data)", "Target Leakage", "Overfitting — High Variance", "Curse of Dimensionality"],
  },
  {
    id: 11,
    phase: "11",
    title: "Baseline Model",
    subtitle: "Simple first, correct CV, trust the score",
    time: "30 min",
    color: "#c084fc",
    icon: "🚀",
    mode: "both",
    summary: "Build the simplest possible model first. A baseline gives you a floor to beat, confirms your pipeline works end-to-end, and — critically — establishes whether your CV correlates with external performance before you invest in complex models.",
    steps: [
      {
        title: "Train a fast, simple baseline",
        detail: "For tabular data, HistGradientBoostingClassifier or LightGBM with default parameters is the best baseline — it handles NaN natively, trains fast, and is hard to beat without deliberate effort. For NLP, LogisticRegression on TF-IDF. For images, Logistic Regression on frozen CNN embeddings.",
        code: `import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# LightGBM baseline — fast, handles NaN, good defaults
baseline_model = lgb.LGBMClassifier(
    n_estimators=500,       # Use early stopping to find the right number
    learning_rate=0.05,     # Conservative — early stopping will compensate
    num_leaves=31,          # Default — don't tune yet
    min_child_samples=20,   # Light regularisation
    random_state=SEED,
    n_jobs=-1,
)

# Run the OOF framework from Phase 09:
oof_preds, baseline_cv = run_oof(
    baseline_model, X_train, y_train, cv=cv
)

print(f"\\n{'='*50}")
print(f"BASELINE CV AUC: {baseline_cv:.5f}")
print(f"{'='*50}")
print(f"This is your floor to beat. Every experiment is measured against this.")

# Save OOF predictions
np.save('oof_baseline.npy', oof_preds)`,
        warnings: ["If your baseline CV score is suspiciously high (> 0.99 for a hard problem), you likely have target leakage. Do not proceed — return to Phase 4 and audit features."],
        tips: ["The baseline model is also a feature importance scanner. After training, immediately plot SHAP summary to understand which features the model is actually using."],
      },
      {
        title: "Verify CV correlation with external performance",
        detail: "For competitions: submit your baseline predictions and compare CV score to public LB. They should be close (within 0.01). If your CV is much higher than LB, you have a CV leakage issue. If LB is much higher than CV, you may have a surprisingly easy test set.",
        code: `# Generate test predictions for submission
def predict_test(model, X_train, y_train, X_test, cv, groups=None):
    """Average predictions from k fold models — do NOT retrain on all data."""
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        model.fit(X_tr, y_tr)
        test_preds += model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

    return test_preds

test_preds = predict_test(baseline_model, X_train, y_train, X_test, cv)

# Create submission
submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv(f'sub_baseline_cv{baseline_cv:.5f}.csv', index=False)
# Expected: |CV - public_LB| < 0.01
# If gap > 0.02 → re-examine your CV strategy and feature construction`,
        warnings: ["Do not average a model retrained on all training data with fold models — this creates OOF train/test inconsistency. Average the k fold-specific models' test predictions."],
        tips: ["Name every submission file with its CV score. 'sub_lgbm_cv0.87234.csv' is vastly more useful than 'sub_v3_final.csv' when reviewing your submission history 3 weeks later."],
      },
    ],
    feRefs: [],
    edaRefs: ["SHAP Summary Plot (Beeswarm)"],
    problemRefs: ["Target Leakage", "Wrong Cross-Validation Strategy", "Overfitting to Public Leaderboard (LB Probing)"],
  },
  {
    id: 12,
    phase: "12",
    title: "Model Iteration & Tuning",
    subtitle: "Improve systematically, trust CV over intuition",
    time: "Variable",
    color: "#fb923c",
    icon: "🔄",
    mode: "both",
    summary: "Iterate in a principled cycle: analyse errors → engineer features → tune hyperparameters → validate with CV. Only commit an engineering decision when it shows a statistically meaningful CV improvement.",
    steps: [
      {
        title: "Error analysis — understand where the model fails",
        detail: "Plot SHAP beeswarm and dependence plots to understand what the model learned. Plot predicted vs actual for regression, and the confusion matrix for classification. Find the patterns in your worst predictions.",
        code: `import shap

# SHAP analysis on validation fold
explainer   = shap.TreeExplainer(trained_lgbm)
shap_values = explainer.shap_values(X_val)

# Summary: global feature importance with direction
shap.summary_plot(shap_values, X_val, max_display=20)

# Dependence: how a specific feature's effect changes
shap.dependence_plot('top_feature', shap_values, X_val, interaction_index='auto')

# Error analysis: worst predictions
errors = pd.DataFrame({
    'y_true': y_val.values,
    'y_pred': oof_preds[val_idx],
    'error':  abs(y_val.values - oof_preds[val_idx]),
}).sort_values('error', ascending=False)

print("Worst 20 predictions:")
print(errors.head(20).merge(X_val.reset_index(), left_index=True, right_index=True))`,
        warnings: ["SHAP values with the MDI importance from tree models will disagree for high-cardinality features. Always trust SHAP over MDI."],
        tips: ["The patterns in your worst predictions are feature engineering instructions. If high errors cluster in a specific category or feature range, that's the next engineering target."],
      },
      {
        title: "Hyperparameter tuning with Optuna",
        detail: "Use Bayesian optimisation (Optuna) rather than grid search. Bayesian search uses prior knowledge of which regions produced good results to guide the next trial — finding good hyperparameters in 10x fewer trials.",
        code: `import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
    }
    model = lgb.LGBMClassifier(**params)
    _, cv_score = run_oof(model, X_train, y_train, cv)
    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best CV: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")`,
        warnings: ["Tuning hyperparameters against the same validation set repeatedly overfits the validation set. After tuning, validate on a truly held-out set to get an unbiased performance estimate."],
        tips: ["Tune learning rate and num_leaves/max_depth first — they have the largest impact on LightGBM. Subsample and colsample_bytree are second. Regularisation terms (alpha/lambda) are third."],
      },
      {
        title: "Handle class imbalance properly",
        detail: "Apply imbalance handling in the correct order: first try class_weight (fastest, no data change). Then threshold tuning. Then SMOTE inside the CV Pipeline. Never oversample before splitting.",
        code: `from imblearn.pipeline import make_pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
import numpy as np

# Option 1 (always try first): class_weight='balanced'
model_balanced = lgb.LGBMClassifier(class_weight='balanced', random_state=SEED)

# Option 2: SMOTE inside imblearn pipeline (prevents leakage)
smote_pipeline = imb_pipeline(
    SMOTE(random_state=SEED),
    lgb.LGBMClassifier(random_state=SEED)
)
# Pass smote_pipeline to cross_validate() — SMOTE only sees training fold data

# Option 3: Threshold tuning from OOF predictions
precision, recall, thresholds = precision_recall_curve(y_train, oof_preds)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
optimal_threshold = thresholds[f1_scores.argmax()]
print(f"Optimal threshold for F1: {optimal_threshold:.3f}")
print(f"(Default was 0.5 — your optimal is {optimal_threshold:.3f})")`,
        warnings: ["SMOTE must be inside imblearn.Pipeline, not applied to X_train before the CV loop. If you see suspiciously high CV scores after adding SMOTE, check that it's not applied to the full training set."],
        tips: ["class_weight='balanced' almost always closes 50-80% of the gap between imbalanced and balanced performance. Try it before SMOTE — it's instantaneous."],
      },
    ],
    feRefs: ["SHAP Values", "Permutation Importance", "Boruta", "Spline Features", "Polynomial Features"],
    edaRefs: ["SHAP Summary Plot (Beeswarm)", "SHAP Dependence Plot", "Residual Plot", "Calibration Curve"],
    problemRefs: ["Overfitting — High Variance", "Hyperparameter Tuning Without Principled Strategy", "Class Imbalance", "SMOTE Applied Before Split", "Poor Probability Calibration"],
  },
  {
    id: 13,
    phase: "13",
    title: "Model Diversity & Ensembling",
    subtitle: "Combine models that make different mistakes",
    time: "Variable",
    color: "#facc15",
    icon: "🏗️",
    mode: "both",
    summary: "Ensembles outperform single models only when the constituent models are diverse — i.e., when they make different mistakes on different samples. Combining five copies of the same model is nearly worthless. Combining GBM + Linear + MLP is powerful.",
    steps: [
      {
        title: "Measure OOF prediction correlation before ensembling",
        detail: "Two models whose OOF predictions have Pearson r > 0.95 will gain almost nothing from being combined. Check pairwise correlations before spending time training additional models.",
        code: `# Collect OOF predictions from each model
oof_dict = {
    'lgbm':    np.load('oof_lgbm.npy'),
    'xgb':     np.load('oof_xgb.npy'),
    'catboost':np.load('oof_cat.npy'),
    'logreg':  np.load('oof_logreg.npy'),
    'mlp':     np.load('oof_mlp.npy'),
}

oof_df   = pd.DataFrame(oof_dict)
corr_mat = oof_df.corr()

print("OOF prediction correlation matrix:")
print(corr_mat.round(3))

# Models with r > 0.95 → barely worth combining
# Models with r < 0.85 → high diversity → strong ensemble candidate

sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', center=0.9)
plt.title("OOF Prediction Correlation — ensemble diversity check")`,
        warnings: ["If all your models are GBMs with different hyperparameters, they will all have correlation > 0.95. True diversity requires different algorithms, different feature sets, or different preprocessing approaches."],
        tips: ["A logistic regression on TF-IDF features combined with a GBM on tabular features often gives excellent diversity — they look at fundamentally different representations."],
      },
      {
        title: "Simple ensembling: weighted average of OOF",
        detail: "Optimise ensemble weights by finding the combination of OOF predictions that maximises the CV metric. Even simple equal-weight averaging of 2-3 diverse models often beats the best single model.",
        code: `from scipy.optimize import minimize

def ensemble_score(weights, oof_preds_list, y_true):
    weights = np.array(weights)
    weights /= weights.sum()  # Normalise to sum to 1
    blended = sum(w * p for w, p in zip(weights, oof_preds_list))
    return -roc_auc_score(y_true, blended)  # Negative because we minimise

oof_list = list(oof_dict.values())
n_models = len(oof_list)

# Optimise weights
result = minimize(
    ensemble_score,
    x0=[1/n_models] * n_models,
    args=(oof_list, y_train),
    method='Nelder-Mead',
    options={'maxiter': 1000}
)
optimal_weights = result.x / result.x.sum()

for name, weight in zip(oof_dict.keys(), optimal_weights):
    print(f"  {name}: {weight:.3f}")

blended_oof = sum(w * p for w, p in zip(optimal_weights, oof_list))
print(f"\\nEnsemble OOF AUC: {roc_auc_score(y_train, blended_oof):.5f}")`,
        warnings: ["Optimising ensemble weights on the full OOF set is susceptible to overfitting if you have many models and few samples. Consider a simple equal-weight average as a sanity check."],
        tips: ["The ensemble weights should be optimised on OOF predictions only, never on test predictions. The test is sacred — you only see it when you submit."],
      },
      {
        title: "Stacking with a meta-learner",
        detail: "For maximum performance, use stacking: OOF predictions from base models become features for a meta-model. The meta-model learns the optimal combination using the error patterns of each base model.",
        code: `# Stacking meta-model using OOF predictions as features
X_meta_train = oof_df.values                # OOF predictions from all base models
y_meta_train = y_train.values

# Simple meta-learner: Logistic Regression (interpretable, less overfitting risk)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

meta_model = LogisticRegression(C=1.0, random_state=SEED)
meta_cv = cross_val_score(meta_model, X_meta_train, y_meta_train,
                           cv=StratifiedKFold(5), scoring='roc_auc').mean()
print(f"Stacking meta-model CV AUC: {meta_cv:.5f}")

# For test predictions: average the k test predictions per base model (from Phase 11)
test_meta = pd.DataFrame({name: test_preds[name] for name in oof_dict.keys()})
meta_model.fit(X_meta_train, y_meta_train)
final_test_preds = meta_model.predict_proba(test_meta.values)[:, 1]`,
        warnings: ["Stacking overfits easily on small datasets. If you have < 5,000 training rows, use simple averaging instead. Stacking gains are most reliable when you have diverse base models and > 10,000 rows."],
        tips: ["The meta-learner should be a simple model (Logistic Regression, Ridge) — not another GBM. If your meta-model overfits, reduce C in LogisticRegression."],
      },
    ],
    feRefs: [],
    edaRefs: [],
    problemRefs: ["Ensemble Diversity Collapse", "OOF Prediction Blending Mistakes", "Overfitting the Validation Set (Repeated Tuning)"],
  },
  {
    id: 14,
    phase: "14",
    title: "Final Evaluation & Calibration",
    subtitle: "Calibrate probabilities, interpret the model",
    time: "30 min",
    color: "#f472b6",
    icon: "🎛️",
    mode: "both",
    summary: "Before delivering a model, verify that its probabilities are calibrated (they match actual frequencies), interpret what drives its decisions, and confirm performance on a truly held-out evaluation set.",
    steps: [
      {
        title: "Check and fix probability calibration",
        detail: "Plot the calibration curve (reliability diagram). If predicted probabilities systematically deviate from the diagonal, apply post-hoc calibration. This matters whenever probabilities are used downstream (risk scoring, expected value, log-loss competitions).",
        code: `from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Plot calibration before correction
CalibrationDisplay.from_predictions(y_val, raw_oof_preds, n_bins=10, ax=ax, label='Before calibration')

# Apply calibration (isotonic for large data, sigmoid/Platt for small)
calibrated = CalibratedClassifierCV(base_estimator=trained_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
cal_preds = calibrated.predict_proba(X_val)[:, 1]

CalibrationDisplay.from_predictions(y_val, cal_preds, n_bins=10, ax=ax, label='After isotonic')
ax.set_title("Calibration Curve (Reliability Diagram)")
plt.show()

from sklearn.metrics import brier_score_loss
print(f"Brier score before: {brier_score_loss(y_val, raw_oof_preds):.4f}")
print(f"Brier score after:  {brier_score_loss(y_val, cal_preds):.4f}")`,
        warnings: ["Random Forests tend to compress probabilities toward 0.5. SVMs output uncalibrated decision scores. Always check calibration for these models before any probability-sensitive application."],
        tips: ["In log-loss competitions, a well-calibrated model with lower ROC-AUC can outperform a poorly-calibrated model with higher ROC-AUC. Calibration is a direct competition lever."],
      },
      {
        title: "Tune decision threshold",
        detail: "The default 0.5 threshold is almost never optimal for imbalanced problems. Use the OOF predictions to find the threshold that maximises your target metric (F1, F-beta, or a custom business cost function).",
        code: `from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# Find optimal F1 threshold from OOF predictions
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_preds)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

best_threshold = thresholds[f1_scores[:-1].argmax()]
best_f1        = f1_scores.max()

print(f"Default threshold (0.5) F1:  {f1_score(y_train, oof_preds > 0.5):.4f}")
print(f"Optimal threshold ({best_threshold:.3f}) F1: {best_f1:.4f}")

# For asymmetric cost (e.g. false negatives cost 5x false positives):
# fbeta_scores = (1+25) * precisions * recalls / (25*precisions + recalls + 1e-9)  # beta=5

# Plot PR curve
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, color='#4ade80', linewidth=2)
plt.axvline(recalls[f1_scores[:-1].argmax()], color='#f87171', linestyle='--',
            label=f'Optimal threshold: {best_threshold:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve'); plt.legend(); plt.show()`,
        warnings: ["Always tune the threshold on OOF predictions (in-sample) and validate on your holdout — do not tune it on holdout, as that leaks the holdout into your decision."],
        tips: ["If you have asymmetric costs (false negatives cost more than false positives), use F-beta with beta > 1 (e.g. F2 or F5) to weight recall more heavily in threshold selection."],
      },
    ],
    feRefs: ["Weight of Evidence (WoE)"],
    edaRefs: ["Calibration Curve (Reliability Diagram)"],
    problemRefs: ["Poor Probability Calibration", "Threshold Set at Default 0.5", "Metric Mismatch: Train Loss ≠ Eval Metric"],
  },
  {
    id: 15,
    phase: "15",
    title: "Submission / Deployment",
    subtitle: "Final checks before delivering your model",
    time: "30 min",
    color: "#fbbf24",
    icon: "🏁",
    mode: "both",
    summary: "The last mile. For competitions: trust CV over public LB and choose your final submissions wisely. For production: validate on held-out data, document the model, set up monitoring, and plan for retraining.",
    steps: [
      {
        title: "Competition: final submission strategy",
        detail: "You have two final submission slots. Use them wisely. Slot 1 = best OOF CV score model. Slot 2 = best ensemble or most diverse alternative. Never use both slots for variants of the same LB-optimised model.",
        code: `# Final submission checklist:
print("FINAL SUBMISSION CHECKLIST")
print("=" * 50)
print(f"1. Best CV single model:      {best_single_cv:.5f}")
print(f"2. Best CV ensemble:          {best_ensemble_cv:.5f}")
print(f"3. Public LB best:            {public_lb_best:.5f}")
print(f"4. CV ↔ LB correlation check: {'GOOD' if abs(best_single_cv - public_lb_best) < 0.01 else 'WARN'}")
print()
print("SUBMISSION 1: Best CV score model (TRUST CV)")
print(f"  → sub_{best_single_name}_cv{best_single_cv:.5f}.csv")
print()
print("SUBMISSION 2: Best ensemble (DIVERSITY INSURANCE)")
print(f"  → sub_ensemble_cv{best_ensemble_cv:.5f}.csv")
print()
print("DO NOT select based on public LB alone.")
print("The private LB rewards CV correlation, not LB chasing.")`,
        warnings: ["The most common mistake: selecting both final submissions based on public LB performance rather than CV performance. The public LB uses only 20-30% of test data — it's noisier than your 5-fold CV."],
        tips: ["If your CV score and public LB are more than 0.01 apart (after the sanity check in Phase 11), return to the CV strategy. Something is wrong and it will hurt you on the private LB."],
      },
      {
        title: "Production: validation, documentation, monitoring",
        detail: "For real-world deployments, final model evaluation must be on a truly held-out test set never seen during development. Document the model card, set up data drift monitoring, and define a retraining trigger.",
        code: `# Production deployment checklist:

# 1. Final evaluation on held-out test set (ONLY LOOK AT THIS ONCE)
final_test_preds  = final_model.predict_proba(X_holdout)[:, 1]
final_test_metric = roc_auc_score(y_holdout, final_test_preds)
print(f"FINAL HOLDOUT AUC: {final_test_metric:.5f}")  # ← This is your reported performance

# 2. Save model + preprocessor together
import joblib
joblib.dump({'model': final_model, 'preprocessor': preprocessor,
             'threshold': best_threshold, 'features': feature_names},
            'model_v1.joblib')

# 3. Set up drift monitoring (check weekly in production)
# from evidently import Report
# data_drift_report = Report(metrics=[DataDriftPreset()])
# data_drift_report.run(reference_data=train, current_data=new_data)

# 4. Define retraining trigger:
#    - KS test p-value < 0.05 on a key feature
#    - Model AUC drops > 3% on recent labelled data
#    - Business events: new product lines, market regime change`,
        warnings: ["Never report performance on the same data used for any development decision. The holdout must be truly held out — no hyperparameter tuning, no threshold selection, no feature selection informed by it."],
        tips: ["In production, the model's score on the most recent window of labelled data is the most reliable performance indicator. Monitor it weekly and alert when it drops below a threshold."],
      },
    ],
    feRefs: [],
    edaRefs: [],
    problemRefs: ["Overfitting to Public Leaderboard (LB Probing)", "Incorrect Final Submission Strategy", "Concept Drift", "Model Not Reproducible Across Runs"],
  },
];

// ─── CHIP COMPONENTS ──────────────────────────────────────────────────────
const REF_COLORS = {
  fe:      { color: "#818cf8", bg: "#818cf818", label: "FE" },
  eda:     { color: "#22d3ee", bg: "#22d3ee18", label: "EDA" },
  problem: { color: "#f87171", bg: "#f8717118", label: "PROB" },
};

function RefChip({ text, type }) {
  const c = REF_COLORS[type];
  return (
    <span style={{ display:"inline-flex", alignItems:"center", gap:"4px", padding:"2px 8px", borderRadius:"12px", fontSize:"0.64rem", fontWeight:600, background:c.bg, color:c.color, border:`1px solid ${c.color}33`, margin:"2px", whiteSpace:"nowrap" }}>
      <span style={{ fontSize:"0.58rem", opacity:0.7 }}>{c.label}</span>
      {text}
    </span>
  );
}

function CodeBlock({ code }) {
  const [show, setShow] = useState(false);
  if (!code) return null;
  return (
    <div style={{ marginTop:"10px" }}>

      <button onClick={() => setShow(s => !s)} style={{ background:"var(--bg-surface)", border:"1px solid var(--text-dim)", borderRadius:"5px", padding:"4px 12px", fontSize:"0.7rem", color:"var(--text-secondary)", cursor:"pointer", fontFamily:"var(--font-mono)" }}>
        {show ? "▲ hide code" : "▶ show code snippet"}
      </button>
      {show && (
        <pre style={{ margin:"6px 0 0", padding:"14px 16px", background:"var(--bg-base)", border:"1px solid var(--border-subtle)", borderRadius:"6px", fontSize:"0.72rem", fontFamily:"var(--font-mono)", color:"#9ab0cc", overflowX:"auto", lineHeight:1.65, whiteSpace:"pre" }}>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────
export default function Play_app() {
  const [activePhase, setActivePhase]     = useState(null);
  const [completed, setCompleted]         = useState({});
  const [mode, setMode]                   = useState("both");    // both | competition | production
  const [openSteps, setOpenSteps]         = useState({});

  const visiblePhases = PHASES.filter(p => mode === "both" || p.mode === "both" || p.mode === mode);

  const toggleCompleted = (id, e) => { e.stopPropagation(); setCompleted(c => ({ ...c, [id]: !c[id] })); };
  const toggleStep = (key) => setOpenSteps(s => ({ ...s, [key]: !s[key] }));

  const completedCount = visiblePhases.filter(p => completed[p.id]).length;
  const pct = Math.round((completedCount / visiblePhases.length) * 100);

  useEffect(() => {
    if (activePhase !== null) {
      const el = document.getElementById(`phase-${activePhase}`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [activePhase]);

  return (
    <div>
      <Header />
    <div style={{ fontFamily:"var(--font-body)", background:"var(--bg-base)", minHeight:"100vh", color:"var(--text-primary)" }}>
      <style>{`
.step-row:hover{background:var(--bg-elevated) !important}
`}</style>
      {/* ── HERO ── */}
      <div style={{ background:"var(--bg-base)", borderBottom:"1px solid var(--border-subtle)", padding:"40px 32px 28px", position:"relative", overflow:"hidden" }}>
        <div style={{ position:"absolute", top:0, left:0, right:0, bottom:0, backgroundImage:"radial-gradient(circle at 20% 50%, #818cf808 0%, transparent 50%), radial-gradient(circle at 80% 20%, #4ade8008 0%, transparent 40%)", pointerEvents:"none" }} />
        <div style={{ position:"relative", maxWidth:"900px" }}>
          <div style={{ fontSize:"0.7rem", fontFamily:"var(--font-mono)", color:"var(--text-tertiary)", letterSpacing:"0.15em", marginBottom:"10px" }}>DATA SCIENCE PLAYBOOK · COMPLETE PROJECT GUIDE</div>
          <h1 style={{ fontFamily:"'Syne', sans-serif", fontSize:"2.2rem", fontWeight:800, margin:"0 0 10px", letterSpacing:"-0.03em", lineHeight:1.1 }}>
            <span style={{ color:"#fff" }}>From Raw Data</span>
            <span style={{ color:"#818cf8" }}> → </span>
            <span style={{ color:"#4ade80" }}>Winning Model</span>
          </h1>
          <p style={{ fontSize:"0.9rem", color:"var(--text-secondary)", margin:"0 0 20px", lineHeight:1.6, maxWidth:"640px" }}>
            A step-by-step operational guide for ML projects and competitions — each phase linked to the EDA, Feature Engineering, and ML Problems tables. Follow it in order. Every step you skip is a problem you'll debug later.
          </p>

          {/* Mode toggle */}
          <div style={{ display:"flex", gap:"8px", alignItems:"center", flexWrap:"wrap" }}>
            <span style={{ fontSize:"0.7rem", color:"var(--text-tertiary)" }}>MODE:</span>
            {[["both","🔬 Full Guide"],["competition","🏆 Competition"],["production","🏢 Production"]].map(([m, label]) => (
              <button key={m} onClick={() => setMode(m)} style={{ padding:"6px 14px", borderRadius:"6px", fontSize:"0.74rem", fontWeight: mode===m ? 700:400, border: mode===m ? "1px solid #818cf8" : "1px solid var(--border-default)", background: mode===m ? "#818cf822" : "transparent", color: mode===m ? "#818cf8" : "var(--text-tertiary)" }}>
                {label}
              </button>
            ))}
            <div style={{ marginLeft:"auto", display:"flex", alignItems:"center", gap:"12px" }}>
              <div style={{ fontSize:"0.72rem", color:"var(--text-tertiary)" }}>{completedCount}/{visiblePhases.length} phases complete</div>
              <div style={{ width:"160px", height:"6px", background:"var(--bg-overlay)", borderRadius:"3px", overflow:"hidden" }}>
                <div style={{ width:`${pct}%`, height:"100%", background:"linear-gradient(90deg, var(--accent-sage), var(--accent-violet))", transition:"width 0.4s", borderRadius:"3px" }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── LEGEND ── */}
      <div style={{ padding:"10px 32px", background:"var(--bg-surface)", borderBottom:"1px solid var(--border-faint)", display:"flex", gap:"20px", flexWrap:"wrap", fontSize:"0.69rem" }}>
        {Object.entries(REF_COLORS).map(([k, v]) => (
          <span key={k} style={{ display:"flex", alignItems:"center", gap:"5px" }}>
            <span style={{ display:"inline-block", padding:"1px 7px", borderRadius:"10px", background:v.bg, color:v.color, border:`1px solid ${v.color}33`, fontSize:"0.63rem", fontWeight:700 }}>{v.label}</span>
            <span style={{ color:"#38386a" }}>= {k==="fe"?"Feature Engineering table":k==="eda"?"EDA table":"ML Problems table"}</span>
          </span>
        ))}
        <span style={{ color:"var(--text-tertiary)", marginLeft:"auto" }}>✓ = mark phase complete · ▶ = expand code snippet</span>
      </div>

      {/* ── LAYOUT ── */}
      <div style={{ display:"flex", minHeight:"calc(100vh - 200px)" }}>

        {/* Sidebar timeline */}
        <div style={{ width:"220px", flexShrink:0, padding:"20px 12px", background:"var(--bg-base)", borderRight:"1px solid var(--border-faint)", position:"sticky", top:"var(--header-h)", height:"calc(100vh - var(--header-h))", overflowY:"auto" }}>
          <div style={{ fontSize:"0.6rem", color:"var(--text-tertiary)", letterSpacing:"0.1em", marginBottom:"12px", padding:"0 4px" }}>PHASES</div>
          {visiblePhases.map((p, i) => (
            <button key={p.id} onClick={() => setActivePhase(p.id)} style={{
              display:"flex", alignItems:"center", gap:"9px",
              width:"100%", padding:"8px 10px", borderRadius:"6px", marginBottom:"2px",
              background: activePhase===p.id ? `${p.color}18` : "transparent",
              border: activePhase===p.id ? `1px solid ${p.color}44` : "1px solid transparent",
              textAlign:"left", color: completed[p.id] ? "var(--text-tertiary)" : activePhase===p.id ? p.color : "var(--text-secondary)",
              fontSize:"0.74rem",
            }}>
              <span style={{ fontSize:"0.7rem", fontFamily:"var(--font-mono)", color: completed[p.id] ? "#2a2a50" : p.color, minWidth:"24px" }}>{p.phase}</span>
              <span style={{ lineHeight:1.3 }}>
                <span style={{ display:"block", fontWeight:600, fontSize:"0.72rem" }}>{p.title}</span>
                {completed[p.id] && <span style={{ fontSize:"0.62rem", color:"#4ade8066" }}>✓ done</span>}
                {p.mode !== "both" && <span style={{ fontSize:"0.58rem", color:"#f8717166", display:"block" }}>◆ {p.mode} only</span>}
              </span>
            </button>
          ))}
        </div>

        {/* Main content */}
        <div style={{ flex:1, padding:"20px 24px", overflowY:"auto" }}>
          {visiblePhases.map((p) => (
            <div key={p.id} id={`phase-${p.id}`} style={{ marginBottom:"12px" }}>

              {/* Phase header card */}
              <div onClick={() => setActivePhase(prev => prev===p.id ? null : p.id)}
                style={{
                  background: activePhase===p.id ? "var(--bg-surface)" : "var(--bg-surface)",
                  border:`1px solid ${activePhase===p.id ? p.color+"44" : "var(--bg-overlay)"}`,
                  borderLeft:`4px solid ${p.color}`,
                  borderRadius:"8px", padding:"16px 20px",
                  cursor:"pointer", transition:"all 0.2s",
                  position:"relative",
                }}>
                <div style={{ display:"flex", alignItems:"center", gap:"14px" }}>
                  <span style={{ fontSize:"0.7rem", fontFamily:"var(--font-mono)", color:p.color, minWidth:"28px" }}>{p.phase}</span>
                  <span style={{ fontSize:"1.4rem" }}>{p.icon}</span>
                  <div style={{ flex:1 }}>
                    <div style={{ display:"flex", alignItems:"baseline", gap:"10px", flexWrap:"wrap" }}>
                      <span style={{ fontFamily:"'Syne',sans-serif", fontSize:"1.05rem", fontWeight:700, color:completed[p.id]?"var(--text-tertiary)":"var(--text-primary)" }}>{p.title}</span>
                      <span style={{ fontSize:"0.76rem", color:"var(--text-tertiary)" }}>{p.subtitle}</span>
                    </div>
                    <p style={{ margin:"4px 0 0", fontSize:"0.8rem", color:"var(--text-secondary)", lineHeight:1.5 }}>{p.summary}</p>
                  </div>
                  <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:"6px", flexShrink:0 }}>
                    <span style={{ fontSize:"0.67rem", color:"var(--text-tertiary)", fontFamily:"var(--font-mono)" }}>⏱ {p.time}</span>
                    {p.mode !== "both" && <span style={{ fontSize:"0.63rem", color:"#f8717166", border:"1px solid #f8717122", borderRadius:"4px", padding:"1px 6px" }}>◆ {p.mode}</span>}
                    <button onClick={(e) => toggleCompleted(p.id, e)} style={{ padding:"3px 10px", borderRadius:"4px", fontSize:"0.67rem", fontWeight:600, border:`1px solid ${completed[p.id]?"#4ade80":"var(--border-default)"}`, background: completed[p.id]?"#4ade8022":"transparent", color: completed[p.id]?"#4ade80":"var(--text-tertiary)" }}>
                      {completed[p.id] ? "✓ done" : "mark done"}
                    </button>
                  </div>
                </div>
              </div>

              {/* Phase expanded content */}
              {activePhase === p.id && (
                <div style={{ background:"var(--bg-base)", border:"1px solid var(--bg-overlay)", borderTop:"none", borderRadius:"0 0 8px 8px", padding:"0 0 16px" }}>

                  {/* Steps */}
                  {p.steps.map((step, si) => {
                    const key = `${p.id}-${si}`;
                    return (
                      <div>
                      <Header />

                                            <div key={si} style={{ margin:"0 16px", borderBottom:"1px solid var(--bg-elevated)" }}>
                        <div className="step-row" onClick={() => toggleStep(key)} style={{ padding:"12px 8px 10px", cursor:"pointer", display:"flex", gap:"12px", alignItems:"flex-start", borderRadius:"4px", background:"transparent", transition:"background 0.1s" }}>
                          <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:"4px", flexShrink:0, paddingTop:"2px" }}>
                            <div style={{ width:"22px", height:"22px", borderRadius:"50%", background:`${p.color}22`, border:`1px solid ${p.color}66`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:"0.65rem", fontWeight:700, color:p.color }}>{si+1}</div>
                            {si < p.steps.length-1 && <div style={{ width:"1px", height:"100%", minHeight:"12px", background:"var(--bg-overlay)" }} />}
                          </div>
                          <div style={{ flex:1 }}>
                            <div style={{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"5px" }}>
                              <span style={{ fontWeight:600, fontSize:"0.88rem", color:"var(--text-primary)" }}>{step.title}</span>
                              <span style={{ fontSize:"0.65rem", color:"var(--text-dim)" }}>{openSteps[key] ? "▲" : "▼"}</span>
                            </div>
                            {openSteps[key] && (
                              <>
                                <p style={{ fontSize:"0.82rem", color:"var(--text-secondary)", lineHeight:1.65, margin:"0 0 8px" }}>{step.detail}</p>
                                <CodeBlock code={step.code} />
                                {step.warnings?.map((w, wi) => (
                                  <div key={wi} style={{ display:"flex", gap:"8px", background:"#1a0808", border:"1px solid #f8717130", borderRadius:"5px", padding:"8px 12px", margin:"8px 0", fontSize:"0.78rem", color:"#f87171", lineHeight:1.55 }}>
                                    <span>⚠️</span><span>{w}</span>
                                  </div>
                                ))}
                                {step.tips?.map((t, ti) => (
                                  <div key={ti} style={{ display:"flex", gap:"8px", background:"#0f1a10", border:"1px solid #4ade8030", borderRadius:"5px", padding:"8px 12px", margin:"8px 0", fontSize:"0.78rem", color:"#86efac", lineHeight:1.55 }}>
                                    <span>💡</span><span>{t}</span>
                                  </div>
                                ))}
                              </>
                            )}
                          </div>
                        </div>
                      </div></div>
                    );
                  })}

                  {/* Cross-references */}
                  {(p.feRefs.length > 0 || p.edaRefs.length > 0 || p.problemRefs.length > 0) && (
                    <div style={{ margin:"12px 20px 0", padding:"12px 14px", background:"var(--bg-surface)", border:"1px solid var(--border-faint)", borderRadius:"6px" }}>
                      <div style={{ fontSize:"0.63rem", color:"var(--text-tertiary)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:"8px" }}>Cross-References → Use These Tables at This Phase</div>
                      <div style={{ display:"flex", flexWrap:"wrap", gap:"0" }}>
                        {p.feRefs.map(r => <RefChip key={r} text={r} type="fe" />)}
                        {p.edaRefs.map(r => <RefChip key={r} text={r} type="eda" />)}
                        {p.problemRefs.map(r => <RefChip key={r} text={r} type="problem" />)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Footer */}
          <div style={{ marginTop:"32px", padding:"24px", background:"var(--bg-base)", borderRadius:"8px", border:"1px solid var(--border-faint)", textAlign:"center" }}>
            <div style={{ fontFamily:"'Syne',sans-serif", fontSize:"1.1rem", fontWeight:700, color:"var(--text-tertiary)", marginBottom:"8px" }}>The Hierarchy of Trust</div>
            <div style={{ fontSize:"0.83rem", color:"#35356a", lineHeight:1.8, maxWidth:"600px", margin:"0 auto" }}>
              Training metrics &lt; Cross-validation score &lt; Held-out test score &lt; Production monitoring<br/>
              <span style={{ color:"var(--text-dim)" }}>Every leakage, every wrong CV strategy, every over-tuned threshold is a violation of this hierarchy.</span>
            </div>
          </div>
        </div>
      </div>
    </div></div>
  );
}