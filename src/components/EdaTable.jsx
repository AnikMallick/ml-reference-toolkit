import { useState, useMemo } from "react";
import Header from "./Header";

const GROUP_ACCENT = {
  "Automated EDA":              "#22d3ee",
  "Univariate · Numeric":       "#60a5fa",
  "Univariate · Categorical":   "#818cf8",
  "Target Analysis":            "#4ade80",
  "Bivariate · Num vs Num":     "#34d399",
  "Bivariate · Cat vs Num":     "#a3e635",
  "Bivariate · Cat vs Cat":     "#c084fc",
  "Missing Data":               "#f87171",
  "Outlier Detection":          "#fb923c",
  "Correlation & Redundancy":   "#facc15",
  "Distribution Comparison":    "#f472b6",
  "Dimensionality Exploration": "#e879f9",
  "Time Series EDA":            "#fbbf24",
  "Text / NLP EDA":             "#86efac",
  "Image / CV EDA":             "#ff9cc2",
  "Feature–Target Relationship":"#7dd3fc",
  "Multivariate / Interaction": "#a78bfa",
};

const PRIORITY_STYLE = {
  "🥇 First": { bg: "#1a2a1a", color: "#4ade80", border: "#4ade80" },
  "🥈 Second": { bg: "var(--border-subtle)", color: "#818cf8", border: "#818cf8" },
  "🥉 Third": { bg: "#2a2a1a", color: "#facc15", border: "#facc15" },
  "🔁 Iterative": { bg: "#2a1a2a", color: "#c084fc", border: "#c084fc" },
};

const EDA = [

  // ─────────────────────────────────────────────────────────────
  // AUTOMATED EDA
  // ─────────────────────────────────────────────────────────────
  {
    name: "ydata-profiling (Pandas Profiling)",
    group: "Automated EDA",
    lib: "ydata-profiling",
    api: "ProfileReport(df, explorative=True)",
    priority: "🥇 First",
    detects: "In one report: distributions, missingness, cardinality, skew, kurtosis, correlations (Pearson/Spearman/Kendall/Cramér's V), duplicate rows, high-cardinality alerts, constant features.",
    action: "Run first on every new dataset. Use the Alerts section to triage problems. Check the Correlations tab to find redundant features. Note every WARNING raised.",
    feLinks: "Rare Label Encoding · Frequency Encoding · Variance Threshold · Correlation-Based Elimination",
    problemLinks: "Missing Value Mishandling · Duplicated Rows · Multicollinear Features · Wrong Evaluation Metric",
    modelBenefit: "All models — the report guides ALL subsequent preprocessing and feature engineering decisions before any model is touched.",
    competitionNote: "The single highest-leverage first step in any competition. Run it before writing a single model line. The Alerts section is a prioritised action list.",
  },
  {
    name: "SweetViz",
    group: "Automated EDA",
    lib: "sweetviz",
    api: "sv.analyze(df, target_feat='y') / sv.compare(train, test)",
    priority: "🥇 First",
    detects: "Target-aware distributions for every feature. Train vs test distribution comparison side-by-side. Associations between all feature pairs and the target. Duplicate and missing value summary.",
    action: "Pass the target column to get target-stratified distributions. Use sv.compare(train, test) to immediately spot train/test distribution shift — one of the most powerful quick checks available.",
    feLinks: "Target / Mean Encoding · Categorical Encoding (all) · Missing Value Indicator",
    problemLinks: "Train/Validation Distribution Mismatch · Covariate Shift · Missing Value Mishandling · Label / Annotation Noise",
    modelBenefit: "All models — particularly valuable before GBM, Logistic Regression, and MLP where feature distributions directly influence training.",
    competitionNote: "sv.compare(train, test) is a competition essential. If features show different distributions across split, adversarial validation is the next step.",
  },
  {
    name: "D-Tale",
    group: "Automated EDA",
    lib: "dtale",
    api: "dtale.show(df)",
    priority: "🥇 First",
    detects: "Interactive GUI for distributions, custom scatter plots, correlation matrix, data filters, custom feature creation, outlier flagging — all without writing code.",
    action: "Use for interactive, ad-hoc exploration. Especially useful for understanding a feature by clicking through its distribution and scatter against target without boilerplate code.",
    feLinks: "Interaction / Product Features · Date/Time Feature Extraction · Ratio Features",
    problemLinks: "Incorrect Data Types / Schema Drift · Outlier Contamination · Feature Scale Sensitivity",
    modelBenefit: "All models. Best used in discovery phase before committing engineering effort.",
    competitionNote: "Fastest way to answer 'what does this feature look like?' questions during initial exploration. The correlation matrix is interactive and immediately actionable.",
  },
  {
    name: "AutoViz",
    group: "Automated EDA",
    lib: "autoviz",
    api: "AutoViz_Class().AutoViz('', dfte=df, depVar='target')",
    priority: "🥇 First",
    detects: "Automatically generates all relevant chart types per column type (histograms, scatter, violin, heatmap). Includes cleaning recommendations: ID columns, constant features, high skew, missing values.",
    action: "Use when you want zero-config chart generation across all columns. The built-in cleaning recommendations are worth reading — they flag obvious issues like ID columns included accidentally.",
    feLinks: "Variance Threshold · Log Transform · Ordinal / Label Encoding",
    problemLinks: "Wrong Model for Problem Structure · Incorrect Data Types / Schema Drift · Underfitting — High Bias",
    modelBenefit: "All models — fast orientation step before any column-by-column analysis.",
    competitionNote: "Faster than ydata-profiling for large datasets. Use it when you need a quick visual sweep and are short on time.",
  },

  // ─────────────────────────────────────────────────────────────
  // UNIVARIATE · NUMERIC
  // ─────────────────────────────────────────────────────────────
  {
    name: "Histogram + KDE",
    group: "Univariate · Numeric",
    lib: "matplotlib · seaborn · plotly",
    api: "sns.histplot(df[col], kde=True) / df[col].hist(bins=50)",
    priority: "🥇 First",
    detects: "Distribution shape: normal, skewed, bimodal, uniform, log-normal, heavy-tailed. Presence of outliers at the tails. Natural breaks or modes in the data.",
    action: "Right-skewed → try log1p or Yeo-Johnson transform before feeding to linear/distance models. Bimodal → possible hidden group → check by category. Heavy tails → RobustScaler. Uniform → fine for all models.",
    feLinks: "Log Transform · Yeo-Johnson / Box-Cox Transform · Standard Scaler · Robust Scaler · Binning / Discretisation",
    problemLinks: "Outlier Contamination · Feature Scale Sensitivity for Distance Models · Wrong Evaluation Metric",
    modelBenefit: "Linear Regression, Logistic Regression, SVM, MLP, KNN — transforms identified here directly improve these models. Tree models are unaffected by distribution shape.",
    competitionNote: "Skewed features are the #1 cause of linear model underperformance. Check every numeric column. The histogram takes 2 seconds and can save hours of debugging.",
  },
  {
    name: "Box Plot",
    group: "Univariate · Numeric",
    lib: "seaborn · plotly · matplotlib",
    api: "sns.boxplot(y=df[col]) / df[col].plot(kind='box')",
    priority: "🥇 First",
    detects: "Median, IQR, whiskers (1.5×IQR), and individual outlier points beyond whiskers. Skewness via median position within box. Spread comparison across features.",
    action: "Outlier points beyond whiskers → investigate: data error or legitimate extreme? If error → Winsorise. If legitimate → use HuberRegressor or log-transform. Asymmetric median → skewed distribution → transform.",
    feLinks: "Log Transform · Robust Scaler · Winsoriser (feature-engine) · Missing Value Indicator",
    problemLinks: "Outlier Contamination · Missing Value Mishandling · Feature Scale Sensitivity for Distance Models",
    modelBenefit: "Same as histogram. Primary insight is for preprocessing decisions for linear/distance models.",
    competitionNote: "Box plots are faster than histograms for outlier detection across many columns. Use df.boxplot() to plot all numeric columns in one go.",
  },
  {
    name: "Violin Plot",
    group: "Univariate · Numeric",
    lib: "seaborn · plotly",
    api: "sns.violinplot(y=df[col]) / sns.violinplot(x='cat', y='num', data=df)",
    priority: "🥈 Second",
    detects: "Full distribution shape (KDE) + box plot statistics in one. Reveals bimodality, heavy tails, and distribution differences across categories simultaneously.",
    action: "Bimodal violin → the column may need to be split by a category or there are two distinct populations. Very different widths by group → heteroscedasticity → use Quantile Regression or log-transform target.",
    feLinks: "Binning / Discretisation · Group Aggregation Features · Log Transform",
    problemLinks: "Wrong Model for Problem Structure · Metric Mismatch: Train Loss ≠ Eval Metric · Underfitting — High Bias",
    modelBenefit: "Informs feature engineering decisions for all models. Reveals group-level structure that suggests group-based features.",
    competitionNote: "Use violin plots when comparing a numeric distribution by class — it's far more informative than a bar chart of means and immediately shows whether the feature is discriminative.",
  },
  {
    name: "QQ Plot (Quantile-Quantile)",
    group: "Univariate · Numeric",
    lib: "scipy · statsmodels",
    api: "scipy.stats.probplot(df[col], plot=plt) / sm.qqplot(df[col], line='s')",
    priority: "🥈 Second",
    detects: "Deviation from normality. S-shaped curve = heavy tails (leptokurtic). Curve bending up = right skew. Straight diagonal line = perfectly normal.",
    action: "S-curve → Yeo-Johnson or PowerTransformer to normalise. After applying a transform, re-plot QQ to verify it worked. Use before Shapiro-Wilk to understand the type of non-normality.",
    feLinks: "Yeo-Johnson / Box-Cox Transform · Log Transform · Quantile Transformer",
    problemLinks: "Overfitting — High Variance (normalisation helps regularised models) · Feature Scale Sensitivity for Distance Models",
    modelBenefit: "Linear Regression, LDA, Gaussian NB, SVM — models whose assumptions depend on Gaussian features improve most after identifying and correcting non-normality here.",
    competitionNote: "Use QQ plot in combination with Shapiro-Wilk test. The plot tells you what type of non-normality exists; the test confirms significance.",
  },
  {
    name: "ECDF (Empirical Cumulative Distribution)",
    group: "Univariate · Numeric",
    lib: "seaborn · statsmodels",
    api: "sns.ecdfplot(df[col]) / plt.plot(np.sort(x), np.arange(1,n+1)/n)",
    priority: "🥈 Second",
    detects: "Full distribution shape without binning. Natural percentile thresholds. Comparing two distributions (train vs test) with precise quantile-level differences — more precise than a histogram.",
    action: "Overlay train and test ECDFs for each feature — where they diverge, covariate shift exists at that quantile. Confirms KS test results with a precise visual.",
    feLinks: "Quantile Transformer · Binning / Discretisation",
    problemLinks: "Covariate Shift (Train/Test Feature Drift) · Train/Validation Distribution Mismatch",
    modelBenefit: "All models — particularly powerful for diagnosing which features drive train/test distribution differences before modelling.",
    competitionNote: "The most precise visual for train vs test distribution comparison. More informative than overlaid histograms. Plot them side-by-side for every important feature.",
  },

  // ─────────────────────────────────────────────────────────────
  // UNIVARIATE · CATEGORICAL
  // ─────────────────────────────────────────────────────────────
  {
    name: "Bar Chart (Value Counts)",
    group: "Univariate · Categorical",
    lib: "matplotlib · seaborn · pandas",
    api: "df[col].value_counts().plot(kind='bar') / sns.countplot(x=col, data=df)",
    priority: "🥇 First",
    detects: "Cardinality of categorical feature. Frequency distribution across categories. Presence of rare categories (long tail). Dominant majority class. Data imbalance.",
    action: "Long tail → apply RareLabelEncoder (group rare values before any encoding). Very high cardinality → use Target/Frequency/Hash encoding, not OHE. Dominant category > 90% → near-constant, consider dropping.",
    feLinks: "One-Hot Encoding · Rare Label Encoding · Frequency / Count Encoding · Target / Mean Encoding · Hashing Encoding",
    problemLinks: "Class Imbalance · Incorrect Data Types / Schema Drift · Multicollinear Features Breaking Linear Models",
    modelBenefit: "All models — cardinality directly determines which encoding strategy to use. Wrong encoding choice can break linear models or create memory issues.",
    competitionNote: "Check every categorical column's cardinality before any encoding. A single 10,000-cardinality column OHE'd creates 10,000 sparse columns — a pipeline-breaking decision.",
  },
  {
    name: "Category Frequency Pareto Chart",
    group: "Univariate · Categorical",
    lib: "matplotlib · pandas",
    api: "vc = df[col].value_counts(); vc.cumsum() / vc.sum() (manually plotted)",
    priority: "🥈 Second",
    detects: "What percentage of rows is covered by the top-k categories (80/20 rule). Which categories dominate vs which are rare. Natural cut-off point for rare-label grouping.",
    action: "If top-10 categories cover 95% of rows → group remaining into 'Other'. Defines the RareLabelEncoder threshold parameter precisely.",
    feLinks: "Rare Label Encoding · Frequency / Count Encoding · One-Hot Encoding",
    problemLinks: "Incorrect Data Types / Schema Drift · Overfitting — High Variance (many rare dummies overfit)",
    modelBenefit: "Linear models, MLP, KNN — reducing encoding dimensionality directly reduces overfitting risk. Tree models benefit less but still gain training speed.",
    competitionNote: "A quick manual step: print df[col].value_counts(normalize=True).head(20). If top 10 cover > 90%, set a rare threshold to keep only those 10.",
  },

  // ─────────────────────────────────────────────────────────────
  // TARGET ANALYSIS
  // ─────────────────────────────────────────────────────────────
  {
    name: "Target Distribution Plot",
    group: "Target Analysis",
    lib: "seaborn · matplotlib",
    api: "df['target'].value_counts(normalize=True).plot(kind='bar') / sns.histplot(df['target'])",
    priority: "🥇 First",
    detects: "Class balance for classification (imbalance ratio). Distribution shape for regression targets (skewness, bimodality). Presence of impossible target values (negative prices). Natural target range.",
    action: "Classification: imbalance > 10:1 → set class_weight='balanced', tune threshold, consider SMOTE inside Pipeline. Regression: right-skewed target → log1p(y), train on log scale, exponentiate predictions.",
    feLinks: "Log Transform (target) · Missing Value Indicator",
    problemLinks: "Class Imbalance · SMOTE Applied Before Split · Threshold Set at Default 0.5 · Wrong Evaluation Metric",
    modelBenefit: "All models — target distribution directly determines loss function, metric choice, and resampling strategy. This is the single most important plot before any modelling.",
    competitionNote: "Always check target distribution first. Log-transforming a skewed regression target routinely improves RMSE by 5-15% for linear/neural models without any other change.",
  },
  {
    name: "Target Mean by Categorical Feature",
    group: "Target Analysis",
    lib: "seaborn · pandas",
    api: "df.groupby('cat_col')['target'].mean().sort_values().plot(kind='barh')",
    priority: "🥇 First",
    detects: "Which categories have above/below-average target values. Encoding signal strength of each category. Whether a categorical feature is predictive and in which direction.",
    action: "Large variance in target mean across categories → feature is predictive → prioritise for target/frequency encoding. Flat line → feature has no signal → candidate for dropping.",
    feLinks: "Target / Mean Encoding · Weight of Evidence (WoE) · Frequency / Count Encoding · CatBoost Encoding",
    problemLinks: "Target Leakage · Wrong Evaluation Metric · Overfitting — High Variance",
    modelBenefit: "GBM, HistGradientBoosting, Linear models — confirms the value of target encoding BEFORE computing it. Tree models benefit from knowing which categories to split on.",
    competitionNote: "Sort by target mean and plot horizontally. Categories with target mean far from the global mean are gold — they carry strong signal for any model.",
  },
  {
    name: "Target Leakage Scatter (Feature vs Target)",
    group: "Target Analysis",
    lib: "seaborn · matplotlib",
    api: "sns.scatterplot(x=df['suspicious_col'], y=df['target'])",
    priority: "🥇 First",
    detects: "Near-perfect linear or monotone relationship between a feature and target — the visual signature of target leakage or extremely strong proxies.",
    action: "Pearson or Spearman correlation > 0.95 with target → investigate immediately for leakage. Check whether this feature could be computed from the target. Audit its temporal construction.",
    feLinks: "Target / Mean Encoding (leakage audit) · Group Aggregation Features (leakage audit)",
    problemLinks: "Target Leakage · Preprocessing Leakage (Fit on Full Data) · Temporal / Future Leakage",
    modelBenefit: "All models — detecting leakage here prevents building a model that looks great locally but fails in production.",
    competitionNote: "Any feature with |correlation with target| > 0.9 is suspicious. Either it's leakage (bad) or an extremely powerful feature (verify it's available at prediction time).",
  },
  {
    name: "Calibration Curve (Reliability Diagram)",
    group: "Target Analysis",
    lib: "sklearn",
    api: "CalibrationDisplay.from_estimator(model, X_val, y_val, n_bins=10)",
    priority: "🔁 Iterative",
    detects: "Whether predicted probabilities align with actual class frequencies. Sigmoid shape = over-confident (SVMs, RF). S-shape reversed = underconfident. Perfect calibration = diagonal line.",
    action: "Deviation from diagonal → apply CalibratedClassifierCV (method='isotonic' for ≥ 1000 samples, method='sigmoid' for less). Required when downstream decisions use probabilities (expected value, risk scoring).",
    feLinks: "Weight of Evidence (WoE) — WoE features improve log-odds calibration of logistic models",
    problemLinks: "Poor Probability Calibration · Wrong Evaluation Metric · Metric Mismatch: Train Loss ≠ Eval Metric",
    modelBenefit: "Logistic Regression (already calibrated), Random Forest, SVM, GBM — the latter three require post-hoc calibration for probability-sensitive tasks.",
    competitionNote: "If competition metric is log-loss or AUC, always check calibration. A better-calibrated model with lower ROC-AUC can outperform on log-loss.",
  },

  // ─────────────────────────────────────────────────────────────
  // BIVARIATE · NUM vs NUM
  // ─────────────────────────────────────────────────────────────
  {
    name: "Scatter Plot",
    group: "Bivariate · Num vs Num",
    lib: "seaborn · matplotlib · plotly",
    api: "sns.scatterplot(x='feat_a', y='feat_b', hue='target', data=df)",
    priority: "🥈 Second",
    detects: "Linear or non-linear relationship between two numeric features. Outlier clusters. Class separability when hue=target. Heteroscedasticity (fan-shaped scatter).",
    action: "Curved pattern → non-linear relationship → add polynomial or interaction feature. Fan shape → heteroscedasticity → log-transform the feature or target. Well-separated by hue → feature is highly discriminative.",
    feLinks: "Polynomial Features · Interaction / Product Features · Log Transform · Ratio Features",
    problemLinks: "Underfitting — High Bias · Feature Scale Sensitivity · Multicollinear Features Breaking Linear Models",
    modelBenefit: "Linear Regression, SVM, MLP — reveals whether feature transformations are needed. Tree models don't need the insights but they confirm what features to create.",
    competitionNote: "Plot your top SHAP features against each other coloured by target. If you see a non-linear boundary, that's a feature interaction worth engineering explicitly.",
  },
  {
    name: "Hexbin / 2D Density Plot",
    group: "Bivariate · Num vs Num",
    lib: "matplotlib · seaborn",
    api: "plt.hexbin(x, y, gridsize=30, cmap='Blues') / sns.kdeplot(x='a', y='b', data=df)",
    priority: "🥈 Second",
    detects: "Density concentrations in 2D feature space. Overplotting obscured by a plain scatter on large datasets. High-density vs sparse regions. Cluster centres.",
    action: "Dense clusters in certain regions → features may benefit from interaction terms. Sparse regions → model may extrapolate there — inspect predictions in sparse zones carefully.",
    feLinks: "Interaction / Product Features · Polynomial Features · Group Aggregation Features",
    problemLinks: "Curse of Dimensionality · Overfitting — High Variance (model memorises dense regions)",
    modelBenefit: "All models — particularly guides KNN and SVM where density patterns affect decision boundaries.",
    competitionNote: "Use hexbin instead of scatter when you have > 50,000 rows — scatter becomes a black blob. Hexbin colour-codes density and reveals the true data structure.",
  },
  {
    name: "Correlation Heatmap (Pearson)",
    group: "Bivariate · Num vs Num",
    lib: "seaborn · pandas",
    api: "sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')",
    priority: "🥇 First",
    detects: "Pairwise linear correlations between all numeric features. Feature multicollinearity clusters. Features correlated with the target. Near-perfect correlation (> 0.95) = redundant features.",
    action: "Cluster of highly correlated features → remove all but one or apply PCA. Features near-zero correlated with everything (including target) → candidate for removal. High correlation with target → investigate for leakage.",
    feLinks: "Correlation-Based Elimination · PCA · Lasso Embedded Selection · Standard Scaler (reduces correlation distortion from different scales)",
    problemLinks: "Multicollinear Features Breaking Linear Models · Target Leakage · Misleading Feature Importance (MDI Bias)",
    modelBenefit: "Linear Regression, Ridge, Lasso, Logistic Regression — directly informs regularisation and feature removal decisions. GBMs are robust to correlation but removal speeds training.",
    competitionNote: "Run this before feature selection. Filter features with |target correlation| > 0.95 as leakage suspects. Identify groups of mutually correlated features for redundancy removal.",
  },
  {
    name: "Spearman Correlation Heatmap",
    group: "Bivariate · Num vs Num",
    lib: "seaborn · pandas",
    api: "sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm')",
    priority: "🥈 Second",
    detects: "Monotone (not just linear) correlations, robust to outliers and non-Gaussian distributions. Better suited than Pearson for skewed or ordinal features.",
    action: "High Spearman but low Pearson → non-linear monotone relationship → log-transform may linearise it. Use Spearman for rank-based or ordinal features.",
    feLinks: "Log Transform · Yeo-Johnson / Box-Cox Transform · Ordinal / Label Encoding",
    problemLinks: "Multicollinear Features Breaking Linear Models · Ignoring Confidence Intervals / Statistical Significance",
    modelBenefit: "All models — more robust than Pearson for exploratory work. Confirms findings from Pearson heatmap.",
    competitionNote: "For competition datasets with heavy-tailed or skewed features, Spearman heatmap is more reliable than Pearson for detecting feature-target relationships.",
  },

  // ─────────────────────────────────────────────────────────────
  // BIVARIATE · CAT vs NUM
  // ─────────────────────────────────────────────────────────────
  {
    name: "Grouped Box Plot / Violin by Category",
    group: "Bivariate · Cat vs Num",
    lib: "seaborn",
    api: "sns.boxplot(x='cat', y='num', data=df) / sns.violinplot(x='cat', y='num', data=df)",
    priority: "🥈 Second",
    detects: "How a numeric feature's distribution differs across categories. Class separability. Presence of category-specific outliers. Heteroscedasticity by group.",
    action: "Well-separated medians → strong categorical-numeric interaction → create group_mean/std aggregation feature. Similar distributions across all groups → categorical feature has no modifying effect on this numeric.",
    feLinks: "Group Aggregation Features · Target Mean by Category · Interaction / Product Features",
    problemLinks: "Class Imbalance · Wrong Model for Problem Structure · Underfitting — High Bias",
    modelBenefit: "GBMs, MLP — confirms value of group-aggregation features. Linear models — confirms whether group × numeric interaction term is worth adding.",
    competitionNote: "Plot your top categorical features against the target as a violin plot. If target distribution is very different across categories, that category is a high-value grouping key for aggregation features.",
  },
  {
    name: "Point Plot with CI (Category Mean + Confidence Interval)",
    group: "Bivariate · Cat vs Num",
    lib: "seaborn",
    api: "sns.pointplot(x='cat', y='target', data=df, capsize=.1)",
    priority: "🥈 Second",
    detects: "Mean target value per category with confidence intervals. Whether differences between category means are statistically significant. Low-sample categories with wide CIs.",
    action: "Overlapping CIs between categories → groups are not significantly different → consider merging them. Wide CI for a category → few samples → apply smoothing/prior in target encoding.",
    feLinks: "Target / Mean Encoding (smoothing parameter) · CatBoost Encoding · Rare Label Encoding",
    problemLinks: "Ignoring Confidence Intervals / Statistical Significance · Label / Annotation Noise",
    modelBenefit: "Linear models, GBMs — directly informs target encoding smoothing settings. Categories with wide CI need stronger regularisation in the encoding.",
    competitionNote: "Use point plots to justify target encoding decisions. Categories where CIs overlap the global mean should be smoothed toward the global mean in your encoding.",
  },
  {
    name: "Strip / Swarm Plot",
    group: "Bivariate · Cat vs Num",
    lib: "seaborn",
    api: "sns.stripplot(x='cat', y='num', data=df, alpha=0.4) / sns.swarmplot(...)",
    priority: "🥉 Third",
    detects: "Individual data point distributions by category. Reveals cluster structure within groups invisible in box plots. Sample size per category is visually obvious.",
    action: "Very few points in a category → that category is a candidate for rare-label grouping. Clear sub-clusters within a group → possible hidden sub-group variable worth investigating.",
    feLinks: "Rare Label Encoding · Group Aggregation Features",
    problemLinks: "Label / Annotation Noise · Class Imbalance",
    modelBenefit: "Informs encoding and aggregation decisions for all models.",
    competitionNote: "Swarm plots are slow for large datasets. Use strip plot with alpha=0.1-0.3 instead. Good for datasets with < 10,000 rows.",
  },

  // ─────────────────────────────────────────────────────────────
  // BIVARIATE · CAT vs CAT
  // ─────────────────────────────────────────────────────────────
  {
    name: "Cramér's V Heatmap",
    group: "Bivariate · Cat vs Cat",
    lib: "pingouin · manual + seaborn",
    api: "pingouin.cramers_v(df[a], df[b]) → sns.heatmap(...)",
    priority: "🥈 Second",
    detects: "Statistical association between pairs of categorical features. Near-duplicate categorical features (Cramér's V > 0.9). Which categoricals are informative vs redundant.",
    action: "V > 0.9 between two features → they carry almost identical information → remove one. V > 0.5 between a feature and the target → highly predictive categorical feature.",
    feLinks: "Correlation-Based Elimination · Rare Label Encoding · One-Hot Encoding",
    problemLinks: "Multicollinear Features Breaking Linear Models · Incorrect Data Types / Schema Drift",
    modelBenefit: "All models — removing near-duplicate categoricals reduces noise. Especially important for linear models and MLP where redundant encoding wastes parameters.",
    competitionNote: "Build a full Cramér's V matrix for all categorical columns. Features with V > 0.9 to each other should be collapsed or one dropped before encoding.",
  },
  {
    name: "Stacked Bar / Mosaic Plot",
    group: "Bivariate · Cat vs Cat",
    lib: "pandas · statsmodels",
    api: "pd.crosstab(df.a, df.b, normalize='index').plot(kind='bar', stacked=True)",
    priority: "🥉 Third",
    detects: "Conditional distribution of one categorical given another. Proportional relationships between category combinations. Deviation from independence (Chi-square pattern).",
    action: "Mosaic shows strong asymmetric pattern → the two categoricals are associated → Chi-square test to confirm → investigate whether one is derivable from the other.",
    feLinks: "Binary Encoding · One-Hot Encoding · Cramér's V",
    problemLinks: "Class Imbalance · Incorrect Data Types / Schema Drift",
    modelBenefit: "Informs encoding strategy for all models. Reveals whether interaction between two categoricals should be encoded explicitly.",
    competitionNote: "Use stacked bar plots to understand how two key categorical features interact before deciding whether to create a combined feature.",
  },

  // ─────────────────────────────────────────────────────────────
  // MISSING DATA
  // ─────────────────────────────────────────────────────────────
  {
    name: "Missingno Matrix",
    group: "Missing Data",
    lib: "missingno",
    api: "import missingno as msno; msno.matrix(df)",
    priority: "🥇 First",
    detects: "Visual pattern of missing values across all columns simultaneously. Whether missingness co-occurs in the same rows (structural / MNAR). Columns that are always missing together.",
    action: "Vertical bands of co-occurring missingness → MNAR pattern → add a binary missingness indicator. Random sparse missingness → MAR/MCAR → simple mean/median imputation suffices.",
    feLinks: "Missing Value Indicator · SimpleImputer · IterativeImputer · KNNImputer",
    problemLinks: "Missing Value Mishandling · Target Leakage · Preprocessing Leakage (Fit on Full Data)",
    modelBenefit: "All models — determines whether missingness itself is a predictive signal (→ add indicator) or just noise (→ impute). HistGradientBoosting handles NaN natively if no indicator is needed.",
    competitionNote: "Run msno.matrix(df) as step one of any competition EDA. A clear co-missingness pattern is often a strong feature in itself — models like HistGBT will exploit it when HistGBT sees NaN as a distinct split value.",
  },
  {
    name: "Missingno Heatmap + Dendrogram",
    group: "Missing Data",
    lib: "missingno",
    api: "msno.heatmap(df) / msno.dendrogram(df)",
    priority: "🥇 First",
    detects: "Correlation between missingness patterns of different columns. Dendrogram shows hierarchical clustering of columns by missingness similarity. Which groups of columns always go missing together.",
    action: "Clustered columns in dendrogram → they share the same data collection mechanism → a single missingness indicator covers the group. Strong heatmap correlation → structural missing data (MNAR) → indicator needed.",
    feLinks: "Missing Value Indicator · Group Aggregation Features",
    problemLinks: "Missing Value Mishandling · Incorrect Data Types / Schema Drift",
    modelBenefit: "Reveals feature groups to create indicators for. Particularly important for linear models, SVM, MLP that cannot handle NaN natively.",
    competitionNote: "Columns that are always missing together in train should be checked in test too. If missingness patterns differ across splits, it signals distribution shift.",
  },
  {
    name: "Missing Value Heatmap by Target Class",
    group: "Missing Data",
    lib: "seaborn · pandas",
    api: "df.isnull().groupby(df['target']).mean().T → sns.heatmap(...)",
    priority: "🥈 Second",
    detects: "Whether missingness rate differs by target class — the clearest signal that missingness is MNAR (Not At Random) and is itself predictive of the target.",
    action: "Very different missingness rates per class → the fact that a value is missing IS the signal → always add a binary missingness indicator for this feature, regardless of imputation.",
    feLinks: "Missing Value Indicator · Target Mean by Category (of the indicator)",
    problemLinks: "Missing Value Mishandling · Target Leakage (if missing driven by post-event coding)",
    modelBenefit: "All models gain signal from the indicator. Especially important for linear models and MLP that can weight the indicator directly.",
    competitionNote: "In medical and financial data, 'not measured' often means 'normal' or 'not suspicious' — which is informative. The indicator captures this.",
  },

  // ─────────────────────────────────────────────────────────────
  // OUTLIER DETECTION
  // ─────────────────────────────────────────────────────────────
  {
    name: "Z-Score / IQR Outlier Flag Plot",
    group: "Outlier Detection",
    lib: "pandas · scipy · matplotlib",
    api: "z = np.abs(zscore(df[num_cols])); (z > 3).sum() / df.plot.scatter + threshold lines",
    priority: "🥇 First",
    detects: "Rows with extreme values beyond 3 standard deviations (Z-score) or 1.5×IQR below Q1 or above Q3. Counts outliers per feature. Identifies most-affected columns.",
    action: "Investigate flagged rows: recording errors → Winsorise or remove. Legitimate extremes → use RobustScaler, HuberRegressor, or log-transform. Never blindly delete without domain check.",
    feLinks: "Log Transform · Robust Scaler · Winsoriser (feature-engine) · IsolationForest",
    problemLinks: "Outlier Contamination · Label / Annotation Noise · Feature Scale Sensitivity for Distance Models",
    modelBenefit: "Linear Regression, SVM, MLP, KNN — outliers severely distort these. GBMs, Random Forests — robust but still benefit from cleaner data.",
    competitionNote: "Print the top-5 outlier rows for each flagged feature. They're often data entry errors (age = 999, price = -1) rather than legitimate extreme values.",
  },
  {
    name: "Isolation Forest Anomaly Score Plot",
    group: "Outlier Detection",
    lib: "sklearn",
    api: "IsolationForest(contamination=0.05).fit_predict(X) → scatter coloured by score",
    priority: "🥈 Second",
    detects: "Multivariate anomalies — rows that are outliers in the joint feature space but not necessarily in any single feature's marginal distribution. Captures compound outliers invisible to univariate methods.",
    action: "Flag rows with anomaly score = −1. Investigate whether they cluster in a specific region. If they're legitimate, train a model on them separately. If errors, remove or impute.",
    feLinks: "Missing Value Indicator · Group Aggregation Features",
    problemLinks: "Outlier Contamination · Label / Annotation Noise · Class Imbalance (anomalies often are the minority class)",
    modelBenefit: "Linear models, SVM, MLP — removing or flagging multivariate outliers significantly improves model performance. Tree models are more robust.",
    competitionNote: "Isolation Forest anomaly scores can themselves be powerful features in fraud/anomaly detection competitions. Add the raw score as a feature.",
  },
  {
    name: "Outlier Fraction Bar Chart by Column",
    group: "Outlier Detection",
    lib: "pandas · matplotlib",
    api: "(np.abs(zscore(df[num_cols])) > 3).mean().sort_values().plot(kind='barh')",
    priority: "🥈 Second",
    detects: "Which features have the highest outlier rates. Columns requiring urgent cleaning. Systematic data quality issues in certain measurements.",
    action: "Columns with > 5% outlier rate need investigation. Check if the high-outlier columns are sensor/measurement data (errors) or financial/economic (legitimate extremes).",
    feLinks: "Log Transform · Robust Scaler · Winsoriser",
    problemLinks: "Outlier Contamination · Slow Training / Scalability Bottleneck (outliers can slow down GBM convergence)",
    modelBenefit: "All models — prioritises which features to clean first.",
    competitionNote: "Sort by outlier fraction and start cleaning the worst offenders. This bar chart makes it easy to argue for or against cleaning specific columns.",
  },

  // ─────────────────────────────────────────────────────────────
  // CORRELATION & REDUNDANCY
  // ─────────────────────────────────────────────────────────────
  {
    name: "Correlation Funnel (Target Correlation Ranked)",
    group: "Correlation & Redundancy",
    lib: "pandas · manual",
    api: "df.corrwith(df['target']).abs().sort_values(ascending=False).plot(kind='barh')",
    priority: "🥇 First",
    detects: "Which features have the strongest linear relationship with the target, ranked. Immediate shortlist for most predictive features. Features with near-zero target correlation (uninformative).",
    action: "Top features → include in baseline model immediately. Near-zero target correlation → candidate for removal (but verify with mutual information — they may have non-linear relationship).",
    feLinks: "Univariate Filter (SelectKBest) · Mutual Information · Lasso Embedded Selection",
    problemLinks: "Target Leakage · Underfitting — High Bias · Misleading Feature Importance (MDI Bias)",
    modelBenefit: "Linear Regression, Logistic Regression — features at the top of this funnel are the most powerful for these models. GBMs benefit less since they capture non-linear relationships.",
    competitionNote: "Plot this before any feature selection. It's the fastest possible feature prioritisation. Build your first model on the top-10 correlated features.",
  },
  {
    name: "Mutual Information Bar Chart",
    group: "Correlation & Redundancy",
    lib: "sklearn",
    api: "mutual_info_classif(X, y) → pd.Series(mi, index=features).sort_values().plot(kind='barh')",
    priority: "🥇 First",
    detects: "Non-linear AND linear feature-target dependencies for every feature. Captures U-shaped, step-function, and any statistical dependency — not just linear. Zero MI = completely uninformative for the target.",
    action: "Features with MI = 0 → drop them. Features with high MI but low Pearson correlation → strong non-linear signal → important for tree-based and neural models. Compare MI ranking vs Pearson ranking to identify non-linear features.",
    feLinks: "Univariate Filter (SelectKBest) · Feature Selection (all embedded/wrapper methods) · Interaction / Product Features (for low-Pearson, high-MI pairs)",
    problemLinks: "Underfitting — High Bias · Misleading Feature Importance (MDI Bias) · Wrong Evaluation Metric",
    modelBenefit: "GBMs, Random Forests, MLP — MI identifies the non-linear signal these models can exploit. Linear models — MI still helps but focus on Pearson too.",
    competitionNote: "Compare MI rankings to Pearson. Features high in MI but low in Pearson correlation are the 'hidden' non-linear gems that tree models will find but linear models miss.",
  },
  {
    name: "VIF (Variance Inflation Factor) Bar Chart",
    group: "Correlation & Redundancy",
    lib: "statsmodels",
    api: "variance_inflation_factor(X, i) for each column i → bar chart",
    priority: "🥈 Second",
    detects: "Multicollinearity severity for each feature in the context of the full feature matrix. VIF > 10 = severe multicollinearity. VIF = 1 = no collinearity.",
    action: "VIF > 10 → either remove one of the correlated group, apply PCA, or switch from OLS to Ridge. Never interpret OLS coefficients when VIF > 5.",
    feLinks: "PCA · Correlation-Based Elimination · Ridge Regression",
    problemLinks: "Multicollinear Features Breaking Linear Models · Misleading Feature Importance (MDI Bias)",
    modelBenefit: "Linear Regression (critical), Logistic Regression, LDA — required diagnostic before fitting these. Ridge/Lasso — tolerant of high VIF but still benefits from knowing.",
    competitionNote: "Only run VIF if you're building linear models or scorecard models. GBM competitions almost never require VIF analysis.",
  },
  {
    name: "Adversarial Validation Plot",
    group: "Distribution Comparison",
    lib: "sklearn · matplotlib",
    api: "Train classifier: train=0, test=1. Plot ROC curve + feature importances of the adversarial model.",
    priority: "🥇 First",
    detects: "Train vs test distribution differences. Which specific features drive the distribution shift. Whether a random CV will give reliable performance estimates.",
    action: "Adversarial AUC > 0.6 → distribution shift exists → identify which features the adversarial model uses most → remove or transform those features. Rerun until adversarial AUC ≈ 0.5.",
    feLinks: "Correlation-Based Elimination · Frequency / Count Encoding (more stable across splits) · Log Transform (reduces scale-driven drift)",
    problemLinks: "Covariate Shift (Train/Test Feature Drift) · Train/Validation Distribution Mismatch · Public/Private LB Split Gap",
    modelBenefit: "All models — removing drift-causing features improves generalisation for every downstream model. This is one of the highest-leverage EDA steps in competitions.",
    competitionNote: "Run adversarial validation before your first model submission. If AUC > 0.65, remove features flagged by the adversarial model — they will hurt your private LB score.",
  },
  {
    name: "Train vs Test Distribution Overlay",
    group: "Distribution Comparison",
    lib: "matplotlib · seaborn",
    api: "for col: sns.kdeplot(train[col], label='train'); sns.kdeplot(test[col], label='test')",
    priority: "🥇 First",
    detects: "Visual covariate shift per feature. Whether train and test share the same feature range, mode, and tail behaviour. Systematic differences in mean, variance, or tail.",
    action: "Features where distributions visually differ → flag for adversarial validation → consider removing. Features perfectly aligned → safe to use without adjustment.",
    feLinks: "Log Transform (reduces scale-driven shift) · Quantile Transformer (forces alignment)",
    problemLinks: "Covariate Shift · Train/Validation Distribution Mismatch · Public/Private LB Split Gap",
    modelBenefit: "All models — a model trained on train distribution will underperform on shifted test distribution.",
    competitionNote: "Plot overlaid KDEs for every numeric feature. Any feature where the two curves look different is a candidate for removal or robust transformation.",
  },

  // ─────────────────────────────────────────────────────────────
  // DIMENSIONALITY EXPLORATION
  // ─────────────────────────────────────────────────────────────
  {
    name: "Pair Plot (Grid Scatter Matrix)",
    group: "Dimensionality Exploration",
    lib: "seaborn · pandas",
    api: "sns.pairplot(df[top_features + ['target']], hue='target', diag_kind='kde')",
    priority: "🥈 Second",
    detects: "All pairwise relationships between selected features simultaneously. Class separability in 2D feature combinations. Non-linear relationships and interaction patterns. Redundant features (diagonal scatter).",
    action: "Pairs that perfectly separate classes → strong interaction feature candidate. Diagonal scatter → those two features are correlated → check for redundancy. Use only top 5-8 features to avoid visual overload.",
    feLinks: "Interaction / Product Features · Polynomial Features · PCA",
    problemLinks: "Underfitting — High Bias · Multicollinear Features · Curse of Dimensionality",
    modelBenefit: "Logistic Regression, SVM, MLP, LDA — pair plots reveal which feature combinations create linear or non-linear separability that these models can exploit.",
    competitionNote: "Use sns.pairplot on your top-8 features by mutual information. The combinations that best separate classes in 2D are candidates for explicit interaction features.",
  },
  {
    name: "PCA Biplot / Explained Variance Plot",
    group: "Dimensionality Exploration",
    lib: "sklearn · matplotlib",
    api: "PCA().fit(X); plt.plot(np.cumsum(pca.explained_variance_ratio_))",
    priority: "🥈 Second",
    detects: "How many principal components explain 90%/95%/99% of variance. Whether the intrinsic dimensionality is much lower than the feature count. Feature contribution to top components (biplot).",
    action: "If 95% variance explained by k << p features → apply PCA dimensionality reduction before KNN, SVM, MLP. If variance decays slowly → data is truly high-dimensional and PCA won't help much.",
    feLinks: "PCA · TruncatedSVD · Kernel PCA",
    problemLinks: "Curse of Dimensionality · Feature Scale Sensitivity for Distance Models · Slow Training / Scalability Bottleneck",
    modelBenefit: "KNN, SVM with RBF, MLP — all benefit from PCA when intrinsic dimensionality is low. Random Forest, GBM — no benefit from PCA.",
    competitionNote: "Check the explained variance curve before applying PCA. If the elbow is at k=5 in a 100-feature dataset, PCA to 5 components removes noise while preserving signal.",
  },
  {
    name: "t-SNE / UMAP 2D Embedding Plot",
    group: "Dimensionality Exploration",
    lib: "sklearn (TSNE) · umap-learn",
    api: "TSNE(n_components=2, random_state=42).fit_transform(X) → scatter(hue=y)",
    priority: "🥈 Second",
    detects: "High-dimensional cluster structure projected into 2D. Whether classes are separable in the feature space. Data manifold topology. Sub-clusters within classes.",
    action: "Well-separated clusters per class → feature space is expressive, deep model or GBM will perform well. Overlapping blobs → problem is hard, feature engineering or different model family needed.",
    feLinks: "Contextual Sentence Embeddings (for NLP) · CNN Embeddings (for images) · PCA",
    problemLinks: "Underfitting — High Bias · Wrong Model for Problem Structure · Class Imbalance",
    modelBenefit: "Informs model selection. Clear separation → simple model may suffice. Mixed → need powerful model + feature engineering.",
    competitionNote: "Always use random_state for reproducible t-SNE plots. UMAP is faster and preserves more global structure — prefer it for large datasets. Colour by target and by cluster to see both structure and separability.",
  },
  {
    name: "Parallel Coordinates Plot",
    group: "Multivariate / Interaction",
    lib: "pandas · plotly",
    api: "pd.plotting.parallel_coordinates(df[cols], class_column='target') / px.parallel_coordinates(df, ...)",
    priority: "🥉 Third",
    detects: "Multi-feature patterns that distinguish classes. Which combination of features simultaneously separate targets. High-dimensional interaction patterns.",
    action: "Lines crossing between two axes → those features have an interaction effect. Features where line colours separate cleanly → strong single-feature discriminators. Use plotly for interactive filtering.",
    feLinks: "Interaction / Product Features · Group Aggregation Features",
    problemLinks: "Underfitting — High Bias · Wrong Model for Problem Structure",
    modelBenefit: "Reveals interactions for linear models to exploit via engineered features. GBMs find these internally.",
    competitionNote: "Use plotly's interactive version — you can brush/filter specific value ranges to isolate which feature combination separates a specific class.",
  },
  {
    name: "SHAP Summary Plot (Beeswarm)",
    group: "Feature–Target Relationship",
    lib: "shap",
    api: "shap.TreeExplainer(model).shap_values(X) → shap.summary_plot(shap_values, X)",
    priority: "🔁 Iterative",
    detects: "Feature importance with direction (positive/negative effect on prediction). Non-linear and interaction patterns. Individual outlier contributions. Which features most consistently push predictions up vs down.",
    action: "Features with near-zero SHAP magnitude → candidates for removal. Features with large SHAP and consistent direction → confirm importance and check for leakage. Wide spread with both colours → feature has complex non-linear effect.",
    feLinks: "SHAP Values (Feature Selection) · Interaction / Product Features · Boruta",
    problemLinks: "Misleading Feature Importance (MDI Bias) · Target Leakage · Overfitting — High Variance",
    modelBenefit: "GBM, Random Forest — TreeExplainer is fast. Any model via KernelExplainer (slower). The plot informs ALL subsequent feature engineering and selection decisions.",
    competitionNote: "The SHAP beeswarm is the most information-dense feature importance plot available. Run it after your first GBM baseline. It replaces the standard feature importance bar chart for every serious analysis.",
  },
  {
    name: "SHAP Dependence Plot",
    group: "Feature–Target Relationship",
    lib: "shap",
    api: "shap.dependence_plot('feature_name', shap_values, X, interaction_index='auto')",
    priority: "🔁 Iterative",
    detects: "The functional relationship between a single feature and its SHAP value (i.e., how its effect on prediction changes across its value range). Interaction effects with a second feature (coloured dots).",
    action: "Non-linear dependence → consider binning or spline transformation for linear models. U-shaped curve → squared term or binning. Interaction colouring by another feature → create explicit interaction feature.",
    feLinks: "Spline Features · Binning / Discretisation · Polynomial Features · Interaction / Product Features",
    problemLinks: "Underfitting — High Bias · Misleading Feature Importance (MDI Bias) · SHAP Values Misleading Due to Correlated Features",
    modelBenefit: "Linear Regression, MLP — the dependence plot reveals exactly what transformation is needed to linearise the feature-prediction relationship for these models.",
    competitionNote: "Run dependence plots for your top-10 SHAP features. For each non-linear relationship you find, engineering that transformation explicitly often gives a measurable CV improvement.",
  },
  {
    name: "Residual Plot (Regression Diagnostics)",
    group: "Feature–Target Relationship",
    lib: "matplotlib · seaborn · statsmodels",
    api: "residuals = y - model.predict(X); sns.residplot(x=y_pred, y=residuals, lowess=True)",
    priority: "🔁 Iterative",
    detects: "Systematic patterns in prediction errors (heteroscedasticity, non-linearity, skewed residuals). Fan shape = heteroscedasticity. Curved pattern = non-linearity not captured. Clustered residuals = missing group variable.",
    action: "Fan shape → log-transform target. Curved residual pattern → add polynomial/interaction features. Clustered residuals by a category → add group-level features. Normal random scatter = good fit.",
    feLinks: "Log Transform (target) · Polynomial Features · Group Aggregation Features · Spline Features",
    problemLinks: "Underfitting — High Bias · Wrong Model for Problem Structure · Metric Mismatch",
    modelBenefit: "Linear Regression (primary use), Logistic Regression, MLP — residual analysis is most directly actionable for gradient-based models whose assumptions are violated by these patterns.",
    competitionNote: "After every regression model fit, plot residuals vs predicted. Patterns in residuals are free feature engineering hints — they show exactly where your model is systematically wrong.",
  },

  // ─────────────────────────────────────────────────────────────
  // TIME SERIES EDA
  // ─────────────────────────────────────────────────────────────
  {
    name: "Time Series Line Plot + Rolling Mean",
    group: "Time Series EDA",
    lib: "pandas · matplotlib · plotly",
    api: "df.set_index('date')['value'].plot(); df['value'].rolling(30).mean().plot()",
    priority: "🥇 First",
    detects: "Overall trend (upward/downward), seasonality cycles, irregular spikes, change points, structural breaks, and regime changes in the series.",
    action: "Upward trend → include trend features (time since start, cumulative count). Clear seasonal cycles → include season/month/dayofweek features + Fourier terms. Spikes → investigate as outliers or events.",
    feLinks: "Date/Time Feature Extraction · Rolling Statistics · Lag Features · Seasonal Decomposition (STL) · Fourier / FFT Features",
    problemLinks: "Temporal / Future Leakage · Concept Drift · Train/Test Split Doesn't Reflect Production",
    modelBenefit: "GBMs on time-series tabular features, SARIMA, Prophet, LSTM — all require understanding the trend/seasonality structure revealed here before feature engineering.",
    competitionNote: "Always plot the raw series before any modelling. If there's a clear trend, your model needs trend features. If there's seasonality, it needs seasonal features. Skipping this step leads to systematic prediction errors.",
  },
  {
    name: "Seasonal Decomposition Plot (STL)",
    group: "Time Series EDA",
    lib: "statsmodels",
    api: "from statsmodels.tsa.seasonal import STL; res = STL(series).fit(); res.plot()",
    priority: "🥇 First",
    detects: "Separates the series into trend, seasonal, and residual components. Quantifies the strength of seasonality vs trend vs noise. Identifies whether seasonality is additive or multiplicative.",
    action: "Strong seasonal component → add Fourier features, month/week/day features. Large residual spikes → anomalous events to investigate. Trend component → add lag of trend or rate of change feature.",
    feLinks: "Seasonal Decomposition (STL) · Fourier / FFT Features · Lag Features · Rolling Statistics",
    problemLinks: "Concept Drift · Temporal / Future Leakage · Wrong Cross-Validation Strategy",
    modelBenefit: "Prophet (uses trend + seasonality directly), SARIMA, and any GBM/MLP on decomposed components.",
    competitionNote: "Decompose the target series on the training set. If trend and seasonality are strong, a simple baseline using only these components will be hard to beat.",
  },
  {
    name: "ACF and PACF Plots",
    group: "Time Series EDA",
    lib: "statsmodels",
    api: "plot_acf(series, lags=40); plot_pacf(series, lags=40)",
    priority: "🥈 Second",
    detects: "Significant autocorrelation lags. ACF: optimal MA order. PACF: optimal AR order. Seasonal spikes at lag k = period length. Slow decay in ACF = non-stationary series.",
    action: "PACF significant at lag 1, 7, 14 → include lags 1, 7, 14 as features. Slow ACF decay → series is non-stationary → apply differencing or include time index. Spikes at lag 12 → annual seasonality.",
    feLinks: "Lag Features · Rolling Statistics · ACF / PACF Features",
    problemLinks: "Wrong Cross-Validation Strategy (confirms need for TimeSeriesSplit) · Temporal / Future Leakage",
    modelBenefit: "ARIMA (directly determines p, d, q orders), LightGBM/XGBoost on lag features (ACF shows which lags to include), LSTM (confirms relevant sequence length).",
    competitionNote: "Read ACF/PACF before engineering lag features. If PACF is significant at lags 1, 7, 30 — include exactly those. Don't blindly include 1-30 lags.",
  },
  {
    name: "Lag Scatter Plot",
    group: "Time Series EDA",
    lib: "pandas · matplotlib",
    api: "pd.plotting.lag_plot(series, lag=1) / sns.scatterplot(x=series.shift(k), y=series)",
    priority: "🥈 Second",
    detects: "Direct visual confirmation of autocorrelation at a specific lag. Linear cluster on diagonal = strong positive autocorrelation. Circular scatter = no autocorrelation.",
    action: "Strong diagonal cluster at lag k → lag_k feature is highly predictive → include it. Circular scatter → random walk or no autocorrelation at that lag → that lag is not informative.",
    feLinks: "Lag Features · Rolling Statistics",
    problemLinks: "Temporal / Future Leakage · Wrong Cross-Validation Strategy",
    modelBenefit: "All time series models. Confirms which specific lag values to include as features for GBMs/MLP.",
    competitionNote: "Plot lag scatter for lags 1, 7, 14, 30 for any daily time series. The diagonal concentration visually confirms which lags are most predictive.",
  },
  {
    name: "Stationarity Test (ADF / KPSS) Plot",
    group: "Time Series EDA",
    lib: "statsmodels",
    api: "adfuller(series) / kpss(series) + plot series with rolling mean/std",
    priority: "🥈 Second",
    detects: "Whether the series is stationary (stable mean, variance, autocorrelation). ADF: null = non-stationary. KPSS: null = stationary. Rolling mean/std plot reveals non-stationarity visually.",
    action: "Non-stationary → difference the series OR include time-index / trend features. Log-transform may stabilise variance. Check whether first or second differencing achieves stationarity.",
    feLinks: "Lag Features · Rolling Statistics · Date/Time Feature Extraction",
    problemLinks: "Concept Drift · Wrong Cross-Validation Strategy · Train/Test Split Doesn't Reflect Production",
    modelBenefit: "ARIMA (must have stationary series), Prophet, VAR — stationarity is a hard requirement. GBMs with lag features handle non-stationarity through trend features.",
    competitionNote: "Non-stationarity in the target is one of the most common reasons time series models underperform. Always test for it before choosing a forecasting approach.",
  },

  // ─────────────────────────────────────────────────────────────
  // TEXT / NLP EDA
  // ─────────────────────────────────────────────────────────────
  {
    name: "Document Length Distribution",
    group: "Text / NLP EDA",
    lib: "matplotlib · seaborn",
    api: "df['text'].str.split().str.len().hist(bins=50)",
    priority: "🥇 First",
    detects: "Distribution of word counts per document. Very short texts (< 5 words). Very long texts that may need truncation. Bimodal length distribution suggesting two populations.",
    action: "Short texts → BernoulliNB may outperform MultinomialNB. Very long texts → truncate or chunk for BERT (512 token limit). Bimodal → text_length itself is a useful feature.",
    feLinks: "Text Statistics Features · Bag of Words · TF-IDF · Contextual Sentence Embeddings (BERT)",
    problemLinks: "Wrong Model for Problem Structure · Memory Errors / OOM on Large Datasets",
    modelBenefit: "All NLP models — length determines which model family is appropriate and what preprocessing is needed.",
    competitionNote: "In NLP competitions, text length is often one of the most predictive simple features. Add word count, char count, and avg word length as explicit features alongside embeddings.",
  },
  {
    name: "Word Frequency / Token Distribution Bar Chart",
    group: "Text / NLP EDA",
    lib: "matplotlib · Counter",
    api: "Counter(' '.join(df['text']).lower().split()).most_common(30) → bar chart",
    priority: "🥇 First",
    detects: "Most frequent tokens. Stop words still present. Domain-specific vocabulary. Vocabulary imbalance between classes. Extremely rare tokens.",
    action: "Stop words dominating → add to stop_words in TfidfVectorizer. Domain vocabulary present → train custom Word2Vec or use domain-specific embeddings. Class-specific vocabulary → TF-IDF will capture it.",
    feLinks: "Bag of Words · TF-IDF · N-grams · Word Embeddings",
    problemLinks: "Wrong Model for Problem Structure · Incorrect Data Types / Schema Drift",
    modelBenefit: "MultinomialNB, Logistic Regression, LinearSVC — guides TF-IDF configuration (stop words, min_df, max_df). Transformer models — confirms whether domain-specific fine-tuning is needed.",
    competitionNote: "Plot word frequency for each class separately. Words that appear much more in one class than others are the most discriminative features — TF-IDF will naturally weight them higher.",
  },
  {
    name: "Word Cloud",
    group: "Text / NLP EDA",
    lib: "wordcloud",
    api: "WordCloud(stopwords=STOPWORDS).generate(' '.join(df['text'])).to_image()",
    priority: "🥇 First",
    detects: "Most prominent terms visually. Domain vocabulary. Unexpected tokens (HTML tags, usernames, special characters still present). Class-distinctive vocabulary when generated per class.",
    action: "Unexpected tokens (URLs, HTML, digits) → add cleaning steps. Generate separately per class → dominant class-specific words reveal discriminative vocabulary to focus on.",
    feLinks: "Bag of Words · TF-IDF · Text Statistics Features",
    problemLinks: "Incorrect Data Types / Schema Drift · Label / Annotation Noise",
    modelBenefit: "All NLP models — confirms text cleaning is complete. Domain-specific words guide vocabulary selection.",
    competitionNote: "Generate per-class word clouds. The words that appear prominently in one class but not others are the feature engineering targets. Check for noise (HTML fragments, usernames).",
  },
  {
    name: "Class-Specific TF-IDF Top Terms",
    group: "Text / NLP EDA",
    lib: "sklearn",
    api: "TfidfVectorizer().fit_transform(docs); top terms per class via mean TF-IDF per class label",
    priority: "🥈 Second",
    detects: "Which specific terms most discriminate each class after IDF weighting. Removes the bias toward common words that raw word counts have. Reveals discriminative vocabulary.",
    action: "Highly discriminative terms confirm the signal in the data. Unexpected discriminative terms may reveal biases or leakage in annotations (e.g., author names correlated with topic).",
    feLinks: "TF-IDF · N-grams · Sentiment Score",
    problemLinks: "Target Leakage · Label / Annotation Noise · Class Imbalance",
    modelBenefit: "Logistic Regression, LinearSVC, MultinomialNB — these linear text models most directly benefit from understanding which terms drive class differences.",
    competitionNote: "If a highly discriminative term is a person's name or a source identifier, it's likely annotation leakage. This check has caught significant data quality issues in real NLP competitions.",
  },

  // ─────────────────────────────────────────────────────────────
  // IMAGE / CV EDA
  // ─────────────────────────────────────────────────────────────
  {
    name: "Sample Image Grid per Class",
    group: "Image / CV EDA",
    lib: "matplotlib",
    api: "fig, axes = plt.subplots(n_classes, k); [axes[i].imshow(sample) for each class]",
    priority: "🥇 First",
    detects: "Visual data quality: blurry images, mislabelled samples, incorrect class assignment, unexpected image content, data leakage patterns, image artefacts.",
    action: "Blurry samples → add blur augmentation during training for robustness. Mislabelled samples visible → use cleanlab to flag and investigate. Very similar images across classes → CNN embeddings + t-SNE to inspect separability.",
    feLinks: "Data Augmentation · CNN Embeddings / Transfer Learning · Pixel Statistics",
    problemLinks: "Label / Annotation Noise · Class Imbalance · Overfitting — High Variance",
    modelBenefit: "All CV models — visual inspection of samples informs augmentation strategy, which is the single largest performance driver in image competitions.",
    competitionNote: "Spend 15 minutes visually inspecting random samples per class. This has consistently revealed mislabelled data, unexpected image artifacts, and class boundary ambiguities that explain model failures.",
  },
  {
    name: "Mean/Std Image per Class",
    group: "Image / CV EDA",
    lib: "numpy · matplotlib",
    api: "np.mean(images_per_class, axis=0) → imshow; np.std(images_per_class, axis=0) → imshow",
    priority: "🥈 Second",
    detects: "Average appearance of each class (where are the relevant regions). High-std regions = areas of within-class variability. Low-std regions = consistent class-specific patterns.",
    action: "High-std uniform across image → no spatial regularity → augmentation is less constrained. Low-std in specific region → that region is the discriminative area → inform attention mechanisms or cropping strategy.",
    feLinks: "Pixel Statistics · CNN Embeddings / Transfer Learning · HOG Features",
    problemLinks: "Wrong Model for Problem Structure · Overfitting — High Variance",
    modelBenefit: "CNN — informs whether spatial attention modules would help. SVM/RF on pixel features — confirms which image regions to use for handcrafted features.",
    competitionNote: "Mean class images often reveal the key discriminative pattern (e.g., texture in specific region). This guides data augmentation — avoid augmentations that destroy the discriminative region.",
  },
  {
    name: "Pixel Intensity Distribution per Class",
    group: "Image / CV EDA",
    lib: "matplotlib · numpy",
    api: "plt.hist(images[class_k].flatten(), bins=50, alpha=0.5) → per class overlay",
    priority: "🥈 Second",
    detects: "Whether classes differ in brightness/contrast. Normalisation requirements (some channels might have very different ranges). Class-specific brightness patterns that are predictive.",
    action: "Very different intensity ranges across classes → normalise per-image (divide by mean pixel). Very similar intensity distributions → brightness is not a discriminative feature → focus augmentation elsewhere.",
    feLinks: "Pixel Statistics (mean, std, entropy) · Color Histograms · Data Augmentation",
    problemLinks: "Feature Scale Sensitivity for Distance Models · Overfitting — High Variance",
    modelBenefit: "CNN, SVM on pixel features, Random Forest on image statistics — normalisation decisions informed here directly affect model performance.",
    competitionNote: "Always normalise images to [0,1] or standardise per channel before training. Different intensity distributions across classes can artificially inflate model performance if not normalised.",
  },
  {
    name: "Image Embedding t-SNE / UMAP",
    group: "Image / CV EDA",
    lib: "sklearn · umap-learn · PyTorch/Keras",
    api: "embeddings = CNN_model(images) → TSNE(2).fit_transform(embeddings) → scatter(hue=class)",
    priority: "🥈 Second",
    detects: "Whether pre-trained CNN embeddings naturally cluster by class. Hard-to-classify examples at class boundaries. Confusion between specific pairs of classes. Mislabelled samples that appear in wrong cluster.",
    action: "Clean clusters → transfer learning will work well. Mixed clusters → need fine-tuning or a different backbone. Isolated points in wrong cluster → likely mislabelled samples → investigate.",
    feLinks: "CNN Embeddings / Transfer Learning · Data Augmentation",
    problemLinks: "Label / Annotation Noise · Wrong Model for Problem Structure · Class Imbalance",
    modelBenefit: "CNN, SVM on embeddings, Logistic Regression on embeddings — confirms whether pre-trained features are sufficient or fine-tuning is required.",
    competitionNote: "Run t-SNE on pre-trained embeddings before training. If classes are already well-separated, a simple classifier on frozen features will work. If not, plan for fine-tuning.",
  },
];

const ALL_GROUPS = ["All", ...Array.from(new Set(EDA.map(e => e.group)))];
const ALL_PRIORITIES = ["All", "🥇 First", "🥈 Second", "🥉 Third", "🔁 Iterative"];

function PriorityBadge({ p }) {
  const s = PRIORITY_STYLE[p] || { bg: "var(--border-subtle)", color: "#888", border: "#888" };
  return (
    <span style={{ display:"inline-block", padding:"2px 9px", borderRadius:"4px", fontSize:"0.64rem", fontWeight:700, letterSpacing:"0.04em", background:s.bg, color:s.color, border:`1px solid ${s.border}44`, whiteSpace:"nowrap" }}>{p}</span>
  );
}

function LibBadge({ text }) {
  return <span style={{ display:"inline-block", padding:"1px 6px", borderRadius:"3px", fontSize:"0.62rem", fontWeight:600, background:"#0e0e1c", color:"var(--text-secondary)", border:"1px solid var(--border-default)", whiteSpace:"nowrap", margin:"1px" }}>{text}</span>;
}

function Card({ icon, label, text, accent }) {
  return (
    <div style={{ padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:`1px solid ${accent}1e` }}>

      <div style={{ fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.09em", color:accent, textTransform:"uppercase", marginBottom:"5px" }}>{icon} {label}</div>
      <div style={{ fontSize:"0.79rem", color:"var(--text-primary)", lineHeight:1.6 }}>{text}</div>
    </div>
  );
}

function LinkChips({ text, accent }) {
  return (
    <div style={{ display:"flex", flexWrap:"wrap", gap:"4px" }}>
      {text.split("·").map(t => (
        <span key={t} style={{ display:"inline-block", padding:"2px 8px", borderRadius:"12px", fontSize:"0.65rem", fontWeight:600, background:accent+"18", color:accent, border:`1px solid ${accent}33` }}>{t.trim()}</span>
      ))}
    </div>
  );
}

const tdS = { padding:"10px 13px", verticalAlign:"middle", borderBottom:"1px solid var(--border-faint)", color:"var(--text-primary)", fontSize:"0.82rem" };
const thS = { padding:"10px 13px", textAlign:"left", fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", color:"var(--text-tertiary)", borderBottom:"2px solid var(--border-subtle)", background:"var(--bg-base)", position:"sticky", zIndex:5 };

function Row({ item, idx }) {
  const [open, setOpen] = useState(false);
  const accent = GROUP_ACCENT[item.group] || "#888";
  return (
    <>
      <tr
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background = "var(--bg-elevated)"}
        onMouseLeave={e => e.currentTarget.style.background = idx % 2 === 0 ? "var(--bg-surface)" : "var(--bg-surface)"}
        style={{ cursor:"pointer", background: idx%2===0?"var(--bg-surface)":"var(--bg-surface)", borderLeft:`3px solid ${accent}`, transition:"background 0.12s" }}
      >
        <td style={tdS}>{open?"▾":"▸"}</td>
        <td style={{ ...tdS, fontWeight:700, color:accent, fontSize:"0.88rem" }}>{item.name}</td>
        <td style={tdS}>{item.lib.split("·").slice(0,2).map(l => <LibBadge key={l} text={l.trim()} />)}</td>
        <td style={tdS}><PriorityBadge p={item.priority} /></td>
        <td style={{ ...tdS, fontFamily:"var(--font-mono)", fontSize:"0.68rem", color:"var(--text-tertiary)", maxWidth:"200px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{item.api}</td>
      </tr>
      {open && (
        <tr style={{ background:"var(--bg-surface)", borderLeft:`3px solid ${accent}` }}>
          <td colSpan={5} style={{ padding:"0 0 0 16px" }}>
            <div style={{ padding:"14px 16px 14px 0" }}>
              {/* Row 1: Detects + Action */}
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                <Card icon="🔍" label="What It Detects" text={item.detects} accent={accent} />
                <Card icon="⚡" label="Action When You Find It" text={item.action} accent="#4ade80" />
              </div>
              {/* Row 2: FE Links + Problem Links */}
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px", marginBottom:"8px" }}>
                <div style={{ padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:`1px solid #818cf822` }}>
                  <div style={{ fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.09em", color:"#818cf8", textTransform:"uppercase", marginBottom:"7px" }}>🔗 → Feature Engineering Table</div>
                  <LinkChips text={item.feLinks} accent="#818cf8" />
                </div>
                <div style={{ padding:"10px 13px", borderRadius:"6px", background:"var(--bg-surface)", border:`1px solid #f8717122` }}>
                  <div style={{ fontSize:"0.63rem", fontWeight:700, letterSpacing:"0.09em", color:"#f87171", textTransform:"uppercase", marginBottom:"7px" }}>⚠️ → ML Problems Table</div>
                  <LinkChips text={item.problemLinks} accent="#f87171" />
                </div>
              </div>
              {/* Row 3: Model Benefit + Competition Note */}
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px" }}>
                <Card icon="🤖" label="Models That Benefit" text={item.modelBenefit} accent="#22d3ee" />
                <Card icon="🏆" label="Competition Note" text={item.competitionNote} accent="#facc15" />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function EDA_app() {
  const [search, setSearch] = useState("");
  const [grp, setGrp] = useState("All");
  const [pri, setPri] = useState("All");

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return EDA.filter(e => {
      const mg = grp === "All" || e.group === grp;
      const mp = pri === "All" || e.priority === pri;
      const mq = !q || [e.name,e.group,e.lib,e.detects,e.action,e.feLinks,e.problemLinks,e.modelBenefit,e.competitionNote]
        .some(s => s.toLowerCase().includes(q));
      return mg && mp && mq;
    });
  }, [search, grp, pri]);

  const groups = useMemo(() => {
    const g = {};
    for (const e of filtered) { if (!g[e.group]) g[e.group] = []; g[e.group].push(e); }
    return g;
  }, [filtered]);

  const priCounts = useMemo(() => {
    const c = {};
    for (const e of EDA) c[e.priority] = (c[e.priority]||0)+1;
    return c;
  }, []);

  return (
    <div>

    <Header />
      
        <div style={{ fontFamily:"var(--font-body)", background:"var(--bg-base)", minHeight:"100vh", color:"var(--text-primary)" }}>
      

      {/* HEADER */}
      <div style={{ position:"sticky",top:"var(--header-h)",zIndex:10,background:"rgba(14,13,12,0.93)",backdropFilter:"blur(10px)",borderBottom:"1px solid var(--border-faint)",padding:"10px 20px 8px" }}>
        <div style={{ display:"flex",alignItems:"center",gap:"16px",marginBottom:"8px",flexWrap:"wrap" }}>
          <div>
            <div style={{ fontSize:"1.05rem",fontWeight:700,color:"#fff",letterSpacing:"-0.02em" }}>
              <span style={{ color:"#22d3ee" }}>EDA</span>
              <span style={{ color:"var(--text-dim)",margin:"0 6px" }}>·</span>
              <span style={{ color:"#4ade80" }}>Visualization</span>
              <span style={{ color:"var(--text-dim)",margin:"0 6px" }}>·</span>
              <span style={{ color:"#818cf8" }}>→ Feature Engineering</span>
              <span style={{ color:"var(--text-dim)",margin:"0 6px" }}>·</span>
              <span style={{ color:"#f87171" }}>→ ML Problems</span>
            </div>
            <div style={{ fontSize:"0.63rem",color:"var(--text-dim)",letterSpacing:"0.05em",marginTop:"1px" }}>
              {EDA.length} techniques · Tabular · Text · Image · Time Series · click any row to expand · cross-referenced to both previous tables
            </div>
          </div>
          <div style={{ flex:1,minWidth:"180px",maxWidth:"300px",marginLeft:"auto" }}>
            <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search: detects, fixes, models, tools…"
              style={{ width:"100%",background:"#0c0c1c",border:"1px solid var(--border-default)",borderRadius:"6px",padding:"7px 11px",color:"var(--text-primary)",fontSize:"0.79rem",outline:"none" }} />
          </div>
          <div style={{ fontSize:"0.7rem",color:"var(--text-dim)",whiteSpace:"nowrap" }}>{filtered.length} shown</div>
        </div>

        {/* Priority filter */}
        <div style={{ display:"flex",gap:"5px",flexWrap:"wrap",marginBottom:"5px" }}>
          <span style={{ fontSize:"0.62rem",color:"var(--text-dim)",alignSelf:"center",marginRight:"4px" }}>WHEN TO RUN:</span>
          {ALL_PRIORITIES.map(p => {
            const active = pri === p;
            const s = PRIORITY_STYLE[p] || { color:"#666",border:"#666",bg:"#111" };
            return <button key={p} onClick={()=>setPri(p)} style={{ padding:"3px 10px",borderRadius:"4px",fontSize:"0.67rem",fontWeight:active?700:400,border:active?`1px solid ${s.border}`:"1px solid var(--border-default)",background:active?s.bg:"transparent",color:active?s.color:"var(--text-tertiary)" }}>
              {p}{p!=="All"&&priCounts[p]?` (${priCounts[p]})`:""}</button>;
          })}
        </div>

        {/* Group filter */}
        <div style={{ display:"flex",gap:"4px",flexWrap:"wrap" }}>
          <span style={{ fontSize:"0.62rem",color:"var(--text-dim)",alignSelf:"center",marginRight:"4px" }}>CATEGORY:</span>
          {ALL_GROUPS.map(g => {
            const active = grp === g;
            const accent = GROUP_ACCENT[g] || "#666";
            return <button key={g} onClick={()=>setGrp(g)} style={{ padding:"3px 9px",borderRadius:"4px",fontSize:"0.65rem",fontWeight:active?700:400,border:active?`1px solid ${accent}`:"1px solid var(--border-default)",background:active?`${accent}22`:"transparent",color:active?accent:"var(--text-tertiary)" }}>{g}</button>;
          })}
        </div>
      </div>

      {/* LEGEND */}
      <div style={{ margin:"10px 20px 0",padding:"8px 14px",borderRadius:"6px",background:"var(--bg-surface)",border:"1px solid var(--border-subtle)",fontSize:"0.73rem",color:"var(--text-secondary)",display:"flex",gap:"20px",flexWrap:"wrap" }}>
        <span><span style={{ color:"#4ade80",fontWeight:700 }}>🥇 First</span> — Run before any model. These are non-negotiable.</span>
        <span><span style={{ color:"#818cf8",fontWeight:700 }}>🥈 Second</span> — After first pass. Deepen understanding of features.</span>
        <span><span style={{ color:"#facc15",fontWeight:700 }}>🥉 Third</span> — Situational. Use when specific questions arise.</span>
        <span><span style={{ color:"#c084fc",fontWeight:700 }}>🔁 Iterative</span> — After model fitting. Diagnose model behaviour.</span>
        <span style={{ marginLeft:"auto" }}><span style={{ color:"#818cf8",fontWeight:700 }}>→ FE Table</span> chips link to feature engineering reference</span>
        <span><span style={{ color:"#f87171",fontWeight:700 }}>→ Problems Table</span> chips link to ML problems reference</span>
      </div>

      {/* TABLE */}
      <div style={{ overflowX:"auto",marginTop:"10px" }}>
        <table style={{ width:"100%",borderCollapse:"collapse",minWidth:"860px" }}>
          <thead>
            <tr>
              <th style={{ ...thS,width:"24px" }}></th>
              <th style={thS}>Technique</th>
              <th style={thS}>Library</th>
              <th style={thS}>When to Run</th>
              <th style={{ ...thS,maxWidth:"200px" }}>API</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([group, items]) => (
              <>
                <tr key={group+"_hdr"}>
                  <td colSpan={5} style={{ padding:"8px 14px 5px",fontSize:"0.65rem",fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",color:GROUP_ACCENT[group]||"#5a5a9a",background:"var(--bg-surface)",borderTop:"2px solid var(--border-faint)",borderBottom:"1px solid var(--border-faint)" }}>
                    ▪ {group} <span style={{ fontWeight:400,color:"#1e1e40",marginLeft:6 }}>({items.length})</span>
                  </td>
                </tr>
                {items.map((item,i) => <Row key={item.name} item={item} idx={i} />)}
              </>
            ))}
            {filtered.length===0 && <tr><td colSpan={5} style={{ padding:"60px",textAlign:"center",color:"var(--text-dim)" }}>No techniques match. Try different search or reset filters.</td></tr>}
          </tbody>
        </table>
      </div>

      <div style={{ padding:"18px",textAlign:"center",fontSize:"0.63rem",color:"var(--border-subtle)",borderTop:"1px solid var(--bg-elevated)" }}>
        Covers matplotlib · seaborn · plotly · missingno · shap · ydata-profiling · sweetviz · dtale · autoviz · statsmodels · sklearn · umap-learn · wordcloud · missingno · scipy
      </div>
    </div></div>
  );
}