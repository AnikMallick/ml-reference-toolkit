import { useState, useMemo } from "react";
import Header from "./Header";
const SEVERITY_STYLE = {
  "🔴 Critical": { bg: "#3a0a0a", color: "#ff6b6b", border: "#ff6b6b" },
  "🟠 High":     { bg: "#2a1a0a", color: "#fb923c", border: "#fb923c" },
  "🟡 Medium":   { bg: "#2a2a0a", color: "#facc15", border: "#facc15" },
  "🔵 Low":      { bg: "#0a1a2a", color: "#60a5fa", border: "#60a5fa" },
};

const GROUP_ACCENT = {
  "Data Quality":         "#f87171",
  "Data Leakage":         "#ff4d4d",
  "Overfitting":          "#fb923c",
  "Underfitting":         "#facc15",
  "Validation & CV":      "#4ade80",
  "Metrics & Evaluation": "#34d399",
  "Class Imbalance":      "#c084fc",
  "Distribution Shift":   "#818cf8",
  "Training Instability": "#f472b6",
  "Competition-Specific": "#22d3ee",
  "Interpretability":     "#a78bfa",
};

const PROBLEMS = [
  // ═══════════════════════════════════════════════
  // DATA QUALITY
  // ═══════════════════════════════════════════════
  {
    name: "Target Leakage",
    group: "Data Leakage",
    severity: "🔴 Critical",
    symptom: "Suspiciously high CV/test accuracy (e.g. 99% on a hard problem). Model collapses to near-random performance in production. Feature importances dominated by a single feature.",
    rootCause: "A feature in your training data is derived from or correlated with the future target. Common sources: post-event flags encoded before event-time, derived features computed on the full dataset before splitting, or ID columns that encode target information.",
    diagnose: "Plot feature importances — a dominant single feature is a red flag. Try removing the top feature and re-evaluate. Check the temporal relationship between every feature and the target. Use adversarial validation on feature subsets.",
    fix: "Strictly reconstruct the prediction point in time. Encode features only using information available before the prediction moment. Use imblearn.Pipeline or sklearn.Pipeline so transformers never see test targets. Audit every feature with domain knowledge.",
    tools: "sklearn.Pipeline · imblearn.Pipeline · pandas (temporal checks) · shap (importance audit)",
    affectedModels: "ALL models — leaky features produce artificially high performance for every algorithm. The model with the cleanest CV score (not the highest) is usually least affected.",
    competition: "Extremely common in Kaggle competitions. A public LB score dramatically higher than expected is the #1 signal. Check aggregation features and any feature correlated > 0.95 with target.",
  },
  {
    name: "Preprocessing Leakage (Fit on Full Data)",
    group: "Data Leakage",
    severity: "🔴 Critical",
    symptom: "CV score is optimistic but test score drops. Scaler, imputer, or encoder was fit on train+test before splitting. Hard to notice because the drop is subtle.",
    rootCause: "Calling fit_transform() on the entire dataset before train/test split leaks test distribution statistics (mean, std, min, max, vocabulary) into the training process.",
    diagnose: "Audit every preprocessing step — trace whether any transformer was fit before the split. Pay attention to: StandardScaler, imputers, target encoders, TF-IDF vocabulary, label encoders.",
    fix: "Always: split first, then fit transformers on train only, transform both. Wrap everything in sklearn.Pipeline or imblearn.Pipeline, then run cross_validate() on the Pipeline object — this guarantees per-fold fitting.",
    tools: "sklearn.Pipeline · sklearn.compose.ColumnTransformer · imblearn.pipeline.make_pipeline",
    affectedModels: "All models. Linear models and SVMs most distorted because they rely on accurate scale statistics.",
    competition: "A classic Kaggle mistake. Always wrap preprocessing in a Pipeline. If CV score is 0.02+ higher than local holdout, suspect this.",
  },
  {
    name: "Temporal / Future Leakage",
    group: "Data Leakage",
    severity: "🔴 Critical",
    symptom: "Model performs perfectly on historical backtest but fails on live deployment. Rolling statistics or lag features accidentally include values from the future.",
    rootCause: "Time series data used without proper temporal ordering. Rolling windows, group aggregations, or target encoding computed without shifting. Test folds contain dates earlier than train folds in random CV.",
    diagnose: "Sort data by time and manually inspect a few rows: does the feature value at time T use information from T+1, T+2, …? Check all .rolling(), .expanding(), .groupby().transform() calls for missing .shift(1).",
    fix: "Always .shift(1) before computing rolling/expanding statistics. Use TimeSeriesSplit for CV on time-indexed data. Reconstruct features with a strict cutoff date. Never use random shuffle in time-ordered data.",
    tools: "pandas .shift() · sklearn.TimeSeriesSplit · tsfresh (built-in temporal handling)",
    affectedModels: "All models trained on time series. GBMs are often the worst offenders because they're so powerful they exploit even tiny leakage.",
    competition: "Sort your dataframe by time and re-check every feature's computation. If your leaderboard gap is huge, this is the first place to look.",
  },
  {
    name: "Group Leakage in Cross-Validation",
    group: "Data Leakage",
    severity: "🟠 High",
    symptom: "CV score is high but generalisation to new groups (users, patients, stores) is poor. Same entity appears in both train and validation fold.",
    rootCause: "Using KFold or StratifiedKFold on data where rows belonging to the same entity (patient, user, product) must stay together. The model memorises entity-level patterns and the validation fold leaks them.",
    diagnose: "Check whether entities (user IDs, patient IDs) are represented across multiple folds. Compute per-entity target mean — does it vary within folds or across folds?",
    fix: "Use GroupKFold(groups=df['entity_id']) or StratifiedGroupKFold. Ensure all rows for one entity are entirely in one fold. This gives a true out-of-group generalisation estimate.",
    tools: "sklearn.GroupKFold · sklearn.StratifiedGroupKFold · sklearn.GroupShuffleSplit",
    affectedModels: "All models. Especially problematic for KNN, MLP, tree models that memorise per-entity patterns.",
    competition: "In medical, NLP, or user-behavior competitions, check whether the eval is 'new users/patients' — if so, you MUST use GroupKFold.",
  },
  {
    name: "Missing Value Mishandling",
    group: "Data Quality",
    severity: "🟠 High",
    symptom: "Model errors on predict(), dtype mismatch in pipeline, performance drop on new data where missingness patterns differ from training.",
    rootCause: "NaN imputed before split (leakage), imputer not included in Pipeline (causes predict-time errors), wrong imputation strategy (e.g. mean imputation on skewed data), or MNAR missingness ignored.",
    diagnose: "df.isnull().mean().sort_values() — inspect missingness rate. Are values Missing At Random (MAR), Completely At Random (MCAR), or Not At Random (MNAR)? Is missingness itself predictive?",
    fix: "1) Add a binary missingness indicator before imputing (for MNAR). 2) Choose strategy: median for skewed, mean for symmetric, most_frequent for categorical, model-based (IterativeImputer) for complex MAR. 3) Put imputer inside Pipeline. 4) Never impute before splitting.",
    tools: "sklearn.SimpleImputer · sklearn.IterativeImputer · sklearn.KNNImputer · feature_engine.MissingIndicator · missingno",
    affectedModels: "Linear models, SVMs, KNN, MLP — require no NaNs. HistGradientBoosting, LightGBM, XGBoost — handle NaN natively. Wrong imputation adds noise to all models.",
    competition: "Always check if test set has more/different missingness than train. HistGradientBoosting is a good default precisely because it handles NaN natively.",
  },
  {
    name: "Outlier Contamination",
    group: "Data Quality",
    severity: "🟡 Medium",
    symptom: "Linear model coefficients dominated by a small number of extreme values. Loss doesn't converge. Mean far from median. Model produces unrealistic predictions at the extremes.",
    rootCause: "Recording errors, sensor faults, data entry mistakes, or legitimately extreme but rare events. Extreme values pull OLS regression disproportionately due to the squared loss.",
    diagnose: "Box plots, Z-score |z| > 3, IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR). Check feature distributions with histograms. Inspect scatter plots of features vs target.",
    fix: "Diagnose first: are they errors or genuine extremes? If errors: cap/clip (Winsorisation) at 1st/99th percentile. If genuine: use robust models (HuberRegressor, QuantileRegressor) or robust scalers. Never blindly delete outliers without domain understanding.",
    tools: "scipy.stats.zscore · sklearn.RobustScaler · sklearn.HuberRegressor · feature_engine.Winsorizer · IsolationForest",
    affectedModels: "Linear Regression, Lasso, Ridge — heavily affected. GBM, Random Forest — moderately robust. HuberRegressor, QuantileRegressor — robust by design.",
    competition: "Winsorisation at 1/99% is a safe default. Log-transforming skewed targets removes the squared-loss outlier problem entirely.",
  },
  {
    name: "Duplicated Rows / Near-Duplicates",
    group: "Data Quality",
    severity: "🟡 Medium",
    symptom: "CV variance is suspiciously low. Model perfectly memorises training patterns. Removing some training rows has no effect. Test performance mysteriously aligns with training rows.",
    rootCause: "Data collection pipelines often produce duplicate records. ETL joins can create row explosions. Test set rows may also exist in the training set, directly leaking answers.",
    diagnose: "df.duplicated().sum() for exact duplicates. For near-duplicates: compute cosine similarity or edit distance on key columns. Check if any test rows exist verbatim in training (exact match on all features).",
    fix: "df.drop_duplicates() for exact. For near-duplicates: deduplication by entity + time window. For train/test overlap: remove overlapping rows from training — keeping them causes severe label leakage if the test has a target.",
    tools: "pandas df.duplicated() · datasketch (MinHash LSH for near-duplicates) · recordlinkage",
    affectedModels: "KNN — catastrophically affected (exact neighbour = exact answer). Tree models — memorise duplicates, inflate importances. All models benefit from deduplication.",
    competition: "Check for train/test overlap before doing anything. In competition datasets, test rows appearing in train is a form of target leakage if the public targets are leaked.",
  },
  {
    name: "Label / Annotation Noise",
    group: "Data Quality",
    severity: "🟠 High",
    symptom: "Training loss decreases but validation loss oscillates. High-confidence predictions on borderline examples are wrong. Model performs unexpectedly well on corrupted labels (memorisation).",
    rootCause: "Human annotation errors, ambiguous class boundaries, crowdsourcing disagreement, systematic annotator bias, or intentional label obfuscation in competition test sets.",
    diagnose: "Inspect low-confidence predictions on training data — they're often mislabelled. Train a strong model and inspect examples where the model disagrees with the label strongly. Use cleanlab to detect noisy labels.",
    fix: "Soft labels / label smoothing (replace 0/1 with 0.05/0.95). Use cleanlab to find and filter noisy examples. Ensemble multiple annotators. Use noise-robust loss functions (symmetric cross-entropy, Taylor cross-entropy). Confident learning (co-training).",
    tools: "cleanlab · label-studio · sklearn label_smoothing (MLP) · torch label smoothing loss",
    affectedModels: "MLP/Deep Learning — severely affected (memorises noise). GBM with low min_child_weight — affected. Strong regularisation and early stopping mitigate label noise for all models.",
    competition: "cleanlab is a Kaggle secret weapon. Training a model and using it to flag its most confused training examples often reveals mislabelled rows that are hurting your CV score.",
  },
  {
    name: "Incorrect Data Types / Schema Drift",
    group: "Data Quality",
    severity: "🟡 Medium",
    symptom: "Model crashes at predict() time. Object dtype columns parsed as numeric or vice versa. Categories in production not seen during training. Pipeline breaks on new data.",
    rootCause: "Pandas inferred dtypes incorrectly. Numeric column stored as string due to a stray character. Categories in test not present in train encoder vocabulary. Schema changes between data versions.",
    diagnose: "df.dtypes + df.describe(include='all'). For categoricals: check unique values in test not in train. Assert column dtypes at pipeline entry. Monitor schema with great_expectations.",
    fix: "Explicit dtype casting at ingestion. Use sklearn OrdinalEncoder(handle_unknown='use_encoded_value') or handle_unknown='ignore' in OHE. Add schema validation to pipeline entry. Log and alert on dtype mismatches.",
    tools: "pandas pd.api.types · sklearn handle_unknown · great_expectations · pandera (schema validation)",
    affectedModels: "All models fail hard on dtype errors. Tree models silently accept wrong ordinal encoding and produce wrong splits.",
    competition: "Always check that test set categories align with train. A single unseen category in test will crash a pipeline if handle_unknown is not set.",
  },

  // ═══════════════════════════════════════════════
  // OVERFITTING
  // ═══════════════════════════════════════════════
  {
    name: "Overfitting — High Variance",
    group: "Overfitting",
    severity: "🟠 High",
    symptom: "Train score ≫ validation score. Large gap between train loss and val loss. CV fold scores have high variance. Model memorises noise.",
    rootCause: "Model complexity too high relative to dataset size. Too many features, too deep trees, too many epochs, insufficient regularisation, small training set.",
    diagnose: "Plot learning curves (train vs val score vs n_samples). Large gap = overfitting. Plot loss curves vs epochs. Compute CV score standard deviation — high std = high variance.",
    fix: "Add regularisation (L1/L2, dropout, weight decay). Reduce model complexity (fewer layers, shallower trees, lower max_features). Get more training data. Increase min_samples_leaf. Use early stopping. Add noise/augmentation. Pruning/feature selection.",
    tools: "sklearn learning_curve() · sklearn early_stopping · optuna (regularisation tuning) · dropout (PyTorch/Keras)",
    affectedModels: "Decision Trees (unconstrained), MLP without dropout, KNN with k=1, GBM with low min_child_weight. Random Forest and regularised GBMs are more robust.",
    competition: "The single most common reason for a gap between public and private LB. Always validate your regularisation with OOF predictions before trusting your score.",
  },
  {
    name: "Overfitting to Public Leaderboard (LB Probing)",
    group: "Competition-Specific",
    severity: "🔴 Critical",
    symptom: "Public LB score keeps improving with each submission but private LB score is worse than earlier submissions. CV score diverges from public LB.",
    rootCause: "Iteratively tuning hyperparameters or feature engineering decisions to maximise public LB score treats the public test set as a validation set, overfitting to the specific sample used for public scoring.",
    diagnose: "Compare your best-CV-score model vs best-public-LB-score model on holdout. If they differ, you're overfitting the LB. Track how many submissions informed each decision.",
    fix: "Trust your local CV score over public LB. Only use the LB to sanity-check (not to tune). Build a robust local CV that correlates with private LB. Use fewer submissions per day. Prefer the model with best CV score for final submission.",
    tools: "Local CV framework · sklearn.model_selection · neptune.ai / wandb (experiment tracking)",
    affectedModels: "No specific model — this is a process problem, not a model problem.",
    competition: "The cardinal sin of Kaggle. The gold medal models are almost always the ones with the best CV correlation, not the highest public LB. Always submit your best-CV model as one of your final two.",
  },
  {
    name: "Overfitting the Validation Set (Repeated Tuning)",
    group: "Overfitting",
    severity: "🟠 High",
    symptom: "Each round of hyperparameter tuning improves validation score, but performance on a truly held-out test set is lower than expected. Over-optimistic evaluation.",
    rootCause: "Using the same validation set repeatedly for hyperparameter tuning causes the tuning process to find settings that work well for that specific random sample, not the underlying distribution.",
    diagnose: "Compare val score vs a completely untouched holdout set. If they differ by > 0.01 after extensive tuning, you've overfit the val set. Check how many tuning rounds were done on the same fold.",
    fix: "Use nested cross-validation (inner CV for tuning, outer CV for evaluation). Maintain a completely untouched holdout set used only for final evaluation. Use Bayesian optimisation with a budget limit rather than grid search.",
    tools: "sklearn.cross_validate + cross_val_score (nested) · optuna (Bayesian) · hyperopt",
    affectedModels: "All models. Models with many hyperparameters (MLP, GBM) are most at risk.",
    competition: "Use a 5-fold OOF (out-of-fold) CV consistently for all decisions. Never look at the same test holdout more than 1-2 times.",
  },

  // ═══════════════════════════════════════════════
  // UNDERFITTING
  // ═══════════════════════════════════════════════
  {
    name: "Underfitting — High Bias",
    group: "Underfitting",
    severity: "🟡 Medium",
    symptom: "Both train and validation scores are low. Learning curve shows both train and val scores converge to a poor value. More data doesn't help. Model predictions cluster around the mean.",
    rootCause: "Model too simple for the problem's complexity. Excessive regularisation. Insufficient features or poor feature engineering. Wrong model family for the task.",
    diagnose: "Learning curve: if train score is already low, the problem is bias not variance. Check residual plots for systematic patterns (indicates missing signal). Check if adding features improves both train and val.",
    fix: "Use a more complex model. Reduce regularisation strength. Add more features / interactions. Increase tree depth, add hidden layers. Engineer features that capture non-linear relationships. Move from linear model to GBM.",
    tools: "sklearn learning_curve() · matplotlib residual plots",
    affectedModels: "Over-regularised Ridge/Lasso, shallow Decision Trees with max_depth=2, LogisticRegression on non-linear data without feature expansion.",
    competition: "Underfitting is less common in competitions but happens when you start with a shallow baseline and forget to increase complexity. Check train score first.",
  },
  {
    name: "Wrong Model for Problem Structure",
    group: "Underfitting",
    severity: "🟡 Medium",
    symptom: "Model performs far below domain baselines. Adding data or features doesn't help. Residuals have clear structure the model can't capture.",
    rootCause: "Model family doesn't match the problem. Using linear regression on multiplicative relationships. Using GBM on image pixels. Using tabular model on sequential data. Ignoring hierarchical / grouped structure.",
    diagnose: "Plot predictions vs actuals. Systematic under/over-prediction in certain ranges = wrong model. Compare to a domain-appropriate simple baseline (e.g. SARIMA for time series).",
    fix: "Match model to data structure: LSTM/Transformer for sequences, CNN for images, GBM for tabular, mixed-effects models for hierarchical data. Try log-transforming the target for multiplicative relationships.",
    tools: "sklearn model zoo · PyTorch/Keras for deep models · statsmodels for time series",
    affectedModels: "Linear models are most commonly the 'wrong choice' when applied to inherently non-linear or structured data without feature engineering.",
    competition: "Read problem description carefully. If data has spatial, temporal, or graph structure, exploit it explicitly — don't treat everything as an i.i.d. tabular problem.",
  },

  // ═══════════════════════════════════════════════
  // CLASS IMBALANCE
  // ═══════════════════════════════════════════════
  {
    name: "Class Imbalance",
    group: "Class Imbalance",
    severity: "🟠 High",
    symptom: "Model predicts majority class almost always. High accuracy but near-zero recall on minority class. Confusion matrix shows all predictions in one column. ROC-AUC seems fine but PR-AUC is poor.",
    rootCause: "Class distribution is skewed (e.g., 99% negative, 1% positive). Standard loss functions optimise overall accuracy, which means predicting the majority class dominates the gradient signal.",
    diagnose: "y.value_counts(normalize=True). Check confusion matrix — are any classes never predicted? Plot PR curve (not just ROC). Compare accuracy vs balanced_accuracy and F1.",
    fix: "In order of preference: (1) class_weight='balanced' in model — easiest, no data change. (2) Threshold tuning from ROC/PR curve. (3) SMOTE on train fold only inside Pipeline. (4) Undersample majority. (5) Reframe as anomaly detection if < 1% minority. (6) Use focal loss for deep models.",
    tools: "sklearn class_weight · imbalanced-learn SMOTE/ADASYN · sklearn threshold_from_estimator · imblearn.pipeline",
    affectedModels: "Logistic Regression, SVM, MLP — severely biased without class_weight. Random Forest, GBM — somewhat robust but still benefit from class_weight. Never resample before splitting.",
    competition: "Never use accuracy as your metric on imbalanced data. Use F1, ROC-AUC, or PR-AUC. class_weight='balanced' is always the first fix to try — it's free and often sufficient.",
  },
  {
    name: "SMOTE Applied Before Train/Test Split",
    group: "Class Imbalance",
    severity: "🔴 Critical",
    symptom: "CV score looks fantastic, but holdout performance is significantly worse. Synthetic minority samples from test data leaked into training through interpolation.",
    rootCause: "SMOTE interpolates between real minority samples. If applied before splitting, it can create synthetic samples between training AND test-set points, leaking test distribution into training.",
    diagnose: "Check code order: is SMOTE/oversampling called before train_test_split or outside the CV pipeline? Is the resampler inside an imblearn.Pipeline?",
    fix: "Apply SMOTE only inside the training fold, using imblearn.pipeline.make_pipeline([SMOTE(), model]). Then pass this Pipeline to cross_validate(). The pipeline will resample only training data per fold.",
    tools: "imblearn.pipeline.make_pipeline · SMOTE · ADASYN · RandomOverSampler",
    affectedModels: "All models — the pipeline fix is required for any resampling technique.",
    competition: "A very easy mistake to make and very hard to detect. If your imbalanced problem CV score is suspiciously good, check for this.",
  },
  {
    name: "Threshold Set at Default 0.5",
    group: "Class Imbalance",
    severity: "🟡 Medium",
    symptom: "Model is evaluated as poor on F1/recall but has reasonable ROC-AUC. Business requirement is high recall (fraud/medical). Default threshold doesn't match cost structure.",
    rootCause: "sklearn defaults to 0.5 probability threshold for all classifiers. In imbalanced or asymmetric-cost settings, the optimal threshold is almost never 0.5.",
    diagnose: "Plot ROC curve and Precision-Recall curve. Identify the threshold that maximises F1 (or minimises your specific cost function). Use sklearn.metrics.precision_recall_curve.",
    fix: "Use precision_recall_curve to find the threshold that maximises F1 (or F-beta for asymmetric costs). Or use sklearn TunedThresholdClassifierCV for CV-based threshold selection. Document the chosen threshold and monitor it.",
    tools: "sklearn.metrics.precision_recall_curve · sklearn.TunedThresholdClassifierCV · sklearn.metrics.roc_curve",
    affectedModels: "All classifiers that output probabilities. Most impactful for Logistic Regression, GBM, Random Forest, MLP.",
    competition: "In binary classification competitions scored on F1, the optimal threshold is often not 0.5. Always tune it on OOF predictions.",
  },

  // ═══════════════════════════════════════════════
  // VALIDATION & CV
  // ═══════════════════════════════════════════════
  {
    name: "Wrong Cross-Validation Strategy",
    group: "Validation & CV",
    severity: "🟠 High",
    symptom: "CV score doesn't correlate with public LB or test performance. Local CV says model A > B but production disagrees. High variance across folds.",
    rootCause: "Using KFold on time-series data (future leakage). Using KFold on grouped data (entity leakage). Using StratifiedKFold on regression. Using too few folds (k=2). Wrong target stratification.",
    diagnose: "Match CV strategy to data structure: Is data time-ordered? → TimeSeriesSplit. Do rows share an entity? → GroupKFold. Is target binary/categorical? → StratifiedKFold. Is there a combination? → StratifiedGroupKFold.",
    fix: "Time series → TimeSeriesSplit or Walk-Forward CV. Groups → GroupKFold(n_splits=5, groups=entity_id). Classification → StratifiedKFold(n_splits=5). Small data → RepeatedStratifiedKFold. Regression → KFold.",
    tools: "sklearn.TimeSeriesSplit · sklearn.GroupKFold · sklearn.StratifiedGroupKFold · sklearn.RepeatedStratifiedKFold",
    affectedModels: "All models — CV strategy determines what you think your model's performance is. Wrong strategy misleads all subsequent decisions.",
    competition: "The most underrated problem in competitions. Before writing a single model, sketch out the correct CV strategy for the problem structure. A good CV that correlates with the private LB is worth more than any model improvement.",
  },
  {
    name: "High CV Fold Variance",
    group: "Validation & CV",
    severity: "🟡 Medium",
    symptom: "Cross-validation scores vary wildly across folds (e.g. fold 1: 0.92, fold 5: 0.71). Hard to compare models. Hyperparameter tuning results are noisy.",
    rootCause: "Too few folds (k=2 or k=3). Small dataset. Class distribution not stratified. A few very informative or very noisy folds exist due to sampling.",
    diagnose: "Report mean ± std across folds. If std > 0.02 for a large dataset, something is wrong. Plot per-fold score distribution. Check if target distribution differs between folds.",
    fix: "Increase k (use k=5 or k=10). Use StratifiedKFold to ensure consistent class ratios. Use RepeatedKFold to average over multiple random splits. On very small datasets, use Leave-One-Out CV.",
    tools: "sklearn.RepeatedStratifiedKFold · sklearn.cross_val_score · sklearn.cross_validate",
    affectedModels: "All models — high fold variance makes it impossible to reliably compare models or tune hyperparameters.",
    competition: "With high fold variance, you can't trust that model A is better than model B. Use repeated CV or more folds before drawing conclusions.",
  },
  {
    name: "Train/Validation Distribution Mismatch",
    group: "Validation & CV",
    severity: "🟠 High",
    symptom: "CV score is good but the model fails on new data batches. The test data distribution has shifted from training. Certain feature ranges are never seen in training.",
    rootCause: "Random splits create folds where some feature distributions differ. In real deployment, new data may have different seasonal patterns, user demographics, or data collection conditions.",
    diagnose: "Run the two-sample KS test or adversarial validation (train a classifier to distinguish train vs test — if it achieves high AUC, distributions differ).",
    fix: "Adversarial validation to identify differing features → remove or transform them. Align CV splits to match production data distribution. Use importance weighting (covariate shift correction) to up-weight test-similar training samples.",
    tools: "scipy.stats.ks_2samp · sklearn adversarial validation (custom) · alibi-detect",
    affectedModels: "All models. Models with low bias (GBM, deep nets) are especially sensitive because they can overfit to training distribution nuances.",
    competition: "Run adversarial validation as a standard step: train a binary classifier on train=0 vs test=1. If AUC > 0.6, your distributions differ and your CV will be optimistic.",
  },
  {
    name: "Nested CV Neglect (Hyperparameter Tuning Bias)",
    group: "Validation & CV",
    severity: "🟡 Medium",
    symptom: "Final reported model performance is optimistic because the same CV folds used for tuning were used for evaluation. Performance drops when deployed to truly new data.",
    rootCause: "Using cross-validation both for hyperparameter selection and final performance estimation. The outer loop and inner loop are conflated, which inflates the reported performance estimate.",
    diagnose: "Compare the CV score from the tuning process to a score from an outer held-out set that was never used in tuning. If they differ significantly, the tuning process has overfit the CV.",
    fix: "Use nested cross-validation: inner k-fold for hyperparameter search, outer k-fold for unbiased performance estimation. Or maintain a completely untouched holdout set used only for final evaluation.",
    tools: "sklearn.cross_val_score (outer loop) + sklearn.GridSearchCV / optuna (inner loop) · sklearn.HalvingGridSearchCV",
    affectedModels: "All models with many hyperparameters: MLP, GBM, SVM (C, gamma), Random Forest.",
    competition: "Competitions have a natural nested CV structure via the private LB. Your local CV is the outer loop — never let your LB submissions inform your CV decisions.",
  },

  // ═══════════════════════════════════════════════
  // METRICS & EVALUATION
  // ═══════════════════════════════════════════════
  {
    name: "Wrong Evaluation Metric",
    group: "Metrics & Evaluation",
    severity: "🟠 High",
    symptom: "Model performs well on accuracy but fails in production (imbalanced data). MSE-optimised model produces bad mean absolute error. Metric doesn't match business objective.",
    rootCause: "Default metric (accuracy, MSE) selected without thinking about the problem structure. Imbalanced classification → accuracy is misleading. Heavy-tailed targets → MSE penalises large errors disproportionately.",
    diagnose: "Ask: does the metric reflect business cost? Is the data balanced? Are large errors worse than small ones in proportion? What does the competition/stakeholder actually care about?",
    fix: "Classification: use F1/ROC-AUC/PR-AUC for imbalanced; accuracy only for balanced. Regression: RMSE for symmetric errors; MAE for robust; RMSLE for log-scale targets; MAPE for percentage errors. Set model loss to match metric where possible.",
    tools: "sklearn.metrics full suite · scipy · Custom metric functions",
    affectedModels: "The model trained to optimise the wrong metric will systematically produce wrong-for-purpose predictions. Align training loss and evaluation metric.",
    competition: "Read the competition metric description very carefully. A model trained with logloss but evaluated with ROC-AUC will underperform vs one calibrated for ROC-AUC.",
  },
  {
    name: "Poor Probability Calibration",
    group: "Metrics & Evaluation",
    severity: "🟡 Medium",
    symptom: "Predicted probabilities don't reflect actual frequencies (e.g. model says 90% confident but is correct only 60% of the time). Log-loss is high even when ROC-AUC is good. Reliability diagram (calibration plot) shows deviation from diagonal.",
    rootCause: "Random Forests tend to compress probabilities toward 0.5. SVMs output uncalibrated decision functions. GBMs can over/underestimate probabilities. Models trained without calibration post-processing.",
    diagnose: "Plot calibration curve: sklearn.calibration.CalibrationDisplay. Compute Brier Score and Expected Calibration Error (ECE). Compare to a perfectly calibrated baseline.",
    fix: "Apply post-hoc calibration: CalibratedClassifierCV with method='isotonic' (more data) or method='sigmoid' (less data). Or use Platt scaling. Logistic Regression and HistGradientBoosting are inherently better calibrated.",
    tools: "sklearn.calibration.CalibratedClassifierCV · sklearn.calibration.CalibrationDisplay · sklearn.metrics.brier_score_loss",
    affectedModels: "Random Forest — tends to compress toward 0.5. SVM — uncalibrated by default. GBM — moderate calibration. Logistic Regression — best calibrated of common models.",
    competition: "If your competition metric is log-loss, calibration is critical — a model with better ROC-AUC but worse calibration will lose to a weaker but better-calibrated competitor.",
  },
  {
    name: "Metric Mismatch: Train Loss ≠ Eval Metric",
    group: "Metrics & Evaluation",
    severity: "🟡 Medium",
    symptom: "Model optimises cross-entropy but competition scores on ROC-AUC. Or model uses MSE but is evaluated on RMSLE. Subtle systematic performance gap that's hard to close.",
    rootCause: "Not all evaluation metrics are differentiable (e.g. F1, ROC-AUC), so models must be trained with a surrogate loss. If the surrogate doesn't align well with the target metric, tuning is inefficient.",
    diagnose: "Write out both the training loss and evaluation metric. Are they correlated? Try a different training loss and compare OOF scores on the target metric.",
    fix: "For RMSLE target: log-transform y, train with MSE, invert transform. For ordinal ranking: use rank-based objectives (LambdaRank). For F1: tune threshold from PR curve after training with log-loss. For custom metrics: use LightGBM/XGBoost custom fobj.",
    tools: "LightGBM custom fobj/feval · XGBoost custom objective · sklearn.metrics · scipy.optimize for threshold tuning",
    affectedModels: "GBM frameworks (LightGBM, XGBoost) support custom objectives — use them. sklearn models are more constrained in loss choices.",
    competition: "Implement the competition metric as your CV metric from day one. Small decisions made against the wrong metric compound over hundreds of experiments.",
  },
  {
    name: "Ignoring Confidence Intervals / Statistical Significance",
    group: "Metrics & Evaluation",
    severity: "🟡 Medium",
    symptom: "Choosing model A over model B based on a CV difference of 0.001 when fold std is ±0.005. Spurious model improvements driven by noise. Wasted engineering effort on non-improvements.",
    rootCause: "Treating mean CV score differences as definitive without accounting for variance across folds. Small differences within noise range are not meaningful.",
    diagnose: "Compute paired t-test between two models' fold scores: scipy.stats.ttest_rel(scores_a, scores_b). If p > 0.05, the difference is not statistically significant.",
    fix: "Report mean ± std for all CV scores. Only act on improvements that exceed 1-2 standard deviations. Use paired tests for model comparisons. Consider McNemar's test for classification.",
    tools: "scipy.stats.ttest_rel · scipy.stats.wilcoxon · sklearn cross_val_score std",
    affectedModels: "Affects interpretation of all models. Especially important when comparing many models in a tuning search.",
    competition: "A 0.0001 improvement on 3 folds is noise, not signal. Use 5 or 10 folds and report std. Only commit engineering effort to improvements > 1σ.",
  },

  // ═══════════════════════════════════════════════
  // DISTRIBUTION SHIFT
  // ═══════════════════════════════════════════════
  {
    name: "Covariate Shift (Train/Test Feature Drift)",
    group: "Distribution Shift",
    severity: "🟠 High",
    symptom: "Model CV is good but test performance drops. Input feature distributions differ between train and test. Certain feature value ranges present in test were rare or absent in train.",
    rootCause: "Data collection bias, seasonal/temporal effects, different user demographics between data collection periods, or deliberate test set construction that samples differently.",
    diagnose: "Adversarial validation: train a classifier to distinguish train=0 vs test=1. High AUC (> 0.6) indicates distribution shift. Per-feature KS test to identify which features drift most.",
    fix: "Remove or transform the most shifted features (identified by adversarial feature importances). Apply importance weighting: weight training samples by P(test) / P(train) using the adversarial model's probabilities. Use more robust features (ratios, ranks).",
    tools: "scipy.stats.ks_2samp · sklearn (adversarial validation, custom) · alibi-detect · alibi",
    affectedModels: "All models — models with low bias (GBM, deep nets) are most affected because they fit training distribution most precisely.",
    competition: "Adversarial validation is one of the highest-leverage analysis steps. Always run it. Features with high adversarial importance are candidates for removal.",
  },
  {
    name: "Concept Drift (Target Relationship Changes Over Time)",
    group: "Distribution Shift",
    severity: "🟠 High",
    symptom: "Model degrades monotonically over weeks/months in production. Retraining on recent data dramatically improves performance. The relationship between features and target has changed.",
    rootCause: "The underlying data-generating process has changed. Consumer preferences shifted. Market regime changed. Pandemic altered behaviour. Seasonal patterns changed. New product category introduced.",
    diagnose: "Plot model performance by time period. If performance degrades over time, drift is present. Compare feature-target correlations computed in old vs new data windows. Use ADWIN or Page-Hinkley drift detection.",
    fix: "Retrain periodically on recent data. Use sliding/expanding window training. Apply exponential sample weighting (recent samples weighted higher). Implement drift detection and trigger retraining. Monitor feature distributions in production.",
    tools: "river (online learning) · alibi-detect · evidently · nannyml (production monitoring)",
    affectedModels: "All static models. Models with high expressiveness (GBM, deep nets) adapt better when retrained but also drift faster. Linear models degrade more gracefully.",
    competition: "Not directly applicable to offline competitions, but in time-aware problems, models trained only on old data and evaluated on new data will show this pattern.",
  },
  {
    name: "Train/Test Split Doesn't Reflect Production",
    group: "Distribution Shift",
    severity: "🟠 High",
    symptom: "Great local CV, mediocre production. Random split used on time-structured data. The evaluation protocol doesn't simulate how the model will be deployed.",
    rootCause: "Random splitting ignores temporal structure — future data leaks into training. Or class ratios in test don't match production rates. Or the test set is too small to be representative.",
    diagnose: "Think: 'What is the model predicting, and what data will it have access to at prediction time?' If there's any time ordering, random split is wrong.",
    fix: "Use time-based split for temporal data (train on T₀→Tₙ₋₁, validate on Tₙ). Match test set class ratios to production. Use stratified splitting for classification. Make the validation set represent the actual deployment scenario.",
    tools: "sklearn.TimeSeriesSplit · sklearn.StratifiedShuffleSplit · pandas .sort_values('date')",
    affectedModels: "All models — this is about the evaluation protocol, not the model.",
    competition: "In time-series competitions, look at how the train/test split was created. If train ends Jan 2023 and test starts Jan 2024, your CV must also use a time gap.",
  },
  {
    name: "Public/Private LB Split Gap",
    group: "Competition-Specific",
    severity: "🔴 Critical",
    symptom: "High public LB score but ranked much lower on private LB. Model that ranked lower on public LB ends up winning on private. Overfitting the public sample.",
    rootCause: "Public LB uses ~20-30% of test data; private uses the rest. If your model (or your submissions) have been tuned to the public sample, it will not generalise to the private sample.",
    diagnose: "If your public/private gap is large: check if you changed model based on public LB feedback. Compute how many decisions were informed by LB scores. Does your CV correlate with public LB?",
    fix: "Trust your local CV above all. Use CV-best submission AND a diverse second submission for final choice. Build a CV that correlates with private LB (correct CV strategy). Use ensembles that average out sample variance.",
    tools: "Local OOF CV · experiment tracking (wandb, neptune) · cross_val_score",
    affectedModels: "Competition-specific. The winning approach is always the one with the best generalisation, not the best public score.",
    competition: "The Kaggle 'shake-up' is real. Many gold-medal solutions ranked 50th on public LB. Always submit your best-CV model as one of your final two choices.",
  },

  // ═══════════════════════════════════════════════
  // TRAINING INSTABILITY
  // ═══════════════════════════════════════════════
  {
    name: "Vanishing Gradients (Deep Networks)",
    group: "Training Instability",
    severity: "🟠 High",
    symptom: "Early layers of a deep network stop learning. Training loss plateaus immediately. Gradient norms near zero for early layers. RNNs fail to learn long-range dependencies.",
    rootCause: "Sigmoid/tanh activations squash gradients to near-zero during backpropagation. Gradients are multiplied across many layers — they decay exponentially toward the input in very deep networks.",
    diagnose: "Monitor gradient norms per layer during training. Early layers with near-zero gradient norms = vanishing. Use tensorboard/wandb gradient histograms.",
    fix: "Use ReLU or LeakyReLU activations (don't saturate). Use batch normalisation between layers. Use residual connections (ResNets skip layers). Use He/Xavier weight initialisation. Use LSTM/GRU for sequences. Reduce network depth if necessary.",
    tools: "PyTorch gradient monitoring · tensorboard · torch.nn.utils.clip_grad_norm_ · batch normalisation",
    affectedModels: "Deep MLP, RNN, LSTM with saturating activations. Modern architectures with BN + ReLU + residual connections largely solve this problem.",
    competition: "For sklearn MLP, this is mitigated automatically with ReLU. In PyTorch/Keras, always use He initialisation + ReLU + BatchNorm for any network > 4 layers.",
  },
  {
    name: "Exploding Gradients (Deep Networks)",
    group: "Training Instability",
    severity: "🟠 High",
    symptom: "Loss suddenly becomes NaN or Inf during training. Model weights become very large. Training is wildly unstable. Loss oscillates or spikes before diverging.",
    rootCause: "Gradient magnitudes grow exponentially during backpropagation. Common in RNNs and deep networks. Often triggered by a bad initialisation, learning rate too high, or a batch containing an extreme outlier.",
    diagnose: "Monitor loss — NaN or Inf = exploding gradients. Check gradient norms: if they reach 1e6+ before the loss explodes, this is the issue. Check for outlier samples in batch.",
    fix: "Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0). Reduce learning rate. Use better weight initialisation (He/Xavier). Use batch normalisation. Check input data for extreme outliers that corrupt the loss.",
    tools: "torch.nn.utils.clip_grad_norm_ · tensorboard gradient monitoring · weight init: torch.nn.init.kaiming_normal_",
    affectedModels: "Deep MLP, RNN, Transformer. Gradient clipping is standard in all modern transformer training.",
    competition: "If you're getting NaN loss in a deep model, gradient clipping with norm=1.0 is the first fix. Then reduce learning rate by 10x.",
  },
  {
    name: "Learning Rate Too High / Too Low",
    group: "Training Instability",
    severity: "🟠 High",
    symptom: "Too high: training loss oscillates or diverges, never converges. Too low: training loss decreases incredibly slowly, model takes forever to converge or gets stuck in suboptimal region.",
    rootCause: "The learning rate is the most sensitive hyperparameter in gradient-based optimisation. It determines the step size in parameter space.",
    diagnose: "Plot loss vs epoch. Oscillating/diverging = too high. Flat/slow = too low. Run a learning rate range test (LR finder): increase LR exponentially over a short run and plot loss vs LR.",
    fix: "Use learning rate finder (fastai-style: start at 1e-7, increase to 10, plot loss). Use learning rate scheduling: cosine annealing, OneCycleLR, ReduceLROnPlateau. Default starting points: Adam: 1e-3 to 1e-4, SGD: 0.01-0.1. Warmup for transformers.",
    tools: "torch.optim.lr_scheduler · PyTorch Lightning LearningRateFinder · keras LearningRateScheduler",
    affectedModels: "All gradient-based models: MLP, CNN, RNN, Transformers, GBM (learning_rate parameter).",
    competition: "For GBM: use learning_rate=0.05 with early stopping. For deep learning: use Adam with lr=1e-4 as a safe default, then tune with LR finder.",
  },
  {
    name: "Early Stopping Misconfiguration",
    group: "Training Instability",
    severity: "🟡 Medium",
    symptom: "GBM stops too early (underfits) or too late (overfits). Validation metric worsens after optimal point but training continues. Best model not saved.",
    rootCause: "patience parameter too small (stops at first sign of non-improvement). restore_best_weights / best model not saved. Monitored metric is noisy.",
    diagnose: "Plot train vs validation metric vs rounds/epochs. Check where validation metric peaks. If model stops before peak (patience too small) or after peak (patience too large), adjust.",
    fix: "Set patience to 50-100 rounds for GBM. Use restore_best_weights=True in Keras. Save best checkpoint manually in PyTorch. Monitor on a smooth metric. Use a validation set, not training set, for early stopping.",
    tools: "LightGBM callbacks.early_stopping · XGBoost early_stopping_rounds · Keras EarlyStopping(restore_best_weights=True) · PyTorch model checkpointing",
    affectedModels: "GBM (LightGBM, XGBoost, CatBoost), MLP/CNN in Keras/PyTorch.",
    competition: "Always set early stopping. It's free regularisation. Use patience=50-100 for GBM, patience=10-20 for deep learning. Save the best checkpoint, not the final epoch.",
  },
  {
    name: "Batch Size Effects",
    group: "Training Instability",
    severity: "🔵 Low",
    symptom: "Too large: model converges to sharp minima that don't generalise. Training fits in memory but validation performance is worse than with smaller batches. Too small: training is extremely slow, loss is noisy.",
    rootCause: "Large batch sizes compute more accurate gradient estimates but may converge to sharp minima. Small batches add noise that acts as regularisation (stochastic gradient descent effect).",
    diagnose: "Compare validation performance across batch sizes. Monitor loss smoothness vs epoch.",
    fix: "Start with batch_size=32 or 64 for most tasks. If using large batches, scale learning rate linearly with batch size (linear scaling rule). Use gradient accumulation to simulate larger batches. For GBMs, subsample acts as batch control.",
    tools: "PyTorch gradient accumulation · torch.utils.data.DataLoader",
    affectedModels: "Deep learning models (MLP, CNN, Transformer). GBM uses subsample/colsample parameters for analogous effect.",
    competition: "For deep learning on image tasks, batch_size=32 or 64 is a safe start. Scale LR with batch size if increasing it.",
  },
  {
    name: "Slow Training / Scalability Bottleneck",
    group: "Training Instability",
    severity: "🟡 Medium",
    symptom: "Training takes hours when it should take minutes. CPU is bottlenecked by data loading. GPU utilisation < 50%. Single-threaded preprocessing is the constraint.",
    rootCause: "Data loading not parallelised. Too much on-the-fly feature computation. Non-vectorised Python loops. Large model with no mixed precision. Unused cores.",
    diagnose: "Profile: time the data loading step separately from model forward/backward pass. Check GPU utilisation with nvidia-smi. Profile Python code with cProfile or line_profiler.",
    fix: "Parallelise data loading (DataLoader num_workers). Precompute and cache features. Vectorise with numpy/pandas. Use mixed precision (torch.cuda.amp). Use HistGradientBoosting instead of GradientBoosting. Reduce feature count.",
    tools: "PyTorch DataLoader num_workers · torch.cuda.amp · sklearn HistGradientBoosting · joblib parallelism · pandas vectorisation",
    affectedModels: "GradientBoostingClassifier (slow) → replace with HistGradientBoosting. sklearn SVC (O(n²)) → replace with LinearSVC. Deep models benefit most from GPU + mixed precision.",
    competition: "HistGradientBoosting is 10-100x faster than GradientBoosting. LightGBM is even faster. Always start with the fastest model for rapid iteration.",
  },

  // ═══════════════════════════════════════════════
  // FEATURE RELATED
  // ═══════════════════════════════════════════════
  {
    name: "Curse of Dimensionality",
    group: "Overfitting",
    severity: "🟡 Medium",
    symptom: "Model performance degrades as features are added past a certain point. KNN, SVM, and distance-based models become very slow. Adding features does not improve validation score.",
    rootCause: "In high-dimensional spaces, all points become approximately equidistant. Distance-based algorithms lose meaning. The volume of the feature space grows exponentially — the training data becomes sparse.",
    diagnose: "Plot validation score vs number of features. If it peaks and then drops, curse of dimensionality is active. Check whether distance-based models (KNN, kernel SVM) are slower and weaker than tree models on your data.",
    fix: "Apply dimensionality reduction (PCA, TruncatedSVD, UMAP) before distance-based models. Feature selection via SHAP/Boruta. Prefer tree-based models (immune to curse of dimensionality) for high-d tabular data. Use embeddings to compress high-d representations.",
    tools: "sklearn.PCA · sklearn.TruncatedSVD · umap-learn · shap · boruta-py",
    affectedModels: "KNN — severely degraded. SVM with RBF kernel — slow and weaker. GBM, Random Forest — relatively immune. Linear models with regularisation — OK at high d.",
    competition: "If you've generated 5000+ features from DFS or automated engineering, always apply feature selection before using distance-based models.",
  },
  {
    name: "Feature Scale Sensitivity for Distance Models",
    group: "Data Quality",
    severity: "🟡 Medium",
    symptom: "KNN or SVM performance is worse than expected. A feature measured in millions (salary) dominates one measured in fractions (ratio). Adding a scaling step dramatically improves performance.",
    rootCause: "Distance computations (Euclidean, Manhattan) and gradient descent are heavily influenced by feature magnitude. Unscaled features with large ranges dominate.",
    diagnose: "Check df.describe() for features with very different ranges. Compare KNN/SVM performance before and after StandardScaler.",
    fix: "Always apply StandardScaler or RobustScaler before KNN, SVM, PCA, MLP, Logistic Regression. Put scaler inside a Pipeline to prevent leakage.",
    tools: "sklearn.StandardScaler · sklearn.RobustScaler · sklearn.Pipeline",
    affectedModels: "KNN, SVM (both C-SVM and Nu-SVM), Logistic Regression, MLP, PCA, KMeans. Tree models (Random Forest, GBM) are completely immune.",
    competition: "A very basic but surprisingly common oversight. Always check if you forgot to scale before a distance-based model.",
  },
  {
    name: "Multicollinear Features Breaking Linear Models",
    group: "Data Quality",
    severity: "🟡 Medium",
    symptom: "Linear regression coefficients are unstable — small data change produces wildly different coefficients. Very large coefficient magnitudes with opposite signs that cancel. VIF > 10 for several features.",
    rootCause: "Two or more features are strongly correlated. OLS regression cannot determine which feature drives the prediction — coefficients become arbitrarily large and unstable.",
    diagnose: "Compute correlation matrix. VIF > 5-10 signals multicollinearity. Condition number of feature matrix > 30 is another indicator.",
    fix: "Use Ridge regression (L2 regularisation handles collinearity gracefully). Or remove one feature from each collinear pair. Or apply PCA to create orthogonal features. feature_engine.SmartCorrelatedSelection for automated deduplication.",
    tools: "statsmodels VIF · pandas corr() · feature_engine.SmartCorrelatedSelection · sklearn.Ridge",
    affectedModels: "Unregularised Linear Regression — severely affected. Ridge/Lasso — robust (regularisation handles it). Tree models — immune (splits on one feature at a time).",
    competition: "Not commonly a direct issue in competitions (most use GBMs) but matters when interpreting linear model coefficients or building scorecard models.",
  },

  // ═══════════════════════════════════════════════
  // INTERPRETABILITY
  // ═══════════════════════════════════════════════
  {
    name: "Misleading Feature Importance (MDI Bias)",
    group: "Interpretability",
    severity: "🟡 Medium",
    symptom: "Feature importances from Random Forest rank high-cardinality or continuous features higher than they should be. A feature that seems unimportant appears important. Importance doesn't match domain knowledge.",
    rootCause: "Mean Decrease in Impurity (MDI) — the default importance in sklearn trees — is biased toward high-cardinality features because they provide more possible split points. Also biased by feature correlations.",
    diagnose: "Compare MDI importances with permutation importances or SHAP values. If rankings differ significantly, MDI is biased. High-cardinality features ranking unusually high is a red flag.",
    fix: "Use permutation importance (sklearn.inspection.permutation_importance) on validation data. Use SHAP values (shap.TreeExplainer) for reliable global and local importances. These are not biased by cardinality.",
    tools: "sklearn.inspection.permutation_importance · shap.TreeExplainer · shap.summary_plot",
    affectedModels: "Random Forest (MDI bias is strongest here). GBM is slightly less biased but still affected. SHAP/permutation importance work correctly for both.",
    competition: "Don't use .feature_importances_ from RandomForest to decide what to remove. Use SHAP or permutation importance on a holdout set instead.",
  },
  {
    name: "SHAP Values Misleading Due to Correlated Features",
    group: "Interpretability",
    severity: "🔵 Low",
    symptom: "Two correlated features each appear less important than expected. Domain knowledge says feature X is critical, but SHAP shows it as unimportant. Removing a 'low SHAP' feature degrades performance.",
    rootCause: "SHAP distributes the predictive credit across correlated features. If A and B are both predictive but correlated, each gets ~half the total credit, appearing less important individually.",
    diagnose: "Check correlation matrix for the features flagged as low-SHAP. Try removing each of the correlated pair individually — if removing one degrades performance, it was important.",
    fix: "Interpret SHAP values at the group level for correlated features. Use SHAP interaction values to understand joint contributions. Consider PCA or manual combination of correlated features before SHAP analysis.",
    tools: "shap.TreeExplainer · shap.plots.bar · shap.plots.beeswarm · shap.interaction_values",
    affectedModels: "SHAP is model-agnostic but this limitation applies when any model is explained.",
    competition: "Use SHAP for directional understanding, not definitive feature removal decisions. Validate every removal with CV score before committing.",
  },
  {
    name: "Model Not Reproducible Across Runs",
    group: "Training Instability",
    severity: "🔵 Low",
    symptom: "Re-running the exact same code produces different CV scores each time. Ensemble weights change. Hard to debug or compare experiments reliably.",
    rootCause: "random_state not set in model, data splitter, or resampler. Non-deterministic operations in GPU (CUDA). Parallel processing with non-deterministic ordering. NumPy/Python random seeds not fixed.",
    diagnose: "Run the same script 3 times and compare outputs. If they differ, trace random_state parameters in every step.",
    fix: "Set random_state=42 (or any constant) in all sklearn estimators, splitters, samplers, and numpy. Set torch.manual_seed() and numpy.random.seed(). For GPU: torch.use_deterministic_algorithms(True) (slower but deterministic).",
    tools: "numpy.random.seed() · sklearn random_state=42 · torch.manual_seed() · random.seed()",
    affectedModels: "All models with stochastic elements: Random Forest, GBM, MLP, KMeans, SGD, data splitters.",
    competition: "Set all seeds at the top of your notebook. Irreproducible experiments make it impossible to know if an improvement is real or lucky. This is non-negotiable for serious ML work.",
  },
  {
    name: "Hyperparameter Tuning Without Principled Strategy",
    group: "Overfitting",
    severity: "🟡 Medium",
    symptom: "Grid search takes forever. Random search doesn't converge. Same hyperparameters tried multiple times. Optimal found in first few tries or not at all.",
    rootCause: "Grid search covers the space uniformly — most of the grid is wasted on unimportant regions. Manual tuning is biased by intuition. No principled exploration-exploitation balance.",
    diagnose: "Track all experiments. If performance doesn't improve after 50+ trials with random search, the search space is wrong or the model architecture is the bottleneck.",
    fix: "Use Bayesian optimisation (optuna, hyperopt) instead of grid/random search. Define search spaces based on log-scale for LR, linear for regularisation. Use HalvingGridSearchCV for early elimination of bad configs. Fix architecture first, then tune.",
    tools: "optuna · hyperopt · sklearn.HalvingGridSearchCV · wandb Sweeps · ray[tune]",
    affectedModels: "All models. GBM: most impactful to tune (n_estimators, max_depth, learning_rate, min_child_weight, subsample). MLP: LR, hidden_dim, dropout.",
    competition: "Optuna with 100-200 trials of Bayesian search typically outperforms 1000+ random search trials. Use it with cross_val_score for objective, not a single fold.",
  },
  {
    name: "Ensemble Diversity Collapse",
    group: "Competition-Specific",
    severity: "🟡 Medium",
    symptom: "Adding a 5th model to an ensemble gives no improvement. All models make the same mistakes on the same samples. Correlation between model predictions is > 0.99.",
    rootCause: "All ensemble members are the same algorithm type with similar hyperparameters trained on the same features. The ensemble is not diverse — errors are correlated so averaging them doesn't help.",
    diagnose: "Compute pairwise OOF prediction correlation. If > 0.95 between two models, they're too similar. Plot agreement/disagreement patterns across models.",
    fix: "Add diverse model families: GBM + Linear + Neural Net. Use different subsets of features. Use different CV fold seeds. Include models with different preprocessing pipelines. Use Stacking where the meta-learner weighs diverse base models.",
    tools: "pandas corr() on OOF predictions · sklearn.StackingClassifier · numpy.corrcoef",
    affectedModels: "Ensemble methods. Homogeneous ensembles (10 GBMs) provide less diversity gain than heterogeneous (GBM + LR + MLP).",
    competition: "Compute OOF prediction correlation between all your models before ensembling. The best ensembles combine the GBM best-performer with a diverse model that makes different mistakes.",
  },
  {
    name: "OOF (Out-of-Fold) Prediction Blending Mistakes",
    group: "Competition-Specific",
    severity: "🟡 Medium",
    symptom: "Stacked model OOF predictions look perfect on training data but test performance is poor. Meta-learner overfits because OOF predictions are inconsistent between train and test.",
    rootCause: "Base model OOF predictions for training are generated with k models (one per fold), but test predictions use all training data or a different configuration. The distribution of OOF train and test predictions differ.",
    diagnose: "Check: are test predictions generated by the same k models (averaged), or by a model retrained on all training data? Do OOF train and test predictions have the same distribution?",
    fix: "Generate test predictions by averaging the k fold models' predictions on the test set. Never retrain a single model on all data for OOF test predictions. Validate that OOF train/test distributions align.",
    tools: "sklearn.cross_val_predict · manual OOF loops",
    affectedModels: "Stacking meta-learner. All base models used in a stacking pipeline.",
    competition: "The correct OOF procedure: for each fold i, train on all folds except i, predict on fold i (train OOF) and on test. Average the k test predictions for final test OOF.",
  },
  {
    name: "Incorrect Final Submission Strategy",
    group: "Competition-Specific",
    severity: "🟠 High",
    symptom: "Selected final submission optimised for public LB rather than CV score. Private LB score is lower than models not selected. Shake-up from public to private.",
    rootCause: "Using the two final submission slots for the two best public LB submissions rather than the best CV submission + a diverse alternative.",
    diagnose: "Look at your submission history: was every submission decision informed by public LB feedback? Is your best-CV-score model one of your final two?",
    fix: "Always include your best-CV-score model as final submission 1. For submission 2, choose a model that is different (ensemble, different architecture) as insurance. Never both be variants of the same LB-optimised model.",
    tools: "Experiment tracking: wandb, neptune, mlflow · CV scoring framework",
    affectedModels: "Competition-specific. Applies to all model types.",
    competition: "The Kaggle 'safe' strategy: submission 1 = best OOF CV score. Submission 2 = best ensemble or most diverse alternative. Both should have been validated with robust local CV.",
  },
  {
    name: "Memory Errors / OOM on Large Datasets",
    group: "Training Instability",
    severity: "🟡 Medium",
    symptom: "Python kernel crashes. CUDA Out of Memory error. Process killed by OS. Pandas operations hang or fail silently.",
    rootCause: "Entire dataset loaded into memory at once. OHE on high-cardinality columns creates wide matrices. Model stores dense matrices instead of sparse. GPU VRAM exceeded by model + batch.",
    diagnose: "df.memory_usage(deep=True).sum(). Track RAM usage with psutil. Check matrix sparsity after encoding. Monitor GPU memory with torch.cuda.memory_summary().",
    fix: "Use appropriate dtypes (float32 not float64, int8 for small integers). Use sparse matrices for OHE (sparse=True in OHE). Use polars or chunked pandas operations. Use HistGradientBoosting instead of GBM for large data. Reduce batch size for GPU.",
    tools: "pandas memory_usage · polars · scipy.sparse · sklearn OHE(sparse_output=True) · gc.collect()",
    affectedModels: "Dense models on large data (SVM, KNN). GradientBoosting. MLP with large embeddings. Deep learning without gradient checkpointing.",
    competition: "Cast all floats to float32 immediately after loading. For very large datasets (> 10M rows), use polars instead of pandas — it's 5-10x more memory-efficient.",
  },
];

const ALL_GROUPS = ["All", ...Array.from(new Set(PROBLEMS.map(p => p.group)))];
const ALL_SEVERITIES = ["All", "🔴 Critical", "🟠 High", "🟡 Medium", "🔵 Low"];

function SevBadge({ s }) {
  const st = SEVERITY_STYLE[s] || { bg: "var(--border-subtle)", color: "#888", border: "#888" };
  return (
    <span style={{
      display: "inline-block", padding: "2px 9px", borderRadius: "4px",
      fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.04em",
      background: st.bg, color: st.color, border: `1px solid ${st.border}44`,
      whiteSpace: "nowrap",
    }}>{s}</span>
  );
}

function GroupBadge({ g }) {
  const accent = GROUP_ACCENT[g] || "#888";
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: "4px",
      fontSize: "0.63rem", fontWeight: 600, letterSpacing: "0.03em",
      background: accent + "18", color: accent,
      border: `1px solid ${accent}33`, whiteSpace: "nowrap",
    }}>{g}</span>
  );
}

function Card({ icon, label, text, accent }) {
  return (
    <div style={{
      padding: "10px 13px", borderRadius: "6px",
      background: "var(--bg-surface)", border: `1px solid ${accent}1e`,
    }}>

      <div style={{ fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.09em", color: accent, textTransform: "uppercase", marginBottom: "5px" }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: "0.79rem", color: "var(--text-primary)", lineHeight: 1.6 }}>{text}</div>
    </div>
  );
}

const tdS = { padding: "10px 13px", verticalAlign: "middle", borderBottom: "1px solid var(--bg-elevated)", color: "var(--text-primary)", fontSize: "0.83rem" };
const thS = { padding: "10px 13px", textAlign: "left", fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-tertiary)", borderBottom: "2px solid var(--bg-overlay)", background: "var(--bg-base)", position: "sticky", zIndex: 5 };

function Row({ item, idx }) {
  const [open, setOpen] = useState(false);
  const accent = GROUP_ACCENT[item.group] || "#888";
  return (
    <>
      <tr
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background = "var(--bg-elevated)"}
        onMouseLeave={e => e.currentTarget.style.background = idx % 2 === 0 ? "var(--bg-surface)" : "var(--bg-surface)"}
        style={{ cursor: "pointer", background: idx % 2 === 0 ? "var(--bg-surface)" : "var(--bg-surface)", borderLeft: `3px solid ${accent}`, transition: "background 0.12s" }}
      >
        <td style={tdS}>{open ? "▾" : "▸"}</td>
        <td style={{ ...tdS, fontWeight: 700, color: "var(--text-primary)", fontSize: "0.9rem" }}>{item.name}</td>
        <td style={tdS}><SevBadge s={item.severity} /></td>
        <td style={tdS}><GroupBadge g={item.group} /></td>
        <td style={{ ...tdS, fontSize: "0.78rem", color: "var(--text-secondary)", fontStyle: "italic", maxWidth: "280px" }}>{item.symptom.slice(0, 90)}…</td>
      </tr>
      {open && (
        <tr style={{ background: "var(--bg-surface)", borderLeft: `3px solid ${accent}` }}>
          <td colSpan={5} style={{ padding: "0 0 0 16px" }}>
            <div style={{ padding: "14px 16px 14px 0" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "8px" }}>
                <Card icon="🩺" label="How to Detect / Symptoms" text={item.symptom} accent={accent} />
                <Card icon="🔎" label="Root Cause" text={item.rootCause} accent="#fb923c" />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "8px" }}>
                <Card icon="🔬" label="How to Diagnose" text={item.diagnose} accent="#60a5fa" />
                <Card icon="🔧" label="How to Fix / Solve" text={item.fix} accent="#4ade80" />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                <Card icon="🛠️" label="Tools & APIs" text={item.tools} accent="#a78bfa" />
                <Card icon="🤖" label="Affected Models" text={item.affectedModels} accent="#22d3ee" />
                <Card icon="🏆" label="Competition Note" text={item.competition} accent="#facc15" />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Prob_app() {
  const [search, setSearch] = useState("");
  const [group, setGroup] = useState("All");
  const [severity, setSeverity] = useState("All");

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return PROBLEMS.filter(p => {
      const mg = group === "All" || p.group === group;
      const ms = severity === "All" || p.severity === severity;
      const mq = !q || [p.name, p.group, p.symptom, p.rootCause, p.diagnose, p.fix, p.tools, p.affectedModels, p.competition]
        .some(s => s.toLowerCase().includes(q));
      return mg && ms && mq;
    });
  }, [search, group, severity]);

  const groups = useMemo(() => {
    const g = {};
    for (const p of filtered) {
      if (!g[p.group]) g[p.group] = [];
      g[p.group].push(p);
    }
    return g;
  }, [filtered]);

  const sev_counts = useMemo(() => {
    const c = {};
    for (const p of PROBLEMS) c[p.severity] = (c[p.severity] || 0) + 1;
    return c;
  }, []);

  return (
    <div>
    <Header />

        <div style={{ fontFamily: "var(--font-body)", background: "var(--bg-base)", minHeight: "100vh", color: "var(--text-primary)" }}>
      
      {/* ── HEADER ── */}
      <div style={{
        position: "sticky", top: "var(--header-h)", zIndex: 10,
        background: "rgba(14,13,12,0.93)", backdropFilter: "blur(10px)",
        borderBottom: "1px solid var(--bg-overlay)", padding: "10px 20px 8px",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "8px", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
              <span style={{ color: "#f87171" }}>ML Problems</span>
              <span style={{ color: "var(--text-dim)", margin: "0 6px" }}>·</span>
              <span style={{ color: "#60a5fa" }}>Diagnostics</span>
              <span style={{ color: "var(--text-dim)", margin: "0 6px" }}>·</span>
              <span style={{ color: "#4ade80" }}>Solutions</span>
            </div>
            <div style={{ fontSize: "0.64rem", color: "var(--text-tertiary)", letterSpacing: "0.05em", marginTop: "1px" }}>
              {PROBLEMS.length} problems · Training · Evaluation · Competition · click any row to expand 9 detail fields
            </div>
          </div>
          <div style={{ flex: 1, minWidth: "180px", maxWidth: "300px", marginLeft: "auto" }}>
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search symptoms, fixes, models, tools…"
              style={{
                width: "100%", background: "#0e0e1c", border: "1px solid var(--border-default)",
                borderRadius: "6px", padding: "7px 11px", color: "var(--text-primary)", fontSize: "0.79rem",
              }}
            />
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--text-tertiary)", whiteSpace: "nowrap" }}>{filtered.length} shown</div>
        </div>

        {/* Severity filter */}
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap", marginBottom: "5px" }}>
          <span style={{ fontSize: "0.63rem", color: "var(--text-tertiary)", alignSelf: "center", marginRight: "2px" }}>SEVERITY:</span>
          {ALL_SEVERITIES.map(s => {
            const active = severity === s;
            const st = SEVERITY_STYLE[s] || { color: "#666", border: "#666", bg: "var(--border-subtle)" };
            return (
              <button key={s} onClick={() => setSeverity(s)} style={{
                padding: "3px 10px", borderRadius: "4px", fontSize: "0.68rem", fontWeight: active ? 700 : 400,
                border: active ? `1px solid ${st.border || "#666"}` : "1px solid var(--border-default)",
                background: active ? (st.bg || "var(--border-subtle)") : "transparent",
                color: active ? (st.color || "#aaa") : "var(--text-tertiary)",
              }}>
                {s}{s !== "All" && sev_counts[s] ? ` (${sev_counts[s]})` : ""}
              </button>
            );
          })}
        </div>

        {/* Group filter */}
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "0.63rem", color: "var(--text-tertiary)", alignSelf: "center", marginRight: "2px" }}>CATEGORY:</span>
          {ALL_GROUPS.map(g => {
            const active = group === g;
            const accent = GROUP_ACCENT[g] || "#666";
            return (
              <button key={g} onClick={() => setGroup(g)} style={{
                padding: "3px 10px", borderRadius: "4px", fontSize: "0.67rem", fontWeight: active ? 700 : 400,
                border: active ? `1px solid ${accent}` : "1px solid var(--border-default)",
                background: active ? accent + "22" : "transparent",
                color: active ? accent : "var(--text-tertiary)",
              }}>{g}</button>
            );
          })}
        </div>
      </div>

      {/* ── CRITICAL ALERT BANNER ── */}
      <div style={{
        margin: "12px 20px 0",
        padding: "10px 16px", borderRadius: "6px",
        background: "rgba(120,64,84,0.10)", border: "1px solid #f8717133",
        fontSize: "0.78rem", color: "#f87171",
        display: "flex", alignItems: "center", gap: "10px",
      }}>
        <span style={{ fontSize: "1rem" }}>🔴</span>
        <span><strong>Critical priority:</strong> Target Leakage, Preprocessing Leakage, SMOTE Before Split, and Public LB Overfitting are the most dangerous problems — they produce models that look great locally but fail silently in production or on the private leaderboard. Address these first.</span>
      </div>

      {/* ── TABLE ── */}
      <div style={{ overflowX: "auto", marginTop: "12px" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "860px" }}>
          <thead>
            <tr>
              <th style={{ ...thS, width: "24px" }}></th>
              <th style={thS}>Problem</th>
              <th style={thS}>Severity</th>
              <th style={thS}>Category</th>
              <th style={{ ...thS, maxWidth: "280px" }}>Symptom Preview</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([grp, items]) => (
              <>
                <tr key={grp + "_hdr"}>
                  <td colSpan={5} style={{
                    padding: "8px 14px 5px", fontSize: "0.67rem", fontWeight: 700,
                    letterSpacing: "0.12em", textTransform: "uppercase",
                    color: GROUP_ACCENT[grp] || "#5a5a9a",
                    background: "var(--bg-surface)",
                    borderTop: "2px solid var(--border-faint)",
                    borderBottom: "1px solid var(--border-faint)",
                  }}>
                    ▪ {grp}
                    <span style={{ fontWeight: 400, color: "var(--text-dim)", marginLeft: 6 }}>({items.length})</span>
                  </td>
                </tr>
                {items.map((item, i) => <Row key={item.name} item={item} idx={i} />)}
              </>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: "60px", textAlign: "center", color: "var(--text-tertiary)" }}>
                  No problems match your search. Try adjusting filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div style={{ padding: "20px", textAlign: "center", fontSize: "0.65rem", color: "var(--text-dim)", borderTop: "1px solid var(--border-faint)" }}>
        Covers Data Leakage · Overfitting · Underfitting · Class Imbalance · CV Strategy · Metrics · Distribution Shift · Training Instability · Competition-Specific Pitfalls
      </div>
    </div></div>
  );
}