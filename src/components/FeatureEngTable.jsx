import { useState, useMemo } from "react";
import Header from "./Header";
import { useStickyOffset } from "../hooks/useStickyOffset";

const DATA_TYPE_COLORS = {
  "Tabular":     { bg: "#1e3a5f", color: "#4a9eff" },
  "Text / NLP":  { bg: "#1a3a2a", color: "#4ade80" },
  "Image / CV":  { bg: "#3a1a2a", color: "#f472b6" },
  "Time Series": { bg: "#2a2a1a", color: "#facc15" },
  "Any":         { bg: "#2a2a3a", color: "#a78bfa" },
};

const GROUP_ACCENT = {
  "Categorical Encoding":    "#4a9eff",
  "Scaling & Transformation":"#60a5fa",
  "Tabular Feature Creation":"#34d399",
  "Text / NLP Features":     "#4ade80",
  "Image / CV Features":     "#f472b6",
  "Time Series Features":    "#facc15",
  "Statistical Tests":       "#fb923c",
  "Feature Selection":       "#c084fc",
};

const FEATURES = [
  // ═══════════════════════════════════════════════════════
  // CATEGORICAL ENCODING
  // ═══════════════════════════════════════════════════════
  {
    name: "One-Hot Encoding",
    lib: "sklearn · pandas",
    api: "OneHotEncoder / pd.get_dummies",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Creates a binary column for each category. The presence of a category is 1, absence is 0. Expands dimensionality by (n_categories − 1) columns per feature.",
    achieve: "Converts nominal categories into a numeric form that linear/distance-based models can interpret without implying false order.",
    useWhen: "Low-to-moderate cardinality categories (< ~30 unique values); linear models, SVMs, neural networks; when category order has no meaning.",
    dontUse: "High-cardinality columns (cities, ZIP codes → hundreds of sparse columns); memory-constrained environments.",
    benefitModels: "Logistic Regression, LinearSVC, SVM, MLP, KNN — models that rely on feature space geometry.",
    watchModels: "Tree-based models (can still use it, but ordinal/target encoding often works better and avoids unnecessary splits).",
  },
  {
    name: "Ordinal / Label Encoding",
    lib: "sklearn · category_encoders",
    api: "OrdinalEncoder / LabelEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Maps each category to an integer (0, 1, 2, …). Implies a numeric order between categories, which may or may not exist in your data.",
    achieve: "Minimal encoding with no dimensionality increase. Required for tree-based models that demand numeric input.",
    useWhen: "Ordinal features with a natural order (e.g., Low < Medium < High); tree-based models that internally split on values.",
    dontUse: "Nominal categories with no order fed into linear models — the false numeric order misleads gradient descent.",
    benefitModels: "Decision Trees, Random Forests, GBM variants — they split on thresholds and don't interpret the magnitude.",
    watchModels: "Linear Regression, Logistic Regression, SVM — will treat numbers as continuous order and produce misleading coefficients.",
  },
  {
    name: "Target / Mean Encoding",
    lib: "category_encoders · feature-engine",
    api: "TargetEncoder / MeanEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Replaces each category with the mean of the target variable within that category, computed on training data. Typically uses cross-fold smoothing to prevent leakage.",
    achieve: "Encodes high-cardinality categoricals into a single, information-rich numeric feature. Very popular in Kaggle competitions.",
    useWhen: "High-cardinality categoricals (user IDs, product IDs, cities); tree-based AND linear models; always use with out-of-fold encoding to avoid target leakage.",
    dontUse: "Small datasets (noisy estimates); without cross-validation-based encoding (target leakage risk is high).",
    benefitModels: "All models, especially GBMs and HistGradientBoosting — a single number carries rich target-signal information.",
    watchModels: "Careful with Logistic Regression — can produce perfect in-sample predictions if leakage occurs.",
  },
  {
    name: "Frequency / Count Encoding",
    lib: "feature-engine · manual (pandas)",
    api: "CountFrequencyEncoder / value_counts map",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Replaces each category with how often it appears in the training set (count or relative frequency). Rare categories naturally get low values.",
    achieve: "Captures popularity/rarity signal without exploding dimensionality. Zero-cost to implement, no leakage risk.",
    useWhen: "High-cardinality features; frequency of a category is meaningful signal (e.g., rare product IDs are suspicious); quick baseline encoding.",
    dontUse: "When two different categories with the same frequency should be treated differently by the model.",
    benefitModels: "GBM, Random Forests, MLP — frequency rank provides a useful real-valued signal.",
    watchModels: "Linear models — the frequency value may not correlate linearly with the target.",
  },
  {
    name: "Binary Encoding",
    lib: "category_encoders",
    api: "BinaryEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "First ordinal-encodes categories, then converts the integer to its binary bit representation, using one column per bit. Achieves O(log₂ n) columns instead of O(n) for OHE.",
    achieve: "Compact representation of high-cardinality categoricals — far fewer columns than OHE but retains more structure than ordinal.",
    useWhen: "High-cardinality features where OHE would be too wide; memory-efficient encoding for linear or distance models.",
    dontUse: "Low-cardinality features (OHE is cleaner); when the bit-level representation doesn't have a semantic interpretation.",
    benefitModels: "Linear models, MLP, KNN — more compact than OHE.",
    watchModels: "Tree models may prefer ordinal or target encoding for simplicity.",
  },
  {
    name: "Weight of Evidence (WoE)",
    lib: "category_encoders · feature-engine",
    api: "WOEEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Encodes each category as log(P(event) / P(non-event)) within that category. Information Value (IV) measures overall predictive power of the feature.",
    achieve: "Maps categorical to a log-odds scale that aligns naturally with logistic regression. Feature selection via IV is a standard risk modelling step.",
    useWhen: "Binary classification problems; risk/credit scoring; when interpretability and regulatory compliance matter.",
    dontUse: "Multi-class targets (not straightforwardly applicable); small sample sizes (noisy estimates).",
    benefitModels: "Logistic Regression — WoE aligns perfectly with log-odds. Scorecard models in banking/insurance.",
    watchModels: "Tree-based models get no geometric benefit from the log-odds scale.",
  },
  {
    name: "Hashing Encoding",
    lib: "category_encoders · sklearn",
    api: "HashingEncoder / FeatureHasher",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Maps category strings to a fixed-size vector via a hash function. Collisions mean different categories may share the same column, but the fixed size is guaranteed.",
    achieve: "Handles unlimited or streaming cardinality without storing a vocabulary. Very memory-efficient.",
    useWhen: "Extremely high cardinality (millions of user IDs); online learning / streaming data; memory-constrained pipelines.",
    dontUse: "When interpretability matters (hash collisions destroy semantics); small cardinality (OHE is cleaner).",
    benefitModels: "LinearSVC, SGDClassifier — standard in large-scale NLP and ad-tech pipelines.",
    watchModels: "Tree models lose the benefit of clean splits when collisions exist.",
  },
  {
    name: "Rare Label Encoding",
    lib: "feature-engine",
    api: "RareLabelEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "Groups infrequent category values (below a frequency threshold) into a single 'Rare' bucket before any other encoding step. Reduces noise and controls cardinality.",
    achieve: "Prevents model overfitting to rare, statistically unreliable categories. Reduces post-encoding dimensionality.",
    useWhen: "Any categorical column with a long tail of rare values; before OHE or target encoding to avoid noise.",
    dontUse: "When rare values have strong predictive power (e.g., rare fraud patterns) — group them deliberately.",
    benefitModels: "All models — especially important before OHE (fewer columns) or target encoding (better estimates).",
    watchModels: "No model is harmed; incorrect threshold choice could merge meaningful rare classes.",
  },
  {
    name: "CatBoost Encoding",
    lib: "category_encoders",
    api: "CatBoostEncoder",
    group: "Categorical Encoding",
    dataType: "Tabular",
    category: "Categorical Encoding",
    description: "An ordered target statistic encoding inspired by CatBoost's native categorical handling. Uses random permutation and cumulative target statistics to prevent leakage without needing cross-validation.",
    achieve: "Target encoding with built-in leakage prevention. Often outperforms plain target encoding, especially on small-to-medium datasets.",
    useWhen: "High-cardinality categoricals; when target encoding shows signs of overfitting; competition settings.",
    dontUse: "Very small datasets (stochastic estimates can be noisy).",
    benefitModels: "GBMs, HistGradientBoosting, and linear models — same as target encoding but safer.",
    watchModels: "Same caveats as target encoding.",
  },

  // ═══════════════════════════════════════════════════════
  // SCALING & TRANSFORMATION
  // ═══════════════════════════════════════════════════════
  {
    name: "Standard Scaler (Z-score)",
    lib: "sklearn",
    api: "StandardScaler",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Removes mean and scales to unit variance: z = (x − μ) / σ. The most common baseline scaling. Assumes an approximately Gaussian distribution.",
    achieve: "Equalises feature scales so that gradient descent converges faster and distance-based models aren't dominated by large-magnitude features.",
    useWhen: "Linear models, SVMs, neural networks, KNN, PCA; when feature distributions are roughly Gaussian; always before regularised models.",
    dontUse: "Tree-based models (invariant to monotone transforms); data with extreme outliers (use RobustScaler).",
    benefitModels: "Linear/Logistic Regression, SVM, MLP, KNN, PCA, Lasso, Ridge — essential for convergence and fairness.",
    watchModels: "Decision Trees, Random Forest, GBM — completely unaffected; scaling wastes computation but causes no harm.",
  },
  {
    name: "Min-Max Scaler",
    lib: "sklearn",
    api: "MinMaxScaler",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Scales each feature to a [0, 1] (or custom) range: x' = (x − min) / (max − min). Preserves zero values — important for sparse data.",
    achieve: "Bounded output, useful for models that expect inputs in a fixed range (e.g., sigmoid-activated neural nets).",
    useWhen: "Neural networks with bounded activations; image pixel values; when feature range must be preserved.",
    dontUse: "Data with outliers (one outlier collapses the rest into a tiny range); use RobustScaler instead.",
    benefitModels: "MLP/Neural Networks, KNN, image CNNs, SVM with RBF kernel.",
    watchModels: "Tree models (invariant); Robust models need RobustScaler instead.",
  },
  {
    name: "Robust Scaler",
    lib: "sklearn",
    api: "RobustScaler",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Scales using median and IQR instead of mean and std. Since it uses robust statistics, extreme outliers have minimal effect on the scaling.",
    achieve: "Stable scaling in the presence of outliers. Better than StandardScaler when your data has heavy tails or anomalies.",
    useWhen: "Financial data, medical data, sensor data — any domain with legitimate extreme values.",
    dontUse: "Perfectly clean data without outliers (StandardScaler is equivalent and simpler).",
    benefitModels: "Linear Regression, SVM, Lasso, MLP — same as StandardScaler but more robust.",
    watchModels: "Tree models — invariant to scaling.",
  },
  {
    name: "Log Transform",
    lib: "numpy · pandas",
    api: "np.log1p(x)",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Applies the natural logarithm to compress large values and reduce right skew. log1p(x) = log(1+x) safely handles zeros. Often used for price, income, count features.",
    achieve: "Makes right-skewed distributions approximately Gaussian. Reduces the influence of very large values. Stabilises variance.",
    useWhen: "Positive, right-skewed features (revenue, counts, prices); linear regression residual normality; stabilising variance.",
    dontUse: "Features with zero or negative values without adjustment; already symmetric distributions.",
    benefitModels: "Linear Regression, Logistic Regression, MLP — log-normal targets fit linear assumptions much better.",
    watchModels: "GBM, Random Forest — may help slightly but not required.",
  },
  {
    name: "Yeo-Johnson / Box-Cox Transform",
    lib: "sklearn · scipy",
    api: "PowerTransformer(method='yeo-johnson')",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "A parametric family of power transforms that finds the optimal λ to make a distribution as Gaussian as possible. Yeo-Johnson works on negative values; Box-Cox requires positivity.",
    achieve: "Systematically normalises any numeric distribution. Outperforms manual log/sqrt choices when you don't know the distribution shape.",
    useWhen: "Linear models needing Gaussian features; automated preprocessing pipelines; when the exact transformation isn't obvious.",
    dontUse: "Tree models (won't benefit); when interpretability of original scale matters.",
    benefitModels: "Linear Regression, Ridge, Lasso, MLP, SVM — better Gaussian alignment improves model assumptions.",
    watchModels: "Tree-based models — monotone transforms have zero effect.",
  },
  {
    name: "Quantile Transformer",
    lib: "sklearn",
    api: "QuantileTransformer(output_distribution='normal')",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Maps features to a normal or uniform distribution by matching quantiles to theoretical quantiles. More aggressive than PowerTransformer — truly forces any distribution into Gaussian.",
    achieve: "Forces any distribution to be Gaussian or uniform. Robust to outliers. Useful for severely non-normal features.",
    useWhen: "Features with very irregular distributions; robust preprocessing; when PowerTransformer is insufficient.",
    dontUse: "Small samples (< 1000) — quantile estimates are noisy; when original distribution has meaning.",
    benefitModels: "Linear models, SVM, KNN, MLP — same as StandardScaler but handles any distribution.",
    watchModels: "Tree models; can destroy ordered structure in small datasets.",
  },
  {
    name: "Binning / Discretisation",
    lib: "sklearn · pandas",
    api: "KBinsDiscretizer / pd.cut / pd.qcut",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Converts continuous features into discrete bins (equal-width, equal-frequency, or by learned decision-tree splits). Can follow with OHE for linear model compatibility.",
    achieve: "Handles non-linear relationships in linear models; reduces sensitivity to outliers; captures threshold effects (e.g., age groups).",
    useWhen: "Non-linear continuous features in linear models; when domain knowledge suggests natural thresholds; reducing overfitting.",
    dontUse: "Tree-based models (they discover optimal splits automatically); when continuous precision matters for the task.",
    benefitModels: "Logistic Regression, Linear Regression — transforms non-linear signal into piecewise-linear terms.",
    watchModels: "GBM, Random Forests, MLP — typically don't benefit; can lose information.",
  },
  {
    name: "Polynomial Features",
    lib: "sklearn",
    api: "PolynomialFeatures(degree=2, interaction_only=False)",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Generates all polynomial combinations of features up to a specified degree (e.g., x², x*y, x²y). Dramatically increases dimensionality.",
    achieve: "Gives linear models the ability to fit non-linear decision boundaries without switching algorithms.",
    useWhen: "Few features (< ~10 — combinatorial explosion otherwise); clear polynomial relationship; extending linear/logistic regression.",
    dontUse: "High-dimensional data (O(dⁿ) feature explosion); when regularisation can't control the expanded space.",
    benefitModels: "Linear Regression, Logistic Regression, Ridge, Lasso — turns linear models into polynomial ones.",
    watchModels: "SVM with RBF kernel (kernel implicitly does this); tree models (unnecessary); MLP (learns polynomial implicitly).",
  },
  {
    name: "Spline Features",
    lib: "sklearn",
    api: "SplineTransformer(n_knots=5, degree=3)",
    group: "Scaling & Transformation",
    dataType: "Tabular",
    category: "Scaling & Transformation",
    description: "Creates a set of piecewise polynomial basis functions (B-splines) for each feature. Captures smooth non-linearity more stably than polynomial features.",
    achieve: "Smooth non-linear transformation with controlled extrapolation. More numerically stable than high-degree polynomials.",
    useWhen: "Smooth, non-linear relationships in linear models; age/dose response curves; time-of-day or seasonal patterns.",
    dontUse: "Discontinuous/step relationships (use binning); tree-based models (built-in).",
    benefitModels: "Linear/Logistic Regression, Ridge — adds smooth non-linearity without full polynomial explosion.",
    watchModels: "Tree models; MLP.",
  },

  // ═══════════════════════════════════════════════════════
  // TABULAR FEATURE CREATION
  // ═══════════════════════════════════════════════════════
  {
    name: "Interaction / Product Features",
    lib: "pandas · featuretools · manual",
    api: "df['a_times_b'] = df.a * df.b",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Manually or automatically creates pairwise products (a×b), ratios (a/b), or differences (a−b) of existing features to capture interaction effects.",
    achieve: "Explicit interactions allow linear models to capture synergistic effects between features. Often the top feature in competition winning solutions.",
    useWhen: "Domain knowledge suggests feature synergies; linear model needs interaction terms; competition feature engineering.",
    dontUse: "High-dimensional data (O(d²) pairs); tree-based models that already learn interactions internally.",
    benefitModels: "Linear Regression, Logistic Regression, MLP — explicitly models interactions they can't learn from additive terms.",
    watchModels: "GBM, Random Forest — they discover interactions through splits automatically; adding manual interactions provides marginal gain.",
  },
  {
    name: "Ratio Features",
    lib: "pandas · manual",
    api: "df['ratio'] = df.a / (df.b + ε)",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Computes meaningful ratios between features — e.g., debt/income, price/sqft, clicks/impressions. Normalises absolute values into relative measures.",
    achieve: "Scale-invariant features that capture relative relationships. Often more predictive than raw values.",
    useWhen: "Domain knowledge confirms ratio is meaningful (financial ratios, CTR, efficiency metrics); normalising by size.",
    dontUse: "When denominator can be zero or near-zero without careful handling; when absolute values are predictive.",
    benefitModels: "All models — especially linear models where the ratio captures a unit-free signal.",
    watchModels: "No harm to any model; numerical instability if not guarded against division by zero.",
  },
  {
    name: "Date/Time Feature Extraction",
    lib: "pandas · feature-engine",
    api: "pd.DatetimeIndex / DatetimeFeatures",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Decomposes timestamps into components: year, month, day, hour, day-of-week, is_weekend, quarter, week-of-year, days-since-epoch, etc.",
    achieve: "Converts a single timestamp into rich temporal features that capture seasonality, trends, and periodicity for any model.",
    useWhen: "Any dataset with datetime columns; tabular ML where time matters; competition datasets almost always benefit from this.",
    dontUse: "When used alone without cyclical encoding for periodic features (e.g., month 12 and month 1 appear far apart numerically).",
    benefitModels: "All models — especially GBMs which natively use these as numeric features. Linear models need cyclical encoding too.",
    watchModels: "Without cyclical encoding (sin/cos), linear/distance models see December → January as a large jump.",
  },
  {
    name: "Cyclical Encoding (sin/cos)",
    lib: "numpy · manual",
    api: "sin(2π·x/period), cos(2π·x/period)",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Encodes periodic features (hour of day, day of week, month) using sine and cosine projections so that the beginning and end of a cycle are geometrically close.",
    achieve: "Preserves the circular topology of periodic features. Month 12 becomes close to month 1. Essential for linear and distance models.",
    useWhen: "Any periodic/cyclical feature (time of day, day of week, month, compass bearing, season) for non-tree models.",
    dontUse: "Tree-based models (they can learn the cyclical pattern from ordinal values via splits, though cyclical encoding doesn't hurt).",
    benefitModels: "Linear Regression, MLP, KNN, SVM — geometrically correct representation of cyclic features.",
    watchModels: "GBM, Random Forest — no benefit but also no harm.",
  },
  {
    name: "Group Aggregation Features",
    lib: "pandas · featuretools",
    api: "df.groupby('cat')['num'].transform('mean')",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Computes summary statistics (mean, std, min, max, count, median, nunique) of a numeric column grouped by a categorical — then joins back as a new feature per row.",
    achieve: "Captures group-level context for each row (e.g., 'average price for this product category'). One of the highest-impact feature engineering techniques in practice.",
    useWhen: "Relational data structures; group context is informative; customer/transaction/product data; competition datasets with ID columns.",
    dontUse: "Groups defined on test data must be computed from train only — otherwise data leakage.",
    benefitModels: "All models — GBMs especially benefit. This technique consistently appears in top competition solutions.",
    watchModels: "Linear models also benefit but may need scaling of the new feature.",
  },
  {
    name: "Missing Value Indicator",
    lib: "sklearn · feature-engine",
    api: "MissingIndicator / AddMissingIndicator",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Creates a binary column flagging whether the original feature was missing (NaN) before imputation. Preserves the information that missingness itself may be predictive.",
    achieve: "Allows models to learn that 'this value was absent' is itself a signal (MNAR — Missing Not At Random).",
    useWhen: "Data is not missing completely at random (MCAR); missingness patterns correlate with the target; medical, financial, survey data.",
    dontUse: "Truly random missingness; when you have very few missing values.",
    benefitModels: "All models — linear models can weight the missingness indicator directly; tree models split on it.",
    watchModels: "Creates noise if missingness is truly random.",
  },
  {
    name: "Deep Feature Synthesis (DFS)",
    lib: "featuretools",
    api: "ft.dfs(entityset=..., target_dataframe_name=...)",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Automated feature engineering for multi-table relational datasets. Stacks aggregation and transformation primitives (mean, sum, count, mode, …) across entity relationships up to a configurable depth.",
    achieve: "Automatically discovers hundreds/thousands of features from raw relational data. Mimics what a senior data scientist would engineer by hand. Huge time saver.",
    useWhen: "Multi-table relational data (customers/transactions/sessions); time-permitting competition preprocessing; automated ML pipelines.",
    dontUse: "Single flat tables with no relationships; when interpretability is critical; compute-constrained environments.",
    benefitModels: "All models — best paired with GBMs. The feature set is too large for direct use without selection.",
    watchModels: "Must follow with feature selection (e.g., Boruta, SHAP) to remove noisy generated features.",
  },
  {
    name: "Target Statistics by Window (Leaky Features)",
    lib: "pandas · manual",
    api: "df.sort_values('date').groupby('id')['target'].shift(1).rolling(7).mean()",
    group: "Tabular Feature Creation",
    dataType: "Tabular",
    category: "Tabular Feature Creation",
    description: "Computes rolling/historical target statistics for entity IDs (e.g. 7-day rolling mean of past sales for each store). The key is using only past data to avoid future leakage.",
    achieve: "Extremely powerful predictors in time-based or ID-based data. Captures individual trajectory/history. Common in top competition solutions.",
    useWhen: "Time-ordered data; prediction of future values per entity; retail/finance/logistics where historical behavior matters.",
    dontUse: "Without strict temporal separation — computing on shuffled or future data causes target leakage and artificially inflated CV scores.",
    benefitModels: "GBM, HistGradientBoosting, Neural Networks — these features are among the most predictive in time-series-flavored tabular tasks.",
    watchModels: "Any model will look artificially good if leakage is not prevented.",
  },

  // ═══════════════════════════════════════════════════════
  // TEXT / NLP FEATURES
  // ═══════════════════════════════════════════════════════
  {
    name: "Bag of Words (CountVectorizer)",
    lib: "sklearn",
    api: "CountVectorizer(max_features=10000)",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Converts text into a document-term matrix of raw word counts. Each column is a vocabulary token; each row is a document. Ignores word order and grammar.",
    achieve: "Fast, simple numeric representation of text. Sufficient for many classification tasks.",
    useWhen: "Baseline text classification; short documents; when speed matters; combined with LinearSVC or MultinomialNB.",
    dontUse: "Long or complex documents where word order and semantics matter; when vocabulary size is huge without feature limits.",
    benefitModels: "MultinomialNB, BernoulliNB, LinearSVC, Logistic Regression — the classic NLP pipeline.",
    watchModels: "Tree-based models struggle with high-dimensional sparse BoW; prefer embeddings + dense features instead.",
  },
  {
    name: "TF-IDF",
    lib: "sklearn",
    api: "TfidfVectorizer(sublinear_tf=True, max_features=50000)",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Term Frequency × Inverse Document Frequency. Down-weights common words across all documents and up-weights rare but discriminative terms. Sparse matrix output.",
    achieve: "Better than raw counts for most classification tasks. The de-facto representation for text in classical ML (pre-deep learning).",
    useWhen: "Text classification, search ranking, topic modelling baseline; medium-to-large corpora; linear models.",
    dontUse: "Very short texts (few meaningful tokens); when semantic similarity matters (use embeddings instead).",
    benefitModels: "Logistic Regression, LinearSVC, MultinomialNB, SGDClassifier — the classic TF-IDF + linear model is still competitive.",
    watchModels: "Tree models (too sparse); dense models like MLP (prefer TruncatedSVD after TF-IDF to reduce to dense).",
  },
  {
    name: "N-grams",
    lib: "sklearn",
    api: "TfidfVectorizer(ngram_range=(1,3))",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Extends BoW/TF-IDF to include contiguous sequences of n words (bigrams: 'New York', trigrams: 'not very good'). Captures local word order and phrases.",
    achieve: "Captures negations, named entities, multi-word expressions that unigrams miss. Often improves classification F1 by 2–5%.",
    useWhen: "Sentiment analysis (negations matter); entity detection; topic classification where phrases are discriminative.",
    dontUse: "Very short texts; large vocabularies without max_features limits (memory explodes with trigrams).",
    benefitModels: "Same as TF-IDF. Always combine ngrams with TF-IDF weighting.",
    watchModels: "Tree models; without max_features the vocabulary becomes unmanageable.",
  },
  {
    name: "Word Embeddings (Word2Vec / GloVe / FastText)",
    lib: "gensim · spaCy · torchtext",
    api: "Word2Vec / gensim.models.FastText",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Distributed representations where each word is a dense vector (50–300 dims) such that semantically similar words are geometrically close. Typically pre-trained on large corpora and fine-tuned or averaged per document.",
    achieve: "Semantic similarity, analogy reasoning, generalisation to unseen words. Foundation of modern NLP before transformers.",
    useWhen: "Semantic similarity tasks; document representation by averaging word vectors; small-to-medium NLP tasks without GPU resources for transformers.",
    dontUse: "When you have GPU resources (contextual embeddings like BERT are strictly better); very domain-specific vocabularies without custom training.",
    benefitModels: "MLP, Logistic Regression, SVM (on averaged doc vectors), LSTM, Random Forests on aggregated embeddings.",
    watchModels: "Averaging word vectors loses word order; unsuitable for tasks where sequence matters.",
  },
  {
    name: "Contextual Sentence Embeddings (BERT / Sentence-Transformers)",
    lib: "sentence-transformers · HuggingFace transformers",
    api: "SentenceTransformer('all-MiniLM-L6-v2').encode(texts)",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Encodes each sentence or document into a dense 384–1024 dim vector using pre-trained transformer models (BERT, RoBERTa, etc.). Captures context, polysemy, and deep semantics.",
    achieve: "State-of-the-art semantic representations. A single embedding often outperforms TF-IDF + feature engineering on most NLP tasks.",
    useWhen: "Semantic search, NLP classification, duplicate detection, clustering; any modern NLP pipeline with GPU or acceptable latency.",
    dontUse: "Strictly latency-bound production systems without GPU; very long documents (chunking required); when interpretability is needed.",
    benefitModels: "Logistic Regression, MLP, cosine-similarity search (FAISS), any classifier on top of embeddings.",
    watchModels: "Tree-based models can underperform on raw embeddings (high-dim dense vectors); may need PCA reduction first.",
  },
  {
    name: "Text Statistics Features",
    lib: "textstat · manual (len, count)",
    api: "textstat.flesch_kincaid_grade / len(text.split())",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Extracts simple surface-level text statistics: word count, char count, avg word length, sentence count, punctuation density, uppercase ratio, exclamation count, etc.",
    achieve: "Cheap, fast features that complement semantic embeddings. Style markers are often surprisingly predictive for authorship, spam, sentiment.",
    useWhen: "Any text task as complementary features; author attribution; spam detection; readability scoring alongside semantic features.",
    dontUse: "As the sole representation; when pure semantic features are sufficient.",
    benefitModels: "All models — treat as additional numeric features alongside TF-IDF or embeddings.",
    watchModels: "No harm; low value alone.",
  },
  {
    name: "TF-IDF + SVD (LSA)",
    lib: "sklearn",
    api: "Pipeline([TfidfVectorizer, TruncatedSVD(n_components=100)])",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Latent Semantic Analysis: applies TF-IDF then reduces the high-dimensional sparse matrix to a dense low-dimensional representation via Truncated SVD. Captures latent topic structure.",
    achieve: "Dense, de-noised document vectors that capture latent topic co-occurrence. Enables any dense model (SVM, MLP) on text.",
    useWhen: "Dense downstream models (SVM, MLP, KNN) that struggle with sparse BoW; want document-level topic features; information retrieval.",
    dontUse: "When BERT embeddings are feasible (LSA is dominated by contextual embeddings on most tasks).",
    benefitModels: "SVM, MLP, KNN, Logistic Regression on dense text features.",
    watchModels: "Linear models work directly on sparse TF-IDF; LSA adds latency without always helping.",
  },
  {
    name: "Sentiment Score (VADER / TextBlob)",
    lib: "nltk (VADER) · textblob",
    api: "SentimentIntensityAnalyzer().polarity_scores(text)",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Assigns a pre-computed sentiment polarity score (positive, negative, neutral, compound) to a text without training. Rule-based (VADER) or lexicon-based (TextBlob).",
    achieve: "Zero-shot sentiment signal as a numeric feature. Useful complement to BoW/embeddings when sentiment is a key signal.",
    useWhen: "Product reviews, social media, customer feedback; as an additional feature; when labelled sentiment data is scarce.",
    dontUse: "Domain-specific jargon where the lexicon fails (legal, medical); as the sole feature for fine-grained sentiment.",
    benefitModels: "All tabular models as an additional feature column.",
    watchModels: "No harm; limited value on technical domains.",
  },
  {
    name: "NER / POS Tag Features",
    lib: "spaCy · HuggingFace",
    api: "nlp(text).ents / token.pos_",
    group: "Text / NLP Features",
    dataType: "Text / NLP",
    category: "Text / NLP",
    description: "Named Entity Recognition extracts entities (PERSON, ORG, GPE, DATE) and counts them. POS tagging counts verbs, nouns, adjectives per document.",
    achieve: "Structured semantic features. Entity counts are powerful for news classification, financial NLP, and information extraction.",
    useWhen: "News categorisation, legal NLP, financial document analysis; as complementary features to embeddings.",
    dontUse: "Informal/social media text where NER accuracy drops sharply; latency-critical pipelines.",
    benefitModels: "All classifiers as additional numeric features; combined with TF-IDF or embeddings.",
    watchModels: "No direct harm; spaCy can be slow at scale — cache results.",
  },

  // ═══════════════════════════════════════════════════════
  // IMAGE / CV FEATURES
  // ═══════════════════════════════════════════════════════
  {
    name: "HOG (Histogram of Oriented Gradients)",
    lib: "scikit-image · opencv",
    api: "skimage.feature.hog / cv2.HOGDescriptor",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Computes the distribution of gradient orientations in localised image patches. Captures shape/edge structure robustly, partially invariant to lighting.",
    achieve: "Compact, discriminative representation of shape and texture. The backbone of pre-deep-learning pedestrian/object detection.",
    useWhen: "Classical ML on images without GPU; detecting object shapes; competition image tasks without deep learning resources.",
    dontUse: "Complex scene understanding; deep learning pipelines (CNN features are always better); very small images.",
    benefitModels: "SVM (the HOG + SVM pipeline was dominant for years), Logistic Regression, Random Forest.",
    watchModels: "Deep learning models — use CNN features instead.",
  },
  {
    name: "Color Histograms",
    lib: "opencv · skimage",
    api: "cv2.calcHist / np.histogram on channels",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Computes frequency distribution of pixel intensities per color channel (R, G, B or H, S, V). A global image descriptor, invariant to spatial arrangement.",
    achieve: "Simple, fast color-based image representation. Effective for color-specific classification tasks (product color, nature scene type).",
    useWhen: "Color is a primary discriminative feature; scene/product colour classification; augmenting CNN embeddings.",
    dontUse: "When spatial layout matters (captures no spatial structure); texture-based or shape-based tasks.",
    benefitModels: "SVM, KNN, Random Forest — as fast feature vectors.",
    watchModels: "Not useful alone for complex scene understanding.",
  },
  {
    name: "CNN Embeddings / Transfer Learning",
    lib: "PyTorch · Keras / TensorFlow",
    api: "ResNet50(include_top=False) / torchvision.models.resnet50",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Extracts feature vectors from intermediate layers of a pre-trained CNN (ResNet, EfficientNet, ViT). The penultimate layer produces rich, semantically meaningful embeddings.",
    achieve: "State-of-the-art image representations without training from scratch. 512–2048 dim dense vectors capture deep semantic content.",
    useWhen: "Any image classification/regression; limited training data (transfer learning); competition computer vision; medical imaging.",
    dontUse: "Extremely domain-shifted data where pre-training distribution is very different (rare); strict latency on edge devices.",
    benefitModels: "Logistic Regression, SVM, MLP on top of frozen CNN features — surprisingly competitive baseline.",
    watchModels: "KNN on raw 2048-dim embeddings suffers curse of dimensionality; apply PCA first.",
  },
  {
    name: "Data Augmentation",
    lib: "albumentations · torchvision · imgaug",
    api: "A.Compose([A.HorizontalFlip(), A.Rotate(limit=30), …])",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Randomly applies geometric and photometric transformations (flip, rotate, crop, brightness, contrast, blur, noise, cutout) during training to generate more diverse training samples.",
    achieve: "Acts as a regulariser — reduces overfitting, improves generalisation. Often provides the single largest accuracy boost in image competitions.",
    useWhen: "Any image model training (especially with limited data); always use in competitions; medical imaging with small datasets.",
    dontUse: "Inference time (train only, unless using test-time augmentation); augmentations that destroy label-relevant features (e.g., colour flip on colour-classification tasks).",
    benefitModels: "All deep learning image models — CNNs, ViTs. Non-deep models can't use online augmentation in the same way.",
    watchModels: "Classical ML models take pre-extracted features; augmentation must be applied before feature extraction.",
  },
  {
    name: "SIFT / ORB Keypoint Descriptors",
    lib: "opencv",
    api: "cv2.SIFT_create() / cv2.ORB_create()",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Detects distinctive local keypoints (corners, blobs) and computes rotation/scale-invariant descriptors. SIFT is more accurate; ORB is faster and free of patents.",
    achieve: "Sparse image representations invariant to scale and rotation. Used for image matching, retrieval, and BoVW (Bag of Visual Words) representations.",
    useWhen: "Image matching, panorama stitching, BoVW for classical image classification; when CNN features are unavailable.",
    dontUse: "Modern deep learning pipelines (CNN features dominate); textureless or flat images.",
    benefitModels: "SVM on BoVW representations, KNN for image retrieval.",
    watchModels: "Not useful directly without further aggregation (BoVW or Fisher vectors).",
  },
  {
    name: "Pixel Statistics (mean, std, entropy)",
    lib: "numpy · skimage",
    api: "np.mean(img), np.std(img), skimage.measure.shannon_entropy",
    group: "Image / CV Features",
    dataType: "Image / CV",
    category: "Image / CV",
    description: "Computes global or patch-level pixel statistics: mean intensity, standard deviation, entropy, contrast, kurtosis, skewness per channel or region.",
    achieve: "Ultra-simple image features. Surprisingly useful when image quality, brightness, or noise level is a discriminative signal.",
    useWhen: "Image quality scoring, medical scan normalisation, simple scene/lighting classification; quick baselines.",
    dontUse: "Complex object classification; when spatial structure is important.",
    benefitModels: "Any tabular model with these as features; useful as diagnostic features.",
    watchModels: "Not useful alone for content-based image tasks.",
  },

  // ═══════════════════════════════════════════════════════
  // TIME SERIES FEATURES
  // ═══════════════════════════════════════════════════════
  {
    name: "Lag Features",
    lib: "pandas",
    api: "df['value'].shift(n)",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Creates new features from the value of the target (or another variable) at n timesteps in the past. The most fundamental time-series feature engineering technique.",
    achieve: "Captures autocorrelation and direct history. Essential for almost every time series regression/forecasting problem.",
    useWhen: "Forecasting, time-series regression; always include lags of the target and key predictors; determine lag lengths from ACF/PACF plots.",
    dontUse: "Cross-sectional (non-time-ordered) data; without proper temporal train/test split (data leakage risk).",
    benefitModels: "GBMs (LightGBM, XGBoost), Random Forests, MLP — direct value history is among the most predictive features for forecasting.",
    watchModels: "ARIMA and classical time series models handle lags internally; adding them as features is redundant for these.",
  },
  {
    name: "Rolling Statistics",
    lib: "pandas",
    api: "df['value'].rolling(window=7).mean()",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Computes moving window statistics (mean, std, min, max, sum, skew, kurt, median) over a fixed past window. Captures local trend and volatility.",
    achieve: "Smoothed trends, local volatility, and momentum signals. A rolling mean reveals trend; rolling std reveals volatility.",
    useWhen: "Any time-series regression or classification; multiple window sizes (7, 14, 30, 90 days) commonly used; always shift by 1 to avoid leakage.",
    dontUse: "Without shifting (would include current value in window); very short time series (insufficient window samples).",
    benefitModels: "All ML models on tabular time series, especially GBMs. Standard in finance, retail, energy forecasting.",
    watchModels: "Neural sequence models (LSTM, Transformer) can compute these implicitly — but adding them as features often still helps.",
  },
  {
    name: "Expanding Window Statistics",
    lib: "pandas",
    api: "df['value'].expanding().mean()",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Computes cumulative statistics from the start of the series up to each time point. Captures long-run historical aggregates that grow with data.",
    achieve: "Long-term historical baseline: cumulative mean, max ever seen, total count so far. Useful for entity-level history in event logs.",
    useWhen: "Cumulative metrics matter (total purchases, all-time max, running count); entity-level behavioural history.",
    dontUse: "Early in the series where expanding stats are based on too few points (add a minimum count filter).",
    benefitModels: "GBMs, MLP — same use case as rolling stats, longer horizon.",
    watchModels: "Early-series instability; not directly useful for pure sequence models.",
  },
  {
    name: "Fourier / FFT Features",
    lib: "numpy · scipy",
    api: "np.fft.rfft(series) / scipy.fft.fft",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Transforms the time series from the time domain into the frequency domain. Dominant frequencies, power spectral density, and phase angles capture periodicity structure.",
    achieve: "Explicit representation of seasonal cycles (daily, weekly, annual). Powerful for detecting and capturing repeating patterns.",
    useWhen: "Strong known or suspected periodicity (energy load, retail sales, IoT sensors); as input features for any ML model.",
    dontUse: "Non-stationary series without detrending first; aperiodic irregular event streams.",
    benefitModels: "Any model where Fourier coefficients are used as tabular features. Also used inside Prophet and neural forecasting models.",
    watchModels: "Raw FFT output has complex numbers — use amplitude/phase or top-k Fourier terms.",
  },
  {
    name: "ACF / PACF Features",
    lib: "statsmodels",
    api: "statsmodels.graphics.tsaplots / acf(series, nlags=20)",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Auto-Correlation Function and Partial Auto-Correlation Function measure the correlation of the series with its own past values at different lags. Primarily a diagnostic but can be used as features.",
    achieve: "Identifies which lag orders are informative → guides lag feature selection. PACF helps choose AR order; ACF helps choose MA order.",
    useWhen: "Diagnosing time series structure before modelling; identifying optimal lag lengths; verifying stationarity of residuals.",
    dontUse: "As raw features for ML models (the statistics themselves are descriptors, not per-row features).",
    benefitModels: "ARIMA/SARIMA, VAR, ARIMAX — directly informs model order selection.",
    watchModels: "ML models use lag features derived from this analysis, not ACF values directly.",
  },
  {
    name: "Seasonal Decomposition (STL)",
    lib: "statsmodels · prophet",
    api: "statsmodels.tsa.seasonal.STL / seasonal_decompose",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Decomposes a time series into three additive (or multiplicative) components: Trend, Seasonality, and Residual. STL is robust to outliers.",
    achieve: "Isolates seasonal pattern, long-term trend, and unexplained residual. Each component can be used as a separate feature or modelled independently.",
    useWhen: "Clear seasonality present; forecasting tasks where deseasonalised values are easier to model; anomaly detection on residuals.",
    dontUse: "Irregular or event-driven series without periodic structure.",
    benefitModels: "Any model on the decomposed components. Prophet, SARIMA, and hybrid ML+decomposition models.",
    watchModels: "Direct ML on non-decomposed seasonal data can still work well with calendar features.",
  },
  {
    name: "tsfresh Automated Extraction",
    lib: "tsfresh",
    api: "tsfresh.extract_features(df, column_id=..., column_sort=...)",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "Automatically extracts 794 (default) features from time series: statistical moments, Fourier coefficients, entropy measures, AR coefficients, peaks, zero crossings, and more. Includes hypothesis-test-based feature selection.",
    achieve: "Exhaustive time series feature discovery without domain knowledge. Transforms raw sequences into rich tabular feature sets for any ML model.",
    useWhen: "Time series classification or regression; when domain-specific feature engineering is time-consuming; industrial sensor data, EEG, finance.",
    dontUse: "Very long series with many entities (can be slow — use tsfresh's 'efficient' settings); when deep sequence models are feasible.",
    benefitModels: "Any tabular ML model (GBMs, Random Forest, Logistic Regression) after dimensionality reduction.",
    watchModels: "Most features will be noisy — always follow with feature selection (tsfresh provides built-in FDR-controlled selection).",
  },
  {
    name: "Catch22 Features",
    lib: "pycatch22",
    api: "pycatch22.catch22_all(time_series)",
    group: "Time Series Features",
    dataType: "Time Series",
    category: "Time Series",
    description: "A curated set of 22 highly discriminative time series features selected from thousands of candidates to be minimally redundant and maximally informative.",
    achieve: "Compact, interpretable 22-feature representation of any time series. Much faster than tsfresh but still broadly informative.",
    useWhen: "Time series classification when you want a fast, compact feature set; real-time or embedded applications; when tsfresh is too slow.",
    dontUse: "When exhaustive feature search is needed; regression tasks (originally designed for classification).",
    benefitModels: "Random Forest, SVM, Logistic Regression on the 22-feature vector.",
    watchModels: "May not capture all relevant structure for complex regression tasks.",
  },

  // ═══════════════════════════════════════════════════════
  // STATISTICAL TESTS
  // ═══════════════════════════════════════════════════════
  {
    name: "Shapiro-Wilk Test",
    lib: "scipy",
    api: "scipy.stats.shapiro(x)",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Tests whether a sample comes from a normally distributed population. Null hypothesis: the data is normally distributed. Sensitive for n < 5000.",
    achieve: "Confirms or rejects normality assumption before applying linear models, t-tests, ANOVA — all of which assume Gaussian distributions.",
    useWhen: "Before applying parametric tests or linear models; checking residuals of regression; small-to-medium samples (n < 2000 most reliable).",
    dontUse: "Very large samples (n > 5000 — the test will almost always reject normality for trivial deviations); use Lilliefors or visual QQ-plot instead.",
    benefitModels: "Informs when to apply PowerTransformer/log transform before feeding to Linear Regression, LDA, Gaussian NB.",
    watchModels: "Tree models are distribution-agnostic; normality test results don't change their behaviour.",
  },
  {
    name: "Kolmogorov-Smirnov Test (KS Test)",
    lib: "scipy",
    api: "scipy.stats.kstest / ks_2samp(a, b)",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Non-parametric test comparing the empirical CDF of a sample against a reference distribution (one-sample) or between two samples (two-sample). Measures the max CDF difference.",
    achieve: "Detects distribution shift between train and test sets (two-sample KS). Validates normality without assumptions. Essential for data drift monitoring.",
    useWhen: "Train/test distribution shift detection; comparing feature distributions across splits or time periods; validating transformed distributions.",
    dontUse: "Small samples (low power); when you only care about mean differences (t-test is more powerful for that).",
    benefitModels: "Informs all models by flagging potential covariate shift. Critical before deploying any model to production.",
    watchModels: "A significant KS test doesn't specify which part of the distribution differs — combine with visual inspection.",
  },
  {
    name: "Pearson Correlation",
    lib: "scipy · pandas",
    api: "scipy.stats.pearsonr / df.corr(method='pearson')",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Measures the linear relationship between two continuous variables. Returns r ∈ [−1, 1] and a p-value. Assumes bivariate normality for the p-value to be valid.",
    achieve: "Identifies linearly predictive features, multicollinearity between features, and correlations between inputs and target.",
    useWhen: "Initial feature-target correlation screening; detecting multicollinearity (|r| > 0.9 between features); linear relationships.",
    dontUse: "Non-linear relationships (Pearson will miss them — use Spearman or mutual information); ordinal/categorical data.",
    benefitModels: "Guides feature selection for Linear Regression, Ridge, Lasso. High inter-feature correlation → problematic for unregularised linear models.",
    watchModels: "GBM, Random Forest handle correlated features better — but removing near-duplicates still helps efficiency.",
  },
  {
    name: "Spearman / Kendall Rank Correlation",
    lib: "scipy · pandas",
    api: "scipy.stats.spearmanr / df.corr(method='spearman')",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Non-parametric rank-based correlation that captures monotone (not just linear) relationships. Spearman is faster; Kendall (tau) is more robust for small samples with many ties.",
    achieve: "Detects any monotone relationship, including non-linear ones. More robust than Pearson for skewed or ordinal data.",
    useWhen: "Ordinal features; skewed distributions; when you suspect non-linear but monotone relationships; robust alternative to Pearson.",
    dontUse: "When detecting bidirectional non-monotone relationships (e.g., U-shaped) — use mutual information instead.",
    benefitModels: "Informs feature selection for all models. Kendall's tau used internally by some non-parametric tests.",
    watchModels: "Rank correlation is insensitive to non-monotone patterns — don't use alone.",
  },
  {
    name: "Chi-Square Test (χ²)",
    lib: "scipy · sklearn",
    api: "scipy.stats.chi2_contingency / sklearn.feature_selection.chi2",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Tests statistical dependence between two categorical variables via their contingency table (scipy), or between categorical features and a categorical target for feature selection (sklearn).",
    achieve: "Screens categorical features for independence from the target. The SelectKBest(chi2) pipeline is a fast filter-based feature selection for text/categorical data.",
    useWhen: "Selecting categorical/count features for text classification; testing independence in a contingency table; survey/medical data analysis.",
    dontUse: "Continuous features (use F-test or mutual information); small expected cell counts (< 5) — test loses validity.",
    benefitModels: "MultinomialNB, Logistic Regression, LinearSVC for text classification with SelectKBest(chi2) preprocessing.",
    watchModels: "Tree models — chi2-selected features are still useful but selection is less necessary.",
  },
  {
    name: "ANOVA F-Test (Feature Relevance)",
    lib: "scipy · sklearn",
    api: "scipy.stats.f_oneway / sklearn.feature_selection.f_classif",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Tests whether the mean of a continuous feature differs significantly across class labels (f_classif) or whether a continuous feature is linearly related to a continuous target (f_regression). Assumes normality.",
    achieve: "Fast linear feature-target relevance score for every feature. The SelectKBest(f_classif) pipeline is the canonical univariate filter selection.",
    useWhen: "Quick linear relevance screening; hundreds of features; complement to mutual information for linear relationships.",
    dontUse: "Non-linear feature-target relationships (F-test will miss them); non-normal features with small samples.",
    benefitModels: "Linear/Logistic Regression, LDA, SVM — features selected by F-test are well-aligned with linear model assumptions.",
    watchModels: "GBMs capture non-linear interactions that F-test misses; MI or SHAP is better for tree model feature screening.",
  },
  {
    name: "Mutual Information (MI)",
    lib: "sklearn",
    api: "mutual_info_classif / mutual_info_regression",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Measures how much information a feature provides about the target variable, capturing ANY dependency (not just linear). Based on entropy estimation.",
    achieve: "Non-linear feature relevance score that detects all statistical dependence, including non-linear, non-monotone relationships.",
    useWhen: "General-purpose feature relevance screening; when F-test or correlation miss non-linear patterns; best default for filter-based selection.",
    dontUse: "Very small samples (entropy estimators are noisy); as the sole selection criterion for highly correlated feature sets (select one per redundancy group).",
    benefitModels: "All models — MI identifies features relevant to the target regardless of model type.",
    watchModels: "High MI doesn't guarantee model improvement if redundant features are selected.",
  },
  {
    name: "Variance Inflation Factor (VIF)",
    lib: "statsmodels",
    api: "variance_inflation_factor(X, feature_index)",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Measures how much the variance of a regression coefficient is inflated due to multicollinearity. VIF = 1/(1 − R²_i). VIF > 10 signals severe collinearity.",
    achieve: "Identifies multicollinear features in linear models, guiding which features to remove or combine. Standard diagnostic before linear regression.",
    useWhen: "Before fitting unregularised Linear Regression or interpreting coefficients; checking that OHE doesn't create linear dependencies.",
    dontUse: "Tree-based models (collinearity doesn't bias their splits); when Ridge/Lasso regularisation is already handling collinearity.",
    benefitModels: "Linear Regression, Logistic Regression, LDA — removing high-VIF features stabilises coefficient estimates.",
    watchModels: "Ridge regression tolerates high VIF; VIF reduction is less critical when regularisation is used.",
  },
  {
    name: "Mann-Whitney U Test",
    lib: "scipy",
    api: "scipy.stats.mannwhitneyu(a, b)",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Non-parametric alternative to the independent t-test. Tests whether two independent samples come from the same distribution without assuming normality. Equivalent to testing whether one group tends to have higher values.",
    achieve: "Robust group comparison for non-normal or ordinal data. Used in A/B testing, medical trials, and feature relevance for binary targets.",
    useWhen: "Comparing feature distributions between two classes; A/B test analysis with non-Gaussian metrics; small samples.",
    dontUse: "More than two groups (use Kruskal-Wallis); when samples are paired (use Wilcoxon signed-rank).",
    benefitModels: "Guides feature selection for binary classification — features that differ between classes are predictive.",
    watchModels: "No direct model effect; informs data preprocessing and feature selection decisions.",
  },
  {
    name: "Cramér's V",
    lib: "scipy (manual) · pingouin",
    api: "pingouin.cramers_v(a, b) or manual via chi2",
    group: "Statistical Tests",
    dataType: "Tabular",
    category: "Statistical Tests",
    description: "Measures the association between two categorical variables. A normalised version of chi-square that returns values in [0, 1] regardless of table dimensions.",
    achieve: "Detects redundant categorical features (high Cramér's V between features means one is largely derivable from the other). Categorical correlation matrix.",
    useWhen: "Checking categorical feature-to-feature correlation; feature selection for categorical datasets; multi-class target correlation.",
    dontUse: "Continuous features (use Pearson/Spearman); very sparse contingency tables.",
    benefitModels: "All models benefit from removing redundant categorical features identified by high Cramér's V.",
    watchModels: "No direct model impact; used for data understanding and selection.",
  },

  // ═══════════════════════════════════════════════════════
  // FEATURE SELECTION
  // ═══════════════════════════════════════════════════════
  {
    name: "Variance Threshold",
    lib: "sklearn",
    api: "VarianceThreshold(threshold=0.0)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Removes all features whose variance falls below a threshold. At threshold=0, removes constant features. At higher thresholds, removes near-constant features.",
    achieve: "Cheapest possible feature selection step. Eliminates features that carry no discriminative information because they don't vary.",
    useWhen: "First step in any feature selection pipeline; removing constant/quasi-constant features; after OHE when some dummy columns are all-zero.",
    dontUse: "As the only selection step; doesn't consider feature-target relationship at all.",
    benefitModels: "All models — constant features add parameters without information.",
    watchModels: "Low-variance features may still be predictive in specific contexts (e.g., a binary indicator that's rarely 1 but always indicates fraud).",
  },
  {
    name: "Univariate Filter (SelectKBest / SelectPercentile)",
    lib: "sklearn",
    api: "SelectKBest(score_func=f_classif, k=50)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Selects the top k features based on a univariate statistical score (F-test, chi-square, mutual information). Each feature is scored independently of others — no interaction considered.",
    achieve: "Fast, scalable filter selection. Good first pass before more expensive wrapper or embedded methods.",
    useWhen: "High-dimensional data where wrapper methods are too slow; preliminary reduction; NLP (TF-IDF + chi2 is a standard pipeline).",
    dontUse: "When feature interactions are important (filter ignores them); as the sole selection method for small-to-medium datasets.",
    benefitModels: "Linear models, MLP — removes noise features that hurt convergence.",
    watchModels: "GBMs can handle many irrelevant features reasonably well, but selection still speeds training.",
  },
  {
    name: "Recursive Feature Elimination (RFE / RFECV)",
    lib: "sklearn",
    api: "RFECV(estimator=LogisticRegression(), step=1, cv=5)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Wrapper method that recursively trains a model, removes the least important feature(s) according to the model's coefficients or importances, and repeats. RFECV adds cross-validation to find the optimal feature count.",
    achieve: "Finds a minimal feature subset that maximises model performance. Considers feature interactions implicitly through the model.",
    useWhen: "Small-to-medium datasets; interpretable model required; when you want a compact, well-performing feature set.",
    dontUse: "Very high-dimensional data (too slow — O(d²) model fits); large datasets (use embedded methods instead).",
    benefitModels: "Linear/Logistic Regression, SVM — RFE is most principled when the base estimator has stable feature importances.",
    watchModels: "GBMs have unstable feature importances that make RFE less reliable; prefer Boruta or SHAP for tree models.",
  },
  {
    name: "Lasso Embedded Selection (SelectFromModel)",
    lib: "sklearn",
    api: "SelectFromModel(Lasso(alpha=0.01))",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Trains a regularised model (Lasso/ElasticNet) and selects features whose coefficients are non-zero. The L1 penalty inherently drives irrelevant feature coefficients to exactly zero.",
    achieve: "Automatic, data-driven feature selection embedded in regularised regression. Produces a sparse model in one step.",
    useWhen: "Linear regression/classification with many features; automatic sparsity needed; when feature selection and modelling should be simultaneous.",
    dontUse: "When correlated features need to be handled together (Lasso arbitrarily picks one); tree-based target models.",
    benefitModels: "Logistic Regression, Ridge — remove Lasso-selected features for interpretability, then refit with Ridge for stability.",
    watchModels: "GBMs — tree feature importance is more appropriate for embedded selection with tree models.",
  },
  {
    name: "Tree-Based Feature Importance",
    lib: "sklearn",
    api: "RandomForestClassifier.feature_importances_ / SelectFromModel",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Uses the mean decrease in impurity (MDI) from a trained tree ensemble to rank features. SelectFromModel with a tree estimator then keeps the top features.",
    achieve: "Non-linear feature relevance ranking that captures both main effects and interactions. Computationally free once the model is trained.",
    useWhen: "Tree-based pipelines; quick importance ranking after Random Forest or GBM training; non-linear feature screening.",
    dontUse: "As the sole criterion — MDI importances are biased toward high-cardinality features; use permutation importance or SHAP for reliable ranking.",
    benefitModels: "GBMs, Random Forest — selecting top importance features reduces memory and speeds up training.",
    watchModels: "Linear models may not benefit from features selected by tree importance (non-linear signal).",
  },
  {
    name: "Permutation Importance",
    lib: "sklearn",
    api: "sklearn.inspection.permutation_importance(model, X_val, y_val)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Shuffles each feature independently and measures the drop in model performance on a validation set. Features whose shuffling hurts performance most are most important. Model-agnostic.",
    achieve: "Reliable, unbiased feature importance on validation data. Not affected by high cardinality or feature correlations (unlike MDI). Works for any model.",
    useWhen: "Post-training importance analysis; when MDI importances seem biased; validating which features truly matter for the fitted model.",
    dontUse: "On training data (overfits); very large datasets (slow — n_features × n_permutations × model evaluations).",
    benefitModels: "All models — model-agnostic. Especially useful for black-box models (MLP, SVMs).",
    watchModels: "Correlated features will split importance between them — neither appears highly important even if the pair is.",
  },
  {
    name: "SHAP Values (SHapley Additive exPlanations)",
    lib: "shap",
    api: "shap.TreeExplainer(model).shap_values(X) / shap.summary_plot",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Game-theory-based method that assigns each feature a Shapley value — its average marginal contribution to the prediction across all feature coalitions. Consistent, locally and globally faithful importances.",
    achieve: "The gold standard for model interpretability AND feature selection. |Mean SHAP| is the most reliable global importance metric. Also reveals feature interaction and directional effects.",
    useWhen: "Final feature importance analysis; competition leaderboard pushes; model explainability for stakeholders; detecting features that hurt performance (negative SHAP impact).",
    dontUse: "Very fast pipelines (SHAP computation adds overhead); as a replacement for EDA (use statistical tests first).",
    benefitModels: "TreeExplainer is fast for GBMs/Random Forests. KernelExplainer works for any model but is slow.",
    watchModels: "Correlated features split SHAP values between them — both may appear less important than the pair truly is.",
  },
  {
    name: "Boruta",
    lib: "boruta-py",
    api: "BorutaPy(estimator=RandomForestClassifier(), n_estimators='auto')",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Wrapper method that compares each feature's importance to the importance of shuffled 'shadow' features (random noise). A feature is confirmed if it consistently beats all shadow features.",
    achieve: "Conservative, statistically principled all-relevant feature selection. Finds ALL features relevant to the target, not just the minimal set. Highly popular in Kaggle competitions.",
    useWhen: "When you want all relevant features (not just the most important); tree-based pipelines; competition feature selection; moderate-size datasets.",
    dontUse: "Very large datasets (slow — runs many RF iterations); when only a minimal feature set is needed (use Lasso instead).",
    benefitModels: "GBMs, Random Forests — Boruta uses RF importances internally. Best output for tree-based downstream models.",
    watchModels: "Results may not transfer well to linear models (different importance landscape).",
  },
  {
    name: "Correlation-Based Elimination",
    lib: "pandas · feature-engine",
    api: "SmartCorrelatedSelection(threshold=0.9)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Identifies groups of highly correlated features (|r| above a threshold) and drops all but the most important within each group. Removes redundancy without losing predictive information.",
    achieve: "Reduces multicollinearity and dimensionality without losing target-relevant signal. Essential step after automated feature generation (DFS, tsfresh).",
    useWhen: "After feature generation that produces correlated features; before linear models to reduce VIF; after DFS to cull redundant aggregations.",
    dontUse: "When all correlated features carry unique but related information (e.g., in ensemble stacking).",
    benefitModels: "Linear Regression, Logistic Regression, MLP, KNN — all benefit from de-correlation.",
    watchModels: "GBMs are relatively robust to correlated features but still benefit from the reduced dimensionality.",
  },
  {
    name: "Forward / Backward Stepwise Selection",
    lib: "mlxtend · sklearn (SequentialFeatureSelector)",
    api: "SequentialFeatureSelector(estimator, direction='forward', cv=5)",
    group: "Feature Selection",
    dataType: "Any",
    category: "Feature Selection",
    description: "Forward: starts from zero features, adds the best one at each step. Backward: starts from all features, removes the worst one at each step. Evaluates using cross-validation score.",
    achieve: "Finds a feature subset optimised for the specific model via greedy search. Considers feature combinations (unlike filter methods).",
    useWhen: "Small feature counts (< 50); when a principled subset with model-in-the-loop selection is needed; interpretable model building.",
    dontUse: "High-dimensional data (O(d²) model fits — extremely slow); when Boruta or SHAP-based selection is feasible.",
    benefitModels: "Linear/Logistic Regression, SVM — most principled when the base model is the final model.",
    watchModels: "GBMs — Boruta is faster and better for tree models.",
  },
];

const ALL_CATEGORIES = ["All", "Categorical Encoding", "Scaling & Transformation", "Tabular Feature Creation",
  "Text / NLP", "Image / CV", "Time Series", "Statistical Tests", "Feature Selection"];

const ALL_DATA_TYPES = ["All", "Tabular", "Text / NLP", "Image / CV", "Time Series", "Any"];

function DataTypeBadge({ type }) {
  const s = DATA_TYPE_COLORS[type] || { bg: "#2a2a3a", color: "#aaa" };
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: "4px",
      fontSize: "0.65rem", fontWeight: 700, letterSpacing: "0.05em",
      textTransform: "uppercase", background: s.bg, color: s.color,
      border: `1px solid ${s.color}33`, whiteSpace: "nowrap",
    }}>{type}</span>
  );
}

function LibBadge({ text }) {
  return (
    <span style={{
      display: "inline-block", padding: "1px 7px", borderRadius: "3px",
      fontSize: "0.63rem", fontWeight: 600, letterSpacing: "0.03em",
      background: "var(--bg-elevated)", color: "#7070aa",
      border: "1px solid var(--border-default)", whiteSpace: "nowrap", margin: "1px",
    }}>{text}</span>
  );
}

function DetailCard({ icon, label, text, accent }) {
  return (
    <div style={{
      padding: "10px 14px", borderRadius: "6px",
      background: "var(--bg-surface)", border: `1px solid ${accent}22`,
      marginRight: "8px",
    }}>

      <div style={{ fontSize: "0.66rem", fontWeight: 700, letterSpacing: "0.08em", color: accent, textTransform: "uppercase", marginBottom: "5px" }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: "0.8rem", color: "var(--text-primary)", lineHeight: 1.55 }}>{text}</div>
    </div>
  );
}

function Row({ item, idx }) {
  const [open, setOpen] = useState(false);
  const accent = GROUP_ACCENT[item.group] || "#888";
  const even = idx % 2 === 0;

  return (
    <>
      <tr
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background = "var(--bg-elevated)"}
        onMouseLeave={e => e.currentTarget.style.background = even ? "var(--bg-surface)" : "#0f0f1a"}
        style={{
          cursor: "pointer",
          background: even ? "var(--bg-surface)" : "#0f0f1a",
          borderLeft: `3px solid ${accent}`,
          transition: "background 0.12s",
        }}
      >
        <td style={tdStyle}>{open ? "▾" : "▸"}</td>
        <td style={{ ...tdStyle, fontWeight: 700, color: accent, fontSize: "0.88rem" }}>{item.name}</td>
        <td style={tdStyle}>
          {item.lib.split("·").map(l => <LibBadge key={l} text={l.trim()} />)}
        </td>
        <td style={tdStyle}><DataTypeBadge type={item.dataType} /></td>
        <td style={{ ...tdStyle, fontFamily: "var(--font-mono)", color: "var(--text-tertiary)", fontSize: "0.72rem" }}>{item.api}</td>
      </tr>
      {open && (
        <tr style={{ background: "var(--bg-surface)", borderLeft: `3px solid ${accent}` }}>
          <td colSpan={5} style={{ padding: "0 0 0 18px" }}>
            <div style={{ padding: "14px 18px 14px 0" }}>
              {/* Top row: Description + Achieve */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "8px" }}>
                <DetailCard icon="📋" label="How It Works" text={item.description} accent={accent} />
                <DetailCard icon="🎯" label="What You Achieve" text={item.achieve} accent="#facc15" />
              </div>
              {/* Middle row: Use When + Don't Use */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "8px" }}>
                <DetailCard icon="✅" label="Use When" text={item.useWhen} accent="#4ade80" />
                <DetailCard icon="⛔" label="Don't Use When" text={item.dontUse} accent="#f87171" />
              </div>
              {/* Bottom row: Model benefit + Watch */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                <DetailCard icon="🚀" label="Models That Benefit Most" text={item.benefitModels} accent="#818cf8" />
                <DetailCard icon="⚠️" label="Models / Caveats to Watch" text={item.watchModels} accent="#fb923c" />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

const tdStyle = {
  padding: "10px 14px",
  verticalAlign: "middle",
  borderBottom: "1px solid var(--border-subtle)",
  fontSize: "0.84rem",
  color: "var(--text-primary)",
};

const thStyle = {
  padding: "11px 14px",
  textAlign: "left",
  fontSize: "0.68rem",
  fontWeight: 700,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  color: "var(--text-tertiary)",
  borderBottom: "2px solid #1e1e3e",
  background: "#09090f",
  position: "sticky",
  // top: "calc(var(--header-h))",
  zIndex: 5,
};

export default function Feature_app() {
  const [search, setSearch] = useState("");
  const [cat, setCat] = useState("All");
  const [dt, setDt] = useState("All");

  const [filterRef, filterH] = useStickyOffset(80);
  const thS = {
    padding: "10px 13px",
    textAlign: "left",
    // ... all your other th styles ...
    position: "sticky",
    // header nav (44px) + actual live height of the filter bar
    top: `calc(var(--header-h) + ${filterH}px)`,
    zIndex: 5,
  };

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return FEATURES.filter(f => {
      const matchCat = cat === "All" || f.category === cat;
      const matchDt = dt === "All" || f.dataType === dt || f.dataType === "Any";
      const matchQ = !q || [f.name, f.lib, f.description, f.achieve, f.useWhen, f.benefitModels, f.watchModels, f.group]
        .some(s => s.toLowerCase().includes(q));
      return matchCat && matchDt && matchQ;
    });
  }, [search, cat, dt]);

  const groups = useMemo(() => {
    const g = {};
    for (const f of filtered) {
      if (!g[f.group]) g[f.group] = [];
      g[f.group].push(f);
    }
    return g;
  }, [filtered]);

  const counts = useMemo(() => {
    const c = {};
    for (const f of FEATURES) c[f.category] = (c[f.category] || 0) + 1;
    return c;
  }, []);

  return (
    <div>
    <Header />

        <div style={{ fontFamily: "var(--font-body)", background: "var(--bg-base)", minHeight: "100vh", color: "var(--text-primary)" }}>
      <style>{`
* { box-sizing: border-box; }
        button { cursor: pointer; font-family: inherit; transition: all 0.15s; }
        input { outline: none; font-family: inherit; }
`}</style>
      {/* ── HEADER ── */}
      <div ref={filterRef} style={{
        position: "sticky", top: "var(--header-h)", zIndex: 10,
        background: "rgba(14,13,12,0.93)",
        backdropFilter: "blur(8px)",
        borderBottom: "1px solid var(--bg-overlay)",
        padding: "10px 20px 8px",
      }}>
        {/* Title row */}
        <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "8px", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
              <span style={{ color: "#4ade80" }}>Feature Engineering</span>
              <span style={{ color: "var(--text-tertiary)", margin: "0 6px" }}>·</span>
              <span style={{ color: "#fb923c" }}>Statistical Tests</span>
              <span style={{ color: "var(--text-tertiary)", margin: "0 6px" }}>·</span>
              <span style={{ color: "#c084fc" }}>Feature Selection</span>
            </div>
            <div style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", letterSpacing: "0.05em", marginTop: "1px" }}>
              {FEATURES.length} techniques · sklearn · scipy · statsmodels · category_encoders · featuretools · tsfresh · shap · and more · click any row to expand
            </div>
          </div>
          <div style={{ flex: 1, minWidth: "180px", maxWidth: "300px", marginLeft: "auto" }}>
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search techniques, models, use-cases…"
              style={{
                width: "100%", background: "var(--bg-surface)", border: "1px solid var(--border-default)",
                borderRadius: "6px", padding: "7px 11px", color: "var(--text-primary)", fontSize: "0.8rem",
              }}
            />
          </div>
          <div style={{ fontSize: "0.72rem", color: "var(--text-tertiary)", whiteSpace: "nowrap" }}>
            {filtered.length} shown
          </div>
        </div>

        {/* Data type filter */}
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap", marginBottom: "5px" }}>
          <span style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", alignSelf: "center", marginRight: "2px" }}>DATA TYPE:</span>
          {ALL_DATA_TYPES.map(d => {
            const s = DATA_TYPE_COLORS[d] || {};
            const active = dt === d;
            return (
              <button key={d} onClick={() => setDt(d)} style={{
                padding: "3px 10px", borderRadius: "4px", fontSize: "0.68rem", fontWeight: active ? 700 : 400,
                border: active ? `1px solid ${s.color || "#888"}` : "1px solid var(--border-default)",
                background: active ? (s.bg || "var(--bg-elevated)") : "transparent",
                color: active ? (s.color || "#ccc") : "var(--text-tertiary)",
              }}>{d}</button>
            );
          })}
        </div>

        {/* Category filter */}
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "0.65rem", color: "var(--text-tertiary)", alignSelf: "center", marginRight: "2px" }}>CATEGORY:</span>
          {ALL_CATEGORIES.map(c => {
            const active = cat === c;
            const accent = c === "All" ? "#888" : (GROUP_ACCENT[c] || "#888");
            return (
              <button key={c} onClick={() => setCat(c)} style={{
                padding: "3px 10px", borderRadius: "4px", fontSize: "0.68rem", fontWeight: active ? 700 : 400,
                border: active ? `1px solid ${accent}` : "1px solid var(--border-default)",
                background: active ? `${accent}22` : "transparent",
                color: active ? accent : "var(--text-tertiary)",
              }}>
                {c}{c !== "All" && counts[c] ? ` (${counts[c]})` : ""}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── TABLE ── */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "820px" }}>
          <thead>
            <tr>
              <th style={{ ...thStyle, width: "24px" }}></th>
              <th style={thStyle}>Technique / Feature</th>
              <th style={thStyle}>Library</th>
              <th style={thStyle}>Data Type</th>
              <th style={thStyle}>API / Usage</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([group, items]) => (
              <>
                <tr key={group + "_hdr"}>
                  <td colSpan={5} style={{
                    padding: "9px 14px 5px",
                    fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: GROUP_ACCENT[group] || "#6a6aaa",
                    background: "var(--bg-surface)",
                    borderTop: "2px solid var(--border-subtle)",
                    borderBottom: "1px solid var(--border-subtle)",
                  }}>
                    ▪ {group}
                    <span style={{ fontWeight: 400, color: "var(--text-dim)", marginLeft: 6 }}>({items.length})</span>
                  </td>
                </tr>
                {items.map((item, i) => <Row key={item.name} item={item} idx={i} />)}
              </>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: "60px", textAlign: "center", color: "var(--text-tertiary)" }}>
                  No techniques match your filter. Try a different search or reset the filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* ── FOOTER ── */}
      <div style={{ padding: "20px", textAlign: "center", fontSize: "0.68rem", color: "var(--border-default)", borderTop: "1px solid var(--border-faint)" }}>
        Covers sklearn · scipy · statsmodels · category_encoders · featuretools · tsfresh · shap · boruta-py · sentence-transformers · opencv · spaCy · nltk · textstat · pycatch22 · mlxtend
      </div>
    </div></div>
  );
}