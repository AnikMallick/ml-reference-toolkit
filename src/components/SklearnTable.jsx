import { useState, useMemo } from "react";
import Header from "./Header";
const MODELS = [
  // ── LINEAR MODELS ──────────────────────────────────────────────────────
  {
    name: "Linear Regression (OLS)",
    cls: "LinearRegression",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Fits a linear relationship between features and a continuous target by minimising the sum of squared residuals (Ordinary Least Squares).",
    useWhen: "Baseline regression; relationship between features and target is approximately linear; data is not too high-dimensional; interpretability is required.",
    dontUse: "Non-linear relationships; multicollinear features (use Ridge/Lasso instead); high-dimensional settings where p ≫ n.",
    realWorld: "Housing price estimation, economic forecasting, A/B test effect sizing, risk scoring baselines.",
  },
  {
    name: "Ridge Regression",
    cls: "Ridge",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Linear regression with L2 regularisation (shrinks coefficients toward zero). Controls overfitting and handles multicollinearity.",
    useWhen: "Many correlated features; want all features to contribute (L2 keeps all); slightly better generalisation than plain OLS.",
    dontUse: "Feature selection is needed (Lasso does sparse selection); extreme non-linearity.",
    realWorld: "Genomics (many correlated gene features), financial risk models, collaborative filtering.",
  },
  {
    name: "Lasso",
    cls: "Lasso",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Linear regression with L1 regularisation. Drives many coefficients exactly to zero, performing automatic feature selection.",
    useWhen: "Sparse signal expected; built-in feature selection is desired; interpretable, parsimonious models.",
    dontUse: "Correlated features (arbitrarily picks one and zeroes rest); very large datasets where coordinate descent is slow.",
    realWorld: "Medical biomarker discovery, text regression, gene expression analysis.",
  },
  {
    name: "ElasticNet",
    cls: "ElasticNet",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Combines L1 and L2 penalties. Inherits sparsity from Lasso and handles correlated features like Ridge.",
    useWhen: "Correlated features AND sparsity is desired; large datasets with many features.",
    dontUse: "When simplicity of pure Ridge or Lasso is preferred; non-linear settings.",
    realWorld: "Genomics with correlated SNPs, NLP feature selection, multi-label regression.",
  },
  {
    name: "Logistic Regression",
    cls: "LogisticRegression",
    module: "linear_model",
    category: "Classification",
    group: "Linear Models",
    description: "Linear model for classification. Models class probabilities via the logistic (sigmoid) function. Supports L1/L2/ElasticNet regularisation and multi-class.",
    useWhen: "Binary or multi-class classification; need calibrated probabilities; interpretable model; large datasets.",
    dontUse: "Complex non-linear decision boundaries; very high-dimensional text (consider LinearSVC); purely non-linear data.",
    realWorld: "Credit scoring, disease diagnosis, spam detection, CTR prediction (as a fast baseline).",
  },
  {
    name: "SGD Classifier / Regressor",
    cls: "SGDClassifier / SGDRegressor",
    module: "linear_model",
    category: "Classification / Regression",
    group: "Linear Models",
    description: "Stochastic Gradient Descent optimiser for linear models. Supports many loss functions (hinge, log, huber, etc.). Extremely scalable.",
    useWhen: "Very large datasets that don't fit in memory; online/streaming learning; when speed matters more than optimal tuning.",
    dontUse: "Small datasets (batch methods converge more reliably); when hyper-sensitivity to learning rate is a problem.",
    realWorld: "Real-time ad click prediction, online fraud detection, text classification at scale.",
  },
  {
    name: "Bayesian Ridge",
    cls: "BayesianRidge",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Bayesian formulation of Ridge regression. Infers the regularisation parameter automatically and outputs uncertainty estimates.",
    useWhen: "Uncertainty quantification needed; small datasets; automatic regularisation without cross-validation.",
    dontUse: "Very large datasets (slow due to matrix inversion); when uncertainty estimates are not needed.",
    realWorld: "Scientific measurement, medical dosage prediction, small-sample economic models.",
  },
  {
    name: "HuberRegressor",
    cls: "HuberRegressor",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Robust linear regression that uses the Huber loss — quadratic for small residuals and linear for large ones, reducing sensitivity to outliers.",
    useWhen: "Dataset has significant outliers; linear relationship holds but data is noisy.",
    dontUse: "Clean, Gaussian-noise data (OLS is fine); non-linear relationships.",
    realWorld: "Sensor data regression, financial returns (fat tails), clinical data with recording errors.",
  },
  {
    name: "Quantile Regressor",
    cls: "QuantileRegressor",
    module: "linear_model",
    category: "Regression",
    group: "Linear Models",
    description: "Estimates a specified quantile (e.g. median or 90th percentile) of the conditional distribution rather than the mean.",
    useWhen: "Prediction intervals needed; asymmetric error costs; heteroscedastic data; outlier-robust median regression.",
    dontUse: "Standard mean prediction suffices; very large datasets (slow LP solver).",
    realWorld: "Insurance claim severity, delivery time estimation (upper quantile), load forecasting.",
  },
  {
    name: "Passive Aggressive",
    cls: "PassiveAggressiveClassifier / Regressor",
    module: "linear_model",
    category: "Classification / Regression",
    group: "Linear Models",
    description: "Online learning algorithm — stays passive on correctly classified samples, updates aggressively on mistakes. No learning rate required.",
    useWhen: "Streaming / online learning; large-scale text classification; concept drift scenarios.",
    dontUse: "Small batch datasets; when probabilistic output is needed.",
    realWorld: "Real-time news categorisation, online spam filtering, sentiment tracking.",
  },

  // ── DISCRIMINANT ANALYSIS ─────────────────────────────────────────────
  {
    name: "Linear Discriminant Analysis",
    cls: "LinearDiscriminantAnalysis",
    module: "discriminant_analysis",
    category: "Classification",
    group: "Discriminant Analysis",
    description: "Finds linear combinations of features that maximise class separation. Assumes Gaussian class-conditional distributions with a shared covariance matrix. Also used for dimensionality reduction.",
    useWhen: "Multi-class classification with Gaussian features; also doubles as supervised dimensionality reduction (max n_classes-1 components).",
    dontUse: "Features are clearly non-Gaussian; covariance matrices differ greatly per class (use QDA); very high-dimensional data.",
    realWorld: "Face recognition preprocessing, EEG brain-computer interfaces, medical triage.",
  },
  {
    name: "Quadratic Discriminant Analysis",
    cls: "QuadraticDiscriminantAnalysis",
    module: "discriminant_analysis",
    category: "Classification",
    group: "Discriminant Analysis",
    description: "Like LDA but allows each class to have its own covariance matrix, creating quadratic (curved) decision boundaries.",
    useWhen: "Class covariance structures differ; moderate-sized datasets; Gaussian-ish features.",
    dontUse: "High-dimensional data (requires inverting per-class covariance matrices); small per-class samples.",
    realWorld: "Multi-class sensor classification, object recognition features, medical diagnosis.",
  },

  // ── SVM ───────────────────────────────────────────────────────────────
  {
    name: "SVC (Support Vector Classifier)",
    cls: "SVC",
    module: "svm",
    category: "Classification",
    group: "Support Vector Machines",
    description: "Finds the maximum-margin hyperplane separating classes. Kernel trick (RBF, polynomial, sigmoid) enables non-linear boundaries. Robust in high-dimensional spaces.",
    useWhen: "Small-to-medium datasets; high-dimensional data (text, images); non-linear boundaries with RBF kernel; data that is not linearly separable.",
    dontUse: "Very large datasets (O(n²) to O(n³) training); need fast prediction at scale; need probabilities without costly Platt scaling.",
    realWorld: "Text classification, image classification (pre-deep learning), bioinformatics, handwriting recognition.",
  },
  {
    name: "LinearSVC",
    cls: "LinearSVC",
    module: "svm",
    category: "Classification",
    group: "Support Vector Machines",
    description: "SVM with a linear kernel optimised via LibLinear — much faster than SVC(kernel='linear') for large datasets. Does not support kernel trick.",
    useWhen: "Large-scale text classification; linearly separable data; fast training needed.",
    dontUse: "Non-linear decision boundaries; need probability estimates.",
    realWorld: "Document classification (TF-IDF features), spam filtering, named entity recognition.",
  },
  {
    name: "SVR (Support Vector Regressor)",
    cls: "SVR",
    module: "svm",
    category: "Regression",
    group: "Support Vector Machines",
    description: "Regression version of SVM. Tries to fit the best hyperplane within an epsilon-margin tolerance. Supports kernels.",
    useWhen: "Non-linear regression on small-to-medium datasets; robust to outliers within epsilon margin.",
    dontUse: "Large datasets; need interpretability.",
    realWorld: "Energy load forecasting, stock price movement, time-series forecasting (small datasets).",
  },
  {
    name: "OneClassSVM",
    cls: "OneClassSVM",
    module: "svm",
    category: "Anomaly Detection",
    group: "Support Vector Machines",
    description: "Unsupervised novelty detector. Learns a decision boundary enclosing the training distribution and flags outliers.",
    useWhen: "Only normal data is available for training; novelty detection; anomaly detection without labelled anomalies.",
    dontUse: "Large datasets (slow kernel SVM); when IsolationForest or LOF would suffice.",
    realWorld: "Fault detection in manufacturing, fraud detection, intrusion detection systems.",
  },

  // ── NEAREST NEIGHBORS ─────────────────────────────────────────────────
  {
    name: "K-Nearest Neighbors Classifier",
    cls: "KNeighborsClassifier",
    module: "neighbors",
    category: "Classification",
    group: "Nearest Neighbors",
    description: "Non-parametric lazy learner. Classifies by majority vote among the k nearest training points in feature space.",
    useWhen: "Small to medium datasets; non-linear boundaries; multi-class; quick baseline; data is dense in meaningful feature space.",
    dontUse: "High-dimensional data (curse of dimensionality); very large datasets (slow inference); sparse data.",
    realWorld: "Recommendation system baseline, anomaly detection, image recognition (low-dim features), medical imaging.",
  },
  {
    name: "K-Nearest Neighbors Regressor",
    cls: "KNeighborsRegressor",
    module: "neighbors",
    category: "Regression",
    group: "Nearest Neighbors",
    description: "Predicts the target as the average (or weighted average) of the k nearest neighbours' values.",
    useWhen: "Local, smooth regression surfaces; quick non-parametric baseline.",
    dontUse: "High-dimensional data; large-scale production (high latency); extrapolation beyond training range.",
    realWorld: "Missing value imputation, house price estimation (geo-spatial), traffic prediction.",
  },

  // ── GAUSSIAN PROCESSES ────────────────────────────────────────────────
  {
    name: "Gaussian Process Regressor",
    cls: "GaussianProcessRegressor",
    module: "gaussian_process",
    category: "Regression",
    group: "Gaussian Processes",
    description: "Non-parametric Bayesian regression. Provides mean predictions AND uncertainty estimates via a kernel-defined covariance structure.",
    useWhen: "Small datasets; uncertainty quantification is critical; Bayesian optimisation surrogate; smooth, continuous functions.",
    dontUse: "Large datasets (O(n³) inversion); high-dimensional inputs; when uncertainty is not needed.",
    realWorld: "Bayesian hyperparameter optimisation (e.g. Optuna internally), scientific experiment modelling, robotics.",
  },
  {
    name: "Gaussian Process Classifier",
    cls: "GaussianProcessClassifier",
    module: "gaussian_process",
    category: "Classification",
    group: "Gaussian Processes",
    description: "GP-based probabilistic classifier. Applies a GP latent function then squashes probabilities. Provides calibrated uncertainty.",
    useWhen: "Small datasets with need for well-calibrated probabilities; active learning; uncertainty-aware classification.",
    dontUse: "Large datasets; fast inference required; many classes.",
    realWorld: "Active learning pipelines, medical decision support, small-scale scientific classification.",
  },

  // ── NAIVE BAYES ───────────────────────────────────────────────────────
  {
    name: "Gaussian Naive Bayes",
    cls: "GaussianNB",
    module: "naive_bayes",
    category: "Classification",
    group: "Naive Bayes",
    description: "Applies Bayes' theorem with the assumption that features are continuous and normally distributed within each class.",
    useWhen: "Continuous features, fast training needed, small datasets, baseline model, real-time prediction.",
    dontUse: "Strong feature correlations (violates independence assumption); complex, non-Gaussian distributions.",
    realWorld: "Sensor classification, medical diagnosis baselines, real-time scoring pipelines.",
  },
  {
    name: "Multinomial Naive Bayes",
    cls: "MultinomialNB",
    module: "naive_bayes",
    category: "Classification",
    group: "Naive Bayes",
    description: "Naive Bayes for discrete, count-based features (e.g. word counts in documents). Very fast and surprisingly effective for text.",
    useWhen: "Text classification with count/TF features; document categorisation; fast, interpretable model.",
    dontUse: "Negative feature values; continuous non-count data; when feature correlations are important.",
    realWorld: "Spam detection, sentiment analysis, news categorisation, intent classification.",
  },
  {
    name: "Bernoulli Naive Bayes",
    cls: "BernoulliNB",
    module: "naive_bayes",
    category: "Classification",
    group: "Naive Bayes",
    description: "Naive Bayes for binary/boolean features. Models the presence or absence of features.",
    useWhen: "Binary feature vectors (e.g. word presence in text); short texts where absence of words is informative.",
    dontUse: "Count/frequency data (MultinomialNB is better); continuous features.",
    realWorld: "Document classification with binary bag-of-words, email spam filtering.",
  },
  {
    name: "Complement Naive Bayes",
    cls: "ComplementNB",
    module: "naive_bayes",
    category: "Classification",
    group: "Naive Bayes",
    description: "Adaptation of MultinomialNB that uses the complement of each class for estimation. Better suited for imbalanced datasets.",
    useWhen: "Text classification with class imbalance; often outperforms MultinomialNB on text tasks.",
    dontUse: "Balanced datasets where standard MultinomialNB works fine.",
    realWorld: "Imbalanced news categorisation, rare category detection in text corpora.",
  },

  // ── DECISION TREES ────────────────────────────────────────────────────
  {
    name: "Decision Tree Classifier",
    cls: "DecisionTreeClassifier",
    module: "tree",
    category: "Classification",
    group: "Decision Trees",
    description: "Recursively partitions the feature space into axis-aligned rectangles to form a tree of if-else rules. Fully interpretable.",
    useWhen: "Interpretability required; mixed feature types; no feature scaling needed; quick baseline.",
    dontUse: "Small training sets (high variance, overfits); smooth regression surfaces; when generalisation matters most (use ensemble).",
    realWorld: "Rule extraction for compliance (insurance, banking), medical decision aids, feature engineering discovery.",
  },
  {
    name: "Decision Tree Regressor",
    cls: "DecisionTreeRegressor",
    module: "tree",
    category: "Regression",
    group: "Decision Trees",
    description: "Regression variant of the decision tree, predicting the mean target value in each leaf. Prone to overfitting without depth limits.",
    useWhen: "Non-linear regression that needs to be explainable; mixed features; quick baseline.",
    dontUse: "Smooth functions; small datasets; when variance reduction is critical (use Random Forest).",
    realWorld: "Healthcare cost prediction, energy consumption modelling, interpretable fraud scoring.",
  },

  // ── ENSEMBLE ─────────────────────────────────────────────────────────
  {
    name: "Random Forest Classifier",
    cls: "RandomForestClassifier",
    module: "ensemble",
    category: "Classification",
    group: "Ensemble Methods",
    description: "Bagging of decorrelated decision trees — each tree is trained on a bootstrap sample with a random feature subset. Averages predictions, dramatically reducing variance.",
    useWhen: "Tabular classification; mixed feature types; high-variance trees that need smoothing; feature importance needed; handles missing data well (with imputation).",
    dontUse: "Very high-dimensional sparse data (LinearSVC is faster); when memory is limited; when you need maximum accuracy on tabular data (try HGBT first).",
    realWorld: "Credit default prediction, medical diagnosis, Kaggle competitions, satellite image classification.",
  },
  {
    name: "Random Forest Regressor",
    cls: "RandomForestRegressor",
    module: "ensemble",
    category: "Regression",
    group: "Ensemble Methods",
    description: "Regression version of Random Forest. Averages leaf-node predictions across all trees.",
    useWhen: "Non-linear tabular regression; robust out-of-the-box without tuning; feature importance required.",
    dontUse: "Extrapolation beyond training range; very sparse/high-dimensional data.",
    realWorld: "Property valuation, demand forecasting, clinical outcome prediction.",
  },
  {
    name: "Gradient Boosting Classifier",
    cls: "GradientBoostingClassifier",
    module: "ensemble",
    category: "Classification",
    group: "Ensemble Methods",
    description: "Sequentially fits shallow trees to the residuals of the previous ensemble, gradually minimising a differentiable loss. High accuracy but slower training than HGBT.",
    useWhen: "High accuracy on moderate-sized tabular data; when HGBT is too fast/unconstrained; custom loss functions.",
    dontUse: "Very large datasets (use HistGradientBoosting or XGBoost); real-time retraining; noisy labels.",
    realWorld: "Search ranking, customer churn, insurance fraud, Kaggle tabular competitions.",
  },
  {
    name: "Hist Gradient Boosting Classifier",
    cls: "HistGradientBoostingClassifier",
    module: "ensemble",
    category: "Classification",
    group: "Ensemble Methods",
    description: "Histogram-based gradient boosting (sklearn's equivalent of LightGBM). Bins features into histograms for dramatically faster training. Natively handles missing values.",
    useWhen: "Large tabular datasets; need speed + accuracy; missing values present (no imputation needed); best default tree-based choice in sklearn today.",
    dontUse: "Very small datasets (overkill); need exact feature splits for interpretability.",
    realWorld: "Industry-grade tabular ML, fraud detection at scale, healthcare predictions, competition winning pipelines.",
  },
  {
    name: "Hist Gradient Boosting Regressor",
    cls: "HistGradientBoostingRegressor",
    module: "ensemble",
    category: "Regression",
    group: "Ensemble Methods",
    description: "Regression counterpart to HistGradientBoostingClassifier. Fast, scalable, handles missing values natively.",
    useWhen: "Large tabular regression; production pipelines; missing data; best default for regression in sklearn.",
    dontUse: "Extrapolation; interpretability is critical.",
    realWorld: "Energy demand forecasting, real-estate pricing, supply chain optimisation.",
  },
  {
    name: "AdaBoost Classifier",
    cls: "AdaBoostClassifier",
    module: "ensemble",
    category: "Classification",
    group: "Ensemble Methods",
    description: "Sequentially reweights training samples to focus on mis-classified examples. Combines many weak learners (usually shallow trees or stumps).",
    useWhen: "Binary classification; weak learners available; relatively clean data.",
    dontUse: "Noisy labels (AdaBoost is sensitive to noise); large datasets; when gradient boosting is available (usually better).",
    realWorld: "Face detection (Viola-Jones algorithm), object detection, early ML competitions.",
  },
  {
    name: "Bagging Classifier / Regressor",
    cls: "BaggingClassifier / BaggingRegressor",
    module: "ensemble",
    category: "Classification / Regression",
    group: "Ensemble Methods",
    description: "General-purpose bootstrap aggregating — wraps ANY base estimator and reduces its variance by averaging across bootstrap samples.",
    useWhen: "High-variance base model (e.g. deep decision tree, kNN); custom base estimator for ensemble; variance reduction.",
    dontUse: "Base estimator already has low variance; computational budget is tight.",
    realWorld: "Reducing variance in custom models, ensemble of SVMs on medium datasets.",
  },
  {
    name: "Voting Classifier / Regressor",
    cls: "VotingClassifier / VotingRegressor",
    module: "ensemble",
    category: "Classification / Regression",
    group: "Ensemble Methods",
    description: "Combines multiple heterogeneous models via majority vote (hard) or averaged probabilities (soft). Simple and interpretable stacking.",
    useWhen: "Combining diverse models (e.g. RF + LR + SVC) for marginal accuracy gain; competition ensembling.",
    dontUse: "When models are highly correlated; when Stacking with meta-learner would give more gains.",
    realWorld: "Kaggle final submissions, production ensembles for risk scoring.",
  },
  {
    name: "Stacking Classifier / Regressor",
    cls: "StackingClassifier / StackingRegressor",
    module: "ensemble",
    category: "Classification / Regression",
    group: "Ensemble Methods",
    description: "Two-level ensemble: base models' out-of-fold predictions are used as features for a meta-learner. More powerful than simple voting.",
    useWhen: "Maximum predictive accuracy; competition settings; diverse base models available.",
    dontUse: "Small datasets (data-hungry); when interpretability matters; production systems with strict latency.",
    realWorld: "Winning ML competition solutions, research benchmarks, production pipelines where accuracy trumps simplicity.",
  },

  // ── NEURAL NETWORKS ───────────────────────────────────────────────────
  {
    name: "MLP Classifier",
    cls: "MLPClassifier",
    module: "neural_network",
    category: "Classification",
    group: "Neural Networks",
    description: "Multi-layer perceptron (fully-connected feedforward neural network). Supports multiple hidden layers and various activation functions.",
    useWhen: "Non-linear classification; tabular data where feature interactions matter; intermediate-sized datasets.",
    dontUse: "Images, sequences, or text (use PyTorch/TensorFlow instead); very large data (GPU-based frameworks are faster); need interpretability.",
    realWorld: "Tabular classification in production (simple cases), XOR-like problems, feature interaction modelling.",
  },
  {
    name: "MLP Regressor",
    cls: "MLPRegressor",
    module: "neural_network",
    category: "Regression",
    group: "Neural Networks",
    description: "Multi-layer perceptron for regression tasks. Learns non-linear mappings from features to a continuous target.",
    useWhen: "Non-linear regression; when tree-based methods underperform; moderate dataset sizes.",
    dontUse: "Very large datasets; unstructured data (images, text); when gradient boosting suffices.",
    realWorld: "Energy prediction, financial modelling, multi-output regression.",
  },

  // ── SEMI-SUPERVISED ──────────────────────────────────────────────────
  {
    name: "Label Propagation",
    cls: "LabelPropagation",
    module: "semi_supervised",
    category: "Classification",
    group: "Semi-Supervised",
    description: "Graph-based semi-supervised algorithm. Propagates labels from labelled points to unlabelled ones through a similarity graph.",
    useWhen: "Few labelled examples but many unlabelled; data lies on a manifold; graph structure captures class relationships.",
    dontUse: "Large datasets (O(n²) graph); when no meaningful manifold structure exists.",
    realWorld: "Image annotation, social network label propagation, biological network analysis.",
  },
  {
    name: "Self-Training Classifier",
    cls: "SelfTrainingClassifier",
    module: "semi_supervised",
    category: "Classification",
    group: "Semi-Supervised",
    description: "Wrapper that wraps any probabilistic classifier, iteratively labels high-confidence unlabelled samples and retrains.",
    useWhen: "Semi-supervised setting with any base classifier; confident pseudo-labelling needed.",
    dontUse: "Base classifier has poor calibration; when label noise would compound across iterations.",
    realWorld: "NLP with scarce annotations, medical imaging with few expert labels, web content classification.",
  },

  // ── CLUSTERING ────────────────────────────────────────────────────────
  {
    name: "K-Means",
    cls: "KMeans",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Partitions data into k spherical clusters by alternately assigning points to the nearest centroid and updating centroids. Simple and fast.",
    useWhen: "Compact, spherical clusters expected; k is known or easily determined; large datasets (use MiniBatchKMeans).",
    dontUse: "Clusters of varying size/density/shape; categorical data; unknown k without domain knowledge.",
    realWorld: "Customer segmentation, image compression (colour quantisation), document clustering, vector quantisation.",
  },
  {
    name: "Mini-Batch K-Means",
    cls: "MiniBatchKMeans",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Scalable variant of K-Means that uses random mini-batches per iteration. Slightly worse cluster quality but much faster for large datasets.",
    useWhen: "Very large datasets; streaming/online clustering; when speed is more important than optimality.",
    dontUse: "Small datasets (full KMeans is fine); when cluster quality must be maximised.",
    realWorld: "Online topic modelling, large-scale user segmentation, real-time anomaly grouping.",
  },
  {
    name: "DBSCAN",
    cls: "DBSCAN",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Density-Based Spatial Clustering. Groups points in dense regions, marks low-density points as noise. No need to specify k. Discovers arbitrary shapes.",
    useWhen: "Arbitrary cluster shapes; outlier/noise detection built-in; don't know number of clusters.",
    dontUse: "Varying density clusters (use HDBSCAN); very high-dimensional data (distance metrics break down); large datasets with uniform density.",
    realWorld: "Geo-spatial clustering (crime hotspots, ride-share zones), anomaly detection, astronomical object detection.",
  },
  {
    name: "HDBSCAN",
    cls: "HDBSCAN",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Hierarchical extension of DBSCAN. Robust to varying density clusters. Extracts a flat clustering from the cluster hierarchy automatically.",
    useWhen: "Varying density clusters; robust outlier detection; don't want to tune epsilon; generally preferred over DBSCAN.",
    dontUse: "Very large, high-dimensional datasets (can be slow); when simple spherical clusters are expected (KMeans is faster).",
    realWorld: "NLP embedding clustering, genomics, anomaly detection in sensor networks.",
  },
  {
    name: "Agglomerative Clustering",
    cls: "AgglomerativeClustering",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Bottom-up hierarchical clustering. Merges clusters iteratively based on linkage criterion (ward, complete, average, single). Produces a dendrogram.",
    useWhen: "Hierarchical structure in data; when a dendrogram is needed; small-to-medium datasets; arbitrary cluster shapes.",
    dontUse: "Very large datasets (O(n² log n)); when k must be chosen dynamically.",
    realWorld: "Gene expression hierarchies, organisational structure analysis, document taxonomy.",
  },
  {
    name: "Mean Shift",
    cls: "MeanShift",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Non-parametric algorithm that iteratively shifts each point toward the mode of the local density. Automatically determines number of clusters.",
    useWhen: "Number of clusters unknown; blob-shaped clusters; smooth continuous density.",
    dontUse: "Large datasets (O(n²)); high-dimensional data; oblong/elongated clusters.",
    realWorld: "Image segmentation, computer vision feature tracking, density mode finding.",
  },
  {
    name: "Gaussian Mixture Model",
    cls: "GaussianMixture",
    module: "mixture",
    category: "Clustering",
    group: "Clustering",
    description: "Probabilistic model assuming data is generated from a mixture of Gaussian distributions. Soft cluster assignments (probabilities). Fitted with EM algorithm.",
    useWhen: "Soft/probabilistic cluster membership needed; ellipsoidal cluster shapes; generative model of data required.",
    dontUse: "Non-Gaussian distributions; very large datasets; when number of components is totally unknown.",
    realWorld: "Speaker identification, image segmentation, density estimation, anomaly scoring.",
  },
  {
    name: "Spectral Clustering",
    cls: "SpectralClustering",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Embeds data via the graph Laplacian eigenvectors then applies K-Means. Can find non-convex cluster shapes. Works well when a similarity graph is natural.",
    useWhen: "Concentric or interleaved clusters; graph-structured data; when KMeans fails on complex shapes.",
    dontUse: "Large datasets (eigen-decomposition is expensive); when cluster shapes are simply spherical.",
    realWorld: "Community detection in social graphs, image segmentation, circuit partitioning.",
  },
  {
    name: "OPTICS",
    cls: "OPTICS",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Generalisation of DBSCAN that handles varying density by ordering points by reachability distance. Can extract clusters at multiple density scales.",
    useWhen: "Multi-scale or varying density clusters; when epsilon is hard to choose in DBSCAN.",
    dontUse: "Large datasets (slow); when HDBSCAN handles the problem equally well.",
    realWorld: "Geospatial analysis with density gradients, network anomaly detection.",
  },
  {
    name: "Birch",
    cls: "Birch",
    module: "cluster",
    category: "Clustering",
    group: "Clustering",
    description: "Builds a Clustering Feature Tree (CF-Tree) incrementally. Very memory-efficient and supports online/streaming updates.",
    useWhen: "Very large datasets; memory is constrained; online clustering needed; spherical clusters.",
    dontUse: "Non-spherical clusters; when cluster quality must be maximised.",
    realWorld: "Large-scale log clustering, streaming event grouping, memory-constrained environments.",
  },

  // ── DIMENSIONALITY REDUCTION ──────────────────────────────────────────
  {
    name: "PCA",
    cls: "PCA",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Principal Component Analysis. Projects data onto orthogonal directions of maximum variance. Linear, unsupervised. The gold-standard for linear dimensionality reduction.",
    useWhen: "Reduce dimensionality before modelling; remove multicollinearity; visualisation (2-3 components); speeding up downstream algorithms.",
    dontUse: "Non-linear manifolds (use UMAP/t-SNE); when feature interpretability must be preserved; sparse data (use TruncatedSVD).",
    realWorld: "Preprocessing genomics data, face recognition (eigenfaces), noise reduction, pre-processing before classification.",
  },
  {
    name: "Truncated SVD",
    cls: "TruncatedSVD",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Computes a truncated singular value decomposition. Like PCA but does NOT centre the data — works directly on sparse matrices (e.g. TF-IDF).",
    useWhen: "Sparse data (text, TF-IDF matrices); large-scale LSA/LSI; when centring would destroy sparsity.",
    dontUse: "Dense data where standard PCA is available; when centring is acceptable.",
    realWorld: "Latent Semantic Analysis (LSA) for NLP, collaborative filtering, text document similarity.",
  },
  {
    name: "t-SNE",
    cls: "TSNE",
    module: "manifold",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "t-Distributed Stochastic Neighbour Embedding. Non-linear, probabilistic method that preserves local neighbourhood structure. Excellent for 2D/3D visualisation.",
    useWhen: "High-dimensional data visualisation (always 2-3D output); exploring cluster structure; understanding embedding spaces.",
    dontUse: "Downstream ML features (non-deterministic, not invertible); large datasets (slow O(n²) unless approximate); inter-cluster distances are not preserved.",
    realWorld: "Visualising word embeddings, exploring image datasets, single-cell RNA-seq visualisation, embedding inspection.",
  },
  {
    name: "Isomap",
    cls: "Isomap",
    module: "manifold",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Extends MDS by computing geodesic distances through the data manifold (via shortest paths on a neighbourhood graph). Preserves global manifold structure.",
    useWhen: "Data lies on a smooth, low-dimensional manifold; global structure preservation needed; scientific datasets.",
    dontUse: "Data with holes or discontinuities in the manifold; large datasets (shortest path computation is expensive).",
    realWorld: "Face pose estimation, robot motion planning, scientific simulation embeddings.",
  },
  {
    name: "Locally Linear Embedding",
    cls: "LocallyLinearEmbedding",
    module: "manifold",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Preserves local linear patches of the manifold. Each point is reconstructed from its neighbours, and the low-dim embedding respects these weights.",
    useWhen: "Smooth manifold data; when local structure is more important than global.",
    dontUse: "Noisy data; when manifold has complex topology; large datasets.",
    realWorld: "Handwriting analysis, motion capture data, scientific manifold exploration.",
  },
  {
    name: "FastICA",
    cls: "FastICA",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Independent Component Analysis. Separates a multivariate signal into additive, statistically independent components by maximising non-Gaussianity.",
    useWhen: "Blind source separation (e.g. separating mixed audio); when components are statistically independent and non-Gaussian.",
    dontUse: "When components are Gaussian (PCA is equivalent); fewer samples than features.",
    realWorld: "EEG/MEG signal decomposition, audio source separation ('cocktail party'), fMRI analysis.",
  },
  {
    name: "NMF (Non-negative Matrix Factorisation)",
    cls: "NMF",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Factorises a non-negative matrix V ≈ WH where both W and H are non-negative. Produces parts-based, interpretable representations.",
    useWhen: "Non-negative data (images, text counts, spectrograms); interpretable components needed; topic modelling alternative.",
    dontUse: "Data with negative values; when computational speed is critical (slower than PCA).",
    realWorld: "Topic modelling, hyperspectral unmixing, music transcription, face part decomposition.",
  },
  {
    name: "Latent Dirichlet Allocation",
    cls: "LatentDirichletAllocation",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Generative probabilistic model for text collections. Discovers latent topics as distributions over words. Produces topic-mixture representations per document.",
    useWhen: "Topic discovery in large text corpora; document representation for clustering/search.",
    dontUse: "Short texts (few words per document); when neural topic models (e.g. BERTopic) are feasible.",
    realWorld: "News topic discovery, scientific literature mining, legal document analysis, content recommendation.",
  },
  {
    name: "Kernel PCA",
    cls: "KernelPCA",
    module: "decomposition",
    category: "Dimensionality Reduction",
    group: "Dimensionality Reduction",
    description: "Non-linear extension of PCA using the kernel trick. Maps data into a high-dim feature space and runs PCA there, enabling non-linear structure capture.",
    useWhen: "Non-linear manifold structure; when linear PCA misses important variance; kernel choice reflects domain knowledge.",
    dontUse: "Very large datasets (O(n²) kernel matrix); when linear PCA suffices.",
    realWorld: "De-noising, non-linear feature extraction, pre-processing for SVM pipelines.",
  },

  // ── OUTLIER DETECTION ─────────────────────────────────────────────────
  {
    name: "Isolation Forest",
    cls: "IsolationForest",
    module: "ensemble",
    category: "Anomaly Detection",
    group: "Outlier Detection",
    description: "Randomly partitions the feature space using isolation trees. Anomalies are isolated in fewer splits (shorter average path length). Fast and scalable.",
    useWhen: "Unsupervised anomaly detection; large datasets; high-dimensional data; no labelled anomalies.",
    dontUse: "When labelled anomalies exist (use supervised classifiers); very small datasets.",
    realWorld: "Network intrusion detection, credit card fraud, manufacturing defect detection, log anomaly detection.",
  },
  {
    name: "Local Outlier Factor",
    cls: "LocalOutlierFactor",
    module: "neighbors",
    category: "Anomaly Detection",
    group: "Outlier Detection",
    description: "Computes the local density of each point relative to its neighbours. Points in low-density regions compared to neighbours are flagged as anomalies.",
    useWhen: "Density-based anomalies; clusters of varying density; when local context matters.",
    dontUse: "Very large datasets (O(n²)); high-dimensional data; when global anomaly detection is needed.",
    realWorld: "Fraud detection with regional patterns, sensor fault detection, quality control.",
  },
  {
    name: "Elliptic Envelope",
    cls: "EllipticEnvelope",
    module: "covariance",
    category: "Anomaly Detection",
    group: "Outlier Detection",
    description: "Fits a robust covariance ellipse to the data and flags points far from the centre as outliers. Assumes data is Gaussian.",
    useWhen: "Gaussian-ish data; need a parametric outlier model; small to medium datasets.",
    dontUse: "Non-Gaussian distributions; high-dimensional data (covariance estimation breaks down); multiple clusters.",
    realWorld: "Quality control with Gaussian measurements, financial risk outlier detection.",
  },

  // ── COVARIANCE / SPECIAL ─────────────────────────────────────────────
  {
    name: "Calibrated Classifier CV",
    cls: "CalibratedClassifierCV",
    module: "calibration",
    category: "Classification",
    group: "Calibration & Special",
    description: "Post-hoc probability calibration wrapper for any classifier. Uses Platt scaling (sigmoid) or isotonic regression to align predicted probabilities with true frequencies.",
    useWhen: "Classifier outputs poorly calibrated probabilities (e.g. SVC, Random Forest); downstream use of probabilities (expected value calculations).",
    dontUse: "Model already well-calibrated (Logistic Regression, GaussianNB); very small datasets.",
    realWorld: "Risk scoring systems, clinical prediction tools, ad bidding where probabilities drive bids.",
  },
  {
    name: "Isotonic Regression",
    cls: "IsotonicRegression",
    module: "isotonic",
    category: "Regression",
    group: "Calibration & Special",
    description: "Non-parametric piecewise constant monotone (isotonic) function fitting. Used for probability calibration and when a monotone non-linear relationship exists.",
    useWhen: "Monotone relationship between feature and target; probability calibration; ordinal data.",
    dontUse: "Non-monotone relationships; multivariate regression (it is 1D only).",
    realWorld: "Probability recalibration, dose-response curve fitting, age-measurement relationships.",
  },
  {
    name: "PLS Regression",
    cls: "PLSRegression",
    module: "cross_decomposition",
    category: "Regression",
    group: "Calibration & Special",
    description: "Partial Least Squares regression. Finds latent components that maximise covariance between X and Y. Handles high-dimensional, correlated features with small n.",
    useWhen: "High-dimensional correlated features, n ≪ p (spectroscopy, chemometrics); multi-output regression.",
    dontUse: "Low-dimensional data (Ridge/Lasso simpler); non-linear relationships.",
    realWorld: "Chemometrics, near-infrared spectroscopy, neuroimaging (fMRI), multi-output sensor prediction.",
  },
  {
    name: "Kernel Ridge Regression",
    cls: "KernelRidge",
    module: "kernel_ridge",
    category: "Regression",
    group: "Calibration & Special",
    description: "Combines Ridge regression with the kernel trick. Equivalent to kernel SVM with a squared loss. Provides non-linear regression with a closed-form solution.",
    useWhen: "Non-linear regression on small-to-medium datasets; when kernel SVR is too slow to tune.",
    dontUse: "Large datasets (O(n²) kernel matrix); need for sparsity.",
    realWorld: "Computational chemistry property prediction, small scientific regression tasks, GP approximation.",
  },
];

const CATEGORIES = ["All", "Regression", "Classification", "Clustering", "Dimensionality Reduction", "Anomaly Detection", "Classification / Regression", "Calibration & Special"];

const GROUP_COLORS = {
  "Linear Models": { bg: "#1e3a5f", accent: "#4a9eff" },
  "Discriminant Analysis": { bg: "#1a3a2a", accent: "#4ade80" },
  "Support Vector Machines": { bg: "#3a1a2a", accent: "#f472b6" },
  "Nearest Neighbors": { bg: "#2a2a1a", accent: "#facc15" },
  "Gaussian Processes": { bg: "#1a2a3a", accent: "#818cf8" },
  "Naive Bayes": { bg: "#2a1a1a", accent: "#fb923c" },
  "Decision Trees": { bg: "#1a3a2a", accent: "#34d399" },
  "Ensemble Methods": { bg: "#2a1a3a", accent: "#c084fc" },
  "Neural Networks": { bg: "#1a2a2a", accent: "#22d3ee" },
  "Semi-Supervised": { bg: "#3a2a1a", accent: "#fbbf24" },
  "Clustering": { bg: "#1a2a3a", accent: "#60a5fa" },
  "Dimensionality Reduction": { bg: "#2a3a1a", accent: "#a3e635" },
  "Outlier Detection": { bg: "#3a1a1a", accent: "#f87171" },
  "Calibration & Special": { bg: "#2a2a3a", accent: "#e879f9" },
};

const CATEGORY_BADGE = {
  "Regression": { bg: "#1e3a5f", color: "#4a9eff" },
  "Classification": { bg: "#1a3a2a", color: "#4ade80" },
  "Clustering": { bg: "#1a2a3a", color: "#60a5fa" },
  "Dimensionality Reduction": { bg: "#2a3a1a", color: "#a3e635" },
  "Anomaly Detection": { bg: "#3a1a1a", color: "#f87171" },
  "Classification / Regression": { bg: "#2a1a3a", color: "#c084fc" },
  "Calibration & Special": { bg: "#2a2a3a", color: "#e879f9" },
};

function Badge({ text }) {
  const style = CATEGORY_BADGE[text] || { bg: "#2a2a2a", color: "#aaa" };
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: "4px",
      fontSize: "0.68rem",
      fontWeight: 600,
      letterSpacing: "0.04em",
      textTransform: "uppercase",
      background: style.bg,
      color: style.color,
      border: `1px solid ${style.color}33`,
      whiteSpace: "nowrap",
    }}>{text}</span>
  );
}

function ModelRow({ model, idx }) {
  const [open, setOpen] = useState(false);
  const gc = GROUP_COLORS[model.group] || { bg: "var(--border-subtle)", accent: "#888" };

  return (
    <>
      <tr
        onClick={() => setOpen(o => !o)}
        style={{
          cursor: "pointer",
          background: idx % 2 === 0 ? "var(--bg-surface)" : "var(--bg-surface)",
          borderLeft: `3px solid ${gc.accent}`,
          transition: "background 0.15s",
        }}
        onMouseEnter={e => e.currentTarget.style.background = "var(--bg-elevated)"}
        onMouseLeave={e => e.currentTarget.style.background = idx % 2 === 0 ? "var(--bg-surface)" : "var(--bg-surface)"}
      >
        <td style={td}>{open ? "▾" : "▸"}</td>
        <td style={{ ...td, fontFamily: "var(--font-mono)", color: gc.accent, fontWeight: 700, fontSize: "0.82rem" }}>
          {model.cls}
        </td>
        <td style={{ ...td, fontWeight: 600, color: "var(--text-primary)", fontSize: "0.88rem" }}>{model.name}</td>
        <td style={td}><Badge text={model.category} /></td>
        <td style={{ ...td, color: "var(--text-secondary)", fontSize: "0.75rem", fontFamily: "var(--font-mono)" }}>
          sklearn.{model.module}
        </td>
      </tr>
      {open && (
        <tr style={{ background: gc.bg + "55", borderLeft: `3px solid ${gc.accent}` }}>
          <td colSpan={5} style={{ padding: "0 0 0 20px" }}>
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: "0",
              padding: "16px 20px 16px 0",
            }}>
              <DetailBox icon="📋" label="Description" text={model.description} accent={gc.accent} />
              <DetailBox icon="✅" label="Use When" text={model.useWhen} accent="#4ade80" />
              <DetailBox icon="⛔" label="Don't Use When" text={model.dontUse} accent="#f87171" />
              <div style={{ gridColumn: "1 / -1", marginTop: "10px" }}>
                <DetailBox icon="🌍" label="Real-World Usage Today" text={model.realWorld} accent="#facc15" fullWidth />
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function DetailBox({ icon, label, text, accent, fullWidth }) {
  return (
    <div style={{
      padding: "10px 14px",
      borderRadius: "6px",
      margin: "0 8px 0 0",
      background: "var(--bg-surface)",
      border: `1px solid ${accent}22`,
    }}>

      <div style={{ fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.08em", color: accent, textTransform: "uppercase", marginBottom: "5px" }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: "0.82rem", color: "var(--text-primary)", lineHeight: 1.55 }}>{text}</div>
    </div>
  );
}

const td = {
  padding: "11px 14px",
  verticalAlign: "middle",
  borderBottom: "1px solid var(--border-subtle)",
  fontSize: "0.85rem",
  color: "var(--text-primary)",
};

const th = {
  padding: "12px 14px",
  textAlign: "left",
  fontSize: "0.7rem",
  fontWeight: 700,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  color: "var(--text-secondary)",
  borderBottom: "2px solid var(--border-default)",
  background: "var(--bg-surface)",
  position: "sticky",
  zIndex: 5,
};

export default function Sklearn_app() {
  const [search, setSearch] = useState("");
  const [cat, setCat] = useState("All");

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return MODELS.filter(m => {
      const matchCat = cat === "All" || m.category === cat;
      const matchSearch = !q || [m.name, m.cls, m.group, m.description, m.useWhen, m.realWorld]
        .some(s => s.toLowerCase().includes(q));
      return matchCat && matchSearch;
    });
  }, [search, cat]);

  const groups = useMemo(() => {
    const g = {};
    for (const m of filtered) {
      if (!g[m.group]) g[m.group] = [];
      g[m.group].push(m);
    }
    return g;
  }, [filtered]);

  return (
    <div>
    <Header />

        <div style={{
      fontFamily: "var(--font-body)",
      background: "var(--bg-base)",
      minHeight: "100vh",
      color: "var(--text-primary)",
    }}>
      {/* Import fonts */}
      <style>{`
* { box-sizing: border-box; }
`}</style>
      {/* Header */}
      <div style={{
        position: "sticky", top: "var(--header-h)", zIndex: 10,
        background: "var(--bg-base)",
        borderBottom: "1px solid var(--bg-elevated)",
        padding: "12px 24px",
        display: "flex",
        alignItems: "center",
        gap: "16px",
        flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
            <span style={{ color: "#4a9eff" }}>sklearn</span> Model Reference
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", letterSpacing: "0.05em", marginTop: "1px" }}>
            scikit-learn 1.8 · {MODELS.length} estimators · click any row to expand
          </div>
        </div>
        <div style={{ flex: 1, minWidth: "200px", maxWidth: "340px" }}>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search models, use-cases, APIs…"
            style={{
              width: "100%",
              background: "var(--bg-surface)",
              border: "1px solid var(--border-default)",
              borderRadius: "6px",
              padding: "8px 12px",
              color: "var(--text-primary)",
              fontSize: "0.83rem",
              outline: "none",
              fontFamily: "inherit",
            }}
          />
        </div>
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {CATEGORIES.map(c => (
            <button
              key={c}
              onClick={() => setCat(c)}
              style={{
                padding: "5px 11px",
                borderRadius: "5px",
                fontSize: "0.72rem",
                fontWeight: cat === c ? 700 : 400,
                border: cat === c ? "1px solid #4a9eff" : "1px solid var(--border-default)",
                background: cat === c ? "#1e3a5f" : "transparent",
                color: cat === c ? "#4a9eff" : "var(--text-secondary)",
                cursor: "pointer",
                letterSpacing: "0.03em",
                fontFamily: "inherit",
                transition: "all 0.15s",
                whiteSpace: "nowrap",
              }}
            >{c}</button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--text-tertiary)" }}>
          {filtered.length} models
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
          <thead>
            <tr>
              <th style={{ ...th, width: "28px" }}></th>
              <th style={th}>Class</th>
              <th style={th}>Model Name</th>
              <th style={th}>Category</th>
              <th style={th}>Module</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(groups).map(([group, models]) => (
              <>
                <tr key={group + "_header"}>
                  <td colSpan={5} style={{
                    padding: "10px 14px 6px",
                    fontSize: "0.7rem",
                    fontWeight: 700,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: GROUP_COLORS[group]?.accent || "#6a6aaa",
                    background: "var(--bg-surface)",
                    borderBottom: "1px solid var(--bg-elevated)",
                    borderTop: "2px solid var(--bg-elevated)",
                  }}>
                    ▪ {group} <span style={{ fontWeight: 400, color: "var(--text-tertiary)", marginLeft: 6 }}>({models.length})</span>
                  </td>
                </tr>
                {models.map((m, i) => (
                  <ModelRow key={m.cls} model={m} idx={i} />
                ))}
              </>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: "40px", textAlign: "center", color: "var(--text-tertiary)" }}>
                  No models match your filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div style={{ padding: "24px", textAlign: "center", fontSize: "0.7rem", color: "var(--border-default)", borderTop: "1px solid var(--bg-elevated)" }}>
        Based on scikit-learn 1.8.0 official documentation · {new Date().getFullYear()}
      </div>
    </div></div>
  );
}