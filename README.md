# Predicting Mental Health Risks in Video Gamers

A comparative study of machine learning, deep learning, and feature engineering techniques applied to mental health risk classification in a population of 13,000+ video gamers.

> Master's Thesis — MSc Data Science & Society, Tilburg University (2025)

---

## Overview

This project investigates whether machine learning models can predict mental health risk levels (Low / Moderate / High) among video gamers based on self-reported psychological assessments, gaming behaviors, and demographic data.

A key contribution is the **comparison of two feature engineering strategies** for processing open-ended survey responses:

- **Experiment 1 — Manual Categorization:** Rule-based keyword matching to convert free-text responses into structured categorical features
- **Experiment 2 — Sentence Embeddings:** Transformer-based embeddings (`all-MiniLM-L6-v2`) with PCA dimensionality reduction

Both experiments share identical preprocessing, feature selection, modeling, and evaluation pipelines, enabling a controlled comparison.

Full thesis report available in `thesis/thesis_report.pdf`.

---

## Research Question

> *To what extent can machine learning models predict mental health risk levels among video gamers based on self-reported gaming behaviors and psychological assessments, and how does the choice of feature representation — manual categorization vs. sentence embeddings — affect predictive performance?*

---

## Dataset

The dataset was obtained from the [Open Science Framework (OSF)](https://doi.org/10.31234/osf.io/mfajz) (Sauter & Draschkow, 2017) and is publicly available.

- ~13,000 video gamers from an international online survey
- 55 features including psychological assessments, demographics, gaming behaviors, and open-ended responses
- Psychological scales: Satisfaction With Life Scale (SWLS), GAD-7, Social Phobia Inventory (SPIN), Narcissism scale
- Target variable derived from SWLS total score

| SWL_T Range | Mental Health Risk |
|---|---|
| 21 – 35 | Low Risk |
| 15 – 20 | Moderate Risk |
| 5 – 14 | High Risk |

Class distribution: Low 47.4% / Moderate 26.3% / High 26.3%

> Download `GamingStudy_data.csv` from [OSF](https://doi.org/10.31234/osf.io/mfajz) and place it in the project root before running the notebooks.

---

## Methodology

| Step | Details |
|---|---|
| Missing data | MICE imputation with Random Forest regressors/classifiers |
| Feature selection | Random Forest importance, ANOVA F-test, Mutual Information, RFECV |
| Models | Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, CatBoost, TabNet |
| Hyperparameter tuning | GridSearchCV (LR, RF, SVM) · RandomizedSearchCV (XGBoost, LightGBM, CatBoost, TabNet) |
| Evaluation metric | Macro F1 (primary), AUC OvR, Precision, Recall |
| Interpretability | SHAP analysis + error analysis |
| Data split | 70% train / 15% validation / 15% test (stratified) |

---

## Results

### Final Test Set Performance

| Experiment | Best Model | F1 Macro | AUC OvR | Accuracy |
|---|---|---|---|---|
| Experiment 1 — Manual Categorization | CatBoost | **0.4882** | **0.6928** | 0.52 |
| Experiment 2 — Sentence Embeddings | Logistic Regression | 0.4824 | 0.6916 | 0.52 |

### Per-Class Performance (Test Set)

| Class | Manual — CatBoost | Embeddings — Logistic Regression |
|---|---|---|
| Low Risk | F1: 0.64 | F1: 0.66 |
| Moderate Risk | F1: 0.29 | F1: 0.32 |
| High Risk | F1: 0.53 | F1: 0.54 |

### Key Findings

- Both approaches achieved comparable performance — sentence embeddings did not outperform manual categorization, contrary to initial expectations
- All models consistently struggled to classify the Moderate Risk group, reflecting its overlapping psychological characteristics with adjacent classes
- Psychological distress indicators (GADE, GAD_T, SPIN_T) and socio-economic factors (employment status, education level) were the strongest predictors across both experiments, confirmed by SHAP analysis
- In Experiment 2, Logistic Regression outperformed more complex models (XGBoost, TabNet), suggesting PCA-reduced embeddings produce sufficiently linearly separable representations
- TabNet underperformed in both experiments, likely due to dataset size and feature variability constraints

---

## Project Structure

```
├── notebooks/
│   ├── 01_preprocessing_eda_embeddings.ipynb    # Shared pipeline + Experiment 2
│   └── 02_manual_categorization.ipynb           # Experiment 1
├── thesis/
│   └── thesis_report.pdf
├── requirements.txt
└── README.md
```

### Notebook Guide

| Notebook | Contents |
|---|---|
| `01_preprocessing_eda_embeddings.ipynb` | **Start here.** Data loading, cleaning, MICE imputation, EDA, feature engineering, sentence embeddings + PCA, feature selection, all model training/tuning, SHAP analysis, and final evaluation (Experiment 2) |
| `02_manual_categorization.ipynb` | Rule-based text categorization, followed by the same modeling pipeline. Uses the cleaned dataset produced by notebook 01 (Experiment 1) |

---

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `scikit-learn==1.6.1` · `xgboost==2.1.4` · `lightgbm==4.5.0` · `catboost==1.2.8` · `sentence-transformers==4.1.0` · `shap==0.47.2` · `pytorch-tabnet`

See `requirements.txt` for full list with versions.

### Running the Notebooks

1. Download `GamingStudy_data.csv` from [OSF](https://doi.org/10.31234/osf.io/mfajz) and place it in the project root
2. Run `01_preprocessing_eda_embeddings.ipynb` — shared preprocessing/EDA + Experiment 2
3. Run `02_manual_categorization.ipynb` — Experiment 1 (uses cleaned data from step 2)

> ⚠️ Full model training (hyperparameter tuning across 7 models with cross-validation) is computationally intensive. Pre-computed outputs are preserved in the notebooks for review.

---

## Technologies

Python 3.11 · scikit-learn · XGBoost · LightGBM · CatBoost · TabNet · SentenceTransformers · SHAP · pandas · NumPy · Matplotlib · Seaborn

---

## Author

**Evangelia Santorinaiou**
MSc Data Science & Society — Tilburg University
Supervised by Dr. Richard Dinga · June 2025

