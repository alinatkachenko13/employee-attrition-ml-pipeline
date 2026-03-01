# Employee Attrition ML Pipeline
Practical HR analytics project focused on predicting job satisfaction and employee attrition with reproducible ML pipelines, clear evaluation metrics, and business-ready insights.

## Why This Project
Attrition is costly and often preventable. This project uses anonymized HR data to:
- predict job satisfaction (early risk signal)
- predict attrition (quit vs stay)
- transform model outputs into actionable retention recommendations

## Key Results (from the notebook)
- **Task 1 (Regression)**: `DecisionTreeRegressor` with `max_depth=15`, `min_samples_split=6`
  - SMAPE: **14.47 (train)**, **13.8 (test)** — meets target **<= 15**
- **Task 2 (Classification)**: `DecisionTreeClassifier` with `max_depth=4`, `min_samples_split=3`
  - ROC-AUC: **0.91 (test)** — meets target **>= 0.91**
- The predicted satisfaction score from Task 1 is added as a feature for Task 2 to improve attrition prediction.

## What I Built
- End-to-end **preprocessing pipeline** (missing values, encoding, scaling)
- **EDA** with distribution checks and feature insights
- **Model comparison** with baselines (Dummy models) and hyperparameter search
- **Reproducible workflow** inside a single notebook
- **Business conclusions** for retention strategy

## Skills Demonstrated (Junior ML-Engineer)
- Data cleaning and exploration (pandas, matplotlib, seaborn)
- Feature engineering and leakage-safe pipelines (scikit-learn)
- Model training and evaluation (SMAPE, ROC-AUC)
- Hyperparameter optimization (RandomizedSearchCV)
- Translating model output to business recommendations

## Tech Stack
- Python, pandas, numpy
- scikit-learn, scipy
- matplotlib, seaborn, phik
- Jupyter Notebook

## Data Availability
The original dataset is **not included** in this repository (confidential HR data).
To keep the project reproducible and transparent:
- the notebook clearly documents feature names and preprocessing
- expected columns are listed below

Expected features:
- `dept`, `level`, `workload`
- `employment_years`, `last_year_promo`, `last_year_violations`
- `supervisor_evaluation`, `salary`
- `job_satisfaction_rate` (target for Task 1)
- `quit` (target for Task 2)

## How to Run
1. Create a Python environment and install dependencies.
2. Open the notebook and run cells in order:
   - `hr-analytics.ipynb`

> Note: the notebook references data paths like `/datasets/...`. You can adapt the paths to your local CSV files with the same schema.

## Project Structure
- `hr-analytics.ipynb` — full analysis, modeling, and conclusions
- `README.md` — project overview (this file)

## Business Takeaways (Short)
- Low satisfaction strongly correlates with attrition.
- Higher risk group: junior employees with 1–2 years of experience, lower salary, no recent promotions, and disciplinary incidents.
- Predicting satisfaction first improves attrition prediction quality.

## Next Steps (if expanded)
- Try ensemble models (Random Forest, XGBoost)
- Calibrate probabilities for HR risk scoring
- Build a small API for batch scoring
