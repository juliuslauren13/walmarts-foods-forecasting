# From History to Forecast: A Machine Learning Approach to Daily Food Demand Forecasting at Walmart CA_3 Using the M5 Dataset

**Author:** Julius Lauren Marasigan

**Dataset:** [M5 Forecasting Accuracy — Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

---

## Project Overview

Using the M5 Walmart dataset, this capstone builds a daily sales forecasting system for the **FOODS** category at **Walmart store CA_3**. Forecasting is done at the **store-category-department level** with one model per department (FOODS_1, FOODS_2, FOODS_3), producing a **28-day ahead daily forecast** to support replenishment planning.

The project is written from the perspective of an **experienced demand planner applying machine learning techniques to forecasting**. Every modeling decision is explained in practical, operational terms alongside the technical implementation, making the project useful both as a forecasting study and as a bridge between supply chain practice and data science.


## Problem Statement

**Business problem:** Demand for perishable food at Walmart CA_3 is driven by multiple overlapping factors — SNAP benefit issuance dates, day-of-week shopping behavior, seasonal events, and price changes. Traditional rule-based  forecasting approaches struggle to capture these signals at the same time. 

Forecast errors in the FOODS category may have direct financial impact. Over-forecasting leads to spoilage and markdown write-offs, while under-forecasting results in stockouts that cannot be recovered. Because of this, getting the forecast right for this store and category has a significant operational and financial impact to the business.

**ML task type:** Time series regression — predicting daily sales quantity (units sold) at the store-category-department level, 28 days ahead, modeled separately for FOODS_1, FOODS_2, and FOODS_3.

**What success looks like:** The model should outperform both Naive and Seasonal Naive baselines in terms of RMSE and MAPE. In addition, the model should not show systematic bias during SNAP benefit periods (fairness consideration), and forecast errors should remain consistent across different demand conditions. This is validated through a structured bias check across four dimensions: event days, SNAP days, day of week, and demand quartiles. 


## Scope Rationale

The scope was deliberately narrowed to one store (CA_3) and one category (FOODS) rather than forecasting all stores or all categories. This keeps the project focused while still delivering meaningful business insights.

CA_3 is the highest-volume store in the M5 dataset — the store where a forecast improvement delivers the most operational value in absolute terms. FOODS is the primary revenue driver of that store and the category where forecast accuracy matters most because food is perishable. 

Forecasting is performed at the department level rather than at the SKU level. Daily SKU demand is often highly erratic and difficult to model reliably, while department-level aggregation produces smoother and more forecastable time series. SKU-level decisions can still be supported through downstream disaggregation if required.


## Success Metrics

| **Metric** | **Role** | **Interpretation** | **Target** |
|------------|----------|---------------------|------------|
| RMSE | Primary technical metric | Penalises large forecast errors | Must beat Naive and Seasonal Naive baselines |
| MAE | Secondary technical metric | Measures average absolute error in units | Must beat Naive and Seasonal Naive baselines |
| MAPE | Operational metric | Percentage error interpretable by planners | Monitored per department |
| Bias audit (MPE by group) | Fairness metric | Detects systematic over/under-forecast | No group exceeds 5pp MPE gap vs overall |
| SNAP day accuracy | Equity metric | Ensures consistent performance during SNAP periods | SNAP MAPE ≤ non-SNAP MAPE per department |


## Evaluation Goal

The final model must outperform Naive and Seasonal Naive baselines while maintaining stable performance across different demand conditions. 

---


## Repository Structure

```
walmarts-foods-forecasting/
│
├── README.md
├── requirements.txt
├── DATA_DICTIONARY.md
│
├── notebooks/
│   ├── 01_data_collection_and_scope_selection.ipynb
│   ├── 02_data_preprocessing_and_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_and_evaluation.ipynb
│   └── 05_critical_thinking_ethical_ai.ipynb
│
├── src/
│   └── modeling.py                    
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       ├── README.md
│       ├── scope_selection.csv
│       ├── store_dept_daily_sales.csv
│       └── modeling_dataset.csv
|
├── models/
│   ├── best_model_FOODS_1.pkl
│   ├── best_model_FOODS_2.pkl
│   └── best_model_FOODS_3.pkl
│
├── reports/
│   ├── 28_day_forecast.csv
│   └── bias_audit_flagged_groups.csv
│
└── presentations/
    ├── deck_technical_presentation.pptx
    └── deck_executive_presentation.pptx
    
```

> **Note:** The raw M5 source files (calendar.csv, sales_train_validation.csv, sell_prices.csv) are not included in this repository due to file size and usage restrictions. Please download them directly from the [Kaggle competition page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place them in the project root directory before running the first notebook. This approach ensures compliance with the dataset’s terms of use, which do not allow redistribution of the M5 data.

---

## Capstone Deliverables

This project follows the six-step capstone structure. All steps 1–6 are complete.

| Step | Deliverable | File |
|------|------------|------|
| 1 — Problem Framing | Problem statement, task type, scope rationale, and success metrics | `walmarts-foods-forecasting/README.md` |
| 2 — Data Collection | Dataset overview, scope selection, event feature creation | `notebooks/01_data_collection_and_scope_selection.ipynb` |
| 3 — EDA & Feature Engineering | STL decomposition, ADF test, ACF/PACF, 39 engineered features | `notebooks/02_data_preprocessing_and_eda.ipynb` · `notebooks/03_feature_engineering.ipynb` |
| 4 — Modeling | Two-stage pipeline, 15 candidate models, GridSearchCV with TimeSeriesSplit | `notebooks/04_modeling_and_evaluation.ipynb` · `src/modeling.py` |
| 5 — Critical Thinking & Ethical AI | SHAP explainability, PDP, four-dimension bias audit, Limitations including SKU disaggregation | `notebooks/05_critical_thinking_ethical_ai.ipynb` |
| 6 — Presentations | Technical deck (9 slides) and Executive deck (9 slides) | `reports/` |

> **Data dictionary:** See `DATA_DICTIONARY.md` for full column-level documentation of all processed files.

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_collection_and_scope_selection.ipynb` | Loads and merges the three raw M5 files. Creates event indicator features. Identifies top store (CA_3) and category (FOODS) through year-on-year sales analysis. |
| 02 | `02_data_preprocessing_and_eda.ipynb` | Filters to CA_3 FOODS. Aggregates from SKU to department level. Performs STL decomposition, ADF tests, ACF/PACF analysis. |
| 03 | `03_feature_engineering.ipynb` | Engineers 39 features across lag, rolling window, EWMA, rolling max/min, calendar, price, trend, and interaction groups. |
| 04 | `04_modeling_and_evaluation.ipynb` | Runs two-stage pipeline. Stage 1 compares 15 candidate models. Stage 2 tunes with GridSearchCV and TimeSeriesSplit. Generates 28-day forecast. Save model artifacts. |
| 05 | `05_critical_thinking_ethical_ai.ipynb` | SHAP explainability, overfitting check, residual analysis, four-dimension bias audit, scope boundary documentation. |

---

## Custom Modeling Library — `modeling.py`

| Category | Functions |
|----------|-----------|
| Data loading | `load_and_prepare_dataset`, `show_predictors` |
| Train/test split | `time_split_dept` |
| Evaluation | `evaluate_forecast` |
| Baseline models | `run_naive`, `run_seasonal_naive` |
| Traditional models | `run_exponential_smoothing`, `run_arima`, `run_sarima` |
| ML models | `run_random_forest`, `run_xgboost`, `run_lightgbm` |
| Stage 1 - Model Comparison per Department | `stage1_model_selection_per_dept`, `get_feature_importance_for_stage1_winner`, `plot_feature_importance_all` |
| Feature count selection | `find_best_feature_count`, `plot_feature_count_search` |
| Stage 2 - Hyperparameter Tuning | `get_ml_grid`, `stage2_tune_best_model_per_dept` |
| Forecasting | `forecast_next_28_days`, `forecast_next_28_days_ml_recursive` |
| Visualization | `plot_forecast_vs_actual` |

---

## Methodology

### Data Scope - (What is Selected?)
- **Store:** CA_3 — highest average yearly sales across all M5 stores
- **Category:** FOODS — primary revenue driver, most perishable, most operationally complex
- **Departments:** FOODS_1, FOODS_2, FOODS_3 — modeled separately
- **Forecast level:** Store-department daily aggregation (not SKU level)
- **Forecast horizon:** 28 days

### Note on PCA / Dimensionality Reduction
PCA was deliberately not applied. The lag, rolling mean, EWMA, and rolling max/min features already capture the principal variance structure of the time series across multiple time horizons. Applying PCA would transform interpretable demand signals — such as `lag_1` or `rolling_mean_7` — into abstract linear combinations, removing the explainability required for operational demand planning. Feature dimensionality was instead reduced through a cross-validated importance-based selection step between Stage 1 and Stage 2.

### Modeling Pipeline

**Stage 1 — Model Selection**
- 80/20 time-based train/test split — no shuffling, strict time order
- 15 candidates per department: Naive, Seasonal Naive, Exponential Smoothing (2), ARIMA (3), SARIMA (3), Random Forest (2), XGBoost (2), LightGBM (2)
- Ranked by RMSE — winner advances to Stage 2

**Feature Count Selection (between stages)**
- Tests top 5 → max non-zero importance features using TimeSeriesSplit (5 folds)
- Selects count with lowest CV RMSE
- Results: FOODS_1 = 15 features, FOODS_2 = 10 features, FOODS_3 = 10 features

**Stage 2 — Hyperparameter Tuning**
- GridSearchCV with TimeSeriesSplit (5 folds, 304 rows/fold)
- Search space: n_estimators, learning_rate, max_depth, subsample, colsample_bytree

**28-Day Forecast**
- Statistical models: built-in forecast method
- ML models: recursive step-by-step — rebuilds all lag, rolling, EWMA, and calendar features at each step

---

## Results

### Stage 1 vs Stage 2

| Department | Stage 1 RMSE | Stage 1 MAPE | Stage 2 RMSE | Stage 2 MAPE | Features |
|-----------|-------------|-------------|-------------|-------------|---------|
| FOODS_1 | 73.94 | 12.26% | 76.70 | 12.41% | 15 |
| FOODS_2 | 73.26 | 8.82% | 74.53 | 8.77% | 10 |
| FOODS_3 | 275.66 | 77.04% | 282.31 | 57.80% | 10 |

**Model Winner all departments:** `XGB_200_0.05_5` → `FINAL_XGB_GRID`

> Stage 2 RMSE slightly higher than Stage 1 while MAPE improved — the tuned model is more regularized and consistently accurate on typical days. MAPE is the operationally relevant metric for daily replenishment planning support.

### Top SHAP Features (Test Set — SHAP Summary Plot Ranking)

| Rank | FOODS_1 | FOODS_2 | FOODS_3 |
|------|---------|---------|---------|
| 1 | day_of_week | lag_1 | day_of_week |
| 2 | lag_1 | day_of_week | lag_1 |
| 3 | rolling_mean_7 | snap_CA | lag_28 |
| 4 | lag_28 | lag_28 | lag_7 |
| 5 | rolling_min_7 | lag_7 | day_of_month |

**Forecast horizon:** 2016-04-25 to 2016-05-22

---

## Bias Audit Summary (4-Dimension — Test Period)

### Flagged Groups (MPE gap > 5pp vs department overall)

| Department | Dimension | Group | MPE | Action |
|-----------|----------|-------|-----|--------|
| FOODS_1 | Event | Event day | -6.21% | Manual review on event days |
| FOODS_1 | Weekday | Saturday | +8.13% | Apply Saturday uplift factor |
| FOODS_1 | Demand quartile | Medium-High | +28.75% | Planner review on above-avg days |
| FOODS_2 | Demand quartile | Low | -17.39% | Low impact — only 16 test days |
| FOODS_3 | Event | Event day | -660.63% | **Critical — override before procurement** |
| FOODS_3 | Weekday | Friday | -359.03% | **Critical — override before procurement** |

### Residual Bias

| Department | Mean Residual | Action |
|-----------|--------------|--------|
| FOODS_1 | +9.4 units/day (under-forecasting) | Apply +9 units/day bias correction |
| FOODS_2 | +13.6 units/day (under-forecasting) | Apply +14 units/day bias correction |
| FOODS_3 | -2.6 units/day (over-forecasting) | Negligible |

### SNAP Fairness — Not a Confirmed Issue
No department under-forecasts on SNAP benefit days. FOODS_2 is more accurate on SNAP days (8.41%) than non-SNAP days (8.95%). FOODS_3 SNAP MAPE (8.32%) is far lower than non-SNAP MAPE (81.94%).

---

## SKU Disaggregation Guidance

This model outputs **department-level daily forecasts only**. Before procurement release:

1. Calculate each item's average daily sales contribution % from the most recent 3–6 months
2. Multiply department forecast × item contribution % to derive item-level demand signals
3. Refresh contribution percentages monthly or after any range change
4. New items with no history → use analogue forecasting from a similar active item
5. Phase-in items → inherit phase-out item's contribution %; do not merge phase-out history into phase-in training data (avoids double-counting)

---

## Installation Instructions and How to Run the Code

```bash
# 1. Clone
git clone https://github.com/juliuslaurenmarasigan/walmart-foods-forecasting.git
cd walmart-foods-forecasting

# 2. Install
pip install -r requirements.txt

# 3. Download raw M5 files from Kaggle and place in project root:
#    calendar.csv · sales_train_validation.csv · sell_prices.csv

# 4. Update BASE_DIR in each notebook to your local path

# 5. Run notebooks in order: 01 → 02 → 03 → 04 → 05
```

---

## Processed Data Files

| File | Rows | Description |
|------|------|-------------|
| `scope_selection.csv` | 1 | Top store (CA_3) and category (FOODS) |
| `store_dept_daily_sales.csv` | 5,739 | Department-level daily aggregation — pre-feature engineering |
| `modeling_dataset.csv` | 5,487 | Final modeling-ready dataset — 53 columns, 1,829 rows per department |

---

## Limitations

| Limitation | Impact |
|-----------|--------|
| Forecast errors compound over 28 days | Refit weekly in production |
| Price and SNAP carried forward unchanged | Manual adjustments for known changes |
| FOODS_3 MAPE 58% | All FOODS_3 forecasts need planner review |
| No future event knowledge (promotions) | Overlay confirmed events manually |
| Trained on CA_3 FOODS only | Retrain before applying elsewhere |
| Department-level output | SKU disaggregation required before ordering |
| PCA not applied | Deliberate — preserves interpretability for demand planning |

---

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition. *International Journal of Forecasting*, 38(4), 1346–1364.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- M5 Competition Dataset — Walmart / Makridakis Open Forecasting Center (MOFC).

---

*Last updated: April 2026*

---

## Author Information

### **Julius Lauren Marasigan**
- Post Graduate Diploma in Artificial Intelligence and Machine Learning
- Asian Institute of Management | Makati City, Philippines

Julius is a demand planning professional with over 10 years of experience in supply chain planning and forecasting across multiple industries, including automotive, energy, and consumer goods. This project demonstrates the application of machine learning techniques to real-world demand forecasting challenges, with a focus on improving replenishment decision support for perishable goods.

His areas of interest include demand forecasting, supply chain analytics, and the practical application of machine learning in business operations.

LinkedIn: https://www.linkedin.com/in/julius-lauren-marasigan-37551367/