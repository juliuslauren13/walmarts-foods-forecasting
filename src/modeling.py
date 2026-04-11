# =============================================================================
# modeling.py
# Custom Modeling Library — M5 Sales Forecasting Capstone
# Store: CA_3 | Category: FOODS | Departments: FOODS_1, FOODS_2, FOODS_3
#
# This library centralizes all modeling functions used across the project:
#   - Data loading and preparation
#   - Train-test splitting (time-based)
#   - Baseline models  : Naive, Seasonal Naive
#   - Traditional models: Exponential Smoothing, ARIMA, SARIMA
#   - ML models        : Random Forest, XGBoost, LightGBM
#   - Stage 1          : Model comparison per department
#   - Stage 2          : Hyperparameter tuning of best Stage 1 model
#   - Forecasting      : 28-day ahead prediction
#   - Evaluation       : MAE, RMSE, MAPE
#   - Visualization    : Forecast vs actual plots, feature importance
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


# DATA LOADING AND PREPARATION

def load_and_prepare_dataset(file_path, target_col="sales_qty"):
    """
    Load the modeling dataset from the processed_data folder, clean it, and identify the numeric predictor columns to be used in ML models.
    """
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["dept_id", "date"]).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    exclude_cols = ["date", "store_id", "dept_id", target_col]
    predictors = [col for col in df.columns if col not in exclude_cols]

    numeric_predictors = [col for col in predictors
                          if pd.api.types.is_numeric_dtype(df[col])]
    dropped_predictors = sorted(set(predictors) - set(numeric_predictors))

    return df, numeric_predictors, target_col, dropped_predictors


def show_predictors(predictors):
    """Print the list of predictor features to be used in ML models."""
    print("Predictor features to be used:")
    for col in predictors:
        print(" -", col)


# TRAIN / TEST SPLIT
def time_split_dept(df, dept_id, features, target_col="sales_qty",
                    train_ratio=0.8):
    """
    Split one department's data into train and test sets using a strict time-based cutoff. No shuffling is applied. The first train_ratio fraction of rows is used for training, and the remainder is the test set.
    """
    dept_df = df[df["dept_id"] == dept_id].sort_values("date").reset_index(drop=True)

    if dept_df.empty:
        raise ValueError(f"No data found for dept_id='{dept_id}'")

    train_size = int(len(dept_df) * train_ratio)

    if train_size < 1:
        raise ValueError(f"Training set is empty for dept_id='{dept_id}'")
    if len(dept_df) - train_size < 1:
        raise ValueError(f"Test set is empty for dept_id='{dept_id}'")

    train_df = dept_df.iloc[:train_size].copy()
    test_df  = dept_df.iloc[train_size:].copy()

    X_train = train_df[features].apply(pd.to_numeric, errors="coerce")
    X_test  = test_df[features].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[target_col].astype(float)
    y_test  = test_df[target_col].astype(float)

    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        raise ValueError(
            f"NaN values found in ML features for dept_id='{dept_id}'. "
            "Check that all lag and rolling features are complete after dropna()."
        )

    return dept_df, train_df, test_df, X_train, X_test, y_train, y_test


# FORECAST EVALUATION

def evaluate_forecast(y_true, y_pred):
    """
    Compute MAE, RMSE, and MAPE between actual and predicted values.
    MAPE is computed only on non-zero actual values to avoid division-by-zero errors, which can occur in sparse demand data.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Clip negative predictions to zero (sales cannot be negative)
    y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero = y_true != 0
    if non_zero.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
        ) * 100

    return mae, rmse, mape


# BASELINE MODELS

def run_naive(train_series, test_series):
    """
    Naive forecast: repeat the last observed training value for every future step. This is the simplest possible benchmark and serves as the lower bound for what a useful model must beat.
    """
    if len(train_series) == 0:
        raise ValueError("Training series is empty in run_naive")
    if len(test_series) == 0:
        raise ValueError("Test series is empty in run_naive")

    last_value = train_series.iloc[-1]
    pred = np.repeat(last_value, len(test_series))
    return np.array(test_series), pred


def run_seasonal_naive(train_series, test_series, seasonal_lag=7):
    """
    Seasonal Naive forecast: repeat the last full seasonal cycle (default: 7 days) to fill the forecast horizon. This captures weekly demand patterns without any parameter fitting.
    If the training series is shorter than one seasonal cycle, it falls back to the plain Naive model.
    """
    if len(train_series) == 0:
        raise ValueError("Training series is empty in run_seasonal_naive")
    if len(test_series) == 0:
        raise ValueError("Test series is empty in run_seasonal_naive")

    if len(train_series) < seasonal_lag:
        return run_naive(train_series, test_series)

    last_season = train_series.iloc[-seasonal_lag:].values
    repeats = int(np.ceil(len(test_series) / seasonal_lag))
    pred = np.tile(last_season, repeats)[:len(test_series)]
    return np.array(test_series), pred


# TRADITIONAL MODELS

def run_exponential_smoothing(train_series, forecast_horizon,
                               trend="add", seasonal="add",
                               seasonal_periods=7):
    """
    Holt-Winters Exponential Smoothing: fits a model with trend and seasonal components on the training series and forecasts forward.
    A classical statistical forecasting method that explicitly models level, trend, and seasonality through smoothing equations.
    """
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()
    pred = model.forecast(forecast_horizon)
    return np.array(pred), model


def run_arima(train_series, forecast_horizon, order=(1, 1, 1)):
    """
    ARIMA model: captures autoregressive patterns, integrates for stationarity, and models residual moving average structure.
    The order (p, d, q) controls AR terms, differencing degree, and MA terms respectively. Differencing (d=1) is appropriate when the ADF test indicates non-stationarity.
    """
    model = ARIMA(train_series, order=order).fit()
    pred = model.forecast(forecast_horizon)
    return np.array(pred), model


def run_sarima(train_series, forecast_horizon,
               order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    SARIMA model: extends ARIMA with seasonal AR, differencing, and MA terms. The seasonal period of 7 captures weekly demand cycles observed in the STL decomposition.
    """
    model = ARIMA(train_series, order=order,
                  seasonal_order=seasonal_order).fit()
    pred = model.forecast(forecast_horizon)
    return np.array(pred), model


# ML MODELS

def run_random_forest(X_train, y_train, X_test,
                      n_estimators=200, max_depth=10, random_state=42):
    """
    Random Forest Regressor: an ensemble of decision trees trained on random feature subsets. It handles nonlinear relationships, is robust to outliers, and naturally provides feature importance.
    Negative predictions are clipped to zero after prediction since sales quantities cannot be negative.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    return pred, model


def run_xgboost(X_train, y_train, X_test,
                n_estimators=200, learning_rate=0.05,
                max_depth=5, random_state=42):
    """
    XGBoost Regressor: gradient boosted trees with regularization. Builds trees sequentially, each correcting the residual error of the previous. It is well-suited for tabular feature-rich forecasting problems.
    Negative predictions are clipped to zero after prediction.
    """
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    return pred, model


def run_lightgbm(X_train, y_train, X_test,
                 n_estimators=300, learning_rate=0.05,
                 max_depth=6, num_leaves=31, random_state=42):
    """
    LightGBM Regressor: a gradient boosting framework that uses leaf-wise (best-first) tree growth rather than level-wise growth. This makes it faster and often more accurate than XGBoost
    on large feature sets such as those with many lag and rolling columns.
    LightGBM is added as an additional ML candidate to determine whether its leaf-wise splits offer a better fit to the irregular demand patterns in the FOODS category compared to XGBoost.
    Negative predictions are clipped to zero after prediction.
    """
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        random_state=random_state,
        verbose=-1
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)
    return pred, model


# STAGE 1: MODEL COMPARISON PER DEPARTMENT

def stage1_model_selection_per_dept(df, dept_id, features,
                                    target_col="sales_qty",
                                    train_ratio=0.8):
    """
    Run all candidate models for one department and rank them by RMSE.
    Candidate models:
      Baseline    : Naive, Seasonal Naive
      Traditional : Exponential Smoothing (add-add, add-mul),
                    ARIMA (3 orders), SARIMA (3 orders)
      ML          : Random Forest (2 configs),
                    XGBoost (2 configs),
                    LightGBM (2 configs)
    """
    _, train_df, test_df, X_train, X_test, y_train, y_test = time_split_dept(
        df, dept_id, features,
        target_col=target_col, train_ratio=train_ratio
    )

    dept_results = []

    # ---- Baseline: Naive ----
    y_true_n, y_pred_n = run_naive(y_train, y_test)
    mae, rmse, mape = evaluate_forecast(y_true_n, y_pred_n)
    dept_results.append({
        "dept_id": dept_id, "model_family": "Baseline", "model_name": "Naive",
        "mae": mae, "rmse": rmse, "mape": mape,
        "model_obj": None, "pred": y_pred_n,
        "actual": y_true_n, "dates": test_df["date"].values
    })

    # ---- Baseline: Seasonal Naive ----
    y_true_sn, y_pred_sn = run_seasonal_naive(y_train, y_test, seasonal_lag=7)
    mae, rmse, mape = evaluate_forecast(y_true_sn, y_pred_sn)
    dept_results.append({
        "dept_id": dept_id, "model_family": "Baseline",
        "model_name": "Seasonal_Naive",
        "mae": mae, "rmse": rmse, "mape": mape,
        "model_obj": None, "pred": y_pred_sn,
        "actual": y_true_sn, "dates": test_df["date"].values
    })

    # ---- Traditional: Exponential Smoothing ----
    for trend, seasonal in [("add", "add"), ("add", "mul")]:
        try:
            pred, model = run_exponential_smoothing(
                y_train, len(test_df),
                trend=trend, seasonal=seasonal, seasonal_periods=7
            )
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "Traditional",
                "model_name": f"ES_{trend}_{seasonal}",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | ES ({trend},{seasonal}) failed: {e}")

    # ---- Traditional: ARIMA ----
    for order in [(1, 1, 1), (2, 1, 1), (1, 1, 2)]:
        try:
            pred, model = run_arima(y_train, len(test_df), order=order)
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "Traditional",
                "model_name": f"ARIMA{order}",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | ARIMA {order} failed: {e}")

    # ---- Traditional: SARIMA ----
    for order in [(1, 1, 1), (2, 1, 1), (1, 1, 2)]:
        try:
            pred, model = run_sarima(
                y_train, len(test_df),
                order=order, seasonal_order=(1, 1, 1, 7)
            )
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "Traditional",
                "model_name": f"SARIMA{order}(1,1,1,7)",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | SARIMA {order}(1,1,1,7) failed: {e}")

    # ---- ML: Random Forest ----
    for n_est, depth in [(100, 5), (200, 10)]:
        try:
            pred, model = run_random_forest(
                X_train, y_train, X_test,
                n_estimators=n_est, max_depth=depth
            )
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "ML",
                "model_name": f"RF_{n_est}_{depth}",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | RF ({n_est},{depth}) failed: {e}")

    # ---- ML: XGBoost ----
    for n_est, lr, depth in [(100, 0.1, 3), (200, 0.05, 5)]:
        try:
            pred, model = run_xgboost(
                X_train, y_train, X_test,
                n_estimators=n_est, learning_rate=lr, max_depth=depth
            )
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "ML",
                "model_name": f"XGB_{n_est}_{lr}_{depth}",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | XGB ({n_est},{lr},{depth}) failed: {e}")

    # ---- ML: LightGBM ----
    for n_est, lr, depth, leaves in [(200, 0.05, 6, 31), (300, 0.03, 8, 63)]:
        try:
            pred, model = run_lightgbm(
                X_train, y_train, X_test,
                n_estimators=n_est, learning_rate=lr,
                max_depth=depth, num_leaves=leaves
            )
            mae, rmse, mape = evaluate_forecast(y_test, pred)
            dept_results.append({
                "dept_id": dept_id, "model_family": "ML",
                "model_name": f"LGB_{n_est}_{lr}_{depth}_{leaves}",
                "mae": mae, "rmse": rmse, "mape": mape,
                "model_obj": model, "pred": pred,
                "actual": y_test.values, "dates": test_df["date"].values
            })
        except Exception as e:
            print(f"{dept_id} | LGB ({n_est},{lr},{depth},{leaves}) failed: {e}")

    # ---- Build summary DataFrame ----
    dept_results_df = pd.DataFrame([
        {k: v for k, v in row.items()
         if k not in ["model_obj", "pred", "actual", "dates"]}
        for row in dept_results
    ]).sort_values("rmse").reset_index(drop=True)

    if dept_results_df.empty:
        raise ValueError(
            f"No model results were produced for dept_id='{dept_id}'"
        )

    best_row_summary = dept_results_df.iloc[0]
    best_row_full = next(
        row for row in dept_results
        if row["model_name"] == best_row_summary["model_name"]
        and np.isclose(row["rmse"], best_row_summary["rmse"])
    )

    stage1_best_pred_df = pd.DataFrame({
        "date":             np.array(best_row_full["dates"]),
        "dept_id":          dept_id,
        "actual":           np.array(best_row_full["actual"]),
        "stage1_best_pred": np.array(best_row_full["pred"])
    })

    return {
        "dept_results_raw":    dept_results,
        "dept_results_df":     dept_results_df,
        "best_stage1_row":     best_row_full,
        "stage1_best_pred_df": stage1_best_pred_df
    }


# FEATURE IMPORTANCE

def get_feature_importance_for_stage1_winner(df, dept_id, best_stage1_row,
                                              features, target_col="sales_qty",
                                              train_ratio=0.8):
    """
    Extract feature importances from the Stage 1 winning model, if it is an ML model (Random Forest, XGBoost, or LightGBM).
    For traditional and baseline models, feature importance is not applicable and the function returns None.
    """
    winner_name = best_stage1_row["model_name"]
    if not any(winner_name.startswith(p) for p in ["RF_", "XGB_", "LGB_"]):
        return None

    _, train_df, test_df, X_train, X_test, y_train, y_test = time_split_dept(
        df, dept_id, features,
        target_col=target_col, train_ratio=train_ratio
    )

    if winner_name.startswith("RF_"):
        parts = winner_name.split("_")
        _, model = run_random_forest(
            X_train, y_train, X_test,
            n_estimators=int(parts[1]), max_depth=int(parts[2])
        )
    elif winner_name.startswith("XGB_"):
        parts = winner_name.split("_")
        _, model = run_xgboost(
            X_train, y_train, X_test,
            n_estimators=int(parts[1]),
            learning_rate=float(parts[2]),
            max_depth=int(parts[3])
        )
    elif winner_name.startswith("LGB_"):
        parts = winner_name.split("_")
        _, model = run_lightgbm(
            X_train, y_train, X_test,
            n_estimators=int(parts[1]),
            learning_rate=float(parts[2]),
            max_depth=int(parts[3]),
            num_leaves=int(parts[4])
        )

    importance_df = pd.DataFrame({
        "dept_id":    dept_id,
        "feature":    features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_feature_importance_all(importance_df, dept_id, top_n=20):
    """
    Plot a horizontal bar chart of the top-N most important features for a given department's winning ML model.
    """
    plot_df = importance_df.head(top_n)
    plt.figure(figsize=(10, max(6, len(plot_df) * 0.35)))
    plt.barh(plot_df["feature"][::-1], plot_df["importance"][::-1])
    plt.title(f"Feature Importance — {dept_id} (top {top_n})")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()



# FEATURE COUNT SELECTION

def find_best_feature_count(imp_df, X_train, y_train, model_name,
                             candidates=None, cv_splits=5):
    """
    Find the optimal number of top features to use in Stage 2 by testing different feature counts using TimeSeriesSplit cross-validation.

    The function trains the same model type that won Stage 1 on subsets of features ranked by importance score, and measures the average CV RMSE for each subset size. The feature count that produces
    the lowest CV RMSE is recommended for Stage 2.

    Why this matters:
    Adding more features does not always improve accuracy. Beyond a certain point, additional features introduce noise that the model tries to fit, which can hurt generalization to new data.
    This function finds the point where adding more features stops helping — the sweet spot between too few features (underfitting) and too many (overfitting).

    The candidates list controls which feature counts are tested. If not provided, it automatically tests 5, 10, 15, 20, 25, 30, and the total number of non-zero importance features.
    """
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    # Features with non-zero importance only
    nonzero_feats = imp_df[imp_df["importance"] > 0]["feature"].tolist()
    max_feats     = len(nonzero_feats)

    if max_feats == 0:
        raise ValueError("No features with non-zero importance found in imp_df.")

    if candidates is None:
        base = [5, 10, 15, 20, 25, 30, 35, 40]
        candidates = sorted(set([n for n in base if n <= max_feats] + [max_feats]))

    tscv    = TimeSeriesSplit(n_splits=cv_splits)
    results = []

    for n in candidates:
        top_feats = imp_df.head(n)["feature"].tolist()
        X_sub     = X_train[top_feats]

        # Use the same model type that won Stage 1
        if model_name.startswith("RF_"):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif model_name.startswith("XGB_"):
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=200, learning_rate=0.05,
                max_depth=5, random_state=42, verbosity=0
            )
        elif model_name.startswith("LGB_"):
            model = LGBMRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=6, num_leaves=31,
                random_state=42, verbose=-1
            )
        else:
            raise ValueError(
                f"find_best_feature_count only supports ML models. "
                f"Got: {model_name}"
            )

        scores = cross_val_score(
            model, X_sub, y_train,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        results.append({
            "n_features": n,
            "cv_rmse":    round(-scores.mean(), 4),
            "cv_std":     round(scores.std(), 4)
        })
        print(f"  Top {n:3d} features — CV RMSE: {-scores.mean():.4f}"
              f" ± {scores.std():.4f}")

    result_df = pd.DataFrame(results).sort_values("n_features").reset_index(drop=True)
    best_n    = int(result_df.loc[result_df["cv_rmse"].idxmin(), "n_features"])

    return result_df, best_n


def plot_feature_count_search(result_df, best_n, dept_id):
    """
    Plot the CV RMSE curve from find_best_feature_count to help visually identify the optimal number of features for a department.
    The shaded band shows the standard deviation across CV folds. When two feature counts have overlapping bands, their difference is within noise — prefer the simpler model (fewer features).
    """
    plt.figure(figsize=(9, 4))
    plt.plot(result_df["n_features"], result_df["cv_rmse"],
             marker="o", linewidth=2, color="steelblue", label="CV RMSE")
    plt.fill_between(
        result_df["n_features"],
        result_df["cv_rmse"] - result_df["cv_std"],
        result_df["cv_rmse"] + result_df["cv_std"],
        alpha=0.2, color="steelblue", label="± 1 std"
    )
    plt.axvline(best_n, color="red", linestyle="--",
                linewidth=1.5, label=f"Best = {best_n} features")
    plt.xlabel("Number of Features (top-N by importance)")
    plt.ylabel("CV RMSE")
    plt.title(f"Feature Count vs CV RMSE — {dept_id}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()



# STAGE 2: HYPERPARAMETER TUNING

def get_ml_grid(model_name):
    """
    Return the hyperparameter search grid for the Stage 2 GridSearchCV, based on the name of the Stage 1 winning model.
    The grids are expanded from Stage 1 to search more parameter combinations, including regularization parameters for XGBoost and LightGBM to reduce overfitting on the training window.
    """
    if model_name.startswith("RF_"):
        return {
            "n_estimators": [100, 200, 300],
            "max_depth":    [5, 10, 12]
        }
    elif model_name.startswith("XGB_"):
        return {
            "n_estimators":      [100, 200, 300],
            "learning_rate":     [0.03, 0.05, 0.1],
            "max_depth":         [3, 5, 6],
            "subsample":         [0.8, 1.0],
            "colsample_bytree":  [0.8, 1.0]
        }
    elif model_name.startswith("LGB_"):
        return {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth":     [4, 6, 8],
            "num_leaves":    [31, 63, 127]
        }
    return None


def stage2_tune_best_model_per_dept(df, dept_id, best_stage1_row,
                                    features, target_col="sales_qty",
                                    train_ratio=0.8, cv_splits=5):
    """
    Tune the Stage 1 winning model using GridSearchCV with TimeSeriesSplit cross-validation.
    TimeSeriesSplit is used instead of regular k-fold because it preserves the temporal order of the data: each validation window is always strictly in the future relative to its training window, preventing data leakage.
    With cv_splits=5, five independent validation windows are created, giving a more reliable RMSE estimate for hyperparameter selection than the previous setting of 3 folds.
    For traditional and baseline models that won Stage 1, a manual grid of order combinations is tested instead.
    """
    _, train_df, test_df, X_train, X_test, y_train, y_test = time_split_dept(
        df, dept_id, features,
        target_col=target_col, train_ratio=train_ratio
    )

    winner_name = best_stage1_row["model_name"]
    best_final       = None
    importance_df    = None
    grid_results_df  = None

    # ---- ML models: GridSearchCV with TimeSeriesSplit ----
    if winner_name.startswith("RF_"):
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = get_ml_grid(winner_name)
        tscv  = TimeSeriesSplit(n_splits=cv_splits)
        grid  = GridSearchCV(base_model, param_grid,
                             scoring="neg_root_mean_squared_error",
                             cv=tscv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        pred = np.clip(best_model.predict(X_test), 0, None)
        mae, rmse, mape = evaluate_forecast(y_test, pred)

        best_final = {
            "dept_id":          dept_id,
            "final_model_name": "FINAL_RF_GRID",
            "mae": mae, "rmse": rmse, "mape": mape,
            "model_obj":  best_model,
            "pred":       pred,
            "actual":     y_test.values,
            "dates":      test_df["date"].values,
            "best_params": grid.best_params_
        }
        importance_df = pd.DataFrame({
            "dept_id":    dept_id,
            "feature":    features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        grid_results_df = (
            pd.DataFrame(grid.cv_results_)
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

    elif winner_name.startswith("XGB_"):
        base_model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42, verbosity=0
        )
        param_grid = get_ml_grid(winner_name)
        tscv  = TimeSeriesSplit(n_splits=cv_splits)
        grid  = GridSearchCV(base_model, param_grid,
                             scoring="neg_root_mean_squared_error",
                             cv=tscv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        pred = np.clip(best_model.predict(X_test), 0, None)
        mae, rmse, mape = evaluate_forecast(y_test, pred)

        best_final = {
            "dept_id":          dept_id,
            "final_model_name": "FINAL_XGB_GRID",
            "mae": mae, "rmse": rmse, "mape": mape,
            "model_obj":  best_model,
            "pred":       pred,
            "actual":     y_test.values,
            "dates":      test_df["date"].values,
            "best_params": grid.best_params_
        }
        importance_df = pd.DataFrame({
            "dept_id":    dept_id,
            "feature":    features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        grid_results_df = (
            pd.DataFrame(grid.cv_results_)
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

    elif winner_name.startswith("LGB_"):
        base_model = LGBMRegressor(random_state=42, verbose=-1)
        param_grid = get_ml_grid(winner_name)
        tscv  = TimeSeriesSplit(n_splits=cv_splits)
        grid  = GridSearchCV(base_model, param_grid,
                             scoring="neg_root_mean_squared_error",
                             cv=tscv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        pred = np.clip(best_model.predict(X_test), 0, None)
        mae, rmse, mape = evaluate_forecast(y_test, pred)

        best_final = {
            "dept_id":          dept_id,
            "final_model_name": "FINAL_LGB_GRID",
            "mae": mae, "rmse": rmse, "mape": mape,
            "model_obj":  best_model,
            "pred":       pred,
            "actual":     y_test.values,
            "dates":      test_df["date"].values,
            "best_params": grid.best_params_
        }
        importance_df = pd.DataFrame({
            "dept_id":    dept_id,
            "feature":    features,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        grid_results_df = (
            pd.DataFrame(grid.cv_results_)
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

    # ---- SARIMA grid ----
    elif winner_name.startswith("SARIMA"):
        for order in [(1,1,1),(2,1,1),(1,1,2)]:
            for seas in [(1,1,1,7),(2,1,1,7),(1,1,2,7)]:
                try:
                    pred, model = run_sarima(y_train, len(test_df),
                                             order=order, seasonal_order=seas)
                    mae, rmse, mape = evaluate_forecast(y_test, pred)
                    result = {
                        "dept_id": dept_id,
                        "final_model_name": f"FINAL_SARIMA{order}{seas}",
                        "mae": mae, "rmse": rmse, "mape": mape,
                        "model_obj": model, "pred": pred,
                        "actual": y_test.values,
                        "dates": test_df["date"].values
                    }
                    if best_final is None or rmse < best_final["rmse"]:
                        best_final = result
                except Exception as e:
                    print(f"{dept_id} | Stage2 SARIMA {order}{seas} failed: {e}")

    # ---- ARIMA grid ----
    elif winner_name.startswith("ARIMA"):
        for order in [(1,1,1),(2,1,1),(1,1,2),(2,1,2)]:
            try:
                pred, model = run_arima(y_train, len(test_df), order=order)
                mae, rmse, mape = evaluate_forecast(y_test, pred)
                result = {
                    "dept_id": dept_id,
                    "final_model_name": f"FINAL_ARIMA{order}",
                    "mae": mae, "rmse": rmse, "mape": mape,
                    "model_obj": model, "pred": pred,
                    "actual": y_test.values,
                    "dates": test_df["date"].values
                }
                if best_final is None or rmse < best_final["rmse"]:
                    best_final = result
            except Exception as e:
                print(f"{dept_id} | Stage2 ARIMA {order} failed: {e}")

    # ---- ES grid ----
    elif winner_name.startswith("ES_"):
        for trend, seasonal in [("add","add"),("add","mul"),("mul","add")]:
            try:
                pred, model = run_exponential_smoothing(
                    y_train, len(test_df),
                    trend=trend, seasonal=seasonal, seasonal_periods=7
                )
                mae, rmse, mape = evaluate_forecast(y_test, pred)
                result = {
                    "dept_id": dept_id,
                    "final_model_name": f"FINAL_ES_{trend}_{seasonal}",
                    "mae": mae, "rmse": rmse, "mape": mape,
                    "model_obj": model, "pred": pred,
                    "actual": y_test.values,
                    "dates": test_df["date"].values
                }
                if best_final is None or rmse < best_final["rmse"]:
                    best_final = result
            except Exception as e:
                print(f"{dept_id} | Stage2 ES ({trend},{seasonal}) failed: {e}")

    # ---- Baseline fallback ----
    else:
        best_final = {
            "dept_id":          dept_id,
            "final_model_name": winner_name,
            "mae":  best_stage1_row["mae"],
            "rmse": best_stage1_row["rmse"],
            "mape": best_stage1_row["mape"],
            "model_obj": None,
            "pred":   best_stage1_row["pred"],
            "actual": best_stage1_row["actual"],
            "dates":  best_stage1_row["dates"]
        }

    if best_final is None:
        best_final = {
            "dept_id":          dept_id,
            "final_model_name": winner_name,
            "mae":  best_stage1_row["mae"],
            "rmse": best_stage1_row["rmse"],
            "mape": best_stage1_row["mape"],
            "model_obj": best_stage1_row.get("model_obj"),
            "pred":   best_stage1_row["pred"],
            "actual": best_stage1_row["actual"],
            "dates":  best_stage1_row["dates"]
        }

    # ---- Build final prediction DataFrame ----
    if "dates" in best_final and "actual" in best_final:
        final_pred_df = pd.DataFrame({
            "date":       np.array(best_final["dates"]),
            "dept_id":    dept_id,
            "actual":     np.array(best_final["actual"]),
            "final_pred": np.array(best_final["pred"])
        })
    else:
        final_pred_df = pd.DataFrame({
            "date":       test_df["date"].values,
            "dept_id":    dept_id,
            "actual":     y_test.values,
            "final_pred": np.array(best_final["pred"])
        })

    return {
        "best_final":           best_final,
        "feature_importance_df": importance_df,
        "grid_results_df":       grid_results_df,
        "final_pred_df":         final_pred_df
    }


# VISUALIZATION

def plot_forecast_vs_actual(pred_df, pred_col, title):
    """
    Plot actual vs forecasted values on a time axis.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["date"], pred_df["actual"],
             label="Actual", linewidth=1.8)
    plt.plot(pred_df["date"], pred_df[pred_col],
             label="Forecast", linewidth=1.8, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales Qty")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 28-DAY FUTURE FORECAST

def forecast_next_28_days(final_model_name, model_obj,
                          history_series, future_dates,
                          future_ml_features=None):
    """
    Generate a 28-day ahead forecast using the final tuned model.

    For ML models (RF, XGB, LGB), future_ml_features must be provided as a pre-built DataFrame with the same columns used during training.

    For statistical models (ES, ARIMA, SARIMA), the model's built-in forecast method is called directly using the fitted state from training.
    """
    horizon = len(future_dates)

    if final_model_name == "Naive":
        pred = np.repeat(history_series.iloc[-1], horizon)

    elif final_model_name == "Seasonal_Naive":
        last_season = history_series.iloc[-7:].values
        repeats = int(np.ceil(horizon / 7))
        pred = np.tile(last_season, repeats)[:horizon]

    elif any(final_model_name.startswith(p)
             for p in ["FINAL_ES", "FINAL_ARIMA", "FINAL_SARIMA"]):
        pred = np.array(model_obj.forecast(horizon))

    elif any(final_model_name.startswith(p)
             for p in ["FINAL_RF", "FINAL_XGB", "FINAL_LGB"]):
        if future_ml_features is None:
            raise ValueError(
                f"future_ml_features is required for {final_model_name} forecasting."
            )
        future_ml_features = (
            future_ml_features
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        if future_ml_features.isnull().sum().sum() > 0:
            raise ValueError(
                "NaN values found in future_ml_features. "
                "Check that all lag/rolling features are complete."
            )
        pred = np.clip(
            model_obj.predict(future_ml_features.to_numpy(dtype=float)), 0, None
        )
    else:
        raise ValueError(f"Unsupported final model name: {final_model_name}")

    return pd.DataFrame({
        "date":         pd.to_datetime(future_dates),
        "forecast_qty": pred
    })


def forecast_next_28_days_ml_recursive(model_obj, history_df,
                                        feature_cols, horizon=28):
    """
    Recursive 28-day forecast for ML models, updating all lag and rolling features at each step using the growing prediction history.

    At each step, a single new row is constructed by:
      1. Advancing the date by 1 day
      2. Updating ALL lag features from the current history
      3. Updating ALL rolling mean and rolling std features
      4. Updating calendar features (weekday, month, year, is_weekend, is_month_end, is_quarter_end, day_of_month)
      5. Carrying forward price and SNAP (no future values available)
      6. Predicting and appending to history for the next step

    This approach correctly propagates predictions into future lag and rolling features rather than leaving them stale.
    Note: prediction error compounds over the 28-step horizon, which is expected in recursive forecasting. This is one reason why the model's in-sample accuracy is typically better than its future forecast accuracy.
    """
    history_df = history_df.copy().sort_values("date").reset_index(drop=True)
    predictions = []

    for _ in range(horizon):
        last_row  = history_df.iloc[-1:].copy()
        next_date = pd.to_datetime(last_row["date"].values[0]) + pd.Timedelta(days=1)
        new_row   = last_row.copy()
        new_row["date"] = next_date

        # Calendar features
        new_row["weekday"]        = next_date.weekday()
        new_row["day_of_week"]    = next_date.weekday()
        new_row["month"]          = next_date.month
        new_row["year"]           = next_date.year
        new_row["is_weekend"]     = int(next_date.weekday() in [5, 6])
        new_row["day_of_month"]   = next_date.day
        new_row["is_month_end"]   = int(next_date == next_date + pd.offsets.MonthEnd(0))
        new_row["is_quarter_end"] = int(next_date == next_date + pd.offsets.QuarterEnd(0))
        new_row["week_of_year"]   = next_date.isocalendar()[1]

        sales_hist = history_df["sales_qty"]

        # All lag features
        for lag in [1, 2, 3, 7, 14, 21, 28, 42, 56, 84]:
            col = f"lag_{lag}"
            if col in feature_cols:
                new_row[col] = sales_hist.iloc[-lag] if len(sales_hist) >= lag else np.nan

        # All rolling mean features
        for window in [7, 14, 28, 42, 56]:
            col = f"rolling_mean_{window}"
            if col in feature_cols:
                new_row[col] = sales_hist.iloc[-window:].mean() if len(sales_hist) >= window else np.nan

        # All rolling std features
        for window in [7, 14, 28, 42, 56]:
            col = f"rolling_std_{window}"
            if col in feature_cols:
                new_row[col] = sales_hist.iloc[-window:].std() if len(sales_hist) >= window else np.nan

        # EWMA features
        for span in [7, 14, 28]:
            col = f"ewma_{span}"
            if col in feature_cols:
                new_row[col] = sales_hist.ewm(span=span).mean().iloc[-1]

        # Rolling max and min
        for window in [7, 28]:
            col_max = f"rolling_max_{window}"
            col_min = f"rolling_min_{window}"
            if col_max in feature_cols:
                new_row[col_max] = sales_hist.iloc[-window:].max() if len(sales_hist) >= window else np.nan
            if col_min in feature_cols:
                new_row[col_min] = sales_hist.iloc[-window:].min() if len(sales_hist) >= window else np.nan

        # Carry-forward: price, SNAP, events (no future values known)
        for carry_col in ["sell_price", "snap_CA", "is_event", "event_count",
                          "is_cultural_event", "is_national_event",
                          "is_religious_event", "is_sport_event",
                          "snap_event_inter"]:
            if carry_col in feature_cols and carry_col in history_df.columns:
                new_row[carry_col] = history_df[carry_col].iloc[-1]

        # Carry-forward price-derived features
        for p_col in ["price_change_7", "price_change_28",
                      "sell_price_lag_7", "sell_price_lag_28"]:
            if p_col in feature_cols and p_col in history_df.columns:
                new_row[p_col] = history_df[p_col].iloc[-1]

        # Interaction
        if "snap_event_inter" in feature_cols:
            new_row["snap_event_inter"] = (
                new_row.get("snap_CA", 0) * new_row.get("is_event", 0)
            )

        # Trend index
        if "trend_index" in feature_cols:
            new_row["trend_index"] = history_df["trend_index"].iloc[-1] + 1

        # Predict
        X_new = (
            new_row[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        y_pred = float(np.clip(
            model_obj.predict(X_new.to_numpy(dtype=float)), 0, None
        )[0])

        new_row["sales_qty"] = y_pred
        predictions.append(y_pred)
        history_df = pd.concat([history_df, new_row], ignore_index=True)

    return predictions
