# Data Dictionary
## From History to Forecast: A Machine Learning Approach to Daily Food Demand Forecasting at Walmart CA_3 Using the M5 Dataset
**Author:** Julius Lauren Marasigan | Post Graduate Diploma in Artificial Intelligence and Machine Learning | April 2026

---

## Raw Input Files (Not Included — Download from Kaggle)

Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download and place these three files in `data/raw/` before running Notebook 01.

---

### `calendar.csv`

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `date` | String | Calendar date | YYYY-MM-DD | Primary key |
| `wm_yr_wk` | Integer | Walmart internal week number | — | Links to sell_prices.csv |
| `weekday` | String | Day of the week | Monday to Sunday | — |
| `wday` | Integer | Numeric day (Walmart encoding) | 1–7 | 1=Saturday, 7=Friday |
| `month` | Integer | Calendar month | 1–12 | — |
| `year` | Integer | Calendar year | 2011–2016 | — |
| `d` | String | Day ID matching sales columns | d_1 to d_1941 | Links to sales file |
| `event_name_1` | String | Name of first event, if any | Various / NaN | Null if no event |
| `event_type_1` | String | Type of first event | Cultural, National, Religious, Sporting / NaN | Null if no event |
| `event_name_2` | String | Name of second event, if any | Various / NaN | Null if no second event |
| `event_type_2` | String | Type of second event, if any | Cultural, National, Religious, Sporting / NaN | Null if no second event |
| `snap_CA` | Binary | SNAP purchases allowed in California | 0=No, 1=Yes | Used as model feature |
| `snap_TX` | Binary | SNAP purchases allowed in Texas | 0=No, 1=Yes | Not used — out of scope |
| `snap_WI` | Binary | SNAP purchases allowed in Wisconsin | 0=No, 1=Yes | Not used — out of scope |

---

### `sales_train_validation.csv`

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `id` | String | Unique row identifier | — | Combines item, store, and suffix |
| `item_id` | String | Item identifier | e.g. FOODS_1_001 | — |
| `dept_id` | String | Department identifier | FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, etc. | Filtered to FOODS_1/2/3 |
| `cat_id` | String | Product category | FOODS, HOBBIES, HOUSEHOLD | Filtered to FOODS |
| `store_id` | String | Store identifier | CA_1 to CA_4, TX_1 to TX_3, WI_1to WI_3 | Filtered to CA_3 |
| `state_id` | String | State | CA, TX, WI | Filtered to CA |
| `d_1` to `d_1941` | Integer | Daily unit sales per day column | 0 and above | Wide format — reshaped to long in Notebook 01 |

---

### `sell_prices.csv`

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `store_id` | String | Store identifier | CA_1 to CA_4, TX_1 to TX_3, WI_1to WI_3 | — |
| `item_id` | String | Item identifier | — | — |
| `wm_yr_wk` | Integer | Walmart week number | — | Links to calendar.csv |
| `sell_price` | Float | Weekly selling price of the item | 0.00 and above | Aggregated to dept avg in Notebook 02 |

---

## Processed Files (`data/processed/`)

---

### `scope_selection.csv`
**Created by:** Notebook 01 | **Used by:** Notebook 02 | **Rows:** 1

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `top_store` | String | Store with highest average yearly sales | CA_3 | Identified programmatically |
| `top_category` | String | Category with highest average yearly sales at top store | FOODS | Identified programmatically |

---

### `store_dept_daily_sales.csv`
**Created by:** Notebook 02 | **Used by:** Notebook 03 | **Rows:** 5,739 | **Date range:** 2011-01-29 to 2016-04-24

Daily sales aggregated from SKU level to department level for CA_3 FOODS. One row per department per day.

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `date` | String | Calendar date | 2011-01-29 to 2016-04-24 | YYYY-MM-DD format |
| `dept_id` | String | Department identifier | FOODS_1, FOODS_2, FOODS_3 | — |
| `weekday` | String | Day of the week | Monday to Sunday | — |
| `month` | Integer | Calendar month | 1–12 | — |
| `year` | Integer | Calendar year | 2011–2016 | — |
| `snap_CA` | Binary | SNAP benefit issuance day in California | 0=No, 1=Yes | From calendar.csv |
| `is_event` | Binary | At least one special event on this date | 0=No, 1=Yes | Derived from event columns |
| `event_count` | Integer | Number of events on this date | 0, 1, 2 | — |
| `is_cultural_event` | Binary | Cultural event on this date | 0=No, 1=Yes | — |
| `is_national_event` | Binary | National event on this date | 0=No, 1=Yes | — |
| `is_religious_event` | Binary | Religious event on this date | 0=No, 1=Yes | — |
| `is_sport_event` | Binary | Sporting event on this date | 0=No, 1=Yes | — |
| `sales_qty` | Integer | Total daily units sold across all items in the department | 0 and above | **Aggregated from SKU level** |
| `sell_price` | Float | Average selling price across all items in the department | 0.00 and above | Weighted by item count |

---

### `modeling_dataset.csv`
**Created by:** Notebook 03 | **Used by:** Notebooks 04 and 05 | **Rows:** 5,487 | **Date range:** 2011-04-23 to 2016-04-24

Final modeling-ready dataset with all engineered features. First 84 rows per department removed due to lag_84 requiring 84 days of prior history. All lag and rolling features use `.shift(1)` — today's actual sales never appear in its own feature row.

**Identifiers and Target**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `date` | String | Calendar date | 2011-04-23 to 2016-04-24 | YYYY-MM-DD format |
| `dept_id` | String | Department identifier | FOODS_1, FOODS_2, FOODS_3 | — |
| `sales_qty` | Integer | Total daily units sold | 0 and above | **Target variable** |

**Original Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `weekday` | String | Day of the week (text) | Monday to Sunday | Not used as model feature — numeric version below |
| `month` | Integer | Calendar month | 1–12 | — |
| `year` | Integer | Calendar year | 2011–2016 | — |
| `snap_CA` | Binary | SNAP benefit day | 0=No, 1=Yes | Top feature for FOODS_2 |
| `is_event` | Binary | Event day flag | 0=No, 1=Yes | — |
| `event_count` | Integer | Number of events | 0, 1, 2 | — |
| `is_cultural_event` | Binary | Cultural event flag | 0=No, 1=Yes | — |
| `is_national_event` | Binary | National event flag | 0=No, 1=Yes | — |
| `is_religious_event` | Binary | Religious event flag | 0=No, 1=Yes | — |
| `is_sport_event` | Binary | Sporting event flag | 0=No, 1=Yes | — |
| `sell_price` | Float | Average department selling price | 0.00 and above | — |

**Lag Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `lag_1` | Float | Sales 1 day ago | 0 and above | Yesterday's demand |
| `lag_2` | Float | Sales 2 days ago | 0 and above | — |
| `lag_3` | Float | Sales 3 days ago | 0 and above | — |
| `lag_7` | Float | Sales 7 days ago | 0 and above | Same weekday last week |
| `lag_14` | Float | Sales 14 days ago | 0 and above | Two weeks ago |
| `lag_21` | Float | Sales 21 days ago | 0 and above | Three weeks ago |
| `lag_28` | Float | Sales 28 days ago | 0 and above | Same period last month |
| `lag_42` | Float | Sales 42 days ago | 0 and above | Six weeks ago |
| `lag_56` | Float | Sales 56 days ago | 0 and above | Eight weeks ago |
| `lag_84` | Float | Sales 84 days ago | 0 and above | Determines row removal threshold |

**Rolling Mean Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `rolling_mean_7` | Float | Average sales over past 7 days | 0 and above | One-week demand level |
| `rolling_mean_14` | Float | Average sales over past 14 days | 0 and above | Two-week demand level |
| `rolling_mean_28` | Float | Average sales over past 28 days | 0 and above | Monthly demand level |
| `rolling_mean_42` | Float | Average sales over past 42 days | 0 and above | Six-week demand level |
| `rolling_mean_56` | Float | Average sales over past 56 days | 0 and above | Eight-week demand level |

**Rolling Standard Deviation Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `rolling_std_7` | Float | Std dev of sales over past 7 days | 0 and above | One-week demand variability |
| `rolling_std_14` | Float | Std dev of sales over past 14 days | 0 and above | Two-week demand variability |
| `rolling_std_28` | Float | Std dev of sales over past 28 days | 0 and above | Monthly demand variability |
| `rolling_std_42` | Float | Std dev of sales over past 42 days | 0 and above | Six-week demand variability |
| `rolling_std_56` | Float | Std dev of sales over past 56 days | 0 and above | Eight-week demand variability |

**EWMA Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `ewma_7` | Float | Exponentially weighted average, span=7, alpha=0.25 | 0 and above | Reacts quickly to recent changes |
| `ewma_14` | Float | Exponentially weighted average, span=14, alpha=0.133 | 0 and above | Medium-term smoothed demand |
| `ewma_28` | Float | Exponentially weighted average, span=28, alpha=0.067 | 0 and above | Slow-moving background level |

**Rolling Max and Min Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `rolling_max_7` | Float | Highest daily sales in past 7 days | 0 and above | Peak demand signal |
| `rolling_min_7` | Float | Lowest daily sales in past 7 days | 0 and above | Floor demand signal |
| `rolling_max_28` | Float | Highest daily sales in past 28 days | 0 and above | Monthly peak signal |
| `rolling_min_28` | Float | Lowest daily sales in past 28 days | 0 and above | Monthly floor signal |

**Calendar Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `day_of_week` | Integer | Numeric day of week | 0=Mon to 6=Sun | Top SHAP feature for FOODS_1 and FOODS_3 |
| `week_of_year` | Integer | ISO calendar week number | 1–53 | — |
| `is_weekend` | Binary | Saturday or Sunday | 0=No, 1=Yes | — |
| `day_of_month` | Integer | Day number within the month | 1–31 | — |
| `is_month_end` | Binary | Last day of the calendar month | 0=No, 1=Yes | — |
| `is_quarter_end` | Binary | Last day of a calendar quarter | 0=No, 1=Yes | — |

**Price Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `price_change_7` | Float | Percentage change in avg sell price over past 7 days | Varies | — |
| `price_change_28` | Float | Percentage change in avg sell price over past 28 days | Varies | — |
| `sell_price_lag_7` | Float | Average sell price 7 days ago | 0.00 and above | Delayed price effect |
| `sell_price_lag_28` | Float | Average sell price 28 days ago | 0.00 and above | Delayed price effect |

**Trend and Interaction Features**

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `trend_index` | Integer | Sequential day counter per department | 0 and above | Gives model a sense of time position |
| `snap_event_inter` | Binary | snap_CA × is_event. 1 only when both occur on the same date | 0, 1 | Captures combined SNAP and event demand effect |

---

## Report Output Files (`reports/`)

---

### `28_day_forecast.csv`
**Created by:** Notebook 04 | **Rows:** 28 | **Forecast period:** 2016-04-25 to 2016-05-22

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `Date` | String | Forecast date | 2016-04-25 to 2016-05-22 | YYYY-MM-DD format |
| `FOODS_1_Forecast_Qty` | Float | Predicted daily sales units for FOODS_1 | 0 and above | Output of FINAL_XGB_GRID |
| `FOODS_2_Forecast_Qty` | Float | Predicted daily sales units for FOODS_2 | 0 and above | Output of FINAL_XGB_GRID |
| `FOODS_3_Forecast_Qty` | Float | Predicted daily sales units for FOODS_3 | 0 and above | Output of FINAL_XGB_GRID |

---

### `bias_audit_flagged_groups.csv`
**Created by:** Notebook 05 | **Rows:** 21

Groups where MPE gap vs department overall exceeds 5 percentage points. These require demand planner review before forecast release to procurement.

| Variable Name | Type | Description | Allowed Values | Notes |
|---------------|------|-------------|----------------|-------|
| `Department` | String | Department identifier | FOODS_1, FOODS_2, FOODS_3 | — |
| `Audit_Dimension` | String | Which audit check produced the flag | is_event, snap_CA, weekday, demand_quartile | Four dimensions from Notebook 05 Section 3 |
| `Group` | String | Specific subgroup within the dimension | e.g. Event day, Saturday, Low | — |
| `MPE_%` | Float | Mean Percentage Error for this group | Varies | Positive = under-forecast, Negative = over-forecast |
| `MAPE_%` | Float | Mean Absolute Percentage Error for this group | 0 and above | Accuracy without direction |
| `Overall_MPE_%` | Float | Department overall MPE across all test days | Varies | Baseline used for gap comparison |
| `MPE_Gap_pp` | Float | Group MPE minus Overall MPE in percentage points | Varies | Groups exceeding ±5pp are flagged |
| `Bias_Direction` | String | Whether model is over or under forecasting for this group | under-forecast, over-forecast | Derived from MPE sign |

---

*Last updated: April 2026*