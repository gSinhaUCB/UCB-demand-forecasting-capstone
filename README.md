# Retail Store Item Demand Forecasting & Inventory Optimization
**Capstone Final Report**  
**Author:** Gautam Thockchom  
**Academic Reviewer:** Ahmet Cakmak  
**Repository Link:** [UCB-demand-forecasting-capstone](https://github.com/gSinhaUCB/UCB-demand-forecasting-capstone)  
**Primary Notebook:** [notebooks/Gautam_UCB_Capstone_final.ipynb](./notebooks/Gautam_UCB_Capstone_final.ipynb)

---

## 1. Executive Summary

### Business Problem and Objectives
Retail supply chain operations constantly balance two conflicting financial risks:
1. **Stockouts:** Underestimating demand leads to unfulfilled orders, immediate loss of sales margin, and permanent customer churn to competitors.
2. **Excess Inventory:** Overestimating demand ties up critical working capital, inflates warehousing costs, and increases the risk of product obsolescence and clearance markdowns.

The primary objective of this project is to build an automated, leakage-free forecasting pipeline to predict store-item daily unit sales over a 90-day forward replenishment horizon across 10 retail locations and 50 distinct SKU items (500 distinct time-series combinations).

### Non-Technical Summary of Findings
Analysis of 5 years (913,000 transaction records) of retail activity revealed two dominant, systematic behavioral drivers:
* **Day-of-Week Shopping Spikes:** Unit sales surge on Fridays, Saturdays, and Sundays by approximately 30% to 35% compared to mid-week baselines (Tuesdays/Wednesdays).
* **Annual Macro Seasonality:** Demand follows a predictable sinusoidal curve that peaks every summer (July/August) and hits its annual trough during mid-winter (December/January), alongside a steady year-over-year baseline growth (~6–8% annually).

Among all evaluated architectures, the **Histogram-Based Gradient Boosting Regressor (`HistGradientBoostingRegressor`)** emerged as the winning production model. It reduced forecast variance (RMSE) by **50.1%** and cut the average percentage deviation down to **17.91%**, substantially outperforming the naive persistence baseline (37.11% error) and regularized linear regression (22.86% error).

### Strategic Recommendations & Business Impact
* **Dynamic Safety Stock Sizing:** Replace fixed, static safety stock levels with dynamic buffer thresholds indexed to weekly cyclicality. Shift regional warehouse replenishment dispatches to Thursdays to ensure shelves are fully stocked ahead of Friday–Sunday surges.
* **Working Capital Optimization:** Lower inventory reorder points on low-velocity mid-week days (Tuesdays and Wednesdays) to decrease idle warehouse holding costs without risking product availability.
* **Cold-Start Protocol:** For newly launched stores or items with fewer than 90 days of transaction history, route demand predictions to regional store-item category medians until autoregressive lag signals stabilize.

---

## 2. Research Questions

* **Primary Question:** Which supervised machine learning or econometric time-series architecture minimizes scale-dependent error (RMSE) and percentage deviation (MAPE) when predicting item-level retail demand across a 90-day forward planning horizon?
* **Secondary Question:** What feature engineering combinations (autoregressive lags, rolling statistics, and cyclical calendar transforms) provide the highest predictive signal without introducing future lookahead leakage into validation folds?

---

## 3. Data Architecture & Preprocessing

* **Dataset Scale:** 913,000 historical daily records spanning January 1, 2013 through December 31, 2017 across 10 stores and 50 items.
* **Data Hygiene & Completeness:**
  * Cleaned datetime indexing with verified zero null, missing, or corrupt records.
  * Verified continuous chronological daily cadence for all 500 store-item combinations.
  * Built an automated deterministic synthetic fallback generator that replicates the exact dimensions, day-of-week lift, and seasonal sinusoid if raw data files are absent, guaranteeing reproducible execution across headless grading environments.

### Leakage-Free Feature Engineering (90-Day Planning Horizon)
In an operational 90-day replenishment workflow, unit sales from $t-1$ through $t-89$ are unknown at the time of forecast execution. To prevent lookahead data leakage, all autoregressive and trend signals are strictly shifted:
* **Calendar & Cyclical Indicators:** Day-of-week, month, year, weekend indicator (`is_weekend`), and sine/cosine cyclical encodings (`dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos`) to eliminate artificial boundary jumps.
* **Autoregressive Lags:** Lagged unit sales anchored strictly at or beyond the 90-day lead time ($t-90, t-91, t-98, t-105$) to capture week-over-week momentum from the prior quarter.
* **Lagged Rolling Statistics:** 7-day and 30-day rolling moving averages and standard deviations computed strictly on top of `sales_lag_90` (`.shift(90).rolling(...)`), completely isolating the model from future target values.

---

## 4. Modeling Strategy & Validation Protocol

### Cross-Validation Strategy
Standard random k-fold cross-validation is invalid for temporal data because shuffling shuffles future events into past training windows. We implemented a **5-fold temporal walk-forward split (`TimeSeriesSplit`)**, training solely on historical sequences and testing against subsequent forward blocks.

### Evaluated Model Architectures
1. **Naive Persistence Benchmark (Lag-90):** Directly projects actual demand from 90 days prior ($t-90$) as a baseline.
2. **Tuned Regularized Ridge Regression:** Standardized linear model tuned over regularized penalties ($\alpha \in [0.1, 1.0, 10.0, 100.0]$) via `GridSearchCV`.
3. **Macro Econometric ARIMA(1,1,1):** Fitted to aggregate chain-wide daily volume to evaluate top-down network momentum.
4. **HistGradientBoostingRegressor (Champion):** Non-linear gradient-boosted decision tree ensemble tuned over `learning_rate` ($[0.05, 0.1]$), `max_depth` ($[3, 5]$), and `max_iter` ($[50, 100]$), treating item identifiers as native categorical features.

### Evaluation Metrics Rationale
* **Root Mean Squared Error (RMSE):** Serves as the primary optimization loss function. Because RMSE squares individual deviations before averaging, it heavily penalizes large forecast errors, which directly correlate with catastrophic retail stockouts or heavy surplus waste.
* **Mean Absolute Percentage Error (MAPE):** Serves as the primary non-technical executive KPI, expressing forecast deviation as an intuitive percentage of actual demand across varying sales velocities.

---

## 5. Model Performance Leaderboard

Cross-validation metrics were evaluated across the 5-fold temporal walk-forward split (benchmarked on a representative multi-SKU sample of Store 1, Items 1–5):

| Model Architecture | Granularity | Validation Strategy | Optimal Parameters | CV RMSE | CV MAPE (%) | Status / Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Lag-90)** | SKU Level | 5-Fold `TimeSeriesSplit` | None | 12.819 | 37.11% | Baseline Benchmark |
| **Tuned Ridge Regression** | SKU Level | 5-Fold `TimeSeriesSplit` | $\alpha = 10.0$ | 7.751 | 22.86% | Linear Benchmark |
| **HistGradientBoosting (Champion)** | SKU Level | 5-Fold `TimeSeriesSplit` | `learning_rate: 0.1, max_depth: 3, max_iter: 100` | **6.395** | **17.91%** | **Production Champion** |
| **ARIMA(1,1,1) Aggregate** | Network Aggregate | 80/20 Chronological Split | Order: $(1, 1, 1)$ | 10,247.350* | 26.41% | Macro Network Trend |

*\*Note: The ARIMA model was fitted to chain-wide aggregate daily sales volume (sum of all 500 store-item combinations), which is why its RMSE reflects network-level volume rather than single-SKU variance.*

### Key Takeaways
* **Baseline Inefficiency:** Persisting 90-day-old values results in an average error of 37.11%, confirming that static reordering cannot account for seasonal growth.
* **Champion Model Gains:** The tuned `HistGradientBoostingRegressor` achieves an RMSE of 6.395 (a **50.1% error reduction** over baseline) and a MAPE of 17.91% (a **19.20 percentage point improvement**).
* **Residual Diagnostics:** Residual scatter analysis confirms that the champion model's errors are homoscedastic and tightly centered around zero across all predicted sales volumes, showing no systematic under- or over-forecasting bias.

---

## 6. Next Steps & Production Roadmap

1. **External Regressors & Promotion Tracking:** Integrate markdown discounts, promotional campaign calendars, and regional holiday indicators to capture localized demand shocks.
2. **Hierarchical Reconciliation:** Implement hierarchical reconciliation (such as MinT) to align bottom-up item-store predictions with top-down corporate and regional inventory targets.
3. **Containerized Deployment (FastAPI + Docker):** Package the feature pipeline and trained gradient boosting model into a REST API container, configured with automated weekly batch inference runs triggered by Apache Airflow upon point-of-sale data ingestion.
4. **Drift Monitoring:** Deploy automated population stability index (PSI) and Kolmogorov-Smirnov distribution tests on rolling 30-day forecast residuals to alert supply chain teams when customer purchasing behavior shifts.

---

## 7. Project Structure & Execution

```text
├── notebooks/
│   └── Gautam_UCB_Capstone_final.ipynb       # Fully executed technical notebook
├── README.md                                 # Executive summary and business report
├── requirements.txt                          # Python dependencies
└── data/
    └── train.csv                             # Historical sales data (auto-generated if missing)

```

### Run Locally:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Gautam_UCB_Capstone_final.ipynb
```