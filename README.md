# UCB-demand-forecasting-capstone
My UCB AI/ML course Capstone Project

# Time-Series Demand Forecasting & Inventory Optimization

**Repository Link:** [UCB-demand-forecasting-capstone](https://github.com/gSinhaUCB/UCB-demand-forecasting-capstone)  
**Notebook Deliverable:** [./notebooks/capstone_initial_eda_baseline.ipynb](./notebooks/capstone_initial_eda_baseline.ipynb)

---

## 1. Executive Summary & Research Question

### The Business Challenge
In modern retail supply chains, inventory management is a delicate balancing act. Relying on manual heuristics or simple moving averages results in two chronic failure modes:
1. **Stockouts:** Depleted shelves result in immediate lost revenue and customer defection to competitors.
2. **Overstocking:** Excess stock ties up working capital, inflates holding/warehousing costs, and causes inventory shrink or severe markdown liquidation.

### Research Question
> **How accurately can an end-to-end time-series forecasting model predict daily, item-level retail sales across multiple store locations to optimize replenishment schedules and minimize inventory holding costs?**

### Key Findings from Initial Exploratory Data Analysis (EDA)
* **Pervasive Seasonality & Upward Trend:** Sales exhibit strong multi-level seasonality—both high-frequency day-of-week patterns (demand consistently peaks on weekends) and annual macro-cycles (mid-summer demand surges).
* **Cross-Entity Demand Uniformity:** Both aggregate store demand and individual item velocity follow structurally consistent seasonal envelopes, indicating that global temporal features (day of week, month, rolling statistics) carry substantial predictive utility.
* **Autocorrelation & Stationarity:** Standard ADF (Augmented Dickey-Fuller) testing reveals strong seasonal unit roots. Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots confirm significant lag correlations at t-1 through t-7 as well as weekly seasonal intervals (t-7, t-14, t-21).

### Initial Baseline Model Results
* **Model:** Baseline Ridge Regression with engineered temporal lags (t-1 through t-14), rolling moving averages (7-day and 30-day), and calendar indicators.
* **Validation Strategy:** Chronological temporal cross-validation (`TimeSeriesSplit`, 5 splits) to prevent look-ahead bias and future data leakage.
* **Evaluation Metrics:**
  * **Test Root Mean Squared Error (RMSE):** ~3.5 to 6.5 units
  * **Test Mean Absolute Percentage Error (MAPE):** ~12% to 15%
* This baseline outperforms the naive persistence baseline and establishes the target to beat in Module 24 using Gradient Boosting and SARIMAX.

---

## 2. Dataset Overview
* **Primary Data Source:** Kaggle Store Item Demand Forecasting Challenge (5 years of daily POS transactions across stores and unique items).
* **Granularity:** Daily unit sales per store-item combination (`date`, `store`, `item`, `sales`).
* **Data Hygiene & Completeness:** Zero duplicate records, continuous calendar dates without missing temporal periods, and complete transactional coverage across all store-item permutations.

---

## 3. Exploratory Data Analysis & Feature Engineering
* **Decomposition:** Additive seasonal-trend decomposition conducted on aggregate and sample item signals to isolate trend, weekly/annual seasonality, and residual stochastic noise.
* **Stationarity & Correlation:** Differencing and autocorrelation analyses confirmed that sales signals become stationary after accounting for weekly differencing and trends.
* **Feature Engineering Pipeline:**
  * **Temporal Attributes:** Year, Quarter, Month, Day of Month, Day of Week, Weekend indicator, Day of Year.
  * **Cyclical Transforms:** Sine/Cosine encoding of day-of-year and day-of-week to preserve calendar cyclicality.
  * **Lags & Window Aggregates:** Shifted lag features (t-1, t-2, t-3, t-7, t-14) and backward rolling aggregates (7-day mean, 7-day standard deviation) engineered strictly on past observations without forward-looking data leakage.

---

## 4. Baseline Modeling & Evaluation Framework
* **Selected Estimator:** Regularized Linear Model (`Ridge(alpha=1.0)`) fitted on lag-engineered features.
* **Evaluation Metric Rationale:**
  * **RMSE (Root Mean Squared Error):** Heavily penalizes large forecasting mistakes, which in retail logistics correspond to acute stockout events or massive overstocking that disrupts warehouse throughput.
  * **MAPE (Mean Absolute Percentage Error):** Delivers an intuitive, scale-independent metric for inventory planners to understand error margins relative to overall sales volume.
* **Leakage Prevention:** Standard k-fold cross-validation is strictly avoided because random partitioning leaks future information into past predictions. All evaluation utilizes forward temporal chaining (`TimeSeriesSplit`).

---

## 5. Roadmap for Module 24
* Implement and benchmark advanced gradient-boosted decision trees (`LightGBM`, `XGBoost`, `CatBoost`) handling multi-series interactions and hierarchical grouping.
* Experiment with statistical and econometric models (`AutoARIMA` / `SARIMAX`) incorporating external regressors (promotional flags, holiday schedules).
* Translate forecast intervals into actionable supply-chain policies: calculating Safety Stock levels, Reorder Points (ROP), and expected holding cost reductions versus current heuristic baselines.

---

## 6. Repository Structure
