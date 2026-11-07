# Store Item Demand Forecasting & Inventory Optimization
My UCB AI/ML course Capstone Project

**Author:** Gautam Thockchom  
**Program:** UC Berkeley Executive Education – AI/ML Professional Certificate  

**Repository Link:** [UCB-demand-forecasting-capstone](https://github.com/gSinhaUCB/UCB-demand-forecasting-capstone)  

**Notebook Deliverable:** [./notebooks/capstone_initial_eda_baseline.ipynb](./notebooks/capstone_initial_eda_baseline.ipynb)

**Dataset Source:** [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

---

## 1. Executive Summary & Business Context

In retail supply chains, inventory replenishment requires balancing working capital with customer availability:
* **Stockouts (Under-forecasting):** Lost revenue, brand damage, and customer attrition.
* **Overstocking (Over-forecasting):** Capital lock-up, holding costs, and margin-eroding markdowns.

### Research Objective
> *How accurately can an end-to-end machine learning time-series pipeline forecast daily, item-level retail demand across multiple store locations to optimize automated replenishment schedules and minimize holding costs?*

---

## 2. Dataset Overview & Data Ingestion

The dataset spans 5 years of daily sales records (**2013-01-01 to 2017-12-31**) across multiple store locations and item categories.
* **Schema:** `date` (timestamp), `store` (store ID), `item` (SKU ID), `sales` (daily unit volume).
* **Dual Ingestion:** Checks for local POS data (`data/train.csv`) and falls back to a deterministic synthetic generator that replicates the schema, weekend volume lifts, and annual seasonality for reproducible execution.

---

## 3. Exploratory Data Analysis & Time-Series Diagnostics

* **Data Hygiene:** 0 missing values, 0 duplicate keys across `['date', 'store', 'item']`, and 21,912 validated records.
* **Distribution:** Unimodal, right-skewed daily sales with stable interquartile ranges across stores. Right-tail spikes reflect genuine holiday demand.
* **Time-Series Analysis:** 
  * 30-day moving average highlights steady year-over-year growth and summer peaks.
  * 7-day additive decomposition reveals recurring Friday–Sunday volume surges.
  * Augmented Dickey-Fuller test confirmed non-stationarity (ADF statistic: -2.0716, p-value: 0.2561), requiring explicit lag features.
  * ACF/PACF analysis justified autoregressive lags ($t-1, t-2, t-3$) and weekly seasonal lags ($t-7, t-14$).

---

## 4. Leakage-Free Feature Engineering

Features were engineered using historical observations without look-ahead bias:
1. **Calendar & Cyclical:** Day of week, month, weekend indicator, and trigonometric sine/cosine transforms of annual and weekly cycles.
2. **Autoregressive Lags:** Grouped sales lags ($t-1, t-2, t-3, t-7, t-14$) per store-item pair.
3. **Rolling Windows:** 7-day and 30-day moving averages and standard deviations calculated with a `.shift(1)` backward shift before aggregation.

---

## 5. Model Progression & Evaluation Leaderboard

All models were evaluated using a 5-fold forward-chaining `TimeSeriesSplit` cross-validation scheme.

| Model Architecture | Validation RMSE | Validation MAPE (%) | Evaluation Status |
| :--- | :---: | :---: | :--- |
| **Naive Baseline (Lag 1)** | 7.849 | 16.25% | Baseline Benchmark |
| **Tuned Ridge ($\alpha=100.0$)** | 3.787 | 9.27% | Linear ML Baseline |
| **ARIMA(1,1,1) Aggregate** | 119.001* | 18.02% | Macro Network Trend |
| **Gradient Boosting (Champion)** | **3.274** | **7.76%** | **Selected Production Model** |

*\*Note: ARIMA models aggregate network-wide daily sales volume, leading to a higher scale.*

**Key Takeaway:** Gradient Boosting delivered the highest accuracy, reducing RMSE error by **58.3%** and MAPE by **52.2%** compared to the baseline benchmark.

---

## 6. Operationalization & Deployment

* **Batch Scoring:** A daily Apache Airflow DAG queries closing sales at 00:00 UTC, updates rolling features, and writes to a central feature store.
* **API Inference:** A containerized FastAPI service serves 7-day rolling forecasts via a `/predict/demand` REST endpoint.
* **Inventory Policy:** Forecasts feed into dynamic Safety Stock and Reorder Point (ROP) formulas:
  $$\text{Safety Stock} = 1.65 \times \sigma_{\text{forecast}} \times \sqrt{L}$$
  $$\text{Reorder Point (ROP)} = (\hat{y}_{\text{daily}} \times L) + \text{Safety Stock}$$
* **Monitoring:** Tracks rolling 30-day MAPE; triggers automated retraining if error degrades beyond 12%.

---

## 7. Project Structure & Setup

**Repository File Structure:**
```text
├── UCB_CapStone.ipynb      # Executed Jupyter Notebook
├── README.md               # Project documentation and summary
├── requirements.txt        # Python library dependencies
└── data/                   # Data directory (optional; fallback included)
```

**Running Locally:**
```bash
pip install -r requirements.txt
jupyter notebook UCB_CapStone.ipynb
```



