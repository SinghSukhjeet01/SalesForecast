# 📈 Sales Forecasting with Python — Random Forest on UK Retail Data

> End-to-end machine learning pipeline that predicts monthly revenue from a real-world UK online retail dataset of 1M+ transactions.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **90.58%** |
| Mean Absolute Error | **£55,344.32** |
| Root Mean Squared Error | **£75,059.14** |
| Sample Prediction | **£575,102.38** (January 2012) |

---

## 📁 Project Structure

```
SalesForecast/
│
├── SalesForecast.ipynb        # Main notebook — full pipeline end to end
└── README.md
```

---

## 🔍 Overview

This project builds a complete sales forecasting system from scratch using the **Online Retail II** dataset — a real-world record of transactions from a UK-based online gift retailer spanning 2009 to 2011.

The goal: given everything we know about past sales behaviour, predict the total revenue the business will generate next month.

The pipeline covers:

- Loading and cleaning 1M+ raw transactions
- Exploratory data analysis across four dimensions
- Feature engineering — 14 predictive inputs including lag variables and seasonal flags
- Training a 200-tree Random Forest Regressor
- Evaluating accuracy with R², MAE, and RMSE
- Generating a concrete next-month revenue forecast
- An interactive ipywidgets UI for no-code predictions

---

## 📦 Dataset

| Column | Description |
|--------|-------------|
| `Invoice` | Unique invoice number; prefix `C` = cancellation |
| `StockCode` | Product identifier |
| `Description` | Product name |
| `Quantity` | Units per transaction |
| `InvoiceDate` | Date and time of transaction |
| `Price` | Unit price in GBP |
| `Customer ID` | Unique customer identifier |
| `Country` | Customer's country |

> **Note:** The dataset contains no `Revenue` column — this is derived as `Quantity × Price` during preprocessing.

---

## ⚙️ Pipeline

### 1. Data Cleaning

- Removed rows with missing `Customer ID` or `Description`
- Dropped cancelled orders (invoices prefixed with `C`)
- Filtered out zero/negative quantities and prices
- Removed outliers: `Price > £500` and `Quantity > 10,000`
- Derived `Revenue = Quantity × Price`
- Parsed `InvoiceDate` to datetime

### 2. Exploratory Data Analysis

Four analyses were conducted before modelling:

- **Revenue by country** — UK dominates; secondary markets across Europe
- **Monthly revenue trend** — clear November–December Christmas spikes each year
- **Day of week & hour of day** — weekday peak with highest activity 11am–2pm
- **Customer segmentation** — three tiers: New (<£1k), Returning (£1k–£5k), VIP (>£5k)

### 3. Feature Engineering

All transactions were aggregated to monthly level. 14 features were engineered:

| Feature | Type | Description |
|---------|------|-------------|
| `Month_Number` | Calendar | Sequential month index |
| `Month_Of_Year` | Calendar | 1–12 |
| `Year` | Calendar | Year number |
| `Is_Christmas` | Seasonal flag | 1 if November or December |
| `Is_Summer` | Seasonal flag | 1 if June, July, or August |
| `Is_Q1` | Seasonal flag | 1 if January, February, or March |
| `Prev_Month_Revenue` | Lag | Revenue from 1 month ago |
| `Prev_2_Month_Revenue` | Lag | Revenue from 2 months ago |
| `Prev_3_Month_Revenue` | Lag | Revenue from 3 months ago |
| `Rolling_3_Month_Avg` | Rolling | Mean of last 3 months' revenue |
| `Monthly_Orders` | Volume | Unique invoices in the month |
| `Monthly_Customers` | Volume | Unique customers in the month |
| `Monthly_Items` | Volume | Total units sold |
| `Avg_Order_Value` | Volume | Mean revenue per invoice |

### 4. Model Training

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)  # 80/20 train-test split
```

### 5. Evaluation

```
R2 Score:  90.58%
MAE:       £55,344.32
RMSE:      £75,059.14

Next Month Prediction:
January 2012: £575,102.38
```

Top features by importance (from Random Forest):
1. `Prev_Month_Revenue`
2. `Rolling_3_Month_Avg`
3. `Prev_2_Month_Revenue`
4. `Is_Christmas`
5. `Monthly_Orders`

---

## 🖥️ Interactive Predictor

The notebook includes an **ipywidgets UI** that lets you select any future year and month and generate a forecast without modifying any code.

```
==================================================
INTERACTIVE SALES PREDICTOR
==================================================
Select a year and month then click Predict Sales
```

Dropdowns for year (2012–2020) and month feed directly into the trained model. Results render inline in the notebook.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

### Run in Google Colab (recommended)

1. Upload `SalesForecast.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload the dataset to your Google Drive at the path referenced in the notebook
3. Run all cells (`Runtime → Run all`)

### Run Locally

```bash
git clone https://github.com/SinghSukhjeet01/SalesForecast.git
cd SalesForecast
jupyter notebook SalesForecast.ipynb
```

Update the dataset path in the data loading cell to your local file path before running.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.3 | Data loading and manipulation |
| `numpy` | ≥1.21 | Numerical computing |
| `matplotlib` | ≥3.4 | Charting |
| `seaborn` | ≥0.11 | Statistical visualisations |
| `scikit-learn` | ≥0.24 | ML model, train/test split, metrics |
| `ipywidgets` | ≥7.6 | Interactive notebook UI |

---

## 🔮 Potential Improvements

- **Gradient boosting** — XGBoost or LightGBM would likely improve accuracy on this tabular dataset
- **Time-series cross-validation** — `TimeSeriesSplit` prevents data leakage across the train/test boundary
- **Hyperparameter tuning** — GridSearchCV on `n_estimators`, `max_depth`, and `min_samples_split`
- **External signals** — UK public holidays, macroeconomic indicators, or seasonal search trends
- **Product-level forecasting** — separate models per category or country for finer-grained predictions
- **Prediction intervals** — outputting a confidence range rather than a point estimate

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

**Sukhjeet Singh**  
[GitHub](https://github.com/SinghSukhjeet01)
