
#  Amazon Dividend Initiation Analysis

This repository contains a comprehensive analysis framework that investigates the likelihood of **Amazon initiating dividends**. The project combines financial metric visualizations, data extraction pipelines, and machine learning models to evaluate Amazon's readiness for dividend distribution based on historical patterns.

---

## File Overview

### 1. **`Charts.py`**
**Purpose**: Visualizes Amazon’s financial trends such as Free Cash Flow, Revenue, EBIT, Capex, and R&D spending.
- **Key Plots**:
  - Free Cash Flow with rolling mean & volatility
  - TTM Revenue, EBIT, and EBIT margin
  - Segment-wise operating margins
  - Generative AI market projections
  - Capex & R&D growth with CAGR trend lines

---

### 2. **`Financial_metrics_extraction.py`**
**Purpose**: Constructs comparative box plots for Amazon vs. dividend-initiating firms across:
- Profitability and Interest Coverage (`Fig 3`)
- Maturity Metrics like Revenue Growth & R&D Intensity (`Fig 6`)
- Financial Health Metrics like Debt-to-Equity & Quick Ratio (`Fig 7`)
- **Inputs**:  
  - `new_dividend_companies_2019Q3_2024Q2.csv`
  - Quarterly financials from SimFin

---

### 3. **`Dividend scraper.py`**
**Purpose**: Scrapes dividend payment data for NASDAQ-listed companies using Nasdaq’s public API.
- Saves aggregated quarterly dividend values to:
  - `DATA/outputs/quarterly_dividends.csv`
- Uses randomized user agents for ethical scraping
- Can resume from previous runs

---

### 4. **`Logistic regression.py`**
**Purpose**: Applies a **Logistic Regression** model with LOOCV to predict dividend initiation based on financial metrics.
- Includes:
  - Feature selection via ANOVA F-stat
  - Confusion matrix and ROC curve
  - Prediction and probability for Amazon
- **Output**: Predicts Amazon's likelihood of initiating dividends (Q1 2025 snapshot)

---

### 5. **`randomforest.py`**
**Purpose**: Implements a **Random Forest Classifier** with 2-instance-per-fold CV.
- Evaluation includes:
  - Confusion matrix, ROC curve, probability distribution
- Outputs Amazon’s prediction using the trained model
- **Complementary to `logistic_regression_model.py` for model comparison**
---

### 6. **`ML dataset constructor.py`**
**Purpose**: Constructs the dataset for the data with the correct featureset for the ML models.
- Data inputs include:
  - `DATA/Financials/us-balance-quarterly.csv`
  - `DATA/Financials/us-income-quarterly-fixed.csv`
- Outputs Dataset for ML models
- `DATA/outputs/dividend_classification_dataset.csv` 
---

## Directory Structure

```
.
├── amazon_visuals.py
├── financial_metrics_extraction.py
├── nasdaq_dividend_scraper.py
├── logistic_regression_model.py
├── random_forest_model.py
├── DATA/
│   ├── NASDAQ_stock_tickers.csv
│   ├── new_dividend_companies_2019Q3_2024Q2.csv
│   ├── Financials/
│   │   ├── us-income-quarterly-fixed.csv
│   │   └── us-balance-quarterly.csv
│   └── outputs/
│       ├── quarterly_dividends.csv
│       └── dividend_classification_dataset.csv
```

---

## Features & Model Inputs

The models rely on the following financial indicators:

| Feature | Description |
|--------|-------------|
| Gross Margin | Profitability before operating expenses |
| Operating Margin | Core operating profitability |
| Net Margin | Bottom-line profitability |
| Interest Coverage | Ability to service interest expense |
| Revenue Growth | TTM revenue YoY growth |
| R&D Intensity | R&D as % of revenue |
| SG&A / Revenue | Operating overhead |
| Quick Ratio | Liquidity measure (excluding inventory) |
| Debt to Equity | Leverage metric |
| Retention Ratio | Retained earnings as % of assets |

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/amazon-dividend-forecast.git
cd amazon-dividend-forecast
```

2. **Install Dependencies**
```txt
pandas
numpy
matplotlib
scikit-learn
requests
fake-useragent
```

3. **Run the scripts**
```bash
python amazon_visuals.py
python financial_metrics_extraction.py
python logistic_regression_model.py
python random_forest_model.py
```

> Ensure `DATA/` folder is structured correctly with financials from SimFin and Nasdaq tickers.

---

## Data Sources
- **SimFin**: Quarterly financials for US equities
- **Nasdaq.com**: Dividend histories
- **Amazon 10-Q Filings**: Segment performance and R&D/Capex data
- **Bloomberg**: Generative AI market projections

---
