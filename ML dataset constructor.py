import pandas as pd
import numpy as np
import random
import os

# === Load data ===
dividend_df = pd.read_csv("DATA/new_dividend_companies_2019Q3_2024Q2.csv")
financials_df = pd.read_csv("DATA/Financials/us-income-quarterly-fixed.csv")
balance_df = pd.read_csv("DATA/Financials/us-balance-quarterly.csv")

# === Construct 'Quarter' ===
def convert_to_quarter(df):
    df["Quarter"] = df.apply(lambda row: f"{row['Fiscal Year']}-{row['Fiscal Period']}", axis=1)
    return df

financials_df = convert_to_quarter(financials_df)
balance_df = convert_to_quarter(balance_df)

# === Precompute ratios ===
financials_df.sort_values(by=["Ticker", "Fiscal Year", "Fiscal Period"], inplace=True)
financials_df["Prior Revenue"] = financials_df.groupby("Ticker")["Revenue"].shift(4)
financials_df["Revenue Growth"] = (financials_df["Revenue"] - financials_df["Prior Revenue"]) / financials_df["Prior Revenue"]
financials_df["R&D Intensity"] = -financials_df["Research & Development"] / financials_df["Revenue"]
financials_df["SG&A / Revenue"] = -financials_df["Selling, General & Administrative"] / financials_df["Revenue"]

sorted_income_quarters = sorted(financials_df["Quarter"].unique())
sorted_balance_quarters = sorted(balance_df["Quarter"].unique())

# === Dividend initiators ===
label_1_rows = []

for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]
    if first_div_q == "2019-Q3" or first_div_q not in sorted_income_quarters or first_div_q not in sorted_balance_quarters:
        continue

    idx_income = sorted_income_quarters.index(first_div_q)
    idx_balance = sorted_balance_quarters.index(first_div_q)
    prev_qs_income = sorted_income_quarters[max(0, idx_income - 3):idx_income]
    prev_qs_balance = sorted_balance_quarters[max(0, idx_balance - 3):idx_balance]

    fin_df = financials_df[(financials_df["Ticker"] == ticker) & (financials_df["Quarter"].isin(prev_qs_income))]
    bal_df = balance_df[(balance_df["Ticker"] == ticker) & (balance_df["Quarter"].isin(prev_qs_balance))]
    merged_df = pd.merge(fin_df, bal_df, on=["Ticker", "Quarter"], how="inner")

    for _, r in merged_df.iterrows():
        label_1_rows.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Gross Margin": r["Gross Profit"] / r["Revenue"] if r["Revenue"] else np.nan,
            "Operating Margin": r["Operating Income (Loss)"] / r["Revenue"] if r["Revenue"] else np.nan,
            "Net Margin": r["Net Income"] / r["Revenue"] if r["Revenue"] else np.nan,
            "Interest Coverage Ratio": r["Operating Income (Loss)"] / (-r["Interest Expense, Net"]) if r["Interest Expense, Net"] else np.nan,
            "Revenue Growth": r["Revenue Growth"],
            "R&D Intensity": r["R&D Intensity"],
            "SG&A / Revenue": r["SG&A / Revenue"],
            "Quick Ratio": (r["Total Current Assets"] - r["Inventories"]) / r["Total Current Liabilities"] if r["Total Current Liabilities"] else np.nan,
            "Debt to Equity": r["Total Liabilities"] / r["Total Equity"] if r["Total Equity"] else np.nan,
            "Retention Ratio": r["Retained Earnings"] / r["Total Assets"] if r["Total Assets"] else np.nan,
            "Class": 1
        })

# === Non-dividend tickers ===
non_div_tickers = [
    "AABA", "ADBE", "ADSK", "AKAM", "ALXN", "AMD", "AMZN", "AN", "AZO", "BIIB", "BRK.B", "BSX", "CBG", "CELG", "CERN",
    "CHK", "CHTR", "CMG", "CNC", "CRM", "CTXS", "CXO", "DISCA", "DISCK", "DISH", "DLTR", "DVA", "EA", "EBAY", "ESRX",
    "ETFC", "EVHC", "EW", "FB", "FCX", "FFIV", "FISV", "FTI", "GOOG", "GOOGL", "HCA", "HOLX", "HSIC", "IDXX", "ILMN",
    "INCY", "ISRG", "IT", "KMX", "KORS", "LH", "LKQ", "LVLT", "MHK", "MNK", "MNST", "MTD", "MU", "MYL", "NFLX", "NFX",
    "ORLY", "PCLN", "PWR", "PYPL", "QRVO", "REGN", "RHT", "RIG", "SNPS", "SRCL", "TDC", "TDG", "TRIP", "UA", "UAA", "UAL",
    "ULTA", "URI", "VAR", "VRSK", "VRSN", "VRTX", "WAT",
    "DASH", "CPRT", "MSTR", "EW", "AXON", "COIN", "NET", "IDXX", "VEEV", "ALNY", "FTNT", "XYZ", "RBLX", "TTWO", "HOOD", "TEAM",
    "AZO", "CSGP", "CVNA", "DDOG", "HUBS", "LULU", "ANSS", "DXCM", "GDDY", "IQV", "TTD", "SNOW", "TYL", "WAT", "OKTA", "ULTA",
    "MOH", "PINS", "BURL", "MDB", "MKL", "NVR", "PODD", "SNAP", "ON", "TRU"
]



label_0_rows = []

for ticker in non_div_tickers:
    available_qs = set(financials_df[financials_df["Ticker"] == ticker]["Quarter"]).intersection(
                   set(balance_df[balance_df["Ticker"] == ticker]["Quarter"]))
    if not available_qs:
        continue
    selected_q = random.choice(list(available_qs))
    fin = financials_df[(financials_df["Ticker"] == ticker) & (financials_df["Quarter"] == selected_q)]
    bal = balance_df[(balance_df["Ticker"] == ticker) & (balance_df["Quarter"] == selected_q)]

    if fin.empty or bal.empty:
        continue

    r1 = fin.iloc[0]
    r2 = bal.iloc[0]
    label_0_rows.append({
        "Ticker": ticker,
        "Quarter": selected_q,
        "Gross Margin": r1["Gross Profit"] / r1["Revenue"] if r1["Revenue"] else np.nan,
        "Operating Margin": r1["Operating Income (Loss)"] / r1["Revenue"] if r1["Revenue"] else np.nan,
        "Net Margin": r1["Net Income"] / r1["Revenue"] if r1["Revenue"] else np.nan,
        "Interest Coverage Ratio": r1["Operating Income (Loss)"] / (-r1["Interest Expense, Net"]) if r1["Interest Expense, Net"] else np.nan,
        "Revenue Growth": r1["Revenue Growth"],
        "R&D Intensity": r1["R&D Intensity"],
        "SG&A / Revenue": r1["SG&A / Revenue"],
        "Quick Ratio": (r2["Total Current Assets"] - r2["Inventories"]) / r2["Total Current Liabilities"] if r2["Total Current Liabilities"] else np.nan,
        "Debt to Equity": r2["Total Liabilities"] / r2["Total Equity"] if r2["Total Equity"] else np.nan,
        "Retention Ratio": r2["Retained Earnings"] / r2["Total Assets"] if r2["Total Assets"] else np.nan,
        "Class": 0
    })

# === Combine and Save ===
all_data = pd.DataFrame(label_1_rows + label_0_rows)
all_data.to_csv("dividend_classification_dataset.csv", index=False)
print("Saved as dividend_classification_dataset.csv")
