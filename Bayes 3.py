import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# === Load Datasets ===
dividend_df = pd.read_csv("DATA/new_dividend_companies_2019Q3_2024Q2.csv")
financials_df = pd.read_csv("DATA/Financials/us-income-quarterly-fixed.csv")
balance_df = pd.read_csv("DATA/Financials/us-balance-quarterly.csv")

# === Quarter Construction ===
def convert_to_quarter(df):
    df["Quarter"] = df.apply(lambda row: f"{row['Fiscal Year']}-{row['Fiscal Period']}", axis=1)
    return df

financials_df = convert_to_quarter(financials_df)
balance_df = convert_to_quarter(balance_df)

# Sort quarters
sorted_income_quarters = sorted(financials_df["Quarter"].unique())
sorted_balance_quarters = sorted(balance_df["Quarter"].unique())

# === File 1: Profitability & Interest Coverage ===
income_metrics = []

for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]
    if first_div_q == "2019-Q3":
        continue
    idx = sorted_income_quarters.index(first_div_q) if first_div_q in sorted_income_quarters else -1
    prev_qs = sorted_income_quarters[max(0, idx - 3):idx] if idx > 0 else []
    df = financials_df[(financials_df["Ticker"] == ticker) & (financials_df["Quarter"].isin(prev_qs))]
    for _, r in df.iterrows():
        rev = r["Revenue"]
        gp = r["Gross Profit"]
        op = r["Operating Income (Loss)"]
        int_exp = r["Interest Expense, Net"]
        net = r["Net Income"]
        income_metrics.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Gross Margin": gp / rev if rev else np.nan,
            "Operating Margin": op / rev if rev else np.nan,
            "Net Margin": net / rev if rev else np.nan,
            "Interest Coverage Ratio": op / (-int_exp) if int_exp else np.nan
        })

income_df = pd.DataFrame(income_metrics)
amazon_income = {
    "Gross Margin": 0.4916,
    "Operating Margin": 0.1102,
    "Net Margin": 0.1014,
    "Interest Coverage Ratio": 34.02
}

# === File 2: Maturity Metrics ===
financials_df.sort_values(by=["Ticker", "Fiscal Year", "Fiscal Period"], inplace=True)
financials_df["Prior Revenue"] = financials_df.groupby("Ticker")["Revenue"].shift(4)
financials_df["Revenue Growth"] = (financials_df["Revenue"] - financials_df["Prior Revenue"]) / financials_df["Prior Revenue"]
financials_df["R&D Intensity"] = -financials_df["Research & Development"] / financials_df["Revenue"]
financials_df["SG&A / Revenue"] = -financials_df["Selling, General & Administrative"] / financials_df["Revenue"]

maturity_metrics = []
for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]
    if first_div_q == "2019-Q3":
        continue
    idx = sorted_income_quarters.index(first_div_q) if first_div_q in sorted_income_quarters else -1
    prev_qs = sorted_income_quarters[max(0, idx - 3):idx] if idx > 0 else []
    df = financials_df[(financials_df["Ticker"] == ticker) & (financials_df["Quarter"].isin(prev_qs))]
    for _, r in df.iterrows():
        maturity_metrics.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Revenue Growth": r["Revenue Growth"],
            "R&D Intensity": r["R&D Intensity"],
            "SG&A / Revenue": r["SG&A / Revenue"]
        })

maturity_df = pd.DataFrame(maturity_metrics)
amazon_maturity = {
    "Revenue Growth": 0.086,
    "R&D Intensity": 0.1478,
    "SG&A / Revenue": 0.0796
}

# === File 3: Financial Health ===
balance_df["Quick Ratio"] = (balance_df["Total Current Assets"] - balance_df["Inventories"]) / balance_df["Total Current Liabilities"]
balance_df["Debt to Equity"] = balance_df["Total Liabilities"] / balance_df["Total Equity"]
balance_df["Retention Ratio"] = balance_df["Retained Earnings"] / balance_df["Total Assets"]

health_metrics = []
for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]
    if first_div_q == "2019-Q3":
        continue
    idx = sorted_balance_quarters.index(first_div_q) if first_div_q in sorted_balance_quarters else -1
    prev_qs = sorted_balance_quarters[max(0, idx - 3):idx] if idx > 0 else []
    df = balance_df[(balance_df["Ticker"] == ticker) & (balance_df["Quarter"].isin(prev_qs))]
    for _, r in df.iterrows():
        health_metrics.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Quick Ratio": r["Quick Ratio"],
            "Debt to Equity": r["Debt to Equity"],
            "Retention Ratio": r["Retention Ratio"]
        })

health_df = pd.DataFrame(health_metrics)
amazon_health = {
    "Quick Ratio": 0.85,
    "Debt to Equity": 1.10,
    "Retention Ratio": 0.295
}

# === Plotting All Three Charts ===
combined = [
    ("Fig 3: Amazon Q1 2025 vs Sample of Companies preparing to distribute dividends", income_df, amazon_income),
    ("Fig 6: Amazon Q1 2025 vs Sample on Maturity-Related Metrics", maturity_df, amazon_maturity),
    ("Fig 7: Amazon Q1 2025 vs Sample on Financial Health Metrics", health_df, amazon_health)
]

for title, df, amazon_metrics in combined:
    metrics = list(amazon_metrics.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = df[metric].dropna()

        # Compute percentiles
        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        amazon_value = amazon_metrics[metric]
        max_visible = max(upper_whisker, amazon_value * 1.05)

        # Boxplot
        ax.boxplot(data, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='lightblue'))
        ax.plot(1, amazon_value, 'ro', label='Amazon Q1 2025')

        # Annotated labels
        label_values = [
            (q1, f"Q1: {q1:.2f}", 'blue'),
            (median, f"Median: {median:.2f}", 'black'),
            (q3, f"Q3: {q3:.2f}", 'blue'),
            (amazon_value, f"Amazon: {amazon_value:.2f}", 'red')
        ]
        text_y_positions = np.linspace(max_visible, lower_whisker, len(label_values)+2)[1:-1][::-1]

        for pos, (val, label, color) in zip(text_y_positions, label_values):
            ax.text(1.15, pos, label, verticalalignment='center', fontsize=9, color=color)

        ax.set_title(metric, fontweight='bold', fontsize=12)
        ax.set_ylim([lower_whisker, max_visible])
        ax.set_xticks([])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.text(0.5, 0.01,
             "Data sourced from SimFin and NASDAQ, see referenced GitHub page",
             ha='center', fontsize=10, fontweight='bold', style='italic')
    plt.legend(loc='upper left')
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()
