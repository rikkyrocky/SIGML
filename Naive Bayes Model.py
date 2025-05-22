import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# Load datasets
dividend_df = pd.read_csv("DATA/new_dividend_companies_2019Q3_2024Q2.csv")
financials_df = pd.read_csv("DATA/Financials/us-income-quarterly-fixed.csv")

# Convert Fiscal Year and Period into "YYYY-QX" format
financials_df["Quarter"] = financials_df.apply(
    lambda row: f"{row['Fiscal Year']}-{row['Fiscal Period']}", axis=1
)

# Keep only necessary columns
financials_df = financials_df[[
    "Ticker", "Quarter", "Revenue", "Gross Profit", "Operating Income (Loss)",
    "Interest Expense, Net", "Net Income"
]]

# Helper to find 3 previous quarters
def get_previous_quarters(target_q, sorted_qs):
    if target_q not in sorted_qs:
        return []
    idx = sorted_qs.index(target_q)
    return sorted_qs[max(0, idx - 3):idx]

# All available quarters sorted
sorted_quarters = sorted(financials_df["Quarter"].unique())

# Container for results
results = []

# Loop through each company and calculate ratios
for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]

    if first_div_q == "2019-Q3":
        continue

    prev_qs = get_previous_quarters(first_div_q, sorted_quarters)
    if not prev_qs:
        continue

    df = financials_df[
        (financials_df["Ticker"] == ticker) &
        (financials_df["Quarter"].isin(prev_qs))
    ].copy()

    if df.empty:
        continue

    df.sort_values("Quarter", inplace=True)

    for _, r in df.iterrows():
        revenue = r["Revenue"]
        gross = r["Gross Profit"]
        op_inc = r["Operating Income (Loss)"]
        interest = r["Interest Expense, Net"]
        net = r["Net Income"]

        interest_coverage = op_inc / (-interest) if interest else np.nan

        results.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Gross Margin": gross / revenue if revenue else np.nan,
            "Operating Margin": op_inc / revenue if revenue else np.nan,
            "Net Margin": net / revenue if revenue else np.nan,
            "Interest Coverage Ratio": interest_coverage
        })

# Create DataFrame from results
metrics_df = pd.DataFrame(results)

# Amazon's Q1 2025 metrics
amazon_metrics = {
    "Gross Margin": 0.4916,
    "Operating Margin": 0.1102,
    "Net Margin": 0.1014,
    "Interest Coverage Ratio": 34.02
}

# Compute and print percentiles
print("Amazon Percentile Rankings Compared to Pre-Dividend Sample:")
for metric in amazon_metrics:
    sample_data = metrics_df[metric].dropna()
    percentile = percentileofscore(sample_data, amazon_metrics[metric], kind='rank')
    print(f"{metric}: {percentile:.2f} percentile")

# Plot setup
metrics = list(amazon_metrics.keys())
fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    data = metrics_df[metric].dropna()

    # IQR statistics
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    # Boxplot
    ax.boxplot(data, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='lightblue'))
    ax.plot(1, amazon_metrics[metric], 'ro', label='Amazon Q1 2025')

    # Evenly spaced annotation positions
    label_values = [
        (q1, f"Q1: {q1:.2f}", 'blue'),
        (median, f"Median: {median:.2f}", 'black'),
        (q3, f"Q3: {q3:.2f}", 'blue'),
        (amazon_metrics[metric], f"Amazon: {amazon_metrics[metric]:.2f}", 'red')
    ]
    text_y_positions = np.linspace(upper_whisker, lower_whisker, len(label_values)+2)[1:-1][::-1]

    for pos, (val, label, color) in zip(text_y_positions, label_values):
        ax.text(1.15, pos, label, verticalalignment='center', fontsize=9, color=color)

    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.set_ylim([lower_whisker, upper_whisker])
    ax.set_xticks([])

# Title and source with enhanced formatting
fig.suptitle("Fig 3: Amazon Q1 2025 vs Sample of Companies preparing to distribute dividends",
             fontsize=16, fontweight='bold')

fig.text(0.5, 0.01,
         "Data sourced from SimFin and NASDAQ, see referenced GitHub page",
         ha='center', fontsize=10, fontweight='bold', style='italic')

plt.legend(loc='upper left')
plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.show()
