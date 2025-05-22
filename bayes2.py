import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# Load datasets
dividend_df = pd.read_csv("DATA/new_dividend_companies_2019Q3_2024Q2.csv")
financials_df = pd.read_csv("DATA/Financials/us-income-quarterly-fixed.csv")

# Add Quarter column
financials_df["Quarter"] = financials_df.apply(
    lambda row: f"{row['Fiscal Year']}-{row['Fiscal Period']}", axis=1
)

# Sort for lag calculations
financials_df.sort_values(by=["Ticker", "Fiscal Year", "Fiscal Period"], inplace=True)
financials_df["Prior Revenue"] = financials_df.groupby("Ticker")["Revenue"].shift(4)

# Fix signs for SG&A and R&D before computing metrics
financials_df["Revenue Growth"] = (financials_df["Revenue"] - financials_df["Prior Revenue"]) / financials_df["Prior Revenue"]
financials_df["R&D Intensity"] = -financials_df["Research & Development"] / financials_df["Revenue"]
financials_df["SG&A / Revenue"] = -financials_df["Selling, General & Administrative"] / financials_df["Revenue"]

# Store final results
results = []
sorted_quarters = sorted(financials_df["Quarter"].unique())

# Get trailing 3 quarters of maturity metrics
for _, row in dividend_df.iterrows():
    ticker = row["Ticker"]
    first_div_q = row["First Dividend Quarter"]
    if first_div_q == "2019-Q3":
        continue

    def get_prev_qtrs(q):
        if q not in sorted_quarters:
            return []
        i = sorted_quarters.index(q)
        return sorted_quarters[max(0, i - 3):i]

    prev_qs = get_prev_qtrs(first_div_q)
    if not prev_qs:
        continue

    df = financials_df[
        (financials_df["Ticker"] == ticker) &
        (financials_df["Quarter"].isin(prev_qs))
    ].copy()

    if df.empty:
        continue

    for _, r in df.iterrows():
        results.append({
            "Ticker": ticker,
            "Quarter": r["Quarter"],
            "Revenue Growth": r["Revenue Growth"],
            "R&D Intensity": r["R&D Intensity"],
            "SG&A / Revenue": r["SG&A / Revenue"]
        })

# Final DataFrame
metrics_df = pd.DataFrame(results)


# Amazon's Q1 2025 maturity metrics
amazon_metrics = {
    "Revenue Growth": 0.086,
    "R&D Intensity": 0.1478,
    "SG&A / Revenue": 0.0796
}
print("\nAmazon Percentile Rankings Compared to Pre-Dividend Sample:")
for metric in amazon_metrics:
    sample_data = metrics_df[metric].dropna()
    percentile = percentileofscore(sample_data, amazon_metrics[metric], kind='rank')
    print(f"{metric}: {percentile:.2f} percentile")

# Plotting
metrics = list(amazon_metrics.keys())
fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    data = metrics_df[metric].dropna()

    # IQR stats
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    # Ensure red dot is visible
    amazon_value = amazon_metrics[metric]
    max_visible = max(upper_whisker, amazon_value * 1.05)

    # Draw boxplot
    ax.boxplot(data, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='lightblue'))
    ax.plot(1, amazon_value, 'ro', label='Amazon Q1 2025')

    # Evenly spaced annotations
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

# Title and source line
fig.suptitle("Fig 6: Amazon Q1 2025 vs Sample on Maturity-Related Metrics", fontsize=16, fontweight='bold')
fig.text(0.5, 0.01, "Data sourced from SimFin and NASDAQ, see referenced GitHub page",
         ha='center', fontsize=10, fontweight='bold', style='italic')

plt.legend(loc='upper left')
plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.show()
