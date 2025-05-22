import pandas as pd
import matplotlib.pyplot as plt

# Data setup
data = {
    "Date": [
        "2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31",
        "2022-12-31", "2022-09-30", "2022-06-30", "2022-03-31", "2021-12-31", "2021-09-30", "2021-06-30", "2021-03-31", "2020-12-31",
        "2020-09-30", "2020-06-30", "2020-03-31", "2019-12-31", "2019-09-30", "2019-06-30", "2019-03-31", "2018-12-31", "2018-09-30",
        "2018-06-30", "2018-03-31", "2017-12-31", "2017-09-30", "2017-06-30", "2017-03-31", "2016-12-31", "2016-09-30", "2016-06-30",
        "2016-03-31", "2015-12-31", "2015-09-30", "2015-06-30", "2015-03-31", "2014-12-31", "2014-09-30", "2014-06-30", "2014-03-31",
        "2013-12-31", "2013-09-30", "2013-06-30", "2013-03-31", "2012-12-31", "2012-09-30", "2012-06-30", "2012-03-31", "2011-12-31",
        "2011-09-30", "2011-06-30", "2011-03-31", "2010-12-31", "2010-09-30", "2010-06-30", "2010-03-31", "2009-12-31", "2009-09-30",
        "2009-06-30", "2009-03-31"
    ],
    "Free_Cash_Flow": [
        -7240, 38219, 18635, 13942, 5054, 36813, 7701, -2218, -8282, -11569, -25302, -21665, -16532, -9069, -14685,
        -7247, -6974, 31020, 13783, 11627, -2364, 25825, 10106, 5599, -875, 19400, 6043, -18, -4518, 8307, -1015,
        -2132, -3480, 10466, 1553, -1265, -3132, 7450, -172, -1586, -2370, 1949, -3621, -4010, -3582, 2031, -2668,
        -3018, -3042, 395, -2660, -2887, -2824, 2092, -1627, -1894, -1884, 2516, -644, -1184, -1238, 2920, 446, -250, -640
    ]
}

# Create and process the DataFrame
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df["Rolling_Mean"] = df["Free_Cash_Flow"].rolling(window=4).mean()
df["Rolling_STD"] = df["Free_Cash_Flow"].rolling(window=4).std()

# Drop NA rows caused by rolling window
df_clean = df.dropna(subset=["Rolling_Mean", "Rolling_STD"])

# Extract necessary series as numeric arrays
dates = df_clean["Date"].values
fcf = df_clean["Free_Cash_Flow"].astype(float).values
rolling_mean = df_clean["Rolling_Mean"].astype(float).values
rolling_std = df_clean["Rolling_STD"].astype(float).values
upper_band = rolling_mean + rolling_std
lower_band = rolling_mean - rolling_std

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(dates, fcf, label="Free Cash Flow", color="blue")
plt.plot(dates, rolling_mean, label="4-Quarter Rolling Mean", color="orange")
plt.fill_between(dates, lower_band, upper_band, color="orange", alpha=0.2, label="±1 Std Dev")

plt.title("Amazon Free Cash Flow with 4-Quarter Rolling Mean and Volatility")
plt.xlabel("Date")
plt.ylabel("Free Cash Flow (in millions USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()




data = {
    "Date": [
        "2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
        "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31", "2022-12-31",
        "2022-09-30", "2022-06-30", "2022-03-31", "2021-12-31", "2021-09-30",
        "2021-06-30", "2021-03-31", "2020-12-31", "2020-09-30", "2020-06-30",
        "2020-03-31"
    ],
    "TTM Revenue": [
        650.31, 637.96, 620.13, 604.33, 590.74, 574.79, 554.03, 538.05, 524.90,
        513.98, 502.19, 485.90, 477.75, 469.82, 457.97, 443.30, 419.13,
        386.06, 347.95, 321.78, 296.27
    ],
    "TTM EBIT": [
        71.69, 68.59, 60.60, 54.38, 47.39, 36.85, 26.38, 17.72, 13.35, 12.25,
        12.97, 15.30, 19.68, 24.88, 28.29, 29.63, 27.78, 22.90, 19.91, 16.87, 14.11
    ],
    "EBIT Margin": [
        11.02, 10.75, 9.77, 9.00, 8.02, 6.41, 4.76, 3.29, 2.54, 2.38,
        2.58, 3.15, 4.12, 5.30, 6.18, 6.68, 6.63, 5.93, 5.72, 5.24, 4.76
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# Filter recent data only
df_recent = df[df["Date"] >= pd.Timestamp("2020-01-01")]
annotations_recent = df_recent[df_recent["Date"].isin([
    pd.Timestamp("2022-12-31"), pd.Timestamp("2025-03-31")
])]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 10))

# Bar for TTM Revenue (primary y-axis)
ax1.bar(df_recent["Date"], df_recent["TTM Revenue"], label="TTM Revenue",
        color="#1f77b4", width=50, alpha=0.8)
ax1.set_ylabel("TTM Revenue (in Billions USD)", color="#1f77b4", fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor="#1f77b4", labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Bar and line for EBIT and EBIT Margin (secondary y-axis)
ax2 = ax1.twinx()
ax2.bar(df_recent["Date"], df_recent["TTM EBIT"], label="TTM EBIT",
        color="#ff7f0e", width=35, alpha=0.85)

# Exaggerate EBIT Margin for visual clarity
scaled_margin = df_recent["EBIT Margin"] * 5
ax2.plot(df_recent["Date"], scaled_margin, color="black",
         label="EBIT Margin (scaled)", linewidth=3, marker="o")
ax2.set_ylabel("TTM EBIT (in Billions USD) & EBIT Margin (%)", color="black", fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelcolor="black", labelsize=12)

# Annotate margin points
for _, row in annotations_recent.iterrows():
    ax2.annotate(f'{row["EBIT Margin"]:.2f}%',
                 xy=(row["Date"], row["EBIT Margin"] * 5),
                 xytext=(row["Date"], row["EBIT Margin"] * 5 + 5),
                 textcoords="data",
                 ha='center',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
                 arrowprops=dict(arrowstyle="->", color='gray'))

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

# Title and formatting
ax1.set_title("Amazon TTM Revenue, EBIT, and EBIT Margin", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Creating the data
years = ['2022', '2023', '2024', '2025']
segments = ['North America', 'International', 'AWS']

# Operating income and net sales for each segment
data = {
    'North America': {
        'Net Sales': [69244, 76881, 86341, 92887],
        'Operating Income': [-1568, 898, 4983, 5841]
    },
    'International': {
        'Net Sales': [28759, 29123, 31935, 33513],
        'Operating Income': [-1281, -1247, 903, 1017]
    },
    'AWS': {
        'Net Sales': [18441, 21354, 25037, 29267],
        'Operating Income': [6518, 5123, 9421, 11547]
    }
}

# Calculate operating margins
margins = {segment: [
    (data[segment]['Operating Income'][i] / data[segment]['Net Sales'][i]) * 100
    for i in range(4)
] for segment in segments}

# Create a DataFrame for plotting
df_margins = pd.DataFrame(margins, index=years)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = range(len(years))

# Plot each segment with offset
for i, segment in enumerate(segments):
    ax.bar([p + i * bar_width for p in x], df_margins[segment], width=bar_width, label=segment)

# Labeling
ax.set_xlabel("Year", fontsize=12, fontweight='bold')
ax.set_ylabel("Operating Margin (%)", fontsize=12, fontweight='bold')
ax.set_title("Operating Margin by Segment (Q1 2022–2025)", fontsize=14, fontweight='bold')
ax.set_xticks([p + bar_width for p in x])
ax.set_xticklabels(years, fontsize=11)
ax.legend()
ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()



# Data
data = {
    "Year": list(range(2020, 2033)),
    "Generative AI Revenue ($B)": [14, 23, 40, 67, 137, 217, 304, 399, 548, 728, 897, 1079, 1304],
    "Generative AI as % of Tech Spend": [0, 0, 0, 0, 3, 4, 5, 6, 7, 9, 10, 11, 12]
}
df = pd.DataFrame(data)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 7))

# Bar chart for Generative AI Revenue
bars = ax1.bar(df["Year"], df["Generative AI Revenue ($B)"], color="#4db8ff", label="Generative AI Revenue")
ax1.set_xlabel("Year", fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel("Revenue ($B)", fontsize=14, fontweight='bold', color='black')
ax1.tick_params(axis='x', labelsize=12, colors='black')
ax1.tick_params(axis='y', labelsize=12, colors='black')

# Add bar labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height + 30, f"${int(height)}B",
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Line chart for % of Tech Spend
ax2 = ax1.twinx()
ax2.plot(df["Year"], df["Generative AI as % of Tech Spend"], color="orange", marker='o',
         label="Generative AI as % of Total Technology Spend", linewidth=2)
ax2.set_ylabel("% of Total Technology Spend", fontsize=14, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelsize=12, colors='black')
ax2.set_ylim(0, 36)

# Label the final data point
final_year = df["Year"].iloc[-1]
final_pct = df["Generative AI as % of Tech Spend"].iloc[-1]
ax2.text(final_year, final_pct + 1, f"{final_pct}%", color="orange", fontsize=12, fontweight='bold', ha='center')

# Title and legends
fig.suptitle("Fig 4: Generative AI Revenue and Share of Total Technology Spend",
             fontsize=16, fontweight='bold', color='black')
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

plt.grid(False)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# R&D data
rd_data = {
    "2025-03-31": 22994, "2024-12-31": 23571, "2024-09-30": 22245, "2024-06-30": 22304,
    "2024-03-31": 20424, "2023-12-31": 22038, "2023-09-30": 21203, "2023-06-30": 21931,
    "2023-03-31": 20450, "2022-12-31": 20814, "2022-09-30": 19485, "2022-06-30": 18072,
    "2022-03-31": 14842, "2021-12-31": 15313, "2021-09-30": 14380, "2021-06-30": 13871,
    "2021-03-31": 12488, "2020-12-31": 12051, "2020-09-30": 10976, "2020-06-30": 10388,
    "2020-03-31": 9325, "2019-12-31": 9739, "2019-09-30": 9200, "2019-06-30": 9065,
    "2019-03-31": 7927, "2018-12-31": 7669, "2018-09-30": 7162, "2018-06-30": 7247,
    "2018-03-31": 6759
}

# Capex data
capex_data = {
    "2025-03-31": 29803, "2024-12-31": 37443, "2024-09-30": 16899, "2024-06-30": 22138,
    "2024-03-31": 17862, "2023-12-31": 12601, "2023-09-30": 11753, "2023-06-30": 9673,
    "2023-03-31": 15806, "2022-12-31": 10821, "2022-09-30": 15608, "2022-06-30": 12078,
    "2022-03-31": 906, "2021-12-31": 12580, "2021-09-30": 14828, "2021-06-30": 22080,
    "2021-03-31": 8666, "2020-12-31": 17037, "2020-09-30": 15876, "2020-06-30": 17804,
    "2020-03-31": 8894, "2019-12-31": 3536, "2019-09-30": 5073, "2019-06-30": 7549,
    "2019-03-31": 8123, "2018-12-31": 3572, "2018-09-30": 5572, "2018-06-30": 2692,
    "2018-03-31": 533
}

# Combine into one DataFrame
df_combined = pd.DataFrame({
    "Date": pd.to_datetime(list(rd_data.keys())),
    "R&D": list(rd_data.values()),
    "Capex": [capex_data.get(k, None) for k in rd_data.keys()]
})
df_combined.sort_values("Date", inplace=True)

# Calculate CAGR from Q1 2019 to Q1 2025
start_date = pd.Timestamp("2019-03-31")
end_date = pd.Timestamp("2025-03-31")
years = (end_date - start_date).days / 365.25

rd_start = df_combined[df_combined["Date"] == start_date]["R&D"].values[0]
rd_end = df_combined[df_combined["Date"] == end_date]["R&D"].values[0]
cagr_rd = ((rd_end / rd_start) ** (1 / years)) - 1

capex_start = df_combined[df_combined["Date"] == start_date]["Capex"].values[0]
capex_end = df_combined[df_combined["Date"] == end_date]["Capex"].values[0]
cagr_capex = ((capex_end / capex_start) ** (1 / years)) - 1

# Generate CAGR trend lines
df_combined["R&D CAGR Trend"] = rd_start * ((1 + cagr_rd) ** ((df_combined["Date"] - start_date).dt.days / 365.25))
df_combined["Capex CAGR Trend"] = capex_start * ((1 + cagr_capex) ** ((df_combined["Date"] - start_date).dt.days / 365.25))

# Plot
plt.figure(figsize=(12, 6))
bar_width = 20
dates = df_combined["Date"]

# Dual bar chart
plt.bar(dates - pd.Timedelta(days=10), df_combined["R&D"], width=bar_width, color="#e75480", label="R&D Expense")
plt.bar(dates + pd.Timedelta(days=10), df_combined["Capex"], width=bar_width, color="#4682b4", label="Capex")

# CAGR trend lines
plt.plot(dates, df_combined["R&D CAGR Trend"], color="darkred", linewidth=2.5, linestyle="--", label=f"R&D CAGR ({cagr_rd:.1%})")
plt.plot(dates, df_combined["Capex CAGR Trend"], color="black", linewidth=2.5, linestyle="--", label=f"Capex CAGR ({cagr_capex:.1%})")

# Chart formatting
plt.title("Fig 5: Amazon R&D Expense and Capex (2018–2025)", fontsize=16, fontweight='bold')
plt.xlabel("Quarter", fontsize=12, fontweight='bold')
plt.ylabel("Expenditure (in Millions USD)", fontsize=12, fontweight='bold')
plt.xticks(rotation=45, fontsize=10)
plt.legend(fontsize=11)
plt.text(dates.min(), -5000, "Source: Amazon 10-Q Filings", fontsize=9, style='italic', ha='left')

plt.tight_layout()
plt.grid(True, axis='y')
plt.show()


