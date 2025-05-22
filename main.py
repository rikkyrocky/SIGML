import os
import time
import random
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict
from fake_useragent import UserAgent

# File paths
DATA_DIR = "DATA"
TICKER_FILE = os.path.join(DATA_DIR, "NASDAQ_stock_tickers.csv")
OUTPUT_FILE = "DATA/outputs/quarterly_dividends.csv"

# Load tickers
tickers = pd.read_csv(TICKER_FILE).iloc[:, 0].dropna().unique().tolist()

# Convert MM/DD/YYYY to YYYY-Qx
def date_to_quarter(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%m/%d/%Y")
        quarter = (date_obj.month - 1) // 3 + 1
        return f"{date_obj.year}-Q{quarter}"
    except:
        return None

# Setup user agent rotator
ua = UserAgent()

# Try to load existing CSV (to allow resuming)
if os.path.exists(OUTPUT_FILE):
    main_df = pd.read_csv(OUTPUT_FILE, index_col="Quarter")
else:
    main_df = pd.DataFrame()

processed = 0
skipped = 0

for idx, ticker in enumerate(tickers):
    if ticker in main_df.columns:
        print(f"[{idx+1}/{len(tickers)}] {ticker} already processed. Skipping.")
        continue

    print(f"\n[{idx+1}/{len(tickers)}] Ticker: {ticker}")
    url = f"https://api.nasdaq.com/api/quote/{ticker}/dividends?assetclass=stocks"
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": ua.random,
        "origin": "https://www.nasdaq.com",
        "referer": "https://www.nasdaq.com/"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        status = response.status_code
        print(f"  - Status Code: {status}")

        if status != 200:
            print(f"  - Skipping due to error status.")
            skipped += 1
            continue

        json_data = response.json()
        rows = json_data.get("data", {}).get("dividends", {}).get("rows", [])
        print(f"  - Raw dividend rows: {len(rows)}")

        ticker_quarters = defaultdict(float)
        valid_count = 0

        for row in rows:
            ex_date = row.get("exOrEffDate")
            amount = row.get("amount")
            if not ex_date or not amount:
                continue

            quarter = date_to_quarter(ex_date)
            if not quarter:
                continue

            try:
                value = float(amount.replace("$", ""))
            except:
                continue

            ticker_quarters[quarter] += value
            valid_count += 1

        if valid_count == 0:
            print("  - No valid dividend entries found.")
            skipped += 1
            continue

        # Merge ticker data into master DataFrame
        ticker_df = pd.DataFrame.from_dict(ticker_quarters, orient='index', columns=[ticker])
        ticker_df.index.name = "Quarter"

        # Merge with existing main_df
        main_df = main_df.combine_first(ticker_df)  # Ensures index union
        main_df.update(ticker_df)  # Updates just this ticker's column

        # Sort index (quarters)
        main_df.sort_index(inplace=True)

        # Save after each ticker
        main_df.to_csv(OUTPUT_FILE)
        print(f"  - Valid entries: {valid_count}")
        print(f"  - ✅ Progress saved to {OUTPUT_FILE}")

        processed += 1

    except Exception as e:
        print(f"  - Error occurred: {e}")
        skipped += 1
        continue

    wait = round(random.uniform(0.5, 1.5), 2)
    print(f"  - Waiting {wait}s before next request...")
    time.sleep(wait)

# Final Summary
print(f"\n✅ Finished processing.")
print(f"  - Tickers processed: {processed}")
print(f"  - Tickers skipped: {skipped}")
print(f"📁 Output saved to: {OUTPUT_FILE}")
