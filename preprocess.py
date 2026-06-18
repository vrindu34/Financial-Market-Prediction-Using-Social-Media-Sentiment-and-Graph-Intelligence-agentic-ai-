import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# ==========================
# HELPER FUNCTION
# ==========================

def find_column(df, possible_names):
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    return None


# ==========================
# MARKET DATA
# ==========================

print("Processing market data...")

market = pd.read_csv("nifty_500.csv")

# Normalize column names
market.columns = market.columns.str.strip().str.lower()

# Detect columns dynamically
symbol_col = find_column(market, ["symbol", "company", "stock"])
close_col = find_column(market, ["close", "price", "closing"])
high_col = find_column(market, ["high"])
low_col = find_column(market, ["low"])
volume_col = find_column(market, ["volume"])

# Compute features if possible
if close_col:
    market["trend"] = market.groupby(symbol_col)[close_col].diff()
else:
    market["trend"] = 0

if high_col and low_col:
    market["volatility"] = market[high_col] - market[low_col]
else:
    market["volatility"] = 0

if volume_col:
    market["volume_change"] = market.groupby(symbol_col)[volume_col].diff()
else:
    market["volume_change"] = 0

market.to_csv("market_processed.csv", index=False)

print("✅ Market processed")

# ==========================
# GENERIC SENTIMENT FUNCTION
# ==========================

def process_text_dataset(file, possible_text_cols, output_name):

    print(f"Processing {file} in chunks...")

    chunks = pd.read_csv(file, chunksize=5000, low_memory=False)

    processed_chunks = []

    for i, chunk in enumerate(chunks):

        print(f"Processing chunk {i+1}...")

        chunk.columns = chunk.columns.str.strip().str.lower()

        text_col = None
        for col in chunk.columns:
            if col.lower() in possible_text_cols:
                text_col = col
                break

        if text_col:
            chunk["score"] = chunk[text_col].astype(str).apply(
                lambda x: sia.polarity_scores(x)["compound"]
            )
        else:
            chunk["score"] = 0

        processed_chunks.append(chunk)

    final_df = pd.concat(processed_chunks)

    final_df.to_csv(output_name, index=False)

    print(f"✅ {file} processed successfully")


# ==========================
# TWITTER / REDDIT / NEWS
# ==========================

process_text_dataset(
    "full_dataset-release.csv",
    ["text", "tweet", "content"],
    "twitter_processed.csv"
)

process_text_dataset(
    "merged_eval_data_50.csv",
    ["text", "comment", "body"],
    "reddit_processed.csv"
)



print("🎯 ALL DATA PROCESSED SUCCESSFULLY")