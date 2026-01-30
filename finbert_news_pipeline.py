import feedparser
import pandas as pd
from transformers import pipeline
from datetime import datetime
from urllib.parse import quote_plus


# ----------------------------
# CONFIG
# ----------------------------

STOCK_LIST = [
    "Infosys",
    "Reliance Industries",
    "HDFC Bank",
    "TCS",
    "ICICI Bank",
    "Nifty 50"
]

OUTPUT_FILE = "market_sentiment.csv"


# ----------------------------
# Load FinBERT
# ----------------------------

print("Loading FinBERT Model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# ----------------------------
# Fetch News from Google RSS
# ----------------------------

import time
import urllib.request

def fetch_news(stock):

    encoded_stock = quote_plus(stock)

    url = f"https://news.google.com/rss/search?q={encoded_stock}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        # Add browser header (prevents blocking)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            feed = feedparser.parse(response)

        headlines = []

        for entry in feed.entries[:8]:
            headlines.append(entry.title)

        return headlines

    except Exception as e:

        print(f"News fetch failed for {stock} — skipping")
        print("Error:", e)

        return []



# ----------------------------
# Sentiment Scoring Logic
# ----------------------------

def sentiment_score(label):
    if label == "positive":
        return 1
    elif label == "negative":
        return -1
    else:
        return 0

# ----------------------------
# MAIN PIPELINE
# ----------------------------

final_data = []

print("Fetching News + Running Sentiment Engine...")

for stock in STOCK_LIST:

    time.sleep(2)

    headlines = fetch_news(stock)

    for news in headlines:

        try:
            result = sentiment_model(news)[0]

            label = result["label"].lower()
            confidence = round(result["score"], 3)
            score = sentiment_score(label)

            final_data.append([
                stock,
                news,
                label.upper(),
                score,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        except:
            continue

# ----------------------------
# Save Output
# ----------------------------

df = pd.DataFrame(final_data, columns=[
    "Stock",
    "Headline",
    "Sentiment",
    "Score",
    "Confidence",
    "Timestamp"
])

df.to_csv(OUTPUT_FILE, index=False)

print("DONE. Sentiment File Updated Successfully.")

