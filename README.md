# Agentic AI Financial Prediction System

A 3-agent pipeline that generates BUY/SELL/HOLD signals for stocks by combining 
live market data with social media sentiment analysis.

---

## What It Does

You enter any stock ticker. The system pulls live price data from Yahoo Finance, 
scores sentiment from a Reddit and Twitter corpus, runs both through a graph-based 
fusion layer, and outputs a trading signal with a calibrated confidence score.

---

## How It Works

Three agents run in sequence:

**Market Agent** — fetches live OHLCV data via yfinance and computes three 
normalized signals: price momentum (% change over the analysis window), RSI 
(relative strength index), and OLS trend slope. Each is scaled to −1..+1 before 
fusion.

**Sentiment Agent** — scores Reddit and Twitter data using VADER (Valence Aware 
Dictionary and sEntiment Reasoner). Filters by stock name where possible, falls 
back to corpus-wide average. Output is tanh-amplified to −1..+1.

**Graph Agent** — builds a small influence graph (stock → Market, stock → 
Sentiment, Market → Macro, Sentiment → News) using NetworkX. Computes degree 
centrality and an agreement bonus based on whether market and sentiment signals 
point in the same direction.

The three scores are fused via a **volatility-adaptive weighting scheme**:
- High volatility (>2% daily range): Market 55%, Sentiment 30%, Graph 15%
- Low volatility: Market 40%, Sentiment 45%, Graph 15%

Confidence is mapped through a sigmoid curve rather than raw |score| × 100 — 
because composite scores in real markets rarely exceed ±0.35, so linear scaling 
would cap confidence at 35% even for strong signals.

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Market data | yfinance |
| Sentiment | NLTK VADER |
| Graph | NetworkX |
| Numerics | NumPy, pandas |
| Visualization | Matplotlib |

---

## Project Structure
├── appp.py                    # Main Streamlit app — all agent logic lives here

├── preprocess.py              # Offline preprocessing pipeline (VADER scoring)

├── requirements.txt           # Python dependencies

├── reddit_processed.csv       # Reddit corpus with pre-scored sentiment

├── reddit_sentiment_cache.csv # Cached VADER scores for Reddit data

├── market_processed.csv       # Processed NIFTY 500 market data

├── merged_eval_data_50.csv    # Holdout evaluation set (50 stocks)

├── nifty_500.csv              # NIFTY 500 stock universe

└── .gitignore

---

## Data Notes

The Twitter sentiment cache (245MB) is hosted on Hugging Face and downloaded 
automatically at runtime — no setup needed:

→ [vrindu34/agentic-ai-sentiment-data](https://huggingface.co/datasets/vrindu34/agentic-ai-sentiment-data)

The raw Twitter dataset (`twitter_processed.csv`, 234MB) is not included — it 
was used only during preprocessing to generate the cache. If you want to 
regenerate the cache from scratch, run `preprocess.py` on the original dataset.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run appp.py
```

On first run the Twitter cache downloads automatically from Hugging Face. 
Reddit data is already in the repo. After the first run both are cached to disk 
so subsequent runs are instant.

---

## Evaluation

Evaluated on a 50-stock NIFTY 500 holdout set (`merged_eval_data_50.csv`).  
Composite score +0.26, average signal confidence 74%.

The system intentionally outputs HOLD for ambiguous signals — a BUY or SELL is 
only triggered when market momentum, sentiment, and graph agreement all point 
in the same direction simultaneously.

---

## Research

This system was developed as part of a research paper submitted to **ICCET 2026** 
— currently under review. The paper formalizes the volatility-adaptive weighting 
scheme and the sigmoid confidence calibration approach.

---

## Live Demo

https://agentic-ai-stock-predictor.streamlit.app/

---
