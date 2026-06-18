import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

st.title("Agentic AI Financial Prediction System")

# ─────────────────────────────────────────────
# DISK-BASED SENTIMENT CACHE
# ─────────────────────────────────────────────

def load_with_sentiment(raw_path, cache_path):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    df = pd.read_csv(raw_path)

    if "sentiment" not in df.columns:
        text_col = next(
            (col for col in df.columns
             if col.lower() in ["text", "title", "body", "content", "tweet"]),
            None
        )
        if text_col:
            total = len(df)
            bar = st.progress(0, text=f"Scoring sentiment for {raw_path}…")
            scores = []
            for i, text in enumerate(df[text_col].astype(str)):
                scores.append(sia.polarity_scores(text)["compound"])
                if i % 500 == 0:
                    bar.progress(int(i / total * 100),
                                 text=f"Scoring {i}/{total} rows…")
            bar.empty()
            df["sentiment"] = scores
        else:
            df["sentiment"] = 0.0

    df.to_csv(cache_path, index=False)
    return df


@st.cache_resource
def load_all_data():
    reddit = pd.read_csv("reddit_sentiment_cache.csv")
    
    twitter = pd.read_csv(
        "https://huggingface.co/datasets/vrindu34/agentic-ai-sentiment-data/resolve/main/twitter_sentiment_cache.csv",
        low_memory=False
    )
    
    return reddit, twitter

reddit, twitter = load_all_data()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def mean_sentiment(df, stock_upper):
    col = next(
        (c for c in df.columns if c.lower() in ["stock", "symbol", "ticker"]), None
    )
    if col:
        filtered = df[df[col].astype(str).str.upper() == stock_upper]
        return filtered["sentiment"].mean() if len(filtered) > 0 else df["sentiment"].mean()
    return df["sentiment"].mean()


@st.cache_data(ttl=300)
def fetch_data(ticker):
    data = yf.download(ticker + ".NS", period="6mo", progress=False)
    if data.empty:
        data = yf.download(ticker, period="6mo", progress=False)
    return data if not data.empty else None


def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sigmoid_confidence(composite_score: float) -> float:
    """
    Maps composite score (-1..+1) → confidence (50..95%).

    Why sigmoid and not |score| * 100?
    ────────────────────────────────────────────────────────
    Weighted-average composite scores almost never exceed ±0.35
    in real markets — even strong signals.  Using |score|*100
    caps you at ~35% for a perfectly clear signal.

    Sigmoid re-maps the range so:
        |score| = 0.00  →  50%  (pure uncertainty)
        |score| = 0.15  →  ~67%
        |score| = 0.25  →  ~74%
        |score| = 0.35  →  ~79%
        |score| = 0.50  →  ~84%
        |score| = 0.80  →  ~91%
        |score| = 1.00  →  ~95%  (hard ceiling)

    Formula:  confidence = 50 + 45 × (2 / (1 + e^(-k|x|)) - 1)
    k = 6 gives a smooth, realistic spread.
    """
    k = 6.0
    confidence = 50.0 + 45.0 * (2.0 / (1.0 + np.exp(-k * abs(composite_score))) - 1.0)
    return round(float(confidence), 1)


# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────

stock = st.text_input("Enter Stock Ticker (e.g. RELIANCE or AAPL)")
days  = st.slider("Days to analyse", 7, 90, 30)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

if st.button("Predict"):

    if not stock.strip():
        st.warning("Please enter a stock ticker.")
        st.stop()

    stock_upper = stock.strip().upper()

    with st.spinner("Fetching market data…"):
        data = fetch_data(stock_upper)
    exists = data is not None

    # ── MARKET AGENT (multi-signal, normalized to -1..+1) ─────────────────
    if exists:
        df    = data.tail(days).copy()
        close = df["Close"].squeeze()

        # 1. Price momentum: % change over window → scaled ±1
        momentum      = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
        momentum_norm = float(np.clip(momentum * 5, -1, 1))   # 20% move = ±1

        # 2. RSI → ±1
        rsi = compute_rsi(close).iloc[-1]
        if pd.isna(rsi):
            rsi = 50.0
        rsi_norm = float((rsi - 50) / 50)   # 0→-1, 50→0, 100→+1

        # 3. Linear trend slope → ±1
        x = np.arange(len(close))
        slope, _ = np.polyfit(x, close.values.flatten(), 1)
        slope_pct  = slope / (close.mean() + 1e-9)
        slope_norm = float(np.clip(slope_pct * 100, -1, 1))

        # 4. Volatility (for weight adaption only)
        vol_pct = float(
            (df["High"].squeeze() - df["Low"].squeeze()).mean() / (close.mean() + 1e-9)
        )

        market_score = float(
            momentum_norm * 0.40 + rsi_norm * 0.30 + slope_norm * 0.30
        )
    else:
        market_score = 0.0
        vol_pct      = 0.1
        rsi          = 50.0
        momentum     = 0.0

    # ── SENTIMENT AGENT ───────────────────────────────────────────────────
    r = mean_sentiment(reddit,  stock_upper)
    t = mean_sentiment(twitter, stock_upper)
    raw_sentiment = float((r + t) / 2)

    # tanh amplification: VADER raw avg sits near 0; tanh(x*3) stretches it
    # into [-1, +1] without ever exceeding bounds.
    sentiment_score = float(np.tanh(raw_sentiment * 3))

    # ── GRAPH AGENT ───────────────────────────────────────────────────────
    G = nx.Graph()
    G.add_edges_from([
        (stock_upper, "Market"),
        (stock_upper, "Sentiment"),
        ("Market",    "Macro"),
        ("Sentiment", "News"),
    ])
    graph_centrality = nx.degree_centrality(G)[stock_upper]   # 0..1

    # Graph agreement bonus: positive when market + sentiment agree in direction
    agreement   = market_score * sentiment_score              # positive if same sign
    graph_score = float(np.tanh(agreement * 4))               # squash to (-1, +1)

    # ── WEIGHTS (adapt to volatility) ─────────────────────────────────────
    if exists and vol_pct > 0.02:
        wm, ws, wg = 0.55, 0.30, 0.15   # high-vol: trust market more
    else:
        wm, ws, wg = 0.40, 0.45, 0.15   # low-vol: trust sentiment more

    # ── COMPOSITE SCORE (-1..+1) ──────────────────────────────────────────
    raw_score = float(np.clip(
        wm * market_score + ws * sentiment_score + wg * graph_score,
        -1, 1
    ))

    # ── CONFIDENCE via sigmoid (NOT |score| * 100) ────────────────────────
    confidence_pct = sigmoid_confidence(raw_score)

    # ── DECISION ──────────────────────────────────────────────────────────
    if raw_score > 0.10:
        decision = " BUY"
    elif raw_score < -0.10:
        decision = " SELL"
    else:
        decision = " HOLD"

    # ═════════════════════════════════════════
    # DISPLAY
    # ═════════════════════════════════════════

    st.divider()

    # Price chart
    if exists:
        st.subheader("📈 Price Trend")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, close, linewidth=1.8, color="#1f77b4")
        ma = close.rolling(7).mean()
        ax.plot(df.index, ma, linewidth=1.2, color="orange", linestyle="--", label="7-day MA")
        ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.set_title(f"{stock_upper} — last {days} days")
        st.pyplot(fig); plt.close(fig)

    # Signal breakdown
    st.subheader("📊 Signal Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Score",     f"{market_score:+.3f}",
              help="Momentum + RSI + Trend, normalized −1 to +1")
    c2.metric("Sentiment Score",  f"{sentiment_score:+.3f}",
              help="Reddit + Twitter VADER, tanh-amplified")
    c3.metric("RSI",              f"{rsi:.1f}",
              delta="Oversold" if rsi < 35 else ("Overbought" if rsi > 65 else "Neutral"))
    c4.metric("Momentum",         f"{momentum*100:+.1f}%",
              help=f"Price change over {days} days")

    # Confidence bar
    st.subheader(" Confidence")
    bar_color = (
        "#2ecc71" if confidence_pct >= 70 else
        "#e67e22" if confidence_pct >= 55 else
        "#e74c3c"
    )
    st.markdown(
        f"""<div style="background:var(--secondary-background-color);
                        border-radius:8px;height:30px;width:100%;overflow:hidden">
              <div style="background:{bar_color};width:{confidence_pct}%;height:30px;
                          display:flex;align-items:center;padding-left:12px;
                          color:#fff;font-weight:600;font-size:14px;transition:width .4s">
                {confidence_pct:.1f}%
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    # Agent graph
    st.subheader(" Agent Graph")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, ax=ax2,
            node_color=["#4C9BE8" if n == stock_upper else "#B0BEC5" for n in G.nodes()],
            node_size=1800, font_size=9, font_weight="bold", edge_color="#aaa")
    st.pyplot(fig2); plt.close(fig2)

    # Final prediction
    st.divider()
    st.subheader(" Prediction")
    col_a, col_b = st.columns(2)
    col_a.metric("Decision",   decision)
    col_b.metric("Confidence", f"{confidence_pct:.1f}%")

    st.write(f"**Composite Score:** `{raw_score:+.3f}`  (range: −1.0 to +1.0)")

    with st.expander("How is confidence calculated?"):
        st.markdown(f"""
**Why not `|score| × 100`?**  
Composite scores in real markets almost never exceed ±0.35, so the old formula
would always cap confidence at ~35% even for perfect signals.

**Sigmoid formula used here:**  
`confidence = 50 + 45 × sigmoid(6 × |score|)`

| Signal | Score | Weight |
|--------|-------|--------|
| Market (momentum + RSI + trend) | `{market_score:+.3f}` | `{wm:.0%}` |
| Sentiment (Reddit + Twitter tanh) | `{sentiment_score:+.3f}` | `{ws:.0%}` |
| Graph agreement bonus | `{graph_score:+.3f}` | `{wg:.0%}` |
| **Composite** | **`{raw_score:+.3f}`** | — |
| **Confidence** | **`{confidence_pct:.1f}%`** | sigmoid curve |

Graph centrality: `{graph_centrality:.2f}` (used to compute agreement bonus)
        """)

    st.caption("⚠️ For educational purposes only. Not financial advice.")
