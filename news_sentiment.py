import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from datetime import datetime, timedelta
import os
import statistics

# only for local use
try:
    import env
except ImportError:
    env = None

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY") or getattr(env, "FINNHUB_API_KEY", None)

# Load Models: Replace FinBERT with DistilBERT SST-2 model to avoid torch 2.6+ issues
finbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
finbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sst2_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Removed cardiff_pipe (Roberta) due to torch version conflicts

vader = SentimentIntensityAnalyzer()

def get_finnhub_news(ticker, lookback_days=1):
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=lookback_days)
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": FINNHUB_API_KEY
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        return []

def score_finbert(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    return probs[1].item() - probs[0].item()  # positive - negative

def normalize_label_score(label):
    return {"LABEL_0": -1.0, "LABEL_1": 1.0, "negative": -1.0, "neutral": 0.0, "positive": 1.0}.get(label, 0.0)

def aggregate_scores(scores):
    try:
        std_dev = statistics.stdev(scores)
    except statistics.StatisticsError:
        std_dev = 1.0  # only one score, fallback
    confidence = max(0.0, 1.0 - std_dev)
    return sum(scores) / len(scores), confidence

def analyze_news_sentiment(ticker, lookback_days=1):
    news_data = get_finnhub_news(ticker, lookback_days)
    if not news_data:
        print("[INFO] No news found.")
        return None

    aggregated_scores = []
    for article in news_data:
        headline = article.get("headline", "").strip()
        if not headline or len(headline.split()) < 5:
            continue

        vader_score = vader.polarity_scores(headline)["compound"]
        finbert_score = score_finbert(headline)
        sst_score = normalize_label_score(sst2_pipe(headline)[0]["label"])

        scores = [vader_score, finbert_score, sst_score]  # cardiff removed
        final_score, confidence = aggregate_scores(scores)

        print(f"[{ticker}] Headline: {headline}\n  VADER: {vader_score:.3f}, FinBERT(SST-2): {finbert_score:.3f}, SST-2: {sst_score:.3f}\n  Final Score: {final_score:.3f}, Confidence: {confidence:.2f}\n")
        aggregated_scores.append((final_score, confidence))

    if not aggregated_scores:
        print("[INFO] No valid headlines.")
        return None

    final_avg_score = sum(s for s, _ in aggregated_scores) / len(aggregated_scores)
    final_confidence = sum(c for _, c in aggregated_scores) / len(aggregated_scores)

    if final_avg_score > 0.3:
        signal = "BUY"
    elif final_avg_score < -0.3:
        signal = "SELL"
    else:
        signal = "HOLD"

    print(f"\n>>> Final sentiment analysis Signal for {ticker}: {signal} | Score: {final_avg_score:.3f} | Confidence: {final_confidence:.2f}")
    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(final_confidence, 2),
        "score": round(final_avg_score, 3)
    }
   
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    analyze_news_sentiment(ticker)
