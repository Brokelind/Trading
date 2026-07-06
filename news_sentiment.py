import os
import requests
import statistics
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Only for local use
try:
    import env
except ImportError:
    env = None

# --- Environment Keys ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY") or getattr(env, "FINNHUB_API_KEY", None)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID") or getattr(env, "REDDIT_CLIENT_ID", None)
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET") or getattr(env, "REDDIT_CLIENT_SECRET", None)
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT") or getattr(env, "REDDIT_USER_AGENT", None)

# --- Reddit setup (lazy loaded) ---
praw = None
reddit = None

# --- Load TF FinBERT sentiment model (optional, may fail in CI) ---
finbert_pipe = None
try:
    from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", from_pt=True)
    finbert_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        framework="tf",  # force TensorFlow
        device=-1        # CPU only
    )
except Exception as e:
    print(f"[WARN] Could not load FinBERT model: {e}")
    print("[INFO] Falling back to VADER-only sentiment analysis")

# --- VADER sentiment ---
vader = SentimentIntensityAnalyzer()

# --- Helper functions ---
def get_reddit_news(ticker, lookback_days=1):
    global reddit, praw
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        print("[WARN] Reddit API credentials missing or invalid, skipping Reddit search.")
        return []

    if reddit is None:
        try:
            # Lazy import praw only when needed
            if praw is None:
                import praw as praw_lib
                praw = praw_lib
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize PRAW: {e}")
            return []

    from_date = datetime.utcnow() - timedelta(days=lookback_days)
    subreddits = ["stocks", "investing", "wallstreetbets"]
    news = []

    for subreddit_name in subreddits:
        print(f"[INFO] Searching r/{subreddit_name} for '{ticker}' posts...")
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(ticker, sort="new", time_filter="day", limit=100):
                if datetime.utcfromtimestamp(submission.created_utc) < from_date:
                    continue
                if submission.upvote_ratio < 0.5 or submission.score < 10:
                    continue
                title_lower = submission.title.lower()
                if f"${ticker.lower()}" not in title_lower and ticker.lower() not in title_lower.split():
                    continue
                news.append({
                    "headline": submission.title.strip(),
                    "reach_weight": submission.score
                })
        except Exception as e:
            print(f"[ERROR] Failed to fetch from r/{subreddit_name}: {e}")

    print(f"[INFO] Collected {len(news)} relevant Reddit posts.")
    return news

def get_finnhub_news(ticker, lookback_days=1):
    if not FINNHUB_API_KEY:
        print("[WARN] FINNHUB_API_KEY not set, skipping Finnhub news")
        return []

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=lookback_days)
    url = "https://finnhub.io/api/v1/company-news"
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
        print(f"[ERROR] Failed to fetch Finnhub news: {e}")
        return []

def score_finbert(text):
    if finbert_pipe is None:
        return 0.0
    try:
        result = finbert_pipe(text)[0]  # returns dict with 'label' and 'score'
        label, score = result["label"], result["score"]
        lbl_lower = label.lower()
        return {"negative": -1.0, "neutral": 0.0, "positive": 1.0}.get(lbl_lower, 0.0) * score
    except Exception as e:
        print(f"[ERROR] FinBERT sentiment scoring failed: {e}")
        return 0.0

def aggregate_scores(scores):
    try:
        std_dev = statistics.stdev(scores)
    except statistics.StatisticsError:
        std_dev = 1.0
    confidence = max(0.0, 1.0 - std_dev)
    return sum(scores) / len(scores), confidence

# --- Main function ---
def analyze_news_sentiment(ticker, lookback_days=1):
    news_data = get_finnhub_news(ticker, lookback_days)
    reddit_news = get_reddit_news(ticker, lookback_days)

    if not news_data and not reddit_news:
        print("[INFO] No news found.")
        return None

    aggregated_scores = []
    for article in news_data + reddit_news:
        headline = article.get("headline", "").strip()
        if not headline or len(headline.split()) < 5:
            continue

        # Scores
        vader_score = vader.polarity_scores(headline)["compound"]
        finbert_score = score_finbert(headline)
        scores = [vader_score, finbert_score]

        # Reddit reach weighting
        reach_weight = article.get("reach_weight")
        if reach_weight is not None:
            adjusted_weight = min(reach_weight / 10, 10)
            scores = [s * adjusted_weight for s in scores]
            headline = f"{headline} (REDDIT Reach: {reach_weight})"

        final_score, confidence = aggregate_scores(scores)
        aggregated_scores.append((final_score, confidence))

    final_avg_score = sum(s for s, _ in aggregated_scores) / len(aggregated_scores)
    final_confidence = sum(c for _, c in aggregated_scores) / len(aggregated_scores)

    if final_avg_score > 0.3:
        signal = "BUY"
    elif final_avg_score < -0.3:
        signal = "SELL"
    else:
        signal = "HOLD"

    print(f"\n>>> Final sentiment analysis for {ticker}: {signal} | Score: {final_avg_score:.3f} | Confidence: {final_confidence:.2f}")
    return {
        "ticker": ticker,
        "signal": signal,
        "score": round(final_avg_score, 3),
        "confidence": round(final_confidence, 2)
    }

# --- CLI ---
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    analyze_news_sentiment(ticker)