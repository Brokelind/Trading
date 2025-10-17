# web_dashboard.py
import os
import json
import pandas as pd
from datetime import datetime
from distribute_results import load_results
import plotly.graph_objects as go

# GitHub Pages output directory
OUTPUT_DIR = "web_dashboard"
RESULTS_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    """Generate clean, professional dashboard"""
    
    # Load current results
    summaries = load_results()
    
    # Generate main dashboard HTML
    html_content = generate_clean_dashboard_html(summaries)
    
    # Save main dashboard
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(html_content)
    
    # Generate individual ticker pages
    for summary in summaries:
         generate_clean_ticker_page(summary)
    
    # Copy assets
    generate_clean_assets()
    
    print(f"✅ Clean dashboard generated: {OUTPUT_DIR}/index.html")
    return len(summaries)

def generate_clean_dashboard_html(summaries):
    """Generate clean, professional dashboard HTML"""
    
    summaries_sorted = sorted(summaries, key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Generate clean cards
    cards_html = ""
    for summary in summaries_sorted:
        ticker = summary['ticker']
        signal = summary['signal']
        confidence = summary.get('confidence', 0)
        pct_diff = summary.get('pct_diff', 0)
        chosen_model = summary.get('chosen_model', 'N/A')
        sentiment_score = summary.get('sentiment', {}).get('score', 0)
        
        # Signal styling
        signal_class = "buy" if signal == "BUY" else "sell" if signal == "SELL" else "hold"
        signal_icon = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "➡️"
        
        # Confidence indicator
        confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        
        cards_html += f"""
        <div class="stock-card {signal_class}">
            <div class="card-header">
                <div class="ticker">{ticker}</div>
                <div class="signal {signal_class}">
                    {signal_icon} {signal}
                </div>
            </div>
            <div class="card-body">
                <div class="metric-row">
                    <div class="metric">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value confidence-{confidence_level}">{confidence:.0%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Predicted</div>
                        <div class="metric-value {'positive' if pct_diff > 0 else 'negative'}">{pct_diff:+.1f}%</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric">
                        <div class="metric-label">Model</div>
                        <div class="metric-value model">{chosen_model}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sentiment</div>
                        <div class="metric-value sentiment">{sentiment_score:+.2f}</div>
                    </div>
                </div>
                <a href="tickers/{ticker}.html" class="details-btn">View Details →</a>
            </div>
        </div>
        """
    
    # Stats
    total_signals = len(summaries)
    buy_signals = len([s for s in summaries if s['signal'] == 'BUY'])
    sell_signals = len([s for s in summaries if s['signal'] == 'SELL'])
    avg_confidence = sum(s.get('confidence', 0) for s in summaries) / total_signals if total_signals else 0
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Signals</title>
    <link rel="stylesheet" href="assets/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-content">
                <h1>🤖 AI Trading Signals</h1>
                <p class="subtitle">Machine Learning Powered Market Analysis</p>
                <div class="last-updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_signals}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-card buy">
                <div class="stat-number">{buy_signals}</div>
                <div class="stat-label">BUY Signals</div>
            </div>
            <div class="stat-card sell">
                <div class="stat-number">{sell_signals}</div>
                <div class="stat-label">SELL Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_confidence:.0%}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        <div class="signals-grid">
            {cards_html if cards_html else '<div class="no-signals">No trading signals available</div>'}
        </div>
        
        <footer class="footer">
            <p>Built with Python & Machine Learning • Real-time Analysis • Automated Trading</p>
        </footer>
    </div>
</body>
</html>
"""

def generate_clean_ticker_page(summary):
    """Generate clean, professional ticker detail page"""
    ticker = summary['ticker']
    ticker_dir = os.path.join(OUTPUT_DIR, "tickers")
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Get predictions
    predictions_html = ""
    for model, pred in summary.get('predictions', {}).items():
        if model != "Ensemble":
            pred_price = pred.get('predicted_price', 'N/A')
            pct_diff = pred.get('pct_diff', 0)
            signal = pred.get('signal', 'N/A')
            
            pred_price_display = f"${pred_price:.2f}" if isinstance(pred_price, (int, float)) else str(pred_price)
            pct_diff_display = f"{pct_diff:+.1f}%" if isinstance(pct_diff, (int, float)) else str(pct_diff)
            
            predictions_html += f"""
            <div class="model-prediction">
                <div class="model-name">{model}</div>
                <div class="prediction-info">
                    <span class="price">{pred_price_display}</span>
                    <span class="signal {signal.lower()}">{signal}</span>
                    <span class="change {'positive' if pct_diff > 0 else 'negative'}">{pct_diff_display}</span>
                </div>
            </div>
            """
    
    # Training metrics
    training_html = ""
    training_metrics = summary.get('training_metrics', [])
    
    if training_metrics:
        training_df = pd.DataFrame(training_metrics)
        
        # Simple metrics display
        training_html = """
        <div class="metrics-section">
            <h3>Model Performance</h3>
            <div class="metrics-grid">
        """
        
        for _, row in training_df.iterrows():
            model_name = row['Model']
            accuracy = row.get('Accuracy', 0) or 0
            precision = row.get('Precision', 0) or 0
            recall = row.get('Recall', 0) or 0
            f1 = row.get('F1', 0) or 0
            
            training_html += f"""
                <div class="performance-card">
                    <div class="model-header">
                        <h4>{model_name}</h4>
                        <span class="accuracy">{accuracy:.1%}</span>
                    </div>
                    <div class="performance-metrics">
                        <div class="metric">
                            <span>Precision</span>
                            <span>{precision:.1%}</span>
                        </div>
                        <div class="metric">
                            <span>Recall</span>
                            <span>{recall:.1%}</span>
                        </div>
                        <div class="metric">
                            <span>F1 Score</span>
                            <span>{f1:.1%}</span>
                        </div>
                    </div>
                </div>
            """
        
        training_html += """
            </div>
        </div>
        """
    
    # Main signal info
    signal = summary['signal']
    signal_class = "buy" if signal == "BUY" else "sell" if signal == "SELL" else "hold"
    confidence = summary.get('confidence', 0)
    pct_diff = summary.get('pct_diff', 0)
    sentiment_score = summary.get('sentiment', {}).get('score', 0)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Analysis</title>
    <link rel="stylesheet" href="../assets/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-content">
                <a href="../index.html" class="back-link">← Back to Dashboard</a>
                <h1>{ticker} Detailed Analysis</h1>
            </div>
        </header>
        
        <div class="ticker-content">
            <!-- Signal Summary -->
            <div class="summary-card">
                <div class="signal-header {signal_class}">
                    <div class="signal-info">
                        <h2>{signal} Signal</h2>
                        <div class="confidence">Confidence: {confidence:.0%}</div>
                    </div>
                    <div class="price-info">
                        <div class="current-price">${summary.get('last_price', 'N/A')}</div>
                        <div class="predicted-change {'positive' if pct_diff > 0 else 'negative'}">{pct_diff:+.1f}%</div>
                    </div>
                </div>
            </div>
            
            <!-- Model Predictions -->
            <div class="section">
                <h3>Model Predictions</h3>
                <div class="predictions-grid">
                    {predictions_html if predictions_html else '<div class="no-data">No predictions available</div>'}
                </div>
            </div>
            
            <!-- Sentiment Analysis -->
            <div class="section">
                <h3>Market Sentiment</h3>
                <div class="sentiment-card">
                    <div class="sentiment-score {'positive' if sentiment_score > 0 else 'negative'}">
                        {sentiment_score:+.2f}
                    </div>
                    <div class="sentiment-info">
                        <div>Confidence: {summary.get('sentiment', {}).get('confidence', 'N/A')}</div>
                        <div>Signal: {summary.get('sentiment', {}).get('signal', 'N/A')}</div>
                    </div>
                </div>
            </div>
            
            <!-- Training Metrics -->
            {training_html}
            
            <!-- Quick Links -->
            <div class="section">
                <h3>Quick Links</h3>
                <div class="links-grid">
                    <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" class="link-card">
                        📊 Yahoo Finance
                    </a>
                    <a href="https://www.tradingview.com/symbols/{ticker}" target="_blank" class="link-card">
                        📈 TradingView
                    </a>
                    <a href="https://www.google.com/finance/quote/{ticker}" target="_blank" class="link-card">
                        🔍 Google Finance
                    </a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(os.path.join(ticker_dir, f"{ticker}.html"), "w", encoding='utf-8') as f:
        f.write(html_content)

def generate_clean_assets():
    """Generate clean, modern CSS"""
    assets_dir = os.path.join(OUTPUT_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    css_content = """
:root {
    --primary-bg: #0f0f23;
    --card-bg: #1a1a2e;
    --card-border: #2d2d4d;
    --text-primary: #ffffff;
    --text-secondary: #a0a0c0;
    --text-muted: #666687;
    --buy-color: #00d092;
    --sell-color: #ff4757;
    --hold-color: #ffa502;
    --positive: #00d092;
    --negative: #ff4757;
    --border-radius: 12px;
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--primary-bg);
    color: var(--text-primary);
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 40px;
    padding: 40px 20px;
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--buy-color), #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-bottom: 12px;
}

.last-updated {
    color: var(--text-muted);
    font-size: 0.9rem;
}

.back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.9rem;
    margin-bottom: 20px;
    display: inline-block;
}

.back-link:hover {
    color: var(--text-primary);
}

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
}

.stat-card {
    background: var(--card-bg);
    padding: 24px;
    border-radius: var(--border-radius);
    text-align: center;
    border: 1px solid var(--card-border);
    transition: transform 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
}

.stat-card.buy {
    border-left: 4px solid var(--buy-color);
}

.stat-card.sell {
    border-left: 4px solid var(--sell-color);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* Signals Grid */
.signals-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 60px;
}

.stock-card {
    background: var(--card-bg);
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    overflow: hidden;
    transition: all 0.3s ease;
}

.stock-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow);
}

.stock-card.buy {
    border-left: 4px solid var(--buy-color);
}

.stock-card.sell {
    border-left: 4px solid var(--sell-color);
}

.stock-card.hold {
    border-left: 4px solid var(--hold-color);
}

.card-header {
    padding: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
}

.ticker {
    font-size: 1.5rem;
    font-weight: 600;
}

.signal {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}

.signal.buy {
    background: rgba(0, 208, 146, 0.2);
    color: var(--buy-color);
}

.signal.sell {
    background: rgba(255, 71, 87, 0.2);
    color: var(--sell-color);
}

.signal.hold {
    background: rgba(255, 165, 2, 0.2);
    color: var(--hold-color);
}

.card-body {
    padding: 20px;
}

.metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
}

.metric {
    text-align: center;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-bottom: 4px;
}

.metric-value {
    font-weight: 600;
    font-size: 1.1rem;
}

.confidence-high { color: var(--buy-color); }
.confidence-medium { color: var(--hold-color); }
.confidence-low { color: var(--sell-color); }

.positive { color: var(--buy-color); }
.negative { color: var(--sell-color); }

.model, .sentiment {
    color: var(--text-primary);
}

.details-btn {
    display: block;
    width: 100%;
    padding: 12px;
    background: transparent;
    border: 1px solid var(--card-border);
    color: var(--text-primary);
    text-decoration: none;
    text-align: center;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
    margin-top: 12px;
}

.details-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: var(--text-secondary);
}

/* Ticker Page */
.ticker-content {
    max-width: 800px;
    margin: 0 auto;
}

.summary-card {
    background: var(--card-bg);
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    margin-bottom: 30px;
    overflow: hidden;
}

.signal-header {
    padding: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.signal-header.buy {
    background: linear-gradient(135deg, rgba(0, 208, 146, 0.1), transparent);
}

.signal-header.sell {
    background: linear-gradient(135deg, rgba(255, 71, 87, 0.1), transparent);
}

.signal-header.hold {
    background: linear-gradient(135deg, rgba(255, 165, 2, 0.1), transparent);
}

.signal-info h2 {
    font-size: 2rem;
    margin-bottom: 8px;
}

.confidence {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

.current-price {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.predicted-change {
    font-size: 1.3rem;
    font-weight: 600;
}

.section {
    margin-bottom: 40px;
}

.section h3 {
    font-size: 1.5rem;
    margin-bottom: 20px;
    color: var(--text-primary);
}

/* Predictions */
.predictions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.model-prediction {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
}

.model-name {
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text-primary);
}

.prediction-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.price {
    font-size: 1.2rem;
    font-weight: 600;
}

/* Sentiment */
.sentiment-card {
    background: var(--card-bg);
    padding: 24px;
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    display: flex;
    align-items: center;
    gap: 20px;
}

.sentiment-score {
    font-size: 2.5rem;
    font-weight: 700;
}

.sentiment-info {
    color: var(--text-secondary);
}

/* Metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.performance-card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
}

.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--card-border);
}

.model-header h4 {
    font-size: 1.1rem;
}

.accuracy {
    font-weight: 600;
    color: var(--buy-color);
}

.performance-metrics {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.performance-metrics .metric {
    display: flex;
    justify-content: space-between;
    text-align: left;
}

/* Links */
.links-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.link-card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: var(--border-radius);
    border: 1px solid var(--card-border);
    text-decoration: none;
    color: var(--text-primary);
    text-align: center;
    transition: all 0.2s ease;
}

.link-card:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: var(--text-secondary);
    transform: translateY(-2px);
}

/* Footer */
.footer {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-muted);
    border-top: 1px solid var(--card-border);
    margin-top: 40px;
}

/* No Data States */
.no-signals, .no-data {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
    font-style: italic;
    grid-column: 1 / -1;
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 16px;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .signals-grid {
        grid-template-columns: 1fr;
    }
    
    .signal-header {
        flex-direction: column;
        text-align: center;
        gap: 16px;
    }
    
    .predictions-grid {
        grid-template-columns: 1fr;
    }
    
    .sentiment-card {
        flex-direction: column;
        text-align: center;
    }
}
"""
    
    with open(os.path.join(assets_dir, "style.css"), "w", encoding='utf-8') as f:
        f.write(css_content)

if __name__ == "__main__":
    generate_dashboard()