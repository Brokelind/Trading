# web_dashboard.py
import os
import json
import pandas as pd
from datetime import datetime
from distribute_results import load_results

RESULTS_DIR = "results"
WEB_DIR = "web_dashboard"
os.makedirs(WEB_DIR, exist_ok=True)
# web_dashboard.py
import os
import json
import pandas as pd
from datetime import datetime
from distribute_results import load_results

# GitHub Pages output directory
OUTPUT_DIR = "web_dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    """Generate dashboard optimized for GitHub Pages"""
    
    # Load current results
    summaries = load_results()
    
    # Generate main dashboard HTML
    html_content = generate_github_pages_html(summaries)
    
    # Save main dashboard
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding='utf-8') as f:
        f.write(html_content)
    
    # Generate individual ticker pages
    for summary in summaries:
        generate_ticker_page(summary)
    
    # Copy assets
    generate_assets()
    
    print(f"✅ GitHub Pages dashboard generated: {OUTPUT_DIR}/index.html")
    return len(summaries)

def generate_github_pages_html(summaries):
    """Generate HTML optimized for GitHub Pages"""
    
    summaries_sorted = sorted(summaries, key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Generate cards
    cards_html = ""
    for summary in summaries_sorted:
        ticker = summary['ticker']
        signal = summary['signal']
        confidence = summary.get('confidence', 0)
        pct_diff = summary.get('pct_diff', 0)
        
        signal_class = "signal-buy" if signal == "BUY" else "signal-sell" if signal == "SELL" else "signal-hold"
        signal_icon = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "➡️"
        
        cards_html += f"""
        <div class="card {signal_class}">
            <div class="card-header">
                <h3>{ticker}</h3>
                <span class="signal-badge {signal_class}">{signal_icon} {signal}</span>
            </div>
            <div class="card-body">
                <div class="metric">
                    <label>Confidence:</label>
                    <span class="value">{confidence:.2f}</span>
                </div>
                <div class="metric">
                    <label>Predicted Change:</label>
                    <span class="value {'positive' if pct_diff > 0 else 'negative'}">{pct_diff:+.2f}%</span>
                </div>
                <div class="metric">
                    <label>Model:</label>
                    <span class="value">{summary.get('chosen_model', 'N/A')}</span>
                </div>
                <a href="tickers/{ticker}.html" class="details-link">View Details →</a>
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
    <title>Trading Signals Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <link rel="icon" type="image/x-icon" href="assets/favicon.ico">
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Trading Signals Dashboard</h1>
            <div class="last-updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            <div class="github-info">
                <a href="https://github.com/yourusername/your-repo" target="_blank">
                    View Source on GitHub
                </a>
            </div>
        </header>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{total_signals}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-card buy">
                <div class="stat-value">{buy_signals}</div>
                <div class="stat-label">BUY Signals</div>
            </div>
            <div class="stat-card sell">
                <div class="stat-value">{sell_signals}</div>
                <div class="stat-label">SELL Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_confidence:.2f}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        <div class="signals-grid">
            {cards_html if cards_html else '<div class="no-signals">No strong signals today</div>'}
        </div>
        
        <div class="info-section">
            <h3>About This Dashboard</h3>
            <p>This dashboard is automatically generated daily using AI trading models. 
               Signals are based on technical analysis, sentiment analysis, and ensemble machine learning models.</p>
            <p><em>Last automated run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</em></p>
        </div>
        
        <footer>
            <p>Automatically generated by Trading AI System | 
               <a href="https://github.com/yourusername/your-repo/actions" target="_blank">View Workflow Status</a>
            </p>
        </footer>
    </div>
</body>
</html>
"""

def generate_main_dashboard(summaries):
    """Generate the main dashboard page"""
    
    # Sort by confidence descending
    summaries_sorted = sorted(summaries, key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Generate summary cards
    cards_html = ""
    for summary in summaries_sorted:
        ticker = summary['ticker']
        signal = summary['signal']
        confidence = summary.get('confidence', 0)
        pct_diff = summary.get('pct_diff', 0)
        
        # Signal styling
        signal_class = "signal-buy" if signal == "BUY" else "signal-sell" if signal == "SELL" else "signal-hold"
        signal_icon = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "➡️"
        
        cards_html += f"""
        <div class="card {signal_class}">
            <div class="card-header">
                <h3>{ticker}</h3>
                <span class="signal-badge {signal_class}">{signal_icon} {signal}</span>
            </div>
            <div class="card-body">
                <div class="metric">
                    <label>Confidence:</label>
                    <span class="value">{confidence:.2f}</span>
                </div>
                <div class="metric">
                    <label>Predicted Change:</label>
                    <span class="value {'positive' if pct_diff > 0 else 'negative'}">{pct_diff:+.2f}%</span>
                </div>
                <div class="metric">
                    <label>Model:</label>
                    <span class="value">{summary.get('chosen_model', 'N/A')}</span>
                </div>
                <a href="tickers/{ticker}.html" class="details-link">View Details →</a>
            </div>
        </div>
        """
    
    # Generate metrics summary
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
    <title>Trading Signals Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Trading Signals Dashboard</h1>
            <div class="last-updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>
        
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{total_signals}</div>
                <div class="stat-label">Total Signals</div>
            </div>
            <div class="stat-card buy">
                <div class="stat-value">{buy_signals}</div>
                <div class="stat-label">BUY Signals</div>
            </div>
            <div class="stat-card sell">
                <div class="stat-value">{sell_signals}</div>
                <div class="stat-label">SELL Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_confidence:.2f}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        <div class="signals-grid">
            {cards_html if cards_html else '<div class="no-signals">No strong signals today</div>'}
        </div>
        
        <footer>
            <p>Generated automatically by Trading AI System</p>
        </footer>
    </div>
    
    <script src="assets/script.js"></script>
</body>
</html>
"""

def generate_ticker_page(summary):
    """Generate detailed page for each ticker"""
    ticker = summary['ticker']
    ticker_dir = os.path.join(WEB_DIR, "tickers")
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Check for existing chart files
    chart_html_path = os.path.join(RESULTS_DIR, f"{ticker}_backtest.html")
    chart_html = ""
    if os.path.exists(chart_html_path):
        with open(chart_html_path, 'r', encoding='utf-8') as f:
            chart_html = f.read()
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} - Detailed Analysis</title>
    <link rel="stylesheet" href="../assets/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1><a href="../index.html">← Back to Dashboard</a> | {ticker} Analysis</h1>
        </header>
        
        <div class="ticker-details">
            <div class="detail-card">
                <h2>Signal Summary</h2>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Signal:</label>
                        <span class="signal-{summary['signal'].lower()}">{summary['signal']}</span>
                    </div>
                    <div class="detail-item">
                        <label>Confidence:</label>
                        <span>{summary.get('confidence', 0):.2f}</span>
                    </div>
                    <div class="detail-item">
                        <label>Predicted Change:</label>
                        <span class="{'positive' if summary.get('pct_diff', 0) > 0 else 'negative'}">
                            {summary.get('pct_diff', 0):+.2f}%
                        </span>
                    </div>
                    <div class="detail-item">
                        <label>Selected Model:</label>
                        <span>{summary.get('chosen_model', 'N/A')}</span>
                    </div>
                </div>
            </div>
            
            <div class="detail-card">
                <h2>Sentiment Analysis</h2>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Sentiment Score:</label>
                        <span>{summary.get('sentiment', {}).get('score', 'N/A')}</span>
                    </div>
                    <div class="detail-item">
                        <label>Sentiment Confidence:</label>
                        <span>{summary.get('sentiment', {}).get('confidence', 'N/A')}</span>
                    </div>
                </div>
            </div>
            
            {chart_html if chart_html else '<div class="no-chart">No chart data available</div>'}
        </div>
    </div>
</body>
</html>
"""
    
    with open(os.path.join(ticker_dir, f"{ticker}.html"), "w", encoding='utf-8') as f:
        f.write(html_content)

def generate_assets():
    """Generate CSS and JS assets"""
    assets_dir = os.path.join(WEB_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # CSS
    css_content = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
    background: linear-gradient(45deg, #00ff99, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.last-updated {
    color: #888;
    font-size: 0.9em;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.stat-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stat-card.buy {
    border-left: 4px solid #00ff99;
}

.stat-card.sell {
    border-left: 4px solid #ff4444;
}

.stat-value {
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 5px;
}

.stat-label {
    color: #ccc;
    font-size: 0.9em;
}

.signals-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.card.signal-buy {
    border-left: 4px solid #00ff99;
}

.card.signal-sell {
    border-left: 4px solid #ff4444;
}

.card.signal-hold {
    border-left: 4px solid #ffaa00;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.card-header h3 {
    font-size: 1.4em;
    color: #fff;
}

.signal-badge {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
}

.signal-badge.signal-buy {
    background: rgba(0, 255, 153, 0.2);
    color: #00ff99;
}

.signal-badge.signal-sell {
    background: rgba(255, 68, 68, 0.2);
    color: #ff4444;
}

.signal-badge.signal-hold {
    background: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
}

.metric {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.metric label {
    color: #ccc;
}

.metric .value {
    font-weight: bold;
}

.positive {
    color: #00ff99;
}

.negative {
    color: #ff4444;
}

.details-link {
    display: inline-block;
    margin-top: 15px;
    color: #00ccff;
    text-decoration: none;
    font-weight: bold;
}

.details-link:hover {
    text-decoration: underline;
}

.ticker-details {
    max-width: 800px;
    margin: 0 auto;
}

.detail-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
}

.detail-card h2 {
    margin-bottom: 20px;
    color: #00ccff;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
}

.detail-item {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.no-signals, .no-chart {
    text-align: center;
    padding: 40px;
    color: #888;
    font-style: italic;
}

footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    color: #666;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

a {
    color: #00ccff;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    h1 {
        font-size: 2em;
    }
    
    .summary-stats {
        grid-template-columns: 1fr 1fr;
    }
    
    .signals-grid {
        grid-template-columns: 1fr;
    }
}
"""
    
    with open(os.path.join(assets_dir, "style.css"), "w", encoding='utf-8') as f:
        f.write(css_content)
    
    # JavaScript (optional enhancements)
    js_content = """
// Auto-refresh dashboard every 5 minutes
setTimeout(() => {
    window.location.reload();
}, 300000);

// Add smooth animations
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

// Add to existing CSS:
// .fade-in { animation: fadeIn 0.5s ease-out forwards; opacity: 0; }
// @keyframes fadeIn { to { opacity: 1; } }
"""
    
    with open(os.path.join(assets_dir, "script.js"), "w", encoding='utf-8') as f:
        f.write(js_content)

if __name__ == "__main__":
    count = generate_dashboard()
    print(f"🎉 Generated dashboard with {count} signals")