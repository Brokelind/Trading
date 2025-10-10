# web_dashboard.py
import os
import json
import pandas as pd
from datetime import datetime
from distribute_results import load_results
import plotly.graph_objects as go
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
    """Generate dashboard optimized for GitHub Pages with enhanced features"""
    
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
    """Generate HTML optimized for GitHub Pages with enhanced metrics"""
    
    summaries_sorted = sorted(summaries, key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Generate cards with enhanced information
    cards_html = ""
    for summary in summaries_sorted:
        ticker = summary['ticker']
        signal = summary['signal']
        confidence = summary.get('confidence', 0)
        pct_diff = summary.get('pct_diff', 0)
        chosen_model = summary.get('chosen_model', 'N/A')
        sentiment_score = summary.get('sentiment', {}).get('score', 0)
        sentiment_conf = summary.get('sentiment', {}).get('confidence', 0)

        # Add training performance indicator
        training_metrics = summary.get('training_metrics', [])
        if training_metrics:
            best_r2 = max([m.get('R2', -10) for m in training_metrics])
            training_indicator = f"<div class='training-score'>Best R²: {best_r2:.3f}</div>"
        else:
            training_indicator = ""
        
        # Signal styling
        signal_class = "signal-buy" if signal == "BUY" else "signal-sell" if signal == "SELL" else "signal-hold"
        signal_icon = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "➡️"
        
        # Confidence indicator
        confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        
        cards_html += f"""
        <div class="card {signal_class}">
            <div class="card-header">
                <h3>{ticker}</h3>
                <span class="signal-badge {signal_class}">{signal_icon} {signal}</span>
            </div>
            <div class="card-body">
                <div class="metric">
                    <label>Confidence:</label>
                    <span class="value confidence-{confidence_level}">{confidence:.2f}</span>
                </div>
                <div class="metric">
                    <label>Predicted Change:</label>
                    <span class="value {'positive' if pct_diff > 0 else 'negative'}">{pct_diff:+.2f}%</span>
                </div>
                <div class="metric">
                    <label>Model:</label>
                    <span class="value model-name">{chosen_model}</span>
                </div>
                <div class="metric">
                    <label>Sentiment:</label>
                    <span class="value sentiment-{'positive' if sentiment_score > 0 else 'negative'}">{sentiment_score:+.2f}</span>
                </div>
                <div class="action-buttons">
                    <a href="tickers/{ticker}.html" class="details-link">View Details →</a>
                    <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" class="yahoo-link">Yahoo Finance ↗</a>
                </div>
            </div>
        </div>
        """
    
    # Enhanced stats
    total_signals = len(summaries)
    buy_signals = len([s for s in summaries if s['signal'] == 'BUY'])
    sell_signals = len([s for s in summaries if s['signal'] == 'SELL'])
    avg_confidence = sum(s.get('confidence', 0) for s in summaries) / total_signals if total_signals else 0
    avg_predicted_change = sum(s.get('pct_diff', 0) for s in summaries) / total_signals if total_signals else 0
    
    # Model distribution
    model_counts = {}
    for summary in summaries:
        model = summary.get('chosen_model', 'Unknown')
        model_counts[model] = model_counts.get(model, 0) + 1
    
    top_model = max(model_counts.items(), key=lambda x: x[1]) if model_counts else ('None', 0)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Signals Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <link rel="icon" type="image/x-icon" href="assets/favicon.ico">
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI Trading Signals Dashboard</h1>
            <div class="last-updated">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            <div class="github-info">
                <a href="https://github.com/yourusername/your-repo" target="_blank" class="github-link">
                    📁 View Source on GitHub
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
            <div class="stat-card">
                <div class="stat-value">{avg_predicted_change:+.2f}%</div>
                <div class="stat-label">Avg Predicted Change</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{top_model[0]}</div>
                <div class="stat-label">Top Model</div>
            </div>
        </div>
        
        <div class="signals-grid">
            {cards_html if cards_html else '<div class="no-signals">No strong signals meeting criteria today</div>'}
        </div>
        
        <div class="info-section">
            <h3>📊 About This Dashboard</h3>
            <p>This dashboard displays AI-generated trading signals based on:</p>
            <ul>
                <li><strong>Ensemble Machine Learning</strong> - Combining LSTM, Random Forest, XGBoost, and Neural Network models</li>
                <li><strong>Technical Analysis</strong> - 40+ technical indicators and price patterns</li>
                <li><strong>Sentiment Analysis</strong> - Real-time Reddit and news sentiment scoring</li>
                <li><strong>Risk Management</strong> - Confidence-based position sizing and stop-losses</li>
            </ul>
            <div class="filter-info">
                <strong>Filter Criteria:</strong> BUY/SELL signals only | Min 0.5% predicted change | Min 40% model confidence | Min 30% sentiment confidence
            </div>
            <p><em>Last automated run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</em></p>
        </div>
        
        <footer>
            <p>🚀 Built with Python, TensorFlow, and Machine Learning | Hugo Brokelind</p>
        </footer>
    </div>
    
    <script src="assets/script.js"></script>
</body>
</html>
"""

def generate_ticker_page(summary):
    """Generate detailed page for each ticker with enhanced information"""
    ticker = summary['ticker']
    ticker_dir = os.path.join(OUTPUT_DIR, "tickers")
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Check for existing chart files
    chart_html_path = os.path.join(RESULTS_DIR, f"{ticker}_backtest_dashboard.html")
    chart_html = ""
    if os.path.exists(chart_html_path):
        with open(chart_html_path, 'r', encoding='utf-8') as f:
            chart_html = f.read()
    
    # Get all model predictions
    predictions_html = ""
    for model, pred in summary.get('predictions', {}).items():
        if model != "Ensemble":  # Ensemble is already the main signal
            # Safely extract and format values
            pred_price = pred.get('predicted_price')
            pred_price_display = f"${pred_price:.2f}" if isinstance(pred_price, (int, float)) else str(pred_price) if pred_price is not None else 'N/A'
            
            pct_diff = pred.get('pct_diff')
            pct_diff_display = f"{pct_diff:.2f}%" if isinstance(pct_diff, (int, float)) else str(pct_diff) if pct_diff is not None else 'N/A'
            
            signal = pred.get('signal', 'N/A')
            
            predictions_html += f"""
            <div class="model-prediction">
                <div class="model-name">{model}</div>
                <div class="prediction-details">
                    <span>Price: {pred_price_display}</span>
                    <span>Signal: {signal}</span>
                    <span>Change: {pct_diff_display}</span>
                </div>
            </div>
            """
    
    # Safely get ensemble predicted price
    ensemble_pred = summary.get('predictions', {}).get('Ensemble', {})
    ensemble_price = ensemble_pred.get('predicted_price', 'N/A')
    if isinstance(ensemble_price, (int, float)):
        ensemble_price_display = f"${ensemble_price:.2f}"
    else:
        ensemble_price_display = str(ensemble_price)
    
    # Safely get sentiment data
    sentiment_data = summary.get('sentiment', {})
    sentiment_score = sentiment_data.get('score', 0)
    sentiment_confidence = sentiment_data.get('confidence', 'N/A')
    sentiment_signal = sentiment_data.get('signal', 'N/A')
    
    # Determine sentiment class
    sentiment_class = "positive" if sentiment_score > 0 else "negative"

    # Enhanced training metrics section
    training_html = ""
    training_metrics_table = ""
    training_metrics = summary.get('training_metrics', [])
    
    if training_metrics:
        training_df = pd.DataFrame(training_metrics)
        
        # Create training performance chart
        fig = go.Figure()
        
        # Add R² scores as bars
        fig.add_trace(go.Bar(
            x=training_df['Model'],
            y=training_df['R2'],
            name='R² Score',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            text=training_df['R2'].round(3),
            textposition='auto',
            textfont=dict(color='white', size=12)
        ))
        
        # Add direction accuracy as line
        fig.add_trace(go.Scatter(
            x=training_df['Model'],
            y=training_df['DirectionAccuracy'],
            mode='lines+markers',
            name='Direction Accuracy',
            line=dict(color='white', width=3),
            marker=dict(size=8, color='white'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            title="Model Training Performance",
            yaxis=dict(
                title="R² Score", 
                range=[min(training_df['R2'].min() - 0.1, -0.5), max(training_df['R2'].max() + 0.1, 0.5)],
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis2=dict(
                title="Direction Accuracy", 
                overlaying='y', 
                side='right', 
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.1)'
            ),
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)'
            )
        )
        
        training_html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Create beautiful metrics table
        training_metrics_table = """
        <div class="metrics-section">
            <h3>📊 Detailed Training Metrics</h3>
            <div class="metrics-grid">
        """
        
        for _, row in training_df.iterrows():
            model_name = row['Model']
            r2 = row.get('R2', 0)
            direction_acc = row.get('DirectionAccuracy', 0)
            mse = row.get('MSE', 0)
            mae = row.get('MAE', 0)
            
            # Determine performance classes
            r2_class = "metric-high" if r2 > 0.7 else "metric-medium" if r2 > 0.3 else "metric-low"
            acc_class = "metric-high" if direction_acc > 0.7 else "metric-medium" if direction_acc > 0.6 else "metric-low"
            
            training_metrics_table += f"""
                <div class="metric-card">
                    <div class="metric-header">
                        <h4>{model_name}</h4>
                        <span class="model-badge">{'🏆 Best' if r2 == training_df['R2'].max() else ''}</span>
                    </div>
                    <div class="metric-values">
                        <div class="metric-item">
                            <span class="metric-label">R² Score:</span>
                            <span class="metric-value {r2_class}">{r2:.3f}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Direction Accuracy:</span>
                            <span class="metric-value {acc_class}">{direction_acc:.1%}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MSE:</span>
                            <span class="metric-value">{mse:.4f}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">{mae:.4f}</span>
                        </div>
                    </div>
                </div>
            """
        
        training_metrics_table += """
            </div>
        </div>
        """
    
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
            <h1><a href="../index.html">← Back to Dashboard</a> | {ticker} Detailed Analysis</h1>
        </header>
        
        <div class="ticker-details">
            <div class="detail-card signal-summary">
                <h2>🎯 Signal Summary</h2>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Signal:</label>
                        <span class="signal-{summary['signal'].lower()}">{summary['signal']}</span>
                    </div>
                    <div class="detail-item">
                        <label>Confidence:</label>
                        <span class="confidence-{'high' if summary.get('confidence', 0) > 0.7 else 'medium' if summary.get('confidence', 0) > 0.5 else 'low'}">
                            {summary.get('confidence', 0):.2f}
                        </span>
                    </div>
                    <div class="detail-item">
                        <label>Predicted Change:</label>
                        <span class="{'positive' if summary.get('pct_diff', 0) > 0 else 'negative'}">
                            {summary.get('pct_diff', 0):+.2f}%
                        </span>
                    </div>
                    <div class="detail-item">
                        <label>Selected Model:</label>
                        <span class="model-name">{summary.get('chosen_model', 'N/A')}</span>
                    </div>
                    <div class="detail-item">
                        <label>Current Price:</label>
                        <span>${summary.get('last_price', 'N/A')}</span>
                    </div>
                    <div class="detail-item">
                        <label>Predicted Price:</label>
                        <span>{ensemble_price_display}</span>
                    </div>
                </div>
            </div>
            
            <div class="detail-card">
                <h2>📊 Sentiment Analysis</h2>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Sentiment Score:</label>
                        <span class="sentiment-{sentiment_class}">
                            {sentiment_score:+.2f}
                        </span>
                    </div>
                    <div class="detail-item">
                        <label>Sentiment Confidence:</label>
                        <span>{sentiment_confidence}</span>
                    </div>
                    <div class="detail-item">
                        <label>Sentiment Signal:</label>
                        <span>{sentiment_signal}</span>
                    </div>
                </div>
            </div>
            
            <div class="detail-card">
                <h2>🤖 Model Predictions</h2>
                <div class="model-predictions">
                    {predictions_html if predictions_html else '<div class="no-predictions">No individual model predictions available</div>'}
                </div>
            </div>

            <div class="detail-card">
                <h2>🎯 Model Training Performance</h2>
                {training_html if training_html else '<div class="no-data">No training metrics available</div>'}
                {training_metrics_table if training_metrics else '<div class="no-data">No training metrics available</div>'}
            </div>
            
            {chart_html if chart_html else '<div class="no-chart">No chart data available</div>'}
            
            <div class="detail-card">
                <h2>🔗 Quick Links</h2>
                <div class="quick-links">
                    <a href="https://finance.yahoo.com/quote/{ticker}" target="_blank" class="external-link">Yahoo Finance</a>
                    <a href="https://www.tradingview.com/symbols/{ticker}" target="_blank" class="external-link">TradingView</a>
                    <a href="https://www.google.com/finance/quote/{ticker}" target="_blank" class="external-link">Google Finance</a>
                </div>
            </div>
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
/* Add to existing CSS */
.metrics-section {
    margin-top: 30px;
}

.metrics-section h3 {
    margin-bottom: 20px;
    color: #00ccff;
    font-size: 1.2em;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 15px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.08);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.metric-header h4 {
    color: #fff;
    font-size: 1.1em;
    margin: 0;
}

.model-badge {
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    color: #000;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    font-weight: bold;
}

.metric-values {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
}

.metric-label {
    color: #ccc;
    font-size: 0.9em;
}

.metric-value {
    font-weight: bold;
    font-size: 0.95em;
}

.metric-high {
    color: #00ff99;
}

.metric-medium {
    color: #ffaa00;
}

.metric-low {
    color: #ff4444;
}

.no-data, .no-predictions {
    text-align: center;
    padding: 40px;
    color: #888;
    font-style: italic;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
}

/* Enhanced table styles */
.metrics-table {
    margin-top: 20px;
    overflow-x: auto;
}

.metrics-table table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    overflow: hidden;
}

.metrics-table th {
    background: rgba(0, 204, 255, 0.2);
    padding: 12px 15px;
    text-align: left;
    font-weight: 600;
    color: #00ccff;
}

.metrics-table td {
    padding: 10px 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.metrics-table tr:hover {
    background: rgba(255, 255, 255, 0.08);
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