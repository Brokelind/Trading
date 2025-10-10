# visualize_results.py
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

RESULTS_DIR = "results"

def visualize_backtest_chart(tickers):
    if isinstance(tickers, str):
        tickers = [tickers]

    print("Generating Enhanced Backtest Charts...")
    out_paths = []
    
    MODEL_NAMES = ["LSTM", "Dense NN", "Random Forest", "XGBoost", "Ensemble"]
    MODEL_COLORS = {
        "LSTM": "#FF6B6B",
        "Dense NN": "#4ECDC4", 
        "Random Forest": "#45B7D1",
        "XGBoost": "#96CEB4",
        "Ensemble": "#FFEAA7"
    }
    
    for ticker in tickers:
        # Load metrics from JSON if available, otherwise from CSV
        json_path = os.path.join(RESULTS_DIR, f"{ticker}_summary.json")
        metrics_df = None
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                summary_data = json.load(f)
                # Convert metrics to DataFrame
                if 'backtest_metrics' in summary_data:
                    metrics_df = pd.DataFrame(summary_data['backtest_metrics'])
        else:
            metrics_path = os.path.join("saved_models", f"{ticker}_metrics.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
        
        if metrics_df is None or metrics_df.empty:
            print(f"No metrics data for {ticker}")
            continue
            
        # Create comprehensive dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=(
                f"{ticker} - Price Predictions vs Actual",
                "Portfolio Performance Comparison",
                "Model Sharpe Ratios",
                "Direction Accuracy",
                "Prediction Error Distribution",
                "Trading Signals Distribution"
            ),
            specs=[
                [{"colspan": 2}, None],  # First row: full width price chart
                [{"type": "xy"}, {"type": "bar"}],  # Second row: portfolio + sharpe
                [{"type": "bar"}, {"type": "pie"}]   # Third row: accuracy + signals
            ],
            row_heights=[0.5, 0.25, 0.25]
        )

        # 1. Price Predictions Chart (Top - Full Width)
        true_price_added = False
        portfolio_data = []
        sharpe_data = []
        accuracy_data = []
        error_data = []
        signal_data = []
        
        for model_name in MODEL_NAMES:
            csv_path = os.path.join("saved_models", f"{ticker}_{model_name}_backtest.csv")
            if not os.path.exists(csv_path):
                continue
                
            df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
            
            # Add true price only once
            if not true_price_added:
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['TruePrice'], 
                    mode="lines", line=dict(width=2, color="#FFFFFF"),
                    name="True Price", opacity=0.7
                ), row=1, col=1)
                true_price_added = True

            # Add model predictions
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['PredictedPrice'], 
                mode="lines", line=dict(width=1.5, dash='dot'),
                name=f"{model_name} Pred", opacity=0.8,
                line_color=MODEL_COLORS.get(model_name, "#888888")
            ), row=1, col=1)

            # Add trading signals
            if "Signal" in df.columns:
                buy_df = df[df["Signal"] == "BUY"]
                sell_df = df[df["Signal"] == "SELL"]
                
                if not buy_df.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_df['Date'], y=buy_df['TruePrice'], 
                        mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"),
                        name=f"{model_name} BUY", showlegend=False,
                        opacity=0.7
                    ), row=1, col=1)
                if not sell_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_df['Date'], y=sell_df['TruePrice'], 
                        mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
                        name=f"{model_name} SELL", showlegend=False,
                        opacity=0.7
                    ), row=1, col=1)
            
            # Collect data for other charts
            if "PortfolioValue" in df.columns:
                portfolio_data.append({
                    'model': model_name,
                    'final_value': df['PortfolioValue'].iloc[-1],
                    'returns': (df['PortfolioValue'].iloc[-1] - 10000) / 10000 * 100,
                    'color': MODEL_COLORS.get(model_name, "#888888")
                })
            
            # Get metrics for this model
            model_metrics = metrics_df[metrics_df['Model'] == model_name]
            if not model_metrics.empty:
                sharpe = model_metrics['Sharpe'].iloc[0] if 'Sharpe' in model_metrics.columns else 0
                direction_acc = model_metrics['DirectionAcc'].iloc[0] if 'DirectionAcc' in model_metrics.columns else 0
                mae = model_metrics['MAE'].iloc[0] if 'MAE' in model_metrics.columns else 0
                
                sharpe_data.append({'model': model_name, 'sharpe': sharpe, 'color': MODEL_COLORS.get(model_name, "#888888")})
                accuracy_data.append({'model': model_name, 'accuracy': direction_acc * 100, 'color': MODEL_COLORS.get(model_name, "#888888")})
                error_data.append({'model': model_name, 'mae': mae, 'color': MODEL_COLORS.get(model_name, "#888888")})
            
            # Count signals
            if "Signal" in df.columns:
                signal_counts = df['Signal'].value_counts()
                signal_data.append({
                    'model': model_name,
                    'buy': signal_counts.get('BUY', 0),
                    'sell': signal_counts.get('SELL', 0),
                    'hold': signal_counts.get('HOLD', 0)
                })

        # 2. Portfolio Performance Chart
        if portfolio_data:
            models = [d['model'] for d in portfolio_data]
            returns = [d['returns'] for d in portfolio_data]
            colors = [d['color'] for d in portfolio_data]
            
            fig.add_trace(go.Bar(
                x=models, y=returns,
                marker_color=colors,
                name="Total Return %",
                text=[f"{r:.1f}%" for r in returns],
                textposition='auto',
            ), row=2, col=1)

        # 3. Sharpe Ratios Chart
        if sharpe_data:
            models = [d['model'] for d in sharpe_data]
            sharpes = [d['sharpe'] for d in sharpe_data]
            colors = [d['color'] for d in sharpe_data]
            
            fig.add_trace(go.Bar(
                x=models, y=sharpes,
                marker_color=colors,
                name="Sharpe Ratio",
                text=[f"{s:.2f}" for s in sharpes],
                textposition='auto',
            ), row=2, col=2)

        # 4. Direction Accuracy Chart
        if accuracy_data:
            models = [d['model'] for d in accuracy_data]
            accuracies = [d['accuracy'] for d in accuracy_data]
            colors = [d['color'] for d in accuracy_data]
            
            fig.add_trace(go.Bar(
                x=models, y=accuracies,
                marker_color=colors,
                name="Direction Accuracy %",
                text=[f"{a:.1f}%" for a in accuracies],
                textposition='auto',
            ), row=3, col=1)

        # 5. Signal Distribution Pie Chart (for best model)
        if signal_data and portfolio_data:
            # Find best performing model
            best_model_data = max(portfolio_data, key=lambda x: x['final_value'])
            best_model = best_model_data['model']
            best_signals = next((s for s in signal_data if s['model'] == best_model), None)
            
            if best_signals:
                labels = ['BUY', 'SELL', 'HOLD']
                values = [best_signals['buy'], best_signals['sell'], best_signals['hold']]
                colors_pie = ['#00FF00', '#FF0000', '#888888']
                
                fig.add_trace(go.Pie(
                    labels=labels, values=values,
                    marker_colors=colors_pie,
                    name=f"{best_model} Signals",
                    hole=0.4
                ), row=3, col=2)

        # Update layout
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            height=1200,
            showlegend=True,
            title_text=f"{ticker} - Backtest Trading Model Analysis",
            font=dict(size=10)
        )

        # Update subplot titles and axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return %", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy %", row=3, col=1)

        # Generate enhanced metrics table
        if not metrics_df.empty:
            # Add performance indicators
            metrics_df['Return_Pct'] = ((metrics_df['FinalPortfolio'] - 10000) / 10000 * 100).round(2)
            metrics_df['DirectionAcc_Pct'] = (metrics_df['DirectionAcc'] * 100).round(2)
            metrics_df['Sharpe_Rounded'] = metrics_df['Sharpe'].round(3)
            metrics_df['MAE_Rounded'] = metrics_df['MAE'].round(4)
            
            display_columns = ['Model', 'Return_Pct', 'Sharpe_Rounded', 'DirectionAcc_Pct', 'MAE_Rounded']
            display_df = metrics_df[display_columns].rename(columns={
                'Return_Pct': 'Return %',
                'Sharpe_Rounded': 'Sharpe',
                'DirectionAcc_Pct': 'Direction Acc %',
                'MAE_Rounded': 'MAE'
            })

        # Save comprehensive HTML report
        out_file = os.path.join(RESULTS_DIR, f"{ticker}_backtest_dashboard.html")
        
        try:
            with open(out_file, "w", encoding='utf-8') as f:
                # Write simplified HTML without complex CSS
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Model Dashboard - {ticker}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #1a1a1a; color: white; }}
        .dashboard-container {{ padding: 20px; }}
        .metrics-table {{ background-color: #2d2d2d; }}
        .performance-card {{ background-color: #2d2d2d; padding: 15px; margin: 10px; border-radius: 10px; }}
        .signal-buy {{ color: green; font-weight: bold; }}
        .signal-sell {{ color: red; font-weight: bold; }}
        .signal-hold {{ color: gray; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <h1 class="text-center mb-4">Trading Model Dashboard - {ticker}</h1>
""")
                
                # Add summary cards if JSON data available
                if os.path.exists(json_path):
                    with open(json_path, 'r') as jf:
                        summary = json.load(jf)
                        current_price = summary.get('last_price', 'N/A')
                        ensemble_signal = summary.get('signal', 'N/A')
                        chosen_model = summary.get('chosen_model', 'N/A')
                        pct_diff = summary.get('pct_diff', 0)
                        
                        signal_class = "signal-buy" if ensemble_signal == "BUY" else "signal-sell" if ensemble_signal == "SELL" else "signal-hold"
                        pct_class = "signal-buy" if pct_diff > 0 else "signal-sell" if pct_diff < 0 else "signal-hold"
                        
                        f.write(f"""
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="performance-card">
                    <h5>Current Price</h5>
                    <h3>${current_price}</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="performance-card">
                    <h5>Current Signal</h5>
                    <h3 class="{signal_class}">{ensemble_signal}</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="performance-card">
                    <h5>Selected Model</h5>
                    <h3>{chosen_model}</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="performance-card">
                    <h5>Predicted Change</h5>
                    <h3 class="{pct_class}">{pct_diff:.2f}%</h3>
                </div>
            </div>
        </div>
""")

                # Add the main chart
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                
                # Add metrics table
                f.write("""
        <div class="row mt-4">
            <div class="col-12">
                <h3>Backtesting Model Performance Metrics</h3>
                <div class="table-responsive">
""")
                f.write(display_df.to_html(classes='table table-dark table-striped', index=False))
                f.write("""
                </div>
            </div>
        </div>
    </div>
</body>
</html>
""")
            print(f"Generated enhanced dashboard for {ticker}: {out_file}")
            out_paths.append(out_file)
            
        except Exception as e:
            print(f"Error creating dashboard for {ticker}: {e}")
            # Fallback: create simple chart
            create_simple_chart(ticker, metrics_df, MODEL_NAMES)
            out_paths.append(os.path.join(RESULTS_DIR, f"{ticker}_backtest_simple.html"))

    return out_paths

def create_simple_chart(ticker, metrics_df, model_names):
    """Fallback function for systems with encoding issues"""
    try:
        fig = go.Figure()
        
        # Simple bar chart with returns
        if not metrics_df.empty and 'FinalPortfolio' in metrics_df.columns:
            returns = ((metrics_df['FinalPortfolio'] - 10000) / 10000 * 100).round(2)
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=returns,
                marker_color='blue',
                text=returns,
                texttemplate='%{text:.1f}%',
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"{ticker} - Model Returns (%)",
                xaxis_title="Model",
                yaxis_title="Return %",
                template="plotly_white"
            )
            
            out_file = os.path.join(RESULTS_DIR, f"{ticker}_backtest_simple.html")
            fig.write_html(out_file)
            print(f"Generated simple chart for {ticker}")
            
    except Exception as e:
        print(f"Could not create simple chart for {ticker}: {e}")
