# visualize_results.py
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "results"

def visualize_backtest_chart(tickers):
    if isinstance(tickers, str):
        tickers = [tickers]

    out_paths = []
    for ticker in tickers:
        metrics_path = os.path.join("saved_models", f"{ticker}_metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"No metrics file for {ticker}")
            continue
        metrics_df = pd.read_csv(metrics_path)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{ticker} Backtest Predictions", "Equity Curve")
        )

        # Plot all models’ predictions
        for model_name in ["LSTM", "Dense NN", "Random Forest", "XGBoost"]:
            csv_path = os.path.join("saved_models", f"{ticker}_{model_name}_backtest.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")

            # Actual price as candlesticks (only once)
            if model_name == "LSTM":
                fig.add_trace(go.Candlestick(
                    x=df['Date'], open=df['TruePrice'], high=df['TruePrice'],
                    low=df['TruePrice'], close=df['TruePrice'], name="True Price"
                ), row=1, col=1)

            # Predicted line
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['PredictedPrice'], mode="lines",
                name=f"{model_name} Prediction"
            ), row=1, col=1)

            # Buy/Sell markers
            if "Signal" in df.columns:
                buy_df = df[df["Signal"] == "BUY"]
                sell_df = df[df["Signal"] == "SELL"]
                fig.add_trace(go.Scatter(
                    x=buy_df['Date'], y=buy_df['TruePrice'], mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color="green"),
                    name=f"{model_name} BUY"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=sell_df['Date'], y=sell_df['TruePrice'], mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color="red"),
                    name=f"{model_name} SELL"
                ), row=1, col=1)

            # Equity curve (if present)
            if "PortfolioValue" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'], y=df['PortfolioValue'], mode="lines",
                    name=f"{model_name} Equity"
                ), row=2, col=1)

        fig.update_layout(template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)

        # Save interactive HTML with metrics table
        metrics_html = metrics_df.to_html(index=False, classes="table table-dark table-striped")
        out_file = os.path.join(RESULTS_DIR, f"{ticker}_backtest.html")
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        with open(out_file, "w") as f:
            f.write("<h1>Model Backtest & Metrics</h1>")
            f.write(fig_html)
            f.write("<h2>Metrics</h2>")
            f.write(metrics_html)

        
        out_pngs= os.path.join(RESULTS_DIR, f"{ticker}_backtest.png")
        fig.write_image(out_pngs, width=1200, height=800)

        out_paths.append(out_pngs)

    return out_paths


def visualize_predictions_chart(tickers=None):
    """
    Generate prediction charts (candlestick + buy/sell signals) for one or multiple tickers.
    Returns dict {ticker: path_to_png}.
    """
    chart_paths = {}

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_history.csv"):
            ticker = file.replace("_history.csv", "")

            # If a list of tickers was provided, skip the rest
            if tickers and ticker not in tickers:
                continue

            csv_path = os.path.join(RESULTS_DIR, f"{ticker}_history.csv")
            if not os.path.exists(csv_path):
                print(f"No historical data for {ticker}")
                continue

            # Robust CSV reading
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
            except ValueError:
                df = pd.read_csv(csv_path)
                date_col = "Date" if "Date" in df.columns else "date"
                df[date_col] = pd.to_datetime(df[date_col])
                df.rename(columns={date_col: "date"}, inplace=True)

            df.sort_values("date", inplace=True)

            fig = go.Figure(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ))

            if "signal" in df.columns:
                buy_df = df[df["signal"] == "BUY"]
                sell_df = df[df["signal"] == "SELL"]

                fig.add_trace(go.Scatter(
                    x=buy_df['date'],
                    y=buy_df['close'],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    name="BUY Signal"
                ))
                fig.add_trace(go.Scatter(
                    x=sell_df['date'],
                    y=sell_df['close'],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                    name="SELL Signal"
                ))

            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                xaxis_rangeslider_visible=False,
                title=f"{ticker} Predictions"
            )

            out_path = os.path.join(RESULTS_DIR, f"{ticker}_predictions.png")
            fig.write_image(out_path, width=1200, height=800)
            chart_paths[ticker] = out_path
            print(f"Prediction chart saved: {out_path}")

    return chart_paths


def visualize_results(tickers=None):
    """
    Generate price/signal charts for one or multiple tickers.
    Returns a dict {ticker: path_to_chart}.
    """
    chart_paths = {}

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_summary.json"):
            ticker = file.replace("_summary.json", "")

            # If user passed a list of tickers, skip others
            if tickers and ticker not in tickers:
                continue

            csv_path = os.path.join(RESULTS_DIR, f"{ticker}_history.csv")
            if not os.path.exists(csv_path):
                print(f"No historical data found for {ticker}")
                continue

            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
            except ValueError:
                # Fallback in case date column name is different (e.g., 'Date')
                df = pd.read_csv(csv_path)
                date_col = "Date" if "Date" in df.columns else "date"
                df[date_col] = pd.to_datetime(df[date_col])
                df.rename(columns={date_col: "date"}, inplace=True)

            df.sort_values("date", inplace=True)

            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                )
            ])

            if "signal" in df.columns:
                buy_df = df[df["signal"] == "BUY"]
                sell_df = df[df["signal"] == "SELL"]

                fig.add_trace(go.Scatter(
                    x=buy_df['date'],
                    y=buy_df['close'],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    name="BUY Signal"
                ))

                fig.add_trace(go.Scatter(
                    x=sell_df['date'],
                    y=sell_df['close'],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                    name="SELL Signal"
                ))

            fig.update_layout(
                title=f"{ticker} Price & Signals",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                hovermode="x unified"
            )

            out_file = os.path.join(RESULTS_DIR, f"{ticker}_plot.html")
            fig.write_html(out_file)
            chart_paths[ticker] = out_file
            print(f"Visualization saved: {out_file}")

    return chart_paths


def visualize_comprehensive(tickers=None):
    html_paths = {}

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_history.csv"):
            ticker = file.replace("_history.csv", "")
            if tickers and ticker not in tickers:
                continue

            csv_path = os.path.join(RESULTS_DIR, f"{ticker}_history.csv")
            if not os.path.exists(csv_path):
                print(f"No historical data for {ticker}")
                continue

            # Handle date column name issues
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
            except ValueError:
                df = pd.read_csv(csv_path)
                date_col = "Date" if "Date" in df.columns else "date"
                df[date_col] = pd.to_datetime(df[date_col])
                df.rename(columns={date_col: "date"}, inplace=True)

            df.sort_values("date", inplace=True)

            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(
                    f"{ticker} Price & Signals",
                    "Predicted vs True Price",
                    "Portfolio Value"
                )
            )

            # 1️⃣ Candlestick + signals
            fig.add_trace(go.Candlestick(
                x=df['date'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name="Price"
            ), row=1, col=1)

            if "signal" in df.columns:
                buy_df = df[df["signal"] == "BUY"]
                sell_df = df[df["signal"] == "SELL"]

                fig.add_trace(go.Scatter(
                    x=buy_df['date'], y=buy_df['close'], mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                    name="BUY Signal"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=sell_df['date'], y=sell_df['close'], mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                    name="SELL Signal"
                ), row=1, col=1)

            # 2️⃣ Predicted vs True Price
            if "TruePrice" in df.columns and "PredictedPrice" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['TruePrice'], mode="lines", name="True Price"
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['PredictedPrice'], mode="lines", name="Predicted Price"
                ), row=2, col=1)

            # 3️⃣ Portfolio value
            if "PortfolioValue" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df['PortfolioValue'], mode="lines", name="Portfolio Value"
                ), row=3, col=1)

            # Layout tweaks
            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                xaxis_rangeslider_visible=False,
                height=900,
                showlegend=True
            )

            # Save HTML
            out_path = os.path.join(RESULTS_DIR, f"{ticker}_comprehensive.html")
            fig.write_html(out_path)
            html_paths[ticker] = out_path
            print(f"Comprehensive HTML saved: {out_path}")

    return html_paths


if __name__ == "__main__":
    visualize_results()
    visualize_backtest("SPY")