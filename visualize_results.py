# visualize_results.py
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "results"

def visualize_backtest_chart(ticker):
    csv_path = os.path.join(RESULTS_DIR, f"{ticker}_backtest.csv")
    if not os.path.exists(csv_path):
        print(f"No backtest data found for {ticker}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} Backtest Candlestick", "Equity Curve")
    )

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price'
    ), row=1, col=1)

    if "signal" in df.columns:
        buy_df = df[df["signal"] == "BUY"]
        sell_df = df[df["signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buy_df['date'], y=buy_df['close'], mode="markers",
                                 marker=dict(symbol="triangle-up", size=12, color="green"), name="BUY Signal"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_df['date'], y=sell_df['close'], mode="markers",
                                 marker=dict(symbol="triangle-down", size=12, color="red"), name="SELL Signal"), row=1, col=1)

    if "equity" in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['equity'], mode="lines", name="Equity", line=dict(color="blue")
        ), row=2, col=1)

    fig.update_layout(template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)

    out_path = os.path.join(RESULTS_DIR, f"{ticker}_backtest.png")
    fig.write_image(out_path, width=1200, height=800)
    return out_path


def visualize_predictions_chart(ticker):
    csv_path = os.path.join(RESULTS_DIR, f"{ticker}_history.csv")
    if not os.path.exists(csv_path):
        print(f"No historical data for {ticker}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")

    fig = go.Figure(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"
    ))

    if "signal" in df.columns:
        buy_df = df[df["signal"] == "BUY"]
        sell_df = df[df["signal"] == "SELL"]
        fig.add_trace(go.Scatter(x=buy_df['date'], y=buy_df['close'], mode="markers",
                                 marker=dict(symbol="triangle-up", size=12, color="green"), name="BUY Signal"))
        fig.add_trace(go.Scatter(x=sell_df['date'], y=sell_df['close'], mode="markers",
                                 marker=dict(symbol="triangle-down", size=12, color="red"), name="SELL Signal"))

    fig.update_layout(template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)

    out_path = os.path.join(RESULTS_DIR, f"{ticker}_predictions.png")
    fig.write_image(out_path, width=1200, height=800)
    return out_path

def visualize_results():
    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_summary.json"):
            ticker = file.replace("_summary.json", "")

            # Look for a CSV with price history and signals
            csv_path = os.path.join(RESULTS_DIR, f"{ticker}_history.csv")
            if not os.path.exists(csv_path):
                print(f"No historical data found for {ticker}")
                continue

            df = pd.read_csv(csv_path, parse_dates=["date"])
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

            # Add buy signals
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
            print(f"Visualization saved: {out_file}")

if __name__ == "__main__":
    visualize_results()
    visualize_backtest("SPY")