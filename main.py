import os
import pandas as pd
from tqdm import tqdm
from analysis import TradingModelSystem
import alpaca_trader
import api_call
from alpaca.trading.enums import TimeInForce  

class TradingExecutor:
    def __init__(self):
        self.analyzer = TradingModelSystem()
        self.ticker_list = [
            "SPY", "QQQ",  # ETFs first
            "AAPL", "MSFT", "AMZN", "GOOG", "TSLA",  # Core tech
            "JPM", "BAC", "XOM", "CVX"  # Financials/energy
        ]
        self.max_trades_per_day = 5
        self.current_trades = 0

    def should_process_ticker(self, ticker):
        """Check if we have sufficient data for analysis"""
        data_file = f"data/{ticker}_data.csv"
        if not os.path.exists(data_file):
            return True
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        return len(df) >= 100  # Minimum data points required

    def execute_strategy(self, ticker):
        """Full pipeline for a single ticker"""
        try:
            # 1. Data Collection - ensure we have required columns
            if self.should_process_ticker(ticker):
                print(f"Fetching data for {ticker}")
                api_call.get_data(ticker)
            
            # Verify data has at least adj_close
            data_path = f"data/{ticker}_data.csv"
            if not os.path.exists(data_path):
                print(f"No data file found for {ticker}")
                return
                
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if 'adj_close' not in df.columns:
                print(f"No adj_close column in data for {ticker}")
                return
            
            # 2. Analysis and Prediction
            results, metrics, predictions = self.analyzer.run_analysis(ticker)
            
            # 3. Strategy Selection
            if metrics is None:
                print(f"No metrics generated for {ticker}")
                return
                
            metrics_df = metrics.copy()
            buy_hold_return = metrics_df.loc["Buy & Hold", "Return (%)"]
            
            # Filter out underperforming strategies
            viable_strategies = metrics_df[metrics_df["Return (%)"] > buy_hold_return]
            if viable_strategies.empty:
                print(f"No strategies outperform Buy & Hold for {ticker}")
                return
            
            # Select best strategy
            best_strategy = viable_strategies["Return (%)"].idxmax()
            strategy_prediction = predictions.get(best_strategy, {})
            
            if not strategy_prediction or self.current_trades >= self.max_trades_per_day:
                return
            
            # 4. Trade Execution
            signal = strategy_prediction.get("signal", "HOLD")
            if signal in ["BUY", "SELL"]:
                pct_diff = strategy_prediction.get("pct_diff", 0)
                last_price = df['adj_close'].iloc[-1]
                
                # Position sizing based on strategy confidence
                strategy_return = metrics_df.loc[best_strategy, "Return (%)"]
                best_return = metrics_df["Return (%)"].max()
                confidence = min(strategy_return / best_return, 1.0)
                
                
                qty = alpaca_trader.qty_to_trade(
                    ticker,
                    signal,
                    confidence,
                    last_price=last_price,
                    predicted_diff=pct_diff
                )
                
                if qty > 0:
                    alpaca_trader.make_trade(
                        ticker,
                        signal,
                        qty,
                        time_in_force=TimeInForce.GTC,      
                        stop_loss_pct=0.03                 
                    )
                    self.current_trades += 1
                    print(f"Executed {signal} for {ticker} (Qty: {qty})")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    def run_daily_trading(self):
        """Main execution loop"""
        print("Starting daily trading process...")
        for ticker in tqdm(self.ticker_list, desc="Processing Tickers"):
            self.execute_strategy(ticker)
        print(f"Trading complete. Executed {self.current_trades} trades today.")

if __name__ == "__main__":
    trader = TradingExecutor()
    trader.run_daily_trading()
