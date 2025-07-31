import os
import pandas as pd
from tqdm import tqdm
from analysis import TradingModelSystem
import alpaca_trader
import call_market
from alpaca.trading.enums import TimeInForce  

class TradingExecutor:
    def __init__(self):
        self.analyzer = TradingModelSystem()
        self.ticker_list = [
            "SPY", "QQQ",  # ETFs first
            "AAPL", "MSFT", "AMZN", "GOOG", "TSLA",  # Core tech
            "JPM", "BAC", "XOM", "CVX"  # Financials/energy
        ]
        self.max_trades_per_day = 50
        self.current_trades = 0

    def should_process_ticker(self, ticker):
        """Check if we have sufficient data for analysis"""
        data_file = f"data/{ticker}_data.csv"
        if not os.path.exists(data_file):
            return True
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        return len(df) >= 100  # Minimum data points required

    def select_best_model(self, metrics_df, predictions, buy_hold_return):
        """
        Enhanced model selection considering both performance and prediction accuracy
        """
        # Filter out underperforming strategies
        viable_strategies = metrics_df[metrics_df["Return (%)"] > buy_hold_return]
        
        if viable_strategies.empty:
            print(f"No strategies outperform Buy & Hold")
            return None, None
        
        # Calculate composite score
        def calculate_composite_score(row):
            weights = {
                'Return (%)': 0.4,
                'Sharpe Ratio': 0.3,
                'Direction Accuracy (%)': 0.2,
                'Max Drawdown (%)': -0.1
            }
            
            score = 0
            for metric, weight in weights.items():
                if metric in row:
                    if metric == 'Max Drawdown (%)':
                        normalized = 1 - (row[metric] / 100)
                    else:
                        if metric == 'Direction Accuracy (%)':
                            normalized = row[metric] / 100
                        else:
                            col_min = metrics_df[metric].min()
                            col_max = metrics_df[metric].max()
                            normalized = (row[metric] - col_min) / (col_max - col_min) if col_max != col_min else 0.5
                    
                    score += normalized * weight
            
            return score
        
        # Apply minimum accuracy threshold if available
        if 'Direction Accuracy (%)' in viable_strategies.columns:
            accurate_strategies = viable_strategies[viable_strategies['Direction Accuracy (%)'] >= 55]
            viable_strategies = accurate_strategies if not accurate_strategies.empty else viable_strategies
        
        # Calculate and select best strategy
        viable_strategies['Composite Score'] = viable_strategies.apply(calculate_composite_score, axis=1)
        best_strategy = viable_strategies['Composite Score'].idxmax()
        best_metrics = viable_strategies.loc[best_strategy].to_dict()
        
        print(f"\nSelected Strategy: {best_strategy}")
        print(f"- Return: {best_metrics.get('Return (%)', 'N/A'):.1f}%")
        print(f"- Sharpe: {best_metrics.get('Sharpe Ratio', 'N/A'):.2f}")
        print(f"- Accuracy: {best_metrics.get('Direction Accuracy (%)', 'N/A'):.1f}%")
        print(f"- Mean absolute error: {best_metrics.get('MAE (%)', 'N/A'):.2f}%")

        return best_strategy, best_metrics

    def execute_strategy(self, ticker):
        """Full pipeline for a single ticker"""
        try:
            # 1. Data Collection
            if self.should_process_ticker(ticker):
                print(f"Fetching data for {ticker}")
                call_market.get_data(ticker)
            
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
            
            if metrics is None:
                print(f"No metrics generated for {ticker}")
                return
                
            # 3. Enhanced Strategy Selection
            buy_hold_return = metrics.loc["Buy & Hold", "Return (%)"]
            best_strategy, strategy_metrics = self.select_best_model(
                metrics, 
                predictions,
                buy_hold_return
            )
            
            if not best_strategy or self.current_trades >= self.max_trades_per_day:
                return
            
            # 4. Trade Execution
            strategy_prediction = predictions.get(best_strategy, {})
            
            signal = strategy_prediction.get("signal", "HOLD")
            print(f"Strategy signal for {ticker}: {signal}")

            if signal in ["BUY", "SELL"]:
                pct_diff = strategy_prediction.get("pct_diff", 0)
                last_price = df['adj_close'].iloc[-1]
                
                # Position sizing based on composite score
                confidence = min(strategy_metrics['Composite Score'], 1.0)
                
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

    print("Running script...")
    trader = TradingExecutor()
    trader.run_daily_trading()
