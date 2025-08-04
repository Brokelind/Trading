import pandas as pd
import logging
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest
import os
import yfinance as yf

# only for local use
try:
    import env
except ImportError:
    env = None
    
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY") or getattr(env, "ALPACA_API_KEY", None)
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY") or getattr(env, "ALPACA_SECRET_KEY", None)
# --------------------
# Setup Logging
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --------------------
# Initialize Client
# --------------------
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# --------------------
# Utility Functions
# --------------------

def get_account_info():
    try:
        account = client.get_account()
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        return cash, buying_power
    except Exception as e:
        log.error(f"Failed to get account info: {e}")
        return 0, 0

def get_position_qty(symbol):
    try:
        position = client.get_position(symbol)
        return int(position.qty)
    except Exception:
        return 0

def qty_to_trade(
    ticker, signal, performance_index, last_price,
    predicted_diff, risk_amount=0.1, min_position_value=50
):
    """
    Determine how many shares to trade for a given signal.
    """
    cash, _ = get_account_info()

    if signal == "HOLD" or performance_index <= 0:
        return 0

    # Clamp predicted_diff to [-0.3, +0.3] range
    predicted_diff = max(min(predicted_diff, 0.3), -0.3)

    perf_adj = min(performance_index, 3)

    risk_capital = cash * risk_amount * perf_adj * (1 + predicted_diff)

    # Add 0.2% slippage cushion
    effective_price = last_price * 1.002

    qty = int(risk_capital // effective_price)

    if qty * last_price < min_position_value:
        log.info(f"Skipping {ticker}. Position size too small: ${qty * last_price:.2f}")
        return 0

    return max(qty, 0)

def get_latest_price_from_csv(ticker, folder="data"):
    path = os.path.join(folder, f"{ticker}_data.csv")
    if not os.path.exists(path):
        log.error(f"Data file not found for {ticker} at {path}")
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        latest_price = df["adj_close"].dropna().iloc[-1]
        return float(latest_price)
    except Exception as e:
        log.error(f"Failed to read latest price from {path}: {e}")
        return None


# --------------------
# Main Trading Function
# --------------------

def make_trade(
    ticker, signal, qty,
    limit_slippage_pct=0.002,
    stop_loss_pct=None,
    time_in_force=TimeInForce.DAY
):
    """
    Place a trade for a ticker based on the signal and qty.
    Supports limit orders and optional stop-loss.
    """

    if signal == "HOLD" or qty == 0:
        log.info(f"HOLD for {ticker}")
        return

    try:
        # ------------------------
        # Check current position
        # ------------------------
        positions = client.get_all_positions()
        position = next((p for p in positions if p.symbol == ticker), None)
        current_qty = float(position.qty) if position else 0

        # ------------------------
        # Check pending orders
        # ------------------------
        pending_orders = client.get_orders()
        pending_sell_qty = 0
        pending_buy_qty = 0

        for order in pending_orders:
            if order.symbol == ticker:
                if order.side == OrderSide.SELL:
                    pending_sell_qty += float(order.qty)
                elif order.side == OrderSide.BUY:
                    pending_buy_qty += float(order.qty)

        # ------------------------
        # Decide Side
        # ------------------------
        side = OrderSide.BUY if signal == "BUY" else OrderSide.SELL

        if signal == "SELL":
            available_to_sell = current_qty - pending_sell_qty
            if available_to_sell < qty:
                adjusted_qty = max(0, available_to_sell)
                if adjusted_qty == 0:
                    log.warning(
                        f"Cannot SELL {qty} of {ticker}. "
                        f"No available shares (Current: {current_qty}, Pending sells: {pending_sell_qty})"
                    )
                    return
                else:
                    log.info(
                        f"Adjusting SELL qty from {qty} to {adjusted_qty} due to limited shares."
                    )
                    qty = adjusted_qty

        elif signal == "BUY":
            if pending_buy_qty > 0:
                log.warning(
                    f"Cannot BUY {qty} of {ticker}. "
                    f"Already have pending BUY orders for {pending_buy_qty} shares."
                )
                return

        # ------------------------
        # Create Order
        # ------------------------
        # ------------------------
        # Get latest market price from local data
        # ------------------------
        last_price = get_latest_price_from_csv(ticker)
        if last_price is None or last_price <= 0:
            log.error(f"Cannot trade {ticker}: no valid price in saved data.")
            return


        # Compute limit price (e.g. 0.2% slippage)
        limit_price = round(
            last_price * (1 + limit_slippage_pct * (1 if side == OrderSide.BUY else -1)),
            2
        )

        # ------------------------
        # Prepare Stop Loss if needed
        # ------------------------
        stop_loss = None
        if stop_loss_pct and signal == "BUY":
            stop_price = round(last_price * (1 - stop_loss_pct), 2)
            stop_loss = StopLossRequest(stop_price=stop_price)
            log.info(f"Setting stop-loss at ${stop_price:.2f}")

        # ------------------------
        # Submit Limit Order
        # ------------------------
        try:
            order_data = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=time_in_force,
                limit_price=limit_price,
                order_class=OrderClass.SIMPLE,
                stop_loss=stop_loss
            )
            order = client.submit_order(order_data=order_data)
            log.info(f"Submitted LIMIT order: {signal} {qty} {ticker} @ {limit_price}")
        except Exception as e:
            log.warning(f"Limit order failed for {ticker}: {e}")
            # Optionally fallback to market order
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=time_in_force,
            )
            order = client.submit_order(order_data=order_data)
            log.info(f"Fallback to MARKET order: {signal} {qty} {ticker}")

    except Exception as e:
        log.error(f"Error placing order for {ticker}: {e}")

# --------------------
# Example Usage
# --------------------

if __name__ == "__main__":
    # Example strategy output
    ticker = "AAPL"
    signal = "BUY"         # "BUY", "SELL", or "HOLD"
    performance_index = 2  # Some score from your model
    last_price = 190.50
    predicted_diff = 0.05  # E.g. +5%

    qty = qty_to_trade(ticker, signal, performance_index, last_price, predicted_diff)

    make_trade(
        ticker,
        signal,
        qty,
        limit_slippage_pct=0.002,
        stop_loss_pct=0.03,    # e.g. 3% stop loss
        time_in_force=TimeInForce.DAY
    )
