#!/usr/bin/env python3
"""Test retraining with improved models on AAPL ticker"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SKIP_TRAINING_ON_CI'] = 'False'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tradingmodelsystem import TradingModelSystem

tms = TradingModelSystem()
print("Testing AAPL with force retrain...")
sys.stdout.flush()

try:
    res = tms.ensure_trained("AAPL", force=True)
    status = {k: v.get("status", "N/A") for k, v in res.get("training_results", {}).items()}
    bt = res.get("backtest_metrics", [])
    print(f"\nStatus: {status}")
    print(f"Training time: {res.get('training_time', 0):.1f}s")
    print(f"Backtest metrics: {len(bt)} models")
    for m in bt:
        print(f"  {m.get('Model','?'):15s} | DirAcc={m.get('DirectionAcc',0):.3f} | FinalPortfolio=${m.get('FinalPortfolio',0):.0f} | Sharpe={m.get('Sharpe',0):.3f} | R2={m.get('R2',0):.3f}")
    # Save results for reference
    with open("test_retrain_result.json", "w") as f:
        json.dump({"status": status, "training_time": res.get("training_time"), "backtest_metrics": bt}, f, indent=2)
    print("\nDone")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()