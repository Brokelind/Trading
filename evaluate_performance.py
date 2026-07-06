#!/usr/bin/env python3
"""
Comprehensive Model Performance Evaluation
Analyses all summary JSON files in results/ and generates:
  1. Aggregated cross-ticker performance metrics by model type
  2. Best/worst performing tickers per model
  3. Model ranking & selection recommendations
  4. Portfolio-level simulated performance comparison
  5. CSV/JSON report exports
"""

import os, json, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

RESULTS_DIR = "results"
OUTPUT_DIR = "evaluation_reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all ticker summary files
# ---------------------------------------------------------------------------
def load_all_summaries():
    summaries = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith("_summary.json") or fname == "crypto_signals.json":
            continue
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)
        ticker = data.get("ticker", fname.replace("_summary.json", ""))
        summaries[ticker] = data
    return summaries

summaries = load_all_summaries()
TICKERS = sorted(summaries.keys())
print(f"Loaded {len(TICKERS)} ticker summaries")

# ---------------------------------------------------------------------------
# 1. AGGREGATE BACKTEST METRICS ACROSS ALL TICKERS
# ---------------------------------------------------------------------------
def aggregate_backtest_metrics(summaries):
    """
    Collect backtest_metrics for every ticker across all models.
    Returns a DataFrame with one row per (ticker, model) combination.
    """
    rows = []
    for ticker, data in summaries.items():
        bt_metrics = data.get("backtest_metrics", [])
        for m in bt_metrics:
            rows.append({
                "Ticker": ticker,
                "Model": m["Model"],
                "MAE": m.get("MAE", np.nan),
                "RMSE": m.get("RMSE", np.nan),
                "R2": m.get("R2", np.nan),
                "DirectionAcc": m.get("DirectionAcc", np.nan),
                "FinalPortfolio": m.get("FinalPortfolio", np.nan),
                "Sharpe": m.get("Sharpe", np.nan),
                "Volatility": m.get("Volatility", np.nan),
            })
    df = pd.DataFrame(rows)
    return df

bt_df = aggregate_backtest_metrics(summaries)
print(f"Backtest metrics rows: {len(bt_df)}")

# ---------------------------------------------------------------------------
# 2. MODEL PERFORMANCE SUMMARY (aggregate stats)
# ---------------------------------------------------------------------------
def compute_model_performance_summary(bt_df):
    """
    For each model type, compute aggregate stats across tickers.
    """
    metric_cols = ["MAE", "RMSE", "R2", "DirectionAcc", "FinalPortfolio", "Sharpe", "Volatility"]
    groups = bt_df.groupby("Model")
    
    agg_rows = []
    for model, grp in groups:
        row = {"Model": model}
        row["Count"] = len(grp)  # number of tickers with this model
        for mc in metric_cols:
            vals = grp[mc].dropna()
            if len(vals) > 0:
                row[f"{mc}_mean"] = vals.mean()
                row[f"{mc}_median"] = vals.median()
                row[f"{mc}_std"] = vals.std()
                row[f"{mc}_min"] = vals.min()
                row[f"{mc}_max"] = vals.max()
                row[f"{mc}_p25"] = vals.quantile(0.25)
                row[f"{mc}_p75"] = vals.quantile(0.75)
            else:
                row[f"{mc}_mean"] = np.nan
                row[f"{mc}_median"] = np.nan
                row[f"{mc}_std"] = np.nan
                row[f"{mc}_min"] = np.nan
                row[f"{mc}_max"] = np.nan
                row[f"{mc}_p25"] = np.nan
                row[f"{mc}_p75"] = np.nan
        agg_rows.append(row)
    
    agg_df = pd.DataFrame(agg_rows)
    # Sort by median Sharpe (best risk-adjusted return)
    agg_df = agg_df.sort_values("Sharpe_median", ascending=False).reset_index(drop=True)
    return agg_df

model_summary = compute_model_performance_summary(bt_df)
print("\n=== MODEL PERFORMANCE SUMMARY (sorted by median Sharpe) ===")
display_cols = ["Model", "Count", "DirectionAcc_mean", "R2_mean", "Sharpe_median", "FinalPortfolio_median", "MAE_mean"]
print(model_summary[display_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. BEST / WORST TICKERS PER MODEL
# ---------------------------------------------------------------------------
def best_worst_per_model(bt_df, metric="Sharpe", top_n=5):
    groups = bt_df.groupby("Model")
    results = {}
    for model, grp in groups:
        valid = grp.dropna(subset=[metric])
        best = valid.nlargest(top_n, metric)[["Ticker", metric]].reset_index(drop=True)
        worst = valid.nsmallest(top_n, metric)[["Ticker", metric]].reset_index(drop=True)
        results[model] = {"best": best, "worst": worst}
    return results

bw = best_worst_per_model(bt_df, "Sharpe")

print("\n=== TOP-5 TICKERS PER MODEL (by Sharpe ratio) ===")
for model, d in bw.items():
    print(f"\n  {model}:")
    print(f"    Best:  {', '.join(f'{r.Ticker} ({r.Sharpe:.2f})' for _, r in d['best'].iterrows())}")
    print(f"    Worst: {', '.join(f'{r.Ticker} ({r.Sharpe:.2f})' for _, r in d['worst'].iterrows())}")

# ---------------------------------------------------------------------------
# 4. BEST MODEL PER TICKER (chosen by the system)
# ---------------------------------------------------------------------------
def best_model_per_ticker(summaries):
    rows = []
    for ticker, data in summaries.items():
        best = data.get("chosen_model", "N/A")
        signal = data.get("signal", "N/A")
        pct = data.get("pct_diff", np.nan)
        rows.append({"Ticker": ticker, "ChosenModel": best, "Signal": signal, "PctDiff": pct})
    df = pd.DataFrame(rows)
    return df

chosen_df = best_model_per_ticker(summaries)
print("\n=== CHOSEN MODEL DISTRIBUTION ===")
print(chosen_df["ChosenModel"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 5. MODEL WIN RATE (which model has the best metrics most often)
# ---------------------------------------------------------------------------
def model_win_rates(bt_df):
    """
    For each metric, count how many tickers each model is the best.
    """
    metric_cols = ["Sharpe", "DirectionAcc", "R2", "FinalPortfolio"]
    results = {}
    for mc in metric_cols:
        valid = bt_df.dropna(subset=[mc]).copy()
        # For each ticker, find the model with max value
        idx = valid.groupby("Ticker")[mc].idxmax()
        winners = valid.loc[idx, "Model"]
        counts = winners.value_counts()
        results[mc] = counts
    return results

win_rates = model_win_rates(bt_df)
print("\n=== MODEL WIN RATES (how often each model is #1 per ticker) ===")
for metric, counts in win_rates.items():
    print(f"\n  {metric}:")
    if not counts.empty:
        total = counts.sum()
        for model, c in counts.items():
            print(f"    {model:20s}: {c:3d} tickers ({c/total*100:.0f}%)")

# ---------------------------------------------------------------------------
# 6. PORTFOLIO SIMULATION: if we had followed each model's signals
# ---------------------------------------------------------------------------
def simulate_model_portfolio(bt_df):
    """
    For each model, compute aggregate portfolio performance across all tickers.
    This simulates an equally-weighted portfolio following each model.
    """
    groups = bt_df.groupby("Model")
    results = []
    for model, grp in groups:
        valid = grp.dropna(subset=["FinalPortfolio", "Sharpe", "Volatility"])
        if len(valid) == 0:
            continue
        total_final = valid["FinalPortfolio"].sum()
        avg_sharpe = valid["Sharpe"].mean()
        avg_dir_acc = valid["DirectionAcc"].mean()
        avg_r2 = valid["R2"].mean()
        avg_mae = valid["MAE"].mean()
        win_count = (valid["FinalPortfolio"] > 10000).sum()
        total_count = len(valid)
        results.append({
            "Model": model,
            "TickersWithData": total_count,
            "TickersProfitable": win_count,
            "WinRate": win_count / total_count * 100,
            "TotalPortfolio": total_final,
            "AvgFinalPerTicker": total_final / len(valid),
            "AvgSharpe": avg_sharpe,
            "AvgDirectionAcc": avg_dir_acc,
            "AvgR2": avg_r2,
            "AvgMAE": avg_mae,
        })
    sim_df = pd.DataFrame(results)
    sim_df = sim_df.sort_values("TotalPortfolio", ascending=False).reset_index(drop=True)
    return sim_df

sim_df = simulate_model_portfolio(bt_df)
print("\n=== PORTFOLIO SIMULATION (equal-weighted across all tickers) ===")
print(sim_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. DIRECTION ACCURACY ANALYSIS
# ---------------------------------------------------------------------------
def direction_accuracy_analysis(bt_df):
    valid = bt_df.dropna(subset=["DirectionAcc"]).copy()
    
    # Summary stats
    print("\n=== DIRECTION ACCURACY STATISTICS ===")
    groups = valid.groupby("Model")["DirectionAcc"]
    for model, vals in groups:
        print(f"  {model:20s}: mean={vals.mean():.3f} median={vals.median():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")
    
    # Count how many tickers have DirectionAcc > 0.5 (better than random)
    print("\n  Models beating random (DirectionAcc > 0.50):")
    for model, grp in valid.groupby("Model"):
        beat_random = (grp["DirectionAcc"] > 0.50).sum()
        total = len(grp)
        pct = beat_random / total * 100
        print(f"    {model:20s}: {beat_random:3d}/{total} ({pct:.0f}%)")
    
    return valid

dir_acc_df = direction_accuracy_analysis(bt_df)

# ---------------------------------------------------------------------------
# 8. SIGNAL DISTRIBUTION
# ---------------------------------------------------------------------------
def signal_distribution(summaries):
    from collections import Counter
    all_signals = Counter()
    for ticker, data in summaries.items():
        signal = data.get("signal", "N/A")
        all_signals[signal] += 1
    print("\n=== FINAL SIGNAL DISTRIBUTION (across all tickers) ===")
    for sig, cnt in all_signals.most_common():
        print(f"  {sig:6s}: {cnt:3d} ({cnt/len(summaries)*100:.0f}%)")
    return all_signals

sig_dist = signal_distribution(summaries)

# ---------------------------------------------------------------------------
# 9. EXPORT REPORTS
# ---------------------------------------------------------------------------
# Full backtest metrics CSV
bt_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_metrics_all.csv"), index=False)
print(f"\n   Exported: backtest_metrics_all.csv")

# Model summary CSV
model_summary.to_csv(os.path.join(OUTPUT_DIR, "model_performance_summary.csv"), index=False)
print(f"   Exported: model_performance_summary.csv")

# Portfolio simulation CSV
sim_df.to_csv(os.path.join(OUTPUT_DIR, "portfolio_simulation.csv"), index=False)
print(f"   Exported: portfolio_simulation.csv")

# JSON report
report = {
    "generated_at": datetime.utcnow().isoformat(),
    "total_tickers_analyzed": len(TICKERS),
    "models_evaluated": list(bt_df["Model"].unique()),
    "model_ranking": model_summary[["Model", "Sharpe_median", "DirectionAcc_mean", "R2_mean", "FinalPortfolio_median"]].to_dict(orient="records"),
    "model_win_rates": {k: v.to_dict() for k, v in win_rates.items()},
    "portfolio_simulation": sim_df.to_dict(orient="records"),
    "signal_distribution": dict(sig_dist),
}
with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"   Exported: evaluation_summary.json")

# ---------------------------------------------------------------------------
# 10. PRINT FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  PERFORMANCE EVALUATION COMPLETE")
print("=" * 70)
print(f"  Tickers analyzed:           {len(TICKERS)}")
print(f"  Models evaluated:           {', '.join(sorted(bt_df['Model'].unique()))}")
print(f"  Evaluation reports saved to: {OUTPUT_DIR}/")
print()

# Best overall model
best_model = model_summary.iloc[0]["Model"]
print(f"  🏆 Best overall model (by median Sharpe): {best_model}")
print()

# Key findings
print("  Key Findings:")
for _, row in model_summary.iterrows():
    m = row["Model"]
    dir_acc = row.get("DirectionAcc_mean", np.nan)
    sharpe = row.get("Sharpe_median", np.nan)
    portfolio = row.get("FinalPortfolio_median", np.nan)
    
    if not np.isnan(dir_acc):
        print(f"    {m:20s}: DirAcc={dir_acc:.1%}, Sharpe={sharpe:.2f}, MedianPortfolio=${portfolio:.0f}")

print("\n  Reports generated:")
for f in os.listdir(OUTPUT_DIR):
    print(f"    {OUTPUT_DIR}/{f}")
