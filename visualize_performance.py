#!/usr/bin/env python3
"""
Generate performance evaluation visualizations from evaluation reports.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = "evaluation_reports"
VIS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(VIS_DIR, exist_ok=True)

# Load data
bt_df = pd.read_csv(os.path.join(OUTPUT_DIR, "backtest_metrics_all.csv"))
model_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "model_performance_summary.csv"))
sim_df = pd.read_csv(os.path.join(OUTPUT_DIR, "portfolio_simulation.csv"))

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json")) as f:
    report = json.load(f)

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

# ---------------------------------------------------------------------------
# Chart 1: Model Performance Radar (Sharpe, DirAcc, Portfolio Return)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_to_plot = [
    ("Sharpe_median", "Median Sharpe Ratio", "Sharpe Ratio"),
    ("DirectionAcc_mean", "Mean Direction Accuracy", "Accuracy"),
    ("FinalPortfolio_median", "Median Final Portfolio ($)", "Portfolio Value ($)"),
]

for ax, (col, title, ylabel) in zip(axes, metrics_to_plot):
    if col in model_summary.columns:
        vals = model_summary[col]
        bars = ax.bar(model_summary["Model"], vals, color=COLORS[:len(vals)], edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        
        # Add value labels
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                if "Sharpe" in col:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                elif "Direction" in col:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                            f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                            f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if "Direction" in col:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
        ax.legend(loc='upper right', fontsize=9)
        
plt.suptitle("Trading Model Performance Comparison", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "model_performance_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: model_performance_comparison.png")

# ---------------------------------------------------------------------------
# Chart 2: Win Rate Comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics_list = ["Sharpe", "DirectionAcc", "FinalPortfolio", "R2"]
titles = ["Sharpe Ratio", "Direction Accuracy", "Final Portfolio Value", "R² Score"]

for ax, metric, title in zip(axes.flat, metrics_list, titles):
    wr = report["model_win_rates"].get(metric, {})
    if wr:
        models = list(wr.keys())
        counts = list(wr.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]
        
        bars = ax.barh(models, percentages, color=COLORS[:len(models)], edgecolor='black', linewidth=0.5)
        ax.set_title(f'Win Rate: {title}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Percentage of Tickers (%)')
        
        for bar, pct, cnt in zip(bars, percentages, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{pct:.0f}% ({cnt})', va='center', fontsize=10, fontweight='bold')

plt.suptitle("Model Win Rates - How Often Each Model is #1", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "model_win_rates.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: model_win_rates.png")

# ---------------------------------------------------------------------------
# Chart 3: Direction Accuracy Distribution (Boxplot)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
model_order = bt_df.groupby("Model")["DirectionAcc"].median().sort_values(ascending=False).index
data_to_plot = [bt_df[bt_df["Model"] == m]["DirectionAcc"].dropna().values for m in model_order]

bp = ax.boxplot(data_to_plot, labels=model_order, patch_artist=True, 
                medianprops={'color': 'black', 'linewidth': 2})

for patch, color in zip(bp['boxes'], COLORS[:len(model_order)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random (50%)')
ax.set_title("Direction Accuracy Distribution by Model", fontsize=15, fontweight='bold')
ax.set_ylabel("Direction Accuracy")
ax.set_xlabel("Model")
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add scatter points
for i, (model, data) in enumerate(zip(model_order, data_to_plot)):
    jitter = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(jitter, data, alpha=0.15, s=15, color=COLORS[i % len(COLORS)])

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "direction_accuracy_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: direction_accuracy_distribution.png")

# ---------------------------------------------------------------------------
# Chart 4: Portfolio Simulation (Total & Win Rate)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Total portfolio value
if "TotalPortfolio" in sim_df.columns:
    bars = ax1.bar(sim_df["Model"], sim_df["TotalPortfolio"], color=COLORS[:len(sim_df)], 
                   edgecolor='black', linewidth=0.5)
    ax1.set_title("Total Portfolio Value (All Tickers Combined)", fontweight='bold', fontsize=13)
    ax1.set_ylabel("Total Portfolio Value ($)")
    ax1.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, sim_df["TotalPortfolio"]):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5000,
                    f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Win rate
if "WinRate" in sim_df.columns:
    bars = ax2.bar(sim_df["Model"], sim_df["WinRate"], color=COLORS[:len(sim_df)],
                   edgecolor='black', linewidth=0.5)
    ax2.set_title("Ticker Profitability Win Rate", fontweight='bold', fontsize=13)
    ax2.set_ylabel("Win Rate (%)")
    ax2.tick_params(axis='x', rotation=30)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    for bar, val in zip(bars, sim_df["WinRate"]):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.legend()

plt.suptitle("Portfolio Simulation Results (Equal-Weighted)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "portfolio_simulation.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: portfolio_simulation.png")

# ---------------------------------------------------------------------------
# Chart 5: MAE vs R2 Scatter (Model Quality)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
models = bt_df["Model"].unique()
for i, model in enumerate(models):
    subset = bt_df[bt_df["Model"] == model].dropna(subset=["MAE", "R2"])
    ax.scatter(subset["MAE"], subset["R2"], label=model, alpha=0.6, s=60, 
               color=COLORS[i % len(COLORS)], edgecolors='black', linewidth=0.3)

ax.set_xlabel("Mean Absolute Error (MAE)", fontsize=13)
ax.set_ylabel("R² Score", fontsize=13)
ax.set_title("Model Quality: Prediction Error vs Explained Variance", fontsize=15, fontweight='bold')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0 (baseline)')
ax.legend(fontsize=11, loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "mae_vs_r2_scatter.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: mae_vs_r2_scatter.png")

# ---------------------------------------------------------------------------
# Chart 6: Signal Distribution Pie
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
sig_dist = report.get("signal_distribution", {})
if sig_dist:
    labels = list(sig_dist.keys())
    sizes = list(sig_dist.values())
    colors_pie = {'BUY': '#4CAF50', 'SELL': '#F44336', 'HOLD': '#FF9800'}
    pie_colors = [colors_pie.get(l, '#9E9E9E') for l in labels]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.0f%%',
                                       colors=pie_colors, startangle=90,
                                       textprops={'fontweight': 'bold', 'fontsize': 14})
    ax.set_title("Final Trading Signal Distribution", fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "signal_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: signal_distribution.png")

# ---------------------------------------------------------------------------
# Summary: Best/Worst tickers heatmap-style table
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

# Create a table showing best/worst tickers per model
table_data = []
col_labels = ["Model", "Best Ticker (Sharpe)", "Best Sharpe", "Worst Ticker (Sharpe)", "Worst Sharpe"]

for _, row in model_summary.iterrows():
    model = row["Model"]
    # Get best/worst from the win rates data
    bw_df = bt_df[bt_df["Model"] == model].dropna(subset=["Sharpe"])
    if not bw_df.empty:
        best_idx = bw_df["Sharpe"].idxmax()
        worst_idx = bw_df["Sharpe"].idxmin()
        best_ticker = bw_df.loc[best_idx, "Ticker"]
        best_sharpe = bw_df.loc[best_idx, "Sharpe"]
        worst_ticker = bw_df.loc[worst_idx, "Ticker"]
        worst_sharpe = bw_df.loc[worst_idx, "Sharpe"]
        table_data.append([model, best_ticker, f"{best_sharpe:.2f}", worst_ticker, f"{worst_sharpe:.2f}"])

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1976D2')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#E3F2FD')

ax.set_title("Best & Worst Performing Tickers by Model (Sharpe Ratio)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "best_worst_tickers_table.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: best_worst_tickers_table.png")

print(f"\n✅ All visualizations saved to {VIS_DIR}/")
