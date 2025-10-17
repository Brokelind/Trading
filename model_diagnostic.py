"""
Model Collapse Diagnostic Tool
Run this to diagnose why LSTM and XGBoost are predicting constant values
"""

import numpy as np
import pandas as pd
import joblib
import os
from scipy.stats import pearsonr

def diagnose_model_collapse(ticker: str, model_dir: str = "saved_models"):
    """Diagnose model collapse issues"""
    
    print(f"\n{'='*70}")
    print(f"MODEL COLLAPSE DIAGNOSTIC: {ticker}")
    print(f"{'='*70}\n")
    
    # 1. Load scalers
    feature_scaler_path = os.path.join(model_dir, f"{ticker}_feature_scaler.joblib")
    
    if not os.path.exists(feature_scaler_path):
        print(f"❌ Feature scaler not found at {feature_scaler_path}")
        return
    
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        print(f"✅ Loaded feature scaler")
        print(f"   Features expected: {feature_scaler.n_features_in_}")
        print(f"   Scaler type: {type(feature_scaler).__name__}")
        
        # For StandardScaler, show mean and scale
        if hasattr(feature_scaler, 'mean_'):
            print(f"   Feature means range: [{feature_scaler.mean_.min():.6f}, {feature_scaler.mean_.max():.6f}]")
        if hasattr(feature_scaler, 'scale_'):
            print(f"   Feature scales range: [{feature_scaler.scale_.min():.6f}, {feature_scaler.scale_.max():.6f}]")
            
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return
    
    # 2. Load and check training data
    data_path = os.path.join("data", f"{ticker}_data.csv")
    if not os.path.exists(data_path):
        print(f"\n❌ Data file not found at {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Check target returns
    df['target_return'] = df['adj_close'].pct_change().shift(-1)
    df = df.dropna()
    
    target_returns = df['target_return'].values
    
    print(f"\n📊 TARGET ANALYSIS:")
    print(f"   Samples: {len(target_returns)}")
    print(f"   Mean: {target_returns.mean():.6f}")
    print(f"   Std: {target_returns.std():.6f}")
    print(f"   Min: {target_returns.min():.6f}")
    print(f"   Max: {target_returns.max():.6f}")
    print(f"   Median: {np.median(target_returns):.6f}")
    
    # Check if target is too centered
    if abs(target_returns.mean()) < 0.0001:
        print(f"\n⚠️  Target mean is very close to zero: {target_returns.mean():.6f}")
        print(f"   This can cause models to predict the mean (near zero)")
    
    if target_returns.std() < 0.005:
        print(f"\n❌ CRITICAL: Target std deviation is very low: {target_returns.std():.6f}")
        print(f"   Models will struggle to learn meaningful patterns")
    else:
        print(f"   ✅ Target variance looks reasonable")
    
    # 3. Check for constant predictions in backtest results
    print(f"\n🔍 BACKTEST PREDICTION ANALYSIS:")
    
    backtest_files = []
    for model_type in ["LSTM", "Dense_NN", "XGBoost", "Random_Forest", "Ensemble"]:
        bt_path = os.path.join(model_dir, f"{ticker}_{model_type}_backtest.csv")
        if os.path.exists(bt_path):
            backtest_files.append((model_type, bt_path))
    
    for model_type, bt_path in backtest_files:
        try:
            bt_df = pd.read_csv(bt_path)
            if 'y_pred' in bt_df.columns:
                preds = bt_df['y_pred'].dropna()
                if len(preds) > 0:
                    pred_std = preds.std()
                    unique_vals = len(preds.unique())
                    
                    print(f"   {model_type:15s}: std={pred_std:.6f}, unique_vals={unique_vals}/{len(preds)}")
                    
                    if pred_std < 0.001:
                        print(f"      ⚠️  PREDICTION COLLAPSE: Standard deviation too low!")
                    if unique_vals < len(preds) * 0.1:
                        print(f"      ⚠️  PREDICTION COLLAPSE: Too many repeated values!")
                else:
                    print(f"   {model_type:15s}: No valid predictions")
            else:
                print(f"   {model_type:15s}: No y_pred column")
        except Exception as e:
            print(f"   {model_type:15s}: Error - {e}")
    
    # 4. Check feature correlation with target
    print(f"\n📈 FEATURE CORRELATION ANALYSIS:")
    
    # Create some basic features for correlation check
    feature_cols = []
    if 'adj_close' in df.columns:
        # Basic price-based features
        df['returns_1d'] = df['adj_close'].pct_change()
        df['returns_5d'] = df['adj_close'].pct_change(5)
        df['price_sma_10'] = df['adj_close'].rolling(10).mean()
        df['price_sma_20'] = df['adj_close'].rolling(20).mean()
        df['volatility_10'] = df['returns_1d'].rolling(10).std()
        
        feature_cols = ['returns_1d', 'returns_5d', 'price_sma_10', 'price_sma_20', 'volatility_10']
    
    if len(feature_cols) > 0:
        # Remove rows with NaN
        analysis_df = df[feature_cols + ['target_return']].dropna()
        X = analysis_df[feature_cols].values
        y = analysis_df['target_return'].values
        
        correlations = []
        for i, col in enumerate(feature_cols):
            try:
                corr, p_value = pearsonr(X[:, i], y)
                correlations.append((col, corr, p_value))
            except:
                pass
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"   Top feature correlations with target:")
        for feat, corr, p_val in correlations[:5]:
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"      {feat:15s}: {corr:7.4f} {significance}")
        
        max_corr = abs(correlations[0][1]) if correlations else 0
        
        if max_corr < 0.05:
            print(f"   ⚠️  Maximum correlation is very low: {max_corr:.4f}")
            print(f"      Features have weak relationship with target")
        else:
            print(f"   ✅ Maximum correlation looks reasonable: {max_corr:.4f}")
    
    # 5. Model-specific diagnostics
    print(f"\n🤖 MODEL-SPECIFIC DIAGNOSTICS:")
    
    model_types = ["LSTM", "Dense NN", "XGBoost", "Random Forest"]
    
    for model_type in model_types:
        model_name = f"{ticker}_{model_type.replace(' ', '_')}"
        model_ext = ".keras" if model_type in ["LSTM", "Dense NN"] else ".joblib"
        model_path = os.path.join(model_dir, model_name + model_ext)
        
        if os.path.exists(model_path):
            print(f"   ✅ {model_type:15s}: Model exists")
            
            # For tree models, check feature importances
            if model_type in ["XGBoost", "Random Forest"]:
                try:
                    model = joblib.load(model_path)
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if importances is not None and len(importances) > 0:
                            zero_importance = (importances < 1e-10).sum()
                            max_importance = importances.max()
                            
                            print(f"      Features used: {len(importances)}")
                            print(f"      Zero importance: {zero_importance}")
                            print(f"      Max importance: {max_importance:.6f}")
                            
                            if zero_importance > len(importances) * 0.8:
                                print(f"      ⚠️  >80% of features have zero importance!")
                            
                            if max_importance < 0.01:
                                print(f"      ⚠️  All features have very low importance!")
                except Exception as e:
                    print(f"      ❌ Could not analyze model: {e}")
        else:
            print(f"   ❌ {model_type:15s}: Model not found")
    
    # 6. Check training configuration
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    
    meta_path = os.path.join(model_dir, f"{ticker}_model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"   ✅ Training metadata found")
            if 'data_split' in meta:
                split = meta['data_split']
                print(f"   Train samples: {split.get('train_size', 'N/A')}")
                print(f"   Validation samples: {split.get('val_size', 'N/A')}")
                print(f"   Backtest samples: {split.get('backtest_size', 'N/A')}")
        except:
            print(f"   ❌ Could not read metadata")
    else:
        print(f"   ❌ No training metadata found")
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")
    
    # 7. Recommendations based on findings
    print(f"💡 RECOMMENDATIONS:")
    
    if target_returns.std() < 0.01:
        print(f"   1. ⚠️  CRITICAL: Increase target variance")
        print(f"      - Predict 5-day returns: df['adj_close'].shift(-5) / df['adj_close'] - 1")
        print(f"      - Use log returns: np.log(df['adj_close'] / df['adj_close'].shift(1))")
        print(f"      - Scale targets: target_scaler = StandardScaler()")
    
    # Check if we found prediction collapse
    prediction_collapse = False
    for model_type, bt_path in backtest_files:
        try:
            bt_df = pd.read_csv(bt_path)
            if 'y_pred' in bt_df.columns:
                preds = bt_df['y_pred'].dropna()
                if len(preds) > 0 and preds.std() < 0.001:
                    prediction_collapse = True
                    break
        except:
            pass
    
    if prediction_collapse:
        print(f"   2. ⚠️  PREDICTION COLLAPSE DETECTED")
        print(f"      - Reduce model complexity (fewer layers/units)")
        print(f"      - Add dropout and regularization")
        print(f"      - Use gradient clipping in training")
        print(f"      - Try different activation functions")
    
    print(f"   3. ✅ Adjust trading strategy")
    print(f"      - Increase signal threshold to 0.5-1.0%")
    print(f"      - Use volatility-based position sizing")
    print(f"      - Implement stop-loss and take-profit")
    
    print(f"   4. ✅ Feature engineering")
    print(f"      - Add more technical indicators (RSI, MACD, Bollinger Bands)")
    print(f"      - Include market regime indicators")
    print(f"      - Add volume-price relationships")


# Add missing import
import json

if __name__ == "__main__":
    # Run diagnostic
    diagnose_model_collapse("AMZN")