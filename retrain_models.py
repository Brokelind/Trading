"""Retrain the ticker universe and refresh predictions without trading or email."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from ticker_config import DEFAULT_TICKERS

MODEL_NAMES = {"LSTM", "Dense NN", "Random Forest", "XGBoost", "LightGBM", "Ensemble"}


def json_safe(data):
    """Represent undefined evaluation statistics as null, never NaN/Infinity."""
    if isinstance(data, dict):
        return {key: json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    if isinstance(data, float) and not math.isfinite(data):
        return None
    return data


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(data), indent=2, default=str, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def train_ticker(ticker, run_dir):
    # Imports stay in the worker so every ticker gets a fresh TensorFlow process.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import pandas as pd
    import tensorflow as tf
    from call_market import get_data
    from tradingmodelsystem import TradingModelSystem

    tf.keras.utils.set_random_seed(42)
    stage = Path(run_dir) / ticker
    model_dir = stage / 'saved_models'
    data_dir = stage / 'data'
    path = get_data(ticker, save_folder=str(data_dir))
    if not path:
        raise RuntimeError('Market data refresh failed')
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    data_as_of = raw.index[-1].date().isoformat()
    age = (datetime.now(timezone.utc).date() - raw.index[-1].date()).days
    if age > 5:
        raise RuntimeError(f'Market data is stale: {data_as_of}')
    system = TradingModelSystem({
        'data_dir': str(data_dir), 'model_dir': str(model_dir),
        'prediction_threshold_pct': 0.25, 'n_jobs': 2,
    })
    result = system.ensure_trained(ticker, force=True)
    if 'error' in result:
        raise RuntimeError(result['error'])
    statuses = {name: value.get('status') for name, value in result.get('training_results', {}).items()}
    if set(statuses) != MODEL_NAMES or any(status != 'trained' for status in statuses.values()):
        raise RuntimeError(f'Incomplete model training: {statuses}')
    predictions = system.predict_tomorrow(ticker)
    if set(predictions) != MODEL_NAMES:
        raise RuntimeError(f'Incomplete predictions: {predictions}')
    for name, pred in predictions.items():
        if 'error' in pred or any(not isinstance(pred.get(key), (int, float)) or not math.isfinite(pred[key])
                                  for key in ('predicted_price', 'pct_diff')):
            raise RuntimeError(f'Invalid {name} prediction: {pred}')
    meta = system.load_meta(ticker)
    chosen = meta.get('best_model')
    if chosen not in predictions:
        chosen = 'Ensemble'
    summary = {
        'ticker': ticker, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'data_as_of': data_as_of, 'last_price': float(raw['adj_close'].iloc[-1]),
        'predictions': predictions, 'chosen_model': chosen,
        'signal': predictions[chosen]['signal'], 'pct_diff': predictions[chosen]['pct_diff'],
        'qty': 0, 'sentiment': {}, 'run_mode': 'retrain_only',
        'training_metrics': meta.get('training_metrics', []),
        'backtest_metrics': meta.get('backtest_metrics', []),
    }
    # Validate JSON before promoting any trained artifacts.
    write_json(stage / f'{ticker}_summary.json', summary)
    meta['model_paths'] = {name: str(Path('saved_models') / Path(value).name)
                           for name, value in meta['model_paths'].items()}
    for key in ('training_metrics_file', 'backtest_metrics_file'):
        meta[key] = str(Path('saved_models') / Path(meta[key]).name)
    meta['data_as_of'] = data_as_of
    system.save_meta(ticker, meta)
    for source_dir, destination in ((model_dir, Path('saved_models')), (data_dir, Path('data'))):
        destination.mkdir(exist_ok=True)
        for source in source_dir.iterdir():
            target = destination / source.name
            if target.exists():
                backup = Path(run_dir) / 'backup' / destination / source.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
    write_json(Path('results') / f'{ticker}_summary.json', summary)
    outcome = {'ticker': ticker, 'status': 'success', 'data_as_of': data_as_of,
               'models': statuses, 'training_seconds': result['training_time']}
    write_json(stage / 'status.json', outcome)
    return outcome


def run_worker(ticker, run_dir):
    log_path = Path(run_dir) / f'{ticker}.log'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.run([sys.executable, '-u', __file__, '--worker', ticker, '--run-dir', str(run_dir)],
                              stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    status_path = Path(run_dir) / ticker / 'status.json'
    if proc.returncode or not status_path.exists():
        return {'ticker': ticker, 'status': 'failed', 'log': str(log_path)}
    return json.loads(status_path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        train_ticker(args.worker, args.run_dir)
        return
    if args.workers < 1:
        parser.error('--workers must be positive')
    tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    if set(tickers) - set(DEFAULT_TICKERS):
        parser.error('Tickers must be in ticker_config.DEFAULT_TICKERS')
    run_dir = Path(args.run_dir or ('logs/retrain_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'expected_tickers': tickers, 'results': []}
    write_json(run_dir / 'status.json', report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_worker, ticker, run_dir): ticker for ticker in tickers}
        for future in as_completed(futures):
            result = future.result()
            report['results'].append(result)
            write_json(run_dir / 'status.json', report)
            print(f"[{len(report['results'])}/{len(tickers)}] {result['ticker']}: {result['status']}", flush=True)
    failed = [r['ticker'] for r in report['results'] if r['status'] != 'success']
    report.update(status='failed' if failed else 'success', failed_tickers=failed,
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(run_dir / 'status.json', report)
    write_json('results/retrain_status.json', report)
    from visualize_results import visualize_backtest_chart
    visualize_backtest_chart([r['ticker'] for r in report['results'] if r['status'] == 'success'])
    from web_dev.web_dashboard import generate_dashboard
    generate_dashboard()
    if failed:
        raise SystemExit(f'Retraining failed for: {", ".join(failed)}; see {run_dir}')


if __name__ == '__main__':
    main()
