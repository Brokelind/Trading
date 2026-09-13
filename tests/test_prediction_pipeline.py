import ast
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

import call_market
import distribute_results
from ticker_config import DEFAULT_TICKERS
from retrain_models import write_json
from web_dev import web_dashboard


class ResultsTests(unittest.TestCase):
    def test_undefined_evaluation_statistics_are_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'summary.json')
            write_json(path, {'metrics': [{'Sharpe': float('nan'), 'R2': 0.1}]})
            result = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertIsNone(result['metrics'][0]['Sharpe'])
            self.assertEqual(result['metrics'][0]['R2'], 0.1)

    def test_dashboard_includes_hold_and_low_confidence_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            for ticker, signal, confidence in [('AAPL', 'HOLD', 0.1), ('MSFT', 'BUY', 0.8)]:
                Path(directory, f'{ticker}_summary.json').write_text(json.dumps({
                    'ticker': ticker, 'signal': signal, 'chosen_model': 'Ensemble',
                    'last_price': 100, 'sentiment': {'confidence': 0.8},
                    'predictions': {'Ensemble': {'predicted_price': 101, 'confidence': confidence}},
                }))
            with patch.object(distribute_results, 'RESULTS_DIR', directory):
                self.assertEqual([r['ticker'] for r in distribute_results.load_results()], ['MSFT'])
                self.assertEqual(len(distribute_results.load_results(strong_only=False)), 2)
                with patch.object(web_dashboard, 'OUTPUT_DIR', directory), \
                     patch.object(web_dashboard, 'generate_ticker_page'), \
                     patch.object(web_dashboard, 'generate_assets'):
                    self.assertEqual(web_dashboard.generate_dashboard(), 2)
                    html = Path(directory, 'index.html').read_text(encoding='utf-8')
                    self.assertIn('tickers/AAPL.html', html)
                    self.assertIn('tickers/MSFT.html', html)

    def test_unconfigured_tickers_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'OLD_summary.json').write_text(json.dumps({'ticker': 'OLD'}))
            with patch.object(distribute_results, 'RESULTS_DIR', directory):
                self.assertEqual(distribute_results.load_results(False, DEFAULT_TICKERS), [])


class DataTests(unittest.TestCase):
    def test_yahoo_multiindex_round_trips_to_model_schema(self):
        columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Volume'], ['AAPL']])
        frame = pd.DataFrame([[100, 102, 99, 101, 500]],
                             index=pd.to_datetime(['2026-09-11']), columns=columns)
        with tempfile.TemporaryDirectory() as directory, patch.object(call_market.yf, 'download', return_value=frame):
            path = call_market.get_data('AAPL', directory)
            loaded = pd.read_csv(path, index_col=0, parse_dates=True)
            self.assertEqual(list(loaded.columns), ['open', 'high', 'low', 'adj_close', 'volume'])
            self.assertEqual(loaded['adj_close'].iloc[-1], 101)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(call_market.yf.download.call_args.kwargs['period'], '10y')

    def test_bad_download_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'AAPL_data.csv')
            path.write_text('existing data')
            with patch.object(call_market.yf, 'download', return_value=pd.DataFrame()), \
                 patch.object(call_market, 'client', None), patch.object(call_market.time, 'sleep'):
                self.assertIsNone(call_market.get_data('AAPL', directory))
            self.assertEqual(path.read_text(), 'existing data')


def executor_namespace():
    # Load the executor without importing brokerage or sentiment clients.
    tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = {'DEFAULT_TICKERS': DEFAULT_TICKERS, 'TradingModelSystem': Mock(),
                 'tqdm': lambda items, **kwargs: items,
                 'crypto_correlation': types.SimpleNamespace(generate_signals=lambda: [])}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'main.py', 'exec'), namespace)
    return namespace


class ExecutorTests(unittest.TestCase):
    def test_ci_loads_best_model_and_saves_fresh_prediction(self):
        namespace = executor_namespace()
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({'adj_close': [100]}, index=pd.to_datetime(['2026-09-11'])).to_csv(
                Path(directory, 'AAPL_data.csv'))
            namespace.update({
                'SKIP_TRAINING_ON_CI': True, 'datetime': datetime, 'os': os, 'json': json, 'np': np,
                'data_path': directory, 'RESULTS_DIR': directory,
                'call_market': types.SimpleNamespace(get_data=lambda ticker: 'refreshed'),
                'analyze_news_sentiment': lambda ticker: {},
                'alpaca_trader': types.SimpleNamespace(qty_to_trade=lambda *args, **kwargs: 0),
            })
            executor = namespace['TradingExecutor'](['AAPL'])
            executor.model_system = Mock()
            executor.model_system.load_meta.return_value = {'best_model': 'XGBoost'}
            executor.model_system.predict_tomorrow.return_value = {
                'XGBoost': {'signal': 'BUY', 'pct_diff': 1, 'predicted_price': 101},
                'Ensemble': {'signal': 'HOLD', 'pct_diff': 0, 'predicted_price': 100},
            }
            result = executor.execute_strategy('AAPL')
            self.assertEqual(result['chosen_model'], 'XGBoost')
            self.assertEqual(result['data_as_of'], '2026-09-11')
            self.assertTrue(Path(directory, 'AAPL_summary.json').exists())
            executor.model_system.ensure_trained.assert_not_called()

    def test_trade_limit_does_not_stop_prediction_loop(self):
        executor = executor_namespace()['TradingExecutor']()
        executor.current_trades = executor.max_trades_per_day
        executor.execute_strategy = Mock()
        executor.run_daily_trading()
        self.assertEqual(executor.execute_strategy.call_count, len(DEFAULT_TICKERS))

    def test_one_ticker_failure_does_not_stop_rest(self):
        executor = executor_namespace()['TradingExecutor'](['AAPL', 'MSFT'])
        executor.execute_strategy = Mock(side_effect=[RuntimeError('failed'), None])
        executor.run_daily_trading()
        self.assertEqual(executor.execute_strategy.call_count, 2)

    def test_ci_false_string_is_false(self):
        tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        statement = next(n for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'SKIP_TRAINING_ON_CI' for t in n.targets))
        for value, expected in [('False', False), ('True', True), ('0', False), ('1', True)]:
            namespace = {'os': types.SimpleNamespace(environ={'SKIP_TRAINING_ON_CI': value}), 'env': None}
            exec(compile(ast.Module(body=[statement], type_ignores=[]), 'main.py', 'exec'), namespace)
            self.assertEqual(namespace['SKIP_TRAINING_ON_CI'], expected)


if __name__ == '__main__':
    unittest.main()
