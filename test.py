from main import TradingExecutor
from call_market import get_data

tms = TradingExecutor()
tms.diagnose_model_issues('SPY')

#tms.post_results()