import os
import ast

MAPPING = {
    "core/types.py": [
        "Bar", "PhasePoint", "CapacityRecord", "TradingSignal", "TradeResult",
        "SymbolSpec", "SimPosition", "ClosedTrade", "TradeRecord", "BacktestResult"
    ],
    "core/config.py": [
        "RiskConfig"
    ],
    "execution/connection.py": [
        "MT5Connection"
    ],
    "execution/risk.py": [
        "RiskManager"
    ],
    "execution/executor.py": [
        "TradingEngine", "MT5TradeExecutor", "AutoTradingEngine"
    ],
    "analytics/trade_analytics.py": [
        "TradeAnalytics"
    ],
    "dashboard/server.py": [
        "DashboardState", "DashboardHTTPRequestHandler", "start_dashboard_server"
    ],
    "models/geometry.py": [
        "get_convex_hull_vertices", "_signed_area", "convex_hull_metrics", "phase_coords",
        "_tda_features_ripser", "_tda_features_approx", "compute_tda", "detect_fair_value_gaps",
        "compute_session_levels", "compute_mtf_betti"
    ],
    "models/validation.py": [
        "ModelValidator", "AdversarialValidator"
    ],
    "models/meta.py": [
        "ConformalPredictor", "BatchedLearner", "RegimeAwareModel", "ScenarioGenerator"
    ],
    "models/base.py": [
        "HyperparameterBandit", "SymplecticOnlineModel"
    ],
    "models/forecaster.py": [
        "SymplecticForecaster"
    ],
    "utils/backtest.py": [
        "simulate_backtest_from_cache", "_spread_price", "_compute_atr_from_bars",
        "_compute_sl_tp_backtest", "_calc_lot_backtest", "BacktestSimulator",
        "_mt5_rates_to_bars", "print_signal_diagnostics", "print_backtest_report",
        "save_equity_chart", "_parse_float_list", "_optimizer_score", "run_optimizer_cli", "run_backtest_cli"
    ],
    "main.py": [
        "is_news_blackout", "run_multi_symbol"
    ]
}

def get_intra_imports(file_path):
    if file_path == "core/types.py": return ""
    if file_path == "core/config.py": return ""
    if file_path == "execution/connection.py": return ""
    if file_path == "execution/risk.py":
        return "from core.config import RiskConfig\nfrom core.types import *\n"
    if file_path == "execution/executor.py":
        return "from core.types import *\nfrom core.config import *\nfrom execution.connection import *\nfrom execution.risk import *\nfrom analytics.trade_analytics import TradeAnalytics\n"
    if file_path == "analytics/trade_analytics.py":
        return "from core.types import *\n"
    if file_path == "dashboard/server.py":
        return "from core.types import *\nfrom analytics.trade_analytics import TradeAnalytics\nimport threading\nimport json\nimport urllib.parse\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n"
    if file_path == "models/geometry.py":
        return "from core.types import *\n"
    if file_path == "models/validation.py":
        return "from core.types import *\n"
    if file_path == "models/meta.py":
        return "from core.types import *\nfrom models.base import SymplecticOnlineModel\nfrom models.validation import *\n"
    if file_path == "models/base.py":
        return "from core.types import *\nfrom models.geometry import *\nfrom models.validation import *\n"
    if file_path == "models/forecaster.py":
        return "from core.types import *\nfrom core.config import *\nfrom models.base import *\nfrom models.meta import *\nfrom models.validation import *\nfrom analytics.trade_analytics import *\nfrom execution.executor import *\n"
    if file_path == "utils/backtest.py":
        return "from core.types import *\nfrom core.config import *\nfrom models.forecaster import *\n"
    if file_path == "main.py":
        return "from core.types import *\nfrom core.config import *\nfrom execution.executor import *\nfrom dashboard.server import *\nfrom models.forecaster import *\nfrom utils.backtest import *\n"
    return ""

def main():
    with open("symplectic_forecaster_backup.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Header is everything before class Bar
    header_idx = content.find("class Bar(NamedTuple):")
    header_content = content[:header_idx]
    
    # Find __main__ block manually as ast doesn't give it easily
    main_idx = content.find("if __name__ == \"__main__\":")
    main_content = content[main_idx:]
    
    tree = ast.parse(content)
    
    blocks = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            # get_source_segment gets the original string including decorators!
            blocks[node.name] = ast.get_source_segment(content, node)
    
    for fp in MAPPING.keys():
        os.makedirs(os.path.dirname(fp) or '.', exist_ok=True)
        if os.path.dirname(fp):
            open(os.path.join(os.path.dirname(fp), "__init__.py"), 'w').close()
            
    for fp, names in MAPPING.items():
        with open(fp, "w", encoding="utf-8") as out:
            out.write(header_content)
            out.write("\n")
            out.write(get_intra_imports(fp))
            out.write("\n")
            
            for name in names:
                if name in blocks:
                    out.write(blocks[name])
                    out.write("\n\n")
                else:
                    print(f"WARNING: Could not find {name} for {fp}")
            
            if fp == "main.py":
                out.write("\n\n")
                out.write(main_content)

if __name__ == "__main__":
    main()
