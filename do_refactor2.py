import os
import re

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
        "ConformalPredictor", "BatchedLearner", "RegimeAwareModel", "HyperparameterBandit", "ScenarioGenerator"
    ],
    "models/base.py": [
        "SymplecticOnlineModel"
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
        "_GLOBAL_RAY_READY", "_RAY_INIT_ATTEMPTED", "_GLOBAL_MAMBA_MODEL", "_GLOBAL_NLP_AGENT", "_INIT_LOCK",
        "is_news_blackout", "run_multi_symbol", "main"
    ]
}

def get_intra_imports(file_path):
    if file_path == "core/types.py": return ""
    if file_path == "core/config.py": return ""
    if file_path == "execution/connection.py": return ""
    if file_path == "execution/risk.py":
        return "from core.config import RiskConfig\n"
    if file_path == "execution/executor.py":
        return "from core.types import *\nfrom core.config import *\nfrom execution.connection import *\nfrom execution.risk import *\nfrom analytics.trade_analytics import TradeAnalytics\n"
    if file_path == "analytics/trade_analytics.py":
        return "from core.types import *\n"
    if file_path == "dashboard/server.py":
        return "from core.types import *\nfrom analytics.trade_analytics import TradeAnalytics\n"
    if file_path == "models/geometry.py":
        return "from core.types import *\n"
    if file_path == "models/validation.py":
        return "from core.types import *\n"
    if file_path == "models/meta.py":
        return "from core.types import *\nfrom models.base import SymplecticOnlineModel\n"
    if file_path == "models/base.py":
        return "from core.types import *\nfrom models.geometry import *\n"
    if file_path == "models/forecaster.py":
        return "from core.types import *\nfrom core.config import *\nfrom models.base import *\nfrom models.meta import *\nfrom models.validation import *\nfrom analytics.trade_analytics import *\nfrom execution.executor import *\n"
    if file_path == "utils/backtest.py":
        return "from core.types import *\nfrom core.config import *\nfrom models.forecaster import *\n"
    if file_path == "main.py":
        return "from core.types import *\nfrom core.config import *\nfrom execution.executor import *\nfrom dashboard.server import *\nfrom models.forecaster import *\nfrom utils.backtest import *\n"
    return ""

def main():
    with open("symplectic_forecaster.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract header: everything before 'class Bar(NamedTuple):'
    header_idx = content.find("class Bar(NamedTuple):")
    header_content = content[:header_idx]
    rest_content = content[header_idx:]
    
    lines = rest_content.splitlines(keepends=True)
    
    blocks = []
    current_block = []
    current_name = None
    
    for line in lines:
        match = re.match(r'^(class|def)\s+([A-Za-z0-9_]+)', line)
        if match:
            if current_block:
                blocks.append({"name": current_name, "lines": current_block})
            current_name = match.group(2)
            current_block = [line]
        elif line.startswith("if __name__ == "):
            if current_block:
                blocks.append({"name": current_name, "lines": current_block})
            current_name = "main"
            current_block = [line]
        elif re.match(r'^_[A-Z0-9_]+\s*=', line) and not line.startswith(" "):
            if current_block:
                blocks.append({"name": current_name, "lines": current_block})
            current_name = line.split('=')[0].strip()
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)
                
    if current_block:
        blocks.append({"name": current_name, "lines": current_block})
        
    # Create directories
    for fp in MAPPING.keys():
        os.makedirs(os.path.dirname(fp) or '.', exist_ok=True)
        if os.path.dirname(fp):
            open(os.path.join(os.path.dirname(fp), "__init__.py"), 'w').close()
            
    # Write files
    for fp, names in MAPPING.items():
        with open(fp, "w", encoding="utf-8") as out:
            out.write(header_content)
            out.write("\n")
            out.write(get_intra_imports(fp))
            out.write("\n")
            
            for name in names:
                found = False
                for b in blocks:
                    if b['name'] == name:
                        out.writelines(b['lines'])
                        out.write("\n")
                        found = True
                        break
                if not found:
                    print(f"WARNING: Could not find block {name} for {fp}")

if __name__ == "__main__":
    main()
