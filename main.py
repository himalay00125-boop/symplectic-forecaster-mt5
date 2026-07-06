"""
symplectic_forecaster.py
========================
Self-Learning Price Forecaster — MetaTrader 5 Edition
Based on Symplectic Phase-Space Geometry and Topological Data Analysis.

Mathematical Foundations
------------------------
1. Financial Phase Space (Mishra 2026 — Stability_lemma_1)
   • Symplectic manifold M = (R², ω = dq ∧ dp)
   • Position coordinate : q(t)  = ln P(t)          [log-price]
   • Momentum coordinate: p(t)  = V(t)·sign(ΔP(t))  [signed order-flow]
   • ECH capacity       : C(t)  = Area(Conv(Dₜ))    [symplectic area of rolling hull]
   • Stability guarantee: |C(t)−C(t′)| ≤ L·dₕ + π·dₕ²  [Lemma 3.1]

2. Topological Data Analysis (Shultz 2023 — ssrn4378151)
   • Persistent homology on the rolling (q,p) point cloud
   • H₀ features: connected-component birth/death structure
   • H₁ features: loop persistence (market cycle detection)

3. Hierarchical Market Structure (Mantegna 1999 — s100510050929)
   • Cross-asset correlation distance d(i,j) = √(2(1−ρᵢⱼ))
   • Minimal Spanning Tree for regime identification

4. Symplectic Capacities (Cieliebak et al. 2005 — 0506191v1)
   • Gromov width = c₁(XΩ) = Area(Ω)  for convex toric domains
   • Capacity-preserving structure as the conservation law

Self-Learning Architecture
--------------------------
• River (online ML library) — single-pass, incremental learners
• Passive-Aggressive Regressor for return forecasting
• Adaptive Scaler — online mean/variance normalization
• Regime detector — capacity threshold above rolling 95th percentile
• Ensemble: PA-Regressor (fast adaptation) + Hoeffding Tree (structural)
• Walk-forward validation baked in — never peeks at the future

MetaTrader 5 Integration
-------------------------
• Direct connection to MT5 terminal for live market data
• Supports any symbol available in your MT5 broker (forex, indices, commodities, crypto)
• All timeframes: M1, M5, M15, H1, H4, D1, W1, MN1, etc.
• Signal-only mode: generates BUY / SELL / HOLD trading signals
• Historical data bootstrap from MT5 server (no CSV files needed)

Usage
-----
  # Interactive mode (prompts for symbol and timeframe):
  python symplectic_forecaster.py

  # Command-line mode:
  python symplectic_forecaster.py --symbol EURUSD --timeframe H1

  # With explicit MT5 login:
  python symplectic_forecaster.py --symbol XAUUSD --timeframe D1 \\
      --account 12345 --password mypass --server "BrokerDemo"
"""

from __future__ import annotations

import math
import sys
import time
import warnings
import random
import collections
import argparse
import pickle
from datetime import datetime, date
import importlib
import os
os.environ["RAY_ENABLE_WINDOWS_ORPHAN_SAFE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._validators").setLevel(logging.ERROR)
import ray

# ---- Module-level singleton caches for shared heavy resources ----
_GLOBAL_RAY_READY = False
_RAY_INIT_ATTEMPTED = False  # True once we've tried (success or fail)
_GLOBAL_MAMBA_MODEL = None
_GLOBAL_NLP_AGENT = None
_INIT_LOCK = __import__("threading").Lock()
import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple, Optional, Dict, List, Tuple, Callable, Any

try:
    from ai_engine import AIEngineActor
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False
    
try:
    from river import linear_model, preprocessing, tree, metrics, optim, ensemble
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    print("[WARN] river not found — falling back to sklearn PARegressor.")
    from sklearn.linear_model import SGDRegressor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Thread-safe global reference to dashboard state
global_dashboard_state = None

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    print("[ERROR] MetaTrader5 package not found.")
    print("        Install via:  pip install MetaTrader5")
    print("        Requires Python 3.8–3.13 on Windows (not 3.14+).")


# ===========================================================================
# MT5 TIMEFRAME MAPPING
# ===========================================================================

TIMEFRAME_MAP: Dict[str, int] = {}
if HAS_MT5:
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,   "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,   "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,   "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,  "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,  "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,   "H2":  mt5.TIMEFRAME_H2,
        "H3":  mt5.TIMEFRAME_H3,   "H4":  mt5.TIMEFRAME_H4,
        "H6":  mt5.TIMEFRAME_H6,   "H8":  mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

# Suggested poll intervals (seconds) per timeframe
_TF_POLL_SECONDS: Dict[str, float] = {
    "M1": 5,    "M2": 10,   "M3": 15,   "M4": 20,    "M5": 30,
    "M6": 30,   "M10": 60,  "M12": 60,  "M15": 60,   "M20": 120,
    "M30": 120, "H1": 300,  "H2": 600,  "H3": 900,   "H4": 900,
    "H6": 1800, "H8": 1800, "H12": 3600,"D1": 3600,
    "W1": 7200, "MN1": 14400,
}


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


from core.types import *
from core.config import *
from execution.executor import *
from dashboard.server import *
from models.forecaster import *
from utils.backtest import *

def is_news_blackout(symbol: str, blackout_minutes: int = 30) -> Tuple[bool, str]:
    """Check if a high-impact news event is near for the given symbol.
    
    Uses MT5's built-in economic calendar. Returns (True, reason) if
    within blackout_minutes of a high-impact event, else (False, '').
    """
    if not HAS_MT5:
        return False, ""
    
    try:
        import datetime as _dt
        now = _dt.datetime.utcnow()
        from_time = now - _dt.timedelta(minutes=blackout_minutes)
        to_time = now + _dt.timedelta(minutes=blackout_minutes)
        
        events = mt5.copy_ticks_from(symbol, from_time, 0, mt5.COPY_TICKS_ALL)
        
        # Try the calendar API (available in newer MT5 builds)
        try:
            calendar_events = mt5.calendar_event_get(
                from_time, to_time
            )
        except (AttributeError, TypeError):
            # Broker/build doesn't support calendar API - gracefully skip
            return False, ""
        
        if calendar_events is None:
            return False, ""
        
        # Extract the base currency from the symbol (first 3 chars for forex)
        base_ccy = symbol[:3].upper()
        quote_ccy = symbol[3:6].upper() if len(symbol) >= 6 else ""
        
        for evt in calendar_events:
            # High impact events only (importance >= 3 in MT5)
            importance = getattr(evt, 'importance', 0)
            country = getattr(evt, 'country_id', '')
            name = getattr(evt, 'name', 'Unknown Event')
            
            if importance >= 3:  # High impact
                # Check if the event's currency matches our symbol
                evt_ccy = getattr(evt, 'currency', '')
                if evt_ccy in (base_ccy, quote_ccy) or not evt_ccy:
                    return True, f"High-impact news: {name} ({evt_ccy})"
        
        return False, ""
    except Exception:
        return False, ""

def run_multi_symbol(
    symbols: List[str],
    tf_str: str,
    args,
    conn: 'MT5Connection',
    dashboard_state = None,
) -> None:
    """Run the symplectic forecaster on multiple symbols concurrently.
    
    Each symbol gets its own SymplecticForecaster + AutoTradingEngine.
    All share the same MT5 connection (thread-safe in MT5).
    """
    import concurrent.futures

    # ---- Pre-initialize shared heavy resources ONCE before threads ----
    shared_nlp = None
    try:
        from nlp_agent import FundamentalAgent
        global _GLOBAL_NLP_AGENT
        if _GLOBAL_NLP_AGENT is None:
            _GLOBAL_NLP_AGENT = FundamentalAgent()
        shared_nlp = _GLOBAL_NLP_AGENT
        shared_nlp.start(symbols)
    except ImportError:
        pass

    def run_symbol(sym: str):
        """Worker function for one symbol."""
        try:
            conn.ensure_symbol(sym)
            fc = SymplecticForecaster(
                window=args.window, alert_pct=0.95, min_train_bars=80,
                shared_nlp_agent=shared_nlp,
            )
            
            tf_mt5 = TIMEFRAME_MAP[tf_str]
            executor = None
            if args.auto_trade:
                risk_cfg = RiskConfig(
                    risk_per_trade_pct=args.risk_pct,
                    max_daily_loss_pct=args.max_daily_loss,
                    reward_risk_ratio=args.reward_risk,
                    atr_sl_multiplier=args.atr_sl,
                    trailing_atr_multiplier=args.trailing_atr,
                    max_positions=args.max_positions,
                    allow_live=args.allow_live,
                )
                executor = MT5TradeExecutor(risk_cfg, connection=conn, forecaster=fc)
                engine = AutoTradingEngine(
                    executor, confidence_threshold=args.confidence,
                    timeframe=tf_mt5, forecaster=fc
                )
                if getattr(args, 'news_filter', False):
                    engine._news_filter_enabled = True
            else:
                engine = TradingEngine(confidence_threshold=args.confidence)
                
            if dashboard_state:
                with dashboard_state.lock:
                    if sym not in dashboard_state.symbols_data:
                        dashboard_state.symbols_data[sym] = {
                            "symbol": sym,
                            "timeframe": tf_str,
                            "chart_records": [],
                            "phase_points": [],
                            "hull_points": [],
                            "scenarios": {},
                            "latest_kpi": {},
                            "hits_records": [],
                            "engine": None,
                            "updates_count": 0
                        }
                    dashboard_state.symbols_data[sym]["engine"] = engine
            
            # Train
            state_path = str(SymplecticForecaster.default_state_path(
                sym, tf_str, args.state_dir))
            if Path(state_path).exists():
                try:
                    fc.load_state(state_path, symbol=sym, timeframe=tf_str, executor=executor)
                    print(f"[{sym}] Resumed from saved state.")
                except Exception:
                    fc.train_on_mt5(sym, tf_str, args.bars, connection=conn)
            else:
                fc.train_on_mt5(sym, tf_str, args.bars, connection=conn)
            
            # Run live
            on_poll = engine.on_poll if args.auto_trade else None
            fc.run_live_mt5(
                sym, tf_str,
                poll_interval=args.poll if args.poll > 0 else 0.0,
                on_signal=engine.on_signal,
                on_poll=on_poll,
                connection=conn,
                dashboard_state=dashboard_state,
            )
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
            import traceback; traceback.print_exc()
    
    print(f"\n{'=' * 66}")
    print(f"  MULTI-SYMBOL PORTFOLIO SCANNER")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Timeframe: {tf_str}")
    print(f"{'=' * 66}\n")
    
    # Run each symbol in its own thread
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(len(symbols), 10)
    ) as pool:
        futures = {pool.submit(run_symbol, sym): sym for sym in symbols}
        try:
            concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            print("\n[MULTI] Stopping all symbol monitors...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Symplectic ML Price Forecaster — MetaTrader 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python symplectic_forecaster.py --symbol EURUSD --timeframe H1
  python symplectic_forecaster.py --symbol XAUUSD --timeframe D1 --bars 2000
  python symplectic_forecaster.py                          (interactive mode)
  python symplectic_forecaster.py --symbol BTCUSD --timeframe M5 --confidence 0.7
        """,
    )
    parser.add_argument("--symbol", type=str, default=None,
                        help="MT5 symbol to trade (e.g., EURUSD, XAUUSD, US500)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of symbols for multi-asset scanning")
    parser.add_argument("--news-filter", action="store_true",
                        help="Enable economic calendar news blackout filter")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Timeframe: M1,M5,M15,M30,H1,H4,D1,W1,MN1")
    parser.add_argument("--bars", type=int, default=1000,
                        help="Historical bars for training (default: 1000)")
    parser.add_argument("--account", type=int, default=None,
                        help="MT5 account number (optional if terminal logged in)")
    parser.add_argument("--password", type=str, default=None,
                        help="MT5 account password (optional)")
    parser.add_argument("--server", type=str, default=None,
                        help="MT5 broker server (optional)")
    parser.add_argument("--mt5-path", type=str, default=None,
                        help="Path to terminal64.exe (optional, auto-detected)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Confidence threshold for BUY/SELL signals (default: 0.4)")
    parser.add_argument("--optimize", action="store_true",
                        help="Grid-search confidence/risk/R:R on backtest data")
    parser.add_argument("--opt-confidence", type=str, default="0.2,0.3,0.4,0.5,0.6",
                        help="Comma-separated confidence values for optimizer")
    parser.add_argument("--opt-risk", type=str, default="0.5,1.0,1.5",
                        help="Comma-separated risk %% values for optimizer")
    parser.add_argument("--opt-reward-risk", type=str, default="1.5,2.0,2.5",
                        help="Comma-separated R:R values for optimizer")
    parser.add_argument("--window", type=int, default=60,
                        help="Symplectic rolling window size (default: 60)")
    parser.add_argument("--poll", type=float, default=0.0,
                        help="Custom poll interval in seconds (0 = auto)")
    parser.add_argument("--auto-trade", action="store_true",
                        help="Enable automatic order execution (demo by default)")
    parser.add_argument("--allow-live", action="store_true",
                        help="Allow trading on live accounts (requires --auto-trade)")
    parser.add_argument("--risk-pct", type=float, default=1.0,
                        help="Risk per trade as %% of equity (default: 1.0)")
    parser.add_argument("--max-daily-loss", type=float, default=3.0,
                        help="Max daily loss %% before halting (default: 3.0)")
    parser.add_argument("--reward-risk", type=float, default=2.0,
                        help="Take-profit / stop-loss ratio (default: 2.0)")
    parser.add_argument("--atr-sl", type=float, default=1.5,
                        help="ATR multiplier for stop-loss (default: 1.5)")
    parser.add_argument("--trailing-atr", type=float, default=2.0,
                        help="ATR multiplier for trailing stop (default: 2.0)")
    parser.add_argument("--max-positions", type=int, default=1,
                        help="Max concurrent positions per symbol (default: 1)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run out-of-sample backtest instead of live monitoring")
    parser.add_argument("--train-bars", type=int, default=1000,
                        help="Training bars for backtest (default: 1000)")
    parser.add_argument("--test-bars", type=int, default=500,
                        help="Out-of-sample bars for backtest (default: 500)")
    parser.add_argument("--initial-balance", type=float, default=10000.0,
                        help="Starting balance for backtest (default: 10000)")
    parser.add_argument("--spread-pips", type=float, default=1.0,
                        help="Simulated spread in pips for backtest (default: 1.0)")
    parser.add_argument("--freeze-model", action="store_true",
                        help="Do not update model during backtest test period")
    parser.add_argument("--state-dir", type=str, default="states",
                        help="Directory for saved model state files (default: states)")
    parser.add_argument("--load-state", type=str, default=None,
                        help="Load model state from pickle file (skips training)")
    parser.add_argument("--no-save-state", action="store_true",
                        help="Disable auto-save of model state on exit")
    args = parser.parse_args()

    # ── Banner ──
    CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"
    if args.optimize:
        mode_str = "Walk-Forward Optimizer"
    elif args.backtest:
        mode_str = "Backtest"
    elif args.auto_trade:
        mode_str = "Auto-Trade"
    else:
        mode_str = "Signal-Only (no auto-execution)"
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}SYMPLECTIC ML PRICE FORECASTER — MetaTrader 5{RESET}")
    print(f"  {DIM}Based on: Mishra (2026) · Shultz (2023) · Mantegna (1999){RESET}")
    print(f"  {DIM}Mode: {mode_str}{RESET}")
    if args.auto_trade:
        print(f"  {DIM}Risk: {args.risk_pct}%/trade | Max daily loss: {args.max_daily_loss}% | "
              f"R:R = 1:{args.reward_risk}{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}\n")

    # ── Step 1: Connect to MT5 ──
    if not HAS_MT5:
        print("[ERROR] MetaTrader5 package not available. Cannot proceed.")
        print("        Install via:  pip install MetaTrader5")
        sys.exit(1)

    conn = MT5Connection()
    try:
        conn.connect(
            account=args.account,
            password=args.password,
            server=args.server,
            path=args.mt5_path,
        )
    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    global_dashboard_state = None
    if not args.backtest and not args.optimize:
        global_dashboard_state = DashboardState()
        srv_thread = threading.Thread(
            target=start_dashboard_server,
            args=(global_dashboard_state, 8080,
                  os.path.join(os.path.dirname(__file__), "dashboard.html")),
            daemon=True
        )
        srv_thread.start()

    engine = None
    fc = None
    symbol = None
    tf_str = None
    state_loaded = False

    try:
        # ── Multi-symbol execution mode bypass ──
        if args.symbols:
            tf_str = args.timeframe
            if not tf_str:
                print(f"\n  {BOLD}Select timeframe:{RESET}")
                print(f"    {DIM}Minutes : M1  M5  M15  M30{RESET}")
                print(f"    {DIM}Hours   : H1  H4  H12{RESET}")
                print(f"    {DIM}Daily+  : D1  W1  MN1{RESET}")
                tf_str = input(f"  {CYAN}Timeframe>{RESET} ").strip().upper()
                if not tf_str:
                    tf_str = "D1"
                    print(f"  {DIM}(defaulting to D1){RESET}")
            else:
                tf_str = tf_str.upper()

            if tf_str not in TIMEFRAME_MAP:
                print(f"[ERROR] Unknown timeframe '{tf_str}'.")
                print(f"        Valid: {', '.join(sorted(TIMEFRAME_MAP.keys()))}")
                conn.disconnect()
                sys.exit(1)

            sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
            if sym_list:
                run_multi_symbol(sym_list, tf_str, args, conn, global_dashboard_state)
                conn.disconnect()
                sys.exit(0)

        # ── Step 2: Get symbol (interactive or CLI) ──
        symbol = args.symbol
        if not symbol:
            print(f"\n  {BOLD}Enter the symbol you want to analyze:{RESET}")
            print(f"  {DIM}(e.g., EURUSD, GBPUSD, XAUUSD, US500, BTCUSD){RESET}")
            symbol = input(f"  {CYAN}Symbol> {RESET} ").strip()
            if not symbol:
                print("[ERROR] No symbol entered.")
                conn.disconnect()
                sys.exit(1)

        # Validate symbol in MT5
        conn.ensure_symbol(symbol)

        # ── Step 3: Get timeframe (interactive or CLI) ──
        tf_str = args.timeframe
        if not tf_str:
            print(f"\n  {BOLD}Select timeframe:{RESET}")
            print(f"    {DIM}Minutes : M1  M5  M15  M30{RESET}")
            print(f"    {DIM}Hours   : H1  H4  H12{RESET}")
            print(f"    {DIM}Daily+  : D1  W1  MN1{RESET}")
            tf_str = input(f"  {CYAN}Timeframe>{RESET} ").strip().upper()
            if not tf_str:
                tf_str = "D1"
                print(f"  {DIM}(defaulting to D1){RESET}")
        else:
            tf_str = tf_str.upper()

        if tf_str not in TIMEFRAME_MAP:
            print(f"[ERROR] Unknown timeframe '{tf_str}'.")
            print(f"        Valid: {', '.join(sorted(TIMEFRAME_MAP.keys()))}")
            conn.disconnect()
            sys.exit(1)

        # ── Step 4: Initialize forecaster + trading engine ──
        fc = SymplecticForecaster(
            window=args.window,
            alert_pct=0.95,
            min_train_bars=80,
        )
        tf_mt5 = TIMEFRAME_MAP[tf_str]
        if args.auto_trade:
            risk_cfg = RiskConfig(
                risk_per_trade_pct=args.risk_pct,
                max_daily_loss_pct=args.max_daily_loss,
                reward_risk_ratio=args.reward_risk,
                atr_sl_multiplier=args.atr_sl,
                trailing_atr_multiplier=args.trailing_atr,
                max_positions=args.max_positions,
                allow_live=args.allow_live,
            )
            executor = MT5TradeExecutor(risk_cfg, connection=conn, forecaster=fc)
            engine = AutoTradingEngine(
                executor, confidence_threshold=args.confidence, timeframe=tf_mt5, forecaster=fc
            )
            if args.news_filter:
                engine._news_filter_enabled = True
        else:
            engine = TradingEngine(confidence_threshold=args.confidence)
            
        if global_dashboard_state:
            global_dashboard_state.engine = engine

        risk_cfg = RiskConfig(
            risk_per_trade_pct=args.risk_pct,
            max_daily_loss_pct=args.max_daily_loss,
            reward_risk_ratio=args.reward_risk,
            atr_sl_multiplier=args.atr_sl,
            trailing_atr_multiplier=args.trailing_atr,
            max_positions=args.max_positions,
            allow_live=args.allow_live,
        )

        # ── Step 5: Load saved state or train ──
        state_path = args.load_state
        if state_path is None and not args.no_save_state and not args.backtest:
            state_path = str(SymplecticForecaster.default_state_path(
                symbol, tf_str, args.state_dir))

        if args.load_state:
            fc.load_state(args.load_state, symbol=symbol, timeframe=tf_str, executor=executor if 'executor' in locals() else None)
            state_loaded = True
        elif (state_path and Path(state_path).exists()
              and not args.backtest and not args.load_state):
            try:
                fc.load_state(state_path, symbol=symbol, timeframe=tf_str, executor=executor if 'executor' in locals() else None)
                state_loaded = True
            except (ValueError, FileNotFoundError) as e:
                print(f"[STATE] Could not load existing state: {e}")
                print(f"[STATE] Will train from scratch.")

        if args.backtest or args.optimize:
            if args.optimize:
                run_optimizer_cli(
                    fc=fc,
                    symbol=symbol,
                    tf_str=tf_str,
                    train_bars=args.train_bars if not state_loaded else 0,
                    test_bars=args.test_bars,
                    connection=conn,
                    confidence_grid=_parse_float_list(
                        args.opt_confidence, [0.2, 0.3, 0.4, 0.5, 0.6]),
                    risk_grid=_parse_float_list(args.opt_risk, [0.5, 1.0, 1.5]),
                    reward_risk_grid=_parse_float_list(
                        args.opt_reward_risk, [1.5, 2.0, 2.5]),
                    initial_balance=args.initial_balance,
                    spread_pips=args.spread_pips,
                    export_path=f"optimize_{symbol}_{tf_str}.csv",
                    chart_path=f"optimize_{symbol}_{tf_str}_equity.png",
                )
            else:
                run_backtest_cli(
                    fc=fc,
                    symbol=symbol,
                    tf_str=tf_str,
                    train_bars=args.train_bars if not state_loaded else 0,
                    test_bars=args.test_bars,
                    engine=engine,
                    risk_config=risk_cfg,
                    connection=conn,
                    initial_balance=args.initial_balance,
                    spread_pips=args.spread_pips,
                    freeze_model=args.freeze_model,
                    export_path=f"backtest_{symbol}_{tf_str}.csv",
                )
            if not args.no_save_state:
                save_path = str(SymplecticForecaster.default_state_path(
                    symbol, tf_str, args.state_dir))
                fc.save_state(save_path, symbol, tf_str, executor=executor if 'executor' in locals() else None)
        else:
            if not state_loaded:
                print()
                rdf = fc.train_on_mt5(symbol, tf_str, args.bars, connection=conn)
                export_path = f"symplectic_{symbol}_{tf_str}.csv"
                fc.export_results(rdf, export_path)
            else:
                print(f"[INFO] Resuming from saved state — skipping historical training.")

            # ── Step 6: Print initial forecast / signal ──
            result = fc.forecast(horizon=5)
            if "error" not in result:
                initial_forecast = {
                    **result,
                    "close": result["current_price"],
                    "alert": result["regime"] == "ALERT",
                    "capacity": result["current_capacity"],
                    "betti_1": result["current_betti_1"],
                }

                if global_dashboard_state:
                    pts = np.array(list(fc._phase_buf), dtype=float)
                    hull_verts = get_convex_hull_vertices(pts)
                    acc_info = conn.get_account_info() if conn else {}
                    global_dashboard_state.update_live_metrics(
                        symbol=symbol,
                        timeframe=tf_str,
                        latest_forecast=initial_forecast,
                        phase_buf=list(fc._phase_buf),
                        hull_points=hull_verts.tolist(),
                        acc_info=acc_info,
                        total_updates=fc._model._n_updates
                    )

                engine.on_signal(initial_forecast, symbol)

            # ── Multi-symbol mode ──
            if args.symbols:
                sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
                if sym_list:
                    run_multi_symbol(sym_list, tf_str, args, conn, global_dashboard_state)
                    raise SystemExit(0)

            # ── Step 8: Start live monitoring ──
            print(f"\n{BOLD}[INFO] Starting live monitoring...{RESET}")
            print(f"{DIM}       The model continues learning from each new bar.{RESET}")
            poll = args.poll if args.poll > 0 else 0.0
            on_poll = engine.on_poll if args.auto_trade else None
            fc.run_live_mt5(
                symbol, tf_str,
                poll_interval=poll,
                on_signal=engine.on_signal,
                on_poll=on_poll,
                connection=conn,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not args.backtest and not args.optimize and fc is not None and symbol and tf_str:
            if not args.no_save_state:
                save_path = args.load_state or str(
                    SymplecticForecaster.default_state_path(
                        symbol, tf_str, args.state_dir))
                try:
                    fc.save_state(save_path, symbol, tf_str, executor=executor if 'executor' in locals() else None)
                except Exception as e:
                    print(f"[STATE] Auto-save failed: {e}")

        if engine is not None and not args.backtest and not args.optimize:
            if args.auto_trade and isinstance(engine, AutoTradingEngine):
                summary = engine.trade_summary()
                s = summary["signals"]
                t = summary["trades"]
                print(f"\n  {BOLD}Signal Summary:{RESET} {s['total_signals']} total — "
                      f"\033[92mBUY: {s['buys']}\033[0m | "
                      f"\033[91mSELL: {s['sells']}\033[0m | "
                      f"\033[93mHOLD: {s['holds']}\033[0m")
                print(f"  {BOLD}Trade Summary:{RESET} "
                      f"Opened: {t['OPEN']} | Closed: {t['CLOSE']} | "
                      f"Modified: {t['MODIFY']} | Skipped: {t['SKIP']} | "
                      f"Errors: {t['ERROR']}")
            else:
                s = engine.summary()
                print(f"\n  {BOLD}Signal Summary:{RESET} {s['total_signals']} total — "
                      f"\033[92mBUY: {s['buys']}\033[0m | "
                      f"\033[91mSELL: {s['sells']}\033[0m | "
                      f"\033[93mHOLD: {s['holds']}\033[0m")

        conn.disconnect()
        if args.no_save_state:
            print(f"\n{BOLD}[DONE]{RESET} Session ended (state not saved).")
        else:
            print(f"\n{BOLD}[DONE]{RESET} Session ended. Model state preserved in "
                  f"'{args.state_dir}/'.")
