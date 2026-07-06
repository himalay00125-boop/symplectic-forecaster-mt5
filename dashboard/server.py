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
    from nlp_agent import FundamentalAgent
    HAS_NLP = True
except ImportError:
    HAS_NLP = False
    
try:
    from causal_discovery import learn_causal_graph
    HAS_CAUSAL = True
except ImportError:
    HAS_CAUSAL = False

# Import microservice workers
try:
    from ai_engine import ai_worker_loop
    from causal_optimizer import causal_optimizer_loop
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")

# Thread-safe global reference to dashboard state
global_dashboard_state = None

# ---------------------------------------------------------------------------
# Optional heavy dependencies — graceful fallback if unavailable
# ---------------------------------------------------------------------------
try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("[WARN] ripser not found — TDA features will be approximated.")

try:
    from river import linear_model, preprocessing, tree, metrics, optim, ensemble
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    print("[WARN] river not found — falling back to sklearn PARegressor.")
    from sklearn.linear_model import SGDRegressor

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
from analytics.trade_analytics import TradeAnalytics
import threading
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.symbol = "--"
        self.timeframe = "--"
        self.chart_records = []  # max length 200
        self.phase_points = []
        self.hull_points = []
        self.account_info = {}
        self.scenarios = {}
        self.updates_count = 0
        self.hits_records = []  # rolling 30-bar accuracy points
        self.engine = None
        self.latest_kpi = {
            "last_close": "--", "date": "--", "forecast_pct": "--", "forecast_sub": "--",
            "capacity": "--", "betti_1": "--", "mae": "--", "updates": "0", "regime": "NORMAL"
        }
        self.symbols_data = {}  # symbol -> dict of state for multi-symbol support

    def update_live_metrics(self, symbol: str, timeframe: str, latest_forecast: dict, phase_buf: list, hull_points: list, acc_info: dict, total_updates: int):
        with self.lock:
            # ── Legacy/single-symbol fields (fallback/default) ──
            self.symbol = symbol
            self.timeframe = timeframe
            self.account_info = acc_info
            self.updates_count = total_updates
            
            # Format scenarios
            if "scenarios" in latest_forecast and latest_forecast["scenarios"]:
                self.scenarios = {
                    "bull_p5": f"${latest_forecast['scenarios']['bull'][-1]:.5f}",
                    "base_p5": f"${latest_forecast['scenarios']['base'][-1]:.5f}",
                    "bear_p5": f"${latest_forecast['scenarios']['bear'][-1]:.5f}",
                    "bull_path_str": " → ".join(f"{p:.5f}" for p in latest_forecast['scenarios']['bull']),
                    "base_path_str": " → ".join(f"{p:.5f}" for p in latest_forecast['scenarios']['base']),
                    "bear_path_str": " → ".join(f"{p:.5f}" for p in latest_forecast['scenarios']['bear']),
                    "upper_band": latest_forecast.get("upper_band", []),
                    "lower_band": latest_forecast.get("lower_band", []),
                }
            else:
                self.scenarios = {
                    "bull_p5": "--", "base_p5": "--", "bear_p5": "--",
                    "bull_path_str": "--", "base_path_str": "--", "bear_path_str": "--",
                    "upper_band": [], "lower_band": []
                }
            
            ts = latest_forecast.get("timestamp", time.time())
            if isinstance(ts, (int, float)):
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            else:
                dt_str = str(ts)
            
            direction = latest_forecast.get("direction", 0)
            confidence = latest_forecast.get("confidence", 0.0)
            pred_ret = latest_forecast.get("predicted_return", latest_forecast.get("forecast", 0.0))
            
            forecast_pct = f"{pred_ret:+.4%}"
            if direction > 0:
                forecast_sub = f"↑ bullish signal ({confidence:.1%} conf)"
            elif direction < 0:
                forecast_sub = f"↓ bearish signal ({confidence:.1%} conf)"
            else:
                forecast_sub = f"→ hold/neutral signal"
            
            mae_val = latest_forecast.get("mae_ht") or latest_forecast.get("mae_pa") or 0.001
            self.latest_kpi = {
                "last_close": f"{latest_forecast.get('close', 0.0):.5f}",
                "date": dt_str,
                "forecast_pct": forecast_pct,
                "forecast_sub": forecast_sub,
                "capacity": f"{latest_forecast.get('capacity', 0.0):.6f}",
                "betti_1": str(latest_forecast.get('betti_1', 0)),
                "mae": f"{mae_val:.6f}",
                "updates": str(total_updates),
                "regime": "ALERT" if latest_forecast.get("alert", False) else "NORMAL",
                "val_status": str(latest_forecast.get("val_status", {})),
                "regime_transition_prob": f"{latest_forecast.get('regime_transition_prob', 0.0):.1%}",
                "feature_attribution": str(latest_forecast.get("feature_attribution", {}))
            }
            
            self.phase_points = [{"q": float(p[0]), "p": float(p[1])} for p in phase_buf]
            self.hull_points = [{"q": float(p[0]), "p": float(p[1])} for p in hull_points]
            
            new_chart_rec = {
                "d": dt_str.split(" ")[-1] if " " in dt_str else dt_str,
                "c": float(latest_forecast.get("close", 0.0)),
                "cap": float(latest_forecast.get("capacity", 0.0)),
                "b1": int(latest_forecast.get("betti_1", 0)),
                "fc": float(latest_forecast.get("forecast", 0.0)),
                "ret": float(latest_forecast.get("log_return", 0.0)),
                "al": 1 if latest_forecast.get("alert", False) else 0,
                "tp": float(latest_forecast.get("tot_pers", 0.0))
            }
            
            if not self.chart_records or self.chart_records[-1]["d"] != new_chart_rec["d"]:
                self.chart_records.append(new_chart_rec)
                if len(self.chart_records) > 200:
                    self.chart_records.pop(0)
            
            self._recalculate_accuracy()

            # ── Multi-symbol specific storage ──
            if symbol not in self.symbols_data:
                self.symbols_data[symbol] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "chart_records": [],
                    "phase_points": [],
                    "hull_points": [],
                    "scenarios": {},
                    "latest_kpi": {},
                    "hits_records": [],
                    "engine": None,
                    "updates_count": 0
                }
            
            sym_data = self.symbols_data[symbol]
            sym_data["timeframe"] = timeframe
            sym_data["updates_count"] = total_updates
            sym_data["phase_points"] = [{"q": float(p[0]), "p": float(p[1])} for p in phase_buf]
            sym_data["hull_points"] = [{"q": float(p[0]), "p": float(p[1])} for p in hull_points]
            sym_data["scenarios"] = self.scenarios.copy()
            sym_data["latest_kpi"] = self.latest_kpi.copy()
            
            sym_chart_recs = sym_data["chart_records"]
            if not sym_chart_recs or sym_chart_recs[-1]["d"] != new_chart_rec["d"]:
                sym_chart_recs.append(new_chart_rec)
                if len(sym_chart_recs) > 200:
                    sym_chart_recs.pop(0)
            
            self._recalculate_sym_accuracy(sym_data)

    def load_historical_dataframe(self, df: pd.DataFrame):
        with self.lock:
            self.chart_records = []
            for i, row in df.iterrows():
                dt_str = row.get("date", str(row.get("timestamp", "")))
                short_date = dt_str.split(" ")[-1] if " " in dt_str else dt_str
                self.chart_records.append({
                    "d": short_date,
                    "c": float(row.get("close", 0.0)),
                    "cap": float(row.get("capacity", 0.0)),
                    "b1": int(row.get("betti_1", 0)),
                    "fc": float(row.get("forecast", 0.0)),
                    "ret": float(row.get("log_return", 0.0)),
                    "al": 1 if row.get("alert", False) else 0,
                    "tp": float(row.get("tot_pers", 0.0))
                })
            
            if len(self.chart_records) > 200:
                self.chart_records = self.chart_records[-200:]
                
            self._recalculate_accuracy()

    def _recalculate_accuracy(self):
        self.hits_records = []
        n_records = len(self.chart_records)
        if n_records < 31:
            if n_records > 1:
                self.hits_records = [{"d": r["d"], "a": 50.0} for r in self.chart_records[1::10]]
            return
            
        step = max(1, (n_records - 30) // 12)
        for end_idx in range(30, n_records, step):
            window_recs = self.chart_records[end_idx - 30 : end_idx]
            correct = 0
            total = 0
            for j in range(len(window_recs) - 1):
                pred_direction = window_recs[j]["fc"]
                actual_return = window_recs[j+1]["ret"]
                
                if abs(pred_direction) > 1e-8:
                    total += 1
                    if (pred_direction > 0 and actual_return > 0) or (pred_direction < 0 and actual_return < 0):
                        correct += 1
            
            acc = (correct / total * 100) if total > 0 else 50.0
            self.hits_records.append({
                "d": self.chart_records[end_idx]["d"],
                "a": round(acc, 1)
            })

    def _recalculate_sym_accuracy(self, sym_data: dict):
        sym_data["hits_records"] = []
        chart_records = sym_data["chart_records"]
        n_records = len(chart_records)
        if n_records < 31:
            if n_records > 1:
                sym_data["hits_records"] = [{"d": r["d"], "a": 50.0} for r in chart_records[1::10]]
            return
            
        step = max(1, (n_records - 30) // 12)
        for end_idx in range(30, n_records, step):
            window_recs = chart_records[end_idx - 30 : end_idx]
            correct = 0
            total = 0
            for j in range(len(window_recs) - 1):
                pred_direction = window_recs[j]["fc"]
                actual_return = window_recs[j+1]["ret"]
                
                if abs(pred_direction) > 1e-8:
                    total += 1
                    if (pred_direction > 0 and actual_return > 0) or (pred_direction < 0 and actual_return < 0):
                        correct += 1
            
            acc = (correct / total * 100) if total > 0 else 50.0
            sym_data["hits_records"].append({
                "d": chart_records[end_idx]["d"],
                "a": round(acc, 1)
            })

    def to_json_dict(self, target_symbol: str = None) -> dict:
        with self.lock:
            symbols_list = list(self.symbols_data.keys())
            
            # Select target symbol
            if not target_symbol:
                if symbols_list:
                    target_symbol = symbols_list[0]
                else:
                    target_symbol = self.symbol
            
            if target_symbol in self.symbols_data:
                sym_data = self.symbols_data[target_symbol]
                symbol = target_symbol
                timeframe = sym_data.get("timeframe", self.timeframe)
                kpis = dict(sym_data.get("latest_kpi", self.latest_kpi))
                scenarios = dict(sym_data.get("scenarios", self.scenarios))
                chart_records = list(sym_data.get("chart_records", self.chart_records))
                hits_records = list(sym_data.get("hits_records", self.hits_records))
                engine = sym_data.get("engine", self.engine)
            else:
                symbol = self.symbol
                timeframe = self.timeframe
                kpis = dict(self.latest_kpi)
                scenarios = dict(self.scenarios)
                chart_records = list(self.chart_records)
                hits_records = list(self.hits_records)
                engine = self.engine
            
            trade_log_data = []
            equity_curve = []
            portfolio_stats = {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": "0.0", "total_pnl_pct": "0.00",
                "open_positions": 0, "profit_factor": "0.00",
                "avg_win": "0.00", "avg_loss": "0.00",
                "best_trade": "0.00", "worst_trade": "0.00",
                "max_drawdown": "0.00", "win_streak": 0, "loss_streak": 0,
                "expectancy": "0.00"
            }
            
            if engine and hasattr(engine, "executor"):
                log = engine.executor.trade_log
                trade_log_data = [
                    {
                        "ticket": t.ticket,
                        "action": t.action,
                        "volume": t.volume,
                        "price": round(t.price, 5),
                        "sl": round(t.sl, 5),
                        "tp": round(t.tp, 5),
                        "success": t.success,
                        "pnl": round(t.realized_pnl * 100, 2) if t.action == "CLOSE" else None,
                        "msg": t.message
                    }
                    for t in reversed(log[-50:])
                ]
                closed_trades = [t for t in log if t.action == "CLOSE" and t.success]
                successful_opens = [t for t in log if t.action == "OPEN" and t.success]
                
                if closed_trades:
                    records = getattr(engine.executor, "trade_records", [])
                    closed_records = [r for r in records if r.is_closed()]
                    
                    pnls = [t.realized_pnl for t in closed_trades]
                    wins_list = [p for p in pnls if p > 0]
                    losses_list = [p for p in pnls if p <= 0]
                    n_wins = len(wins_list)
                    n_losses = len(losses_list)
                    win_rate = n_wins / len(pnls) * 100
                    total_pnl = sum(pnls)
                    gross_profit = sum(wins_list) if wins_list else 0.0
                    gross_loss = abs(sum(losses_list)) if losses_list else 0.0
                    pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
                    avg_win = (gross_profit / n_wins * 100) if n_wins > 0 else 0.0
                    avg_loss = (gross_loss / n_losses * 100) if n_losses > 0 else 0.0
                    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
                    
                    mean_pnl = sum(pnls) / len(pnls) if pnls else 0.0
                    variance = sum((p - mean_pnl)**2 for p in pnls) / len(pnls) if len(pnls) > 1 else 0.0
                    std_dev = variance ** 0.5
                    sharpe = (mean_pnl / std_dev * (252**0.5)) if std_dev > 0.0 else 0.0
                    
                    longs_won = sum(1 for r in closed_records if r.direction == "BUY" and r.is_win())
                    total_longs = sum(1 for r in closed_records if r.direction == "BUY")
                    shorts_won = sum(1 for r in closed_records if r.direction == "SELL" and r.is_win())
                    total_shorts = sum(1 for r in closed_records if r.direction == "SELL")
                    
                    avg_hold = sum(r.holding_bars for r in closed_records) / len(closed_records) if closed_records else 0
                    total_lots = sum(r.volume for r in closed_records)
                    
                    cumulative = 0.0
                    peak = 0.0
                    max_dd = 0.0
                    for i, p in enumerate(pnls):
                        cumulative += p * 100
                        equity_curve.append(round(cumulative, 2))
                        if cumulative > peak:
                            peak = cumulative
                        dd = peak - cumulative
                        if dd > max_dd:
                            max_dd = dd
                            
                    cur_win_streak = 0
                    max_win_streak = 0
                    cur_loss_streak = 0
                    max_loss_streak = 0
                    wins_seq = []
                    for p in pnls:
                        wins_seq.append(1 if p > 0 else 0)
                        if p > 0:
                            cur_win_streak += 1
                            cur_loss_streak = 0
                            if cur_win_streak > max_win_streak:
                                max_win_streak = cur_win_streak
                        else:
                            cur_loss_streak += 1
                            cur_win_streak = 0
                            if cur_loss_streak > max_loss_streak:
                                max_loss_streak = cur_loss_streak
                                
                    R = sum(1 for i in range(1, len(wins_seq)) if wins_seq[i] != wins_seq[i-1]) + 1
                    P = 2 * n_wins * n_losses
                    N = len(wins_seq)
                    z_score = 0.0
                    if N > 1 and P > 0:
                        exp_R = (P / N) + 1
                        std_R = ((P * (P - N)) / ((N ** 2) * (N - 1))) ** 0.5
                        if std_R > 0:
                            z_score = (R - exp_R) / std_R
                            
                    portfolio_stats = {
                        "total_trades": len(pnls),
                        "wins": n_wins,
                        "losses": n_losses,
                        "win_rate": f"{win_rate:.1f}",
                        "total_pnl_pct": f"{total_pnl * 100:.2f}",
                        "open_positions": max(0, len(successful_opens) - len(closed_trades)),
                        "profit_factor": f"{pf:.2f}",
                        "avg_win": f"{avg_win:.2f}",
                        "avg_loss": f"{avg_loss:.2f}",
                        "best_trade": f"{max(pnls) * 100:.2f}",
                        "worst_trade": f"{min(pnls) * 100:.2f}",
                        "max_drawdown": f"{max_dd:.2f}",
                        "win_streak": max_win_streak,
                        "loss_streak": max_loss_streak,
                        "expectancy": f"{expectancy:.2f}",
                        "std_dev": f"{std_dev * 100:.2f}",
                        "sharpe": f"{sharpe:.2f}",
                        "longs_won": longs_won,
                        "total_longs": total_longs,
                        "shorts_won": shorts_won,
                        "total_shorts": total_shorts,
                        "z_score": f"{z_score:.2f}",
                        "avg_hold": f"{avg_hold:.1f}",
                        "total_lots": f"{total_lots:.2f}"
                    }
                else:
                    portfolio_stats["open_positions"] = max(0, len(successful_opens))
            
            return {
                "meta": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "account": self.account_info.get("login", ""),
                    "balance": f"{self.account_info.get('balance', 0.0):.2f}" if 'balance' in self.account_info else "--",
                    "equity": f"{self.account_info.get('equity', 0.0):.2f}" if 'equity' in self.account_info else "--",
                    "currency": self.account_info.get("currency", "USD"),
                    "leverage": str(self.account_info.get("leverage", "100")),
                    "server": self.account_info.get("server", ""),
                    "trade_mode": self.account_info.get("trade_mode", "Demo"),
                    "symbols": symbols_list,
                    "active_symbol": symbol
                },
                "kpis": kpis,
                "scenarios": scenarios,
                "CHART": chart_records,
                "HITS": hits_records,
                "TRADES": trade_log_data,
                "EQUITY_CURVE": equity_curve,
                "PORTFOLIO": portfolio_stats
            }

class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    state = None
    dashboard_html_path = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.path)
        
        if parsed_url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            try:
                with open(self.dashboard_html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            except Exception as e:
                self.wfile.write(f"Error loading dashboard: {e}".encode("utf-8"))
        elif parsed_url.path == "/data":
            query_params = parse_qs(parsed_url.query)
            target_symbol = query_params.get("symbol", [None])[0]
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            data_dict = self.state.to_json_dict(target_symbol=target_symbol)
            self.wfile.write(json.dumps(data_dict).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

def start_dashboard_server(state: DashboardState, port: int = 8080, html_path: str = "dashboard.html"):
    DashboardHTTPRequestHandler.state = state
    DashboardHTTPRequestHandler.dashboard_html_path = html_path
    
    server = HTTPServer(("127.0.0.1", port), DashboardHTTPRequestHandler)
    print(f"\n\033[92m[DASHBOARD] Live web dashboard available at: http://localhost:{port}\033[0m")
    server.serve_forever()

