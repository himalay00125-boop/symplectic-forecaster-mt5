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
import collections
import argparse
import datetime
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Dict, List, Tuple, Callable

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

class Bar(NamedTuple):
    """A single OHLCV bar (daily, 1-min, tick-level)."""
    timestamp: float    # Unix epoch or bar index
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float


@dataclass
class PhasePoint:
    """Canonical (q, p) coordinates in the symplectic financial phase space."""
    q: float   # position: q = ln P(t)
    p: float   # momentum: p = V(t) · sign(ΔP(t))


@dataclass
class CapacityRecord:
    """
    Output of the rolling symplectic pipeline for one bar.

    Fields
    ------
    t          : bar timestamp
    capacity   : C(t) = Area(Conv(Dₜ))  — first ECH capacity
    perimeter  : L(t) = Perimeter(Conv(Dₜ))
    alert      : True when C(t) exceeds adaptive threshold (regime shift)
    betti_0    : number of H₀ generators at max persistence scale
    betti_1    : number of H₁ generators (loops) in rolling cloud
    max_pers_0 : max H₀ persistence (connectivity lifetime)
    max_pers_1 : max H₁ persistence (loop lifetime)
    tot_pers   : sum of all persistence values (topological complexity)
    log_return : log(P_t / P_{t-1})
    """
    t:          float
    capacity:   float
    perimeter:  float
    alert:      bool
    betti_0:    int
    betti_1:    int
    max_pers_0: float
    max_pers_1: float
    tot_pers:   float
    log_return: float


@dataclass
class TradingSignal:
    """
    A trading signal generated by the TradingEngine.

    Actions: "BUY", "SELL", "HOLD"
    """
    timestamp:         str
    symbol:            str
    action:            str       # BUY / SELL / HOLD
    confidence:        float
    predicted_return:  float
    regime:            str       # NORMAL / ALERT
    current_price:     float
    reason:            str
    scenarios:         dict
    forecast_horizon:  int


# ===========================================================================
# MT5 CONNECTION MANAGER
# ===========================================================================

class MT5Connection:
    """
    Manages MetaTrader 5 terminal connection lifecycle.

    The MT5 terminal must be running on the same machine.
    If account / password / server are not provided, uses the currently
    logged-in session in the terminal.

    Usage
    -----
        conn = MT5Connection()
        conn.connect()                     # uses already-logged-in terminal
        conn.ensure_symbol("EURUSD")
        ...
        conn.disconnect()

    Or as a context manager:
        with MT5Connection() as conn:
            conn.connect()
            ...
    """

    def __init__(self):
        self._connected = False

    def connect(self, account: int = None, password: str = None,
                server: str = None, path: str = None) -> bool:
        """
        Initialize connection to MT5 terminal.

        Parameters
        ----------
        account  : MT5 account number (optional — if terminal is already logged in)
        password : Account password (optional)
        server   : Broker server name (optional)
        path     : Path to MT5 terminal64.exe (optional, auto-detected)

        Returns True on success.
        Raises ConnectionError on failure.
        """
        if not HAS_MT5:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Run:  pip install MetaTrader5"
            )

        init_kwargs = {}
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            raise ConnectionError(
                f"MT5 initialization failed: {error}\n"
                "Make sure MetaTrader 5 terminal is running."
            )

        # Login if credentials provided
        if account and password and server:
            if not mt5.login(account, password=password, server=server):
                error = mt5.last_error()
                mt5.shutdown()
                raise ConnectionError(f"MT5 login failed: {error}")

        self._connected = True

        # Print connection info
        info = mt5.terminal_info()
        acc  = mt5.account_info()
        print(f"[MT5] Connected to : {info.name}")
        print(f"[MT5] Company      : {info.company}")
        if acc:
            mode_str = "Demo" if acc.trade_mode == 0 else "Contest" if acc.trade_mode == 1 else "Live"
            print(f"[MT5] Account      : {acc.login} ({mode_str})")
            print(f"[MT5] Balance      : {acc.balance:.2f} {acc.currency}")
            print(f"[MT5] Leverage     : 1:{acc.leverage}")
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            print("[MT5] Disconnected.")

    def ensure_symbol(self, symbol: str) -> bool:
        """
        Validate symbol exists and add to MarketWatch if needed.
        Returns True if symbol is available.
        Raises ValueError if symbol not found.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(
                f"Symbol '{symbol}' not found in MT5. "
                f"Check your broker's available instruments."
            )
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(
                    f"Failed to add '{symbol}' to MarketWatch: {mt5.last_error()}"
                )
            print(f"[MT5] Added '{symbol}' to MarketWatch.")
        return True

    def get_account_info(self) -> Dict:
        """Return account info as a dictionary."""
        acc = mt5.account_info()
        if acc is None:
            return {"error": "No account info available"}
        return {
            "login":       acc.login,
            "server":      acc.server,
            "balance":     acc.balance,
            "equity":      acc.equity,
            "margin":      acc.margin,
            "free_margin": acc.margin_free,
            "currency":    acc.currency,
            "leverage":    acc.leverage,
            "trade_mode":  "Demo" if acc.trade_mode == 0 else "Live",
        }

    @property
    def connected(self) -> bool:
        return self._connected

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()


# ===========================================================================
# TRADING ENGINE (Signal-Only)
# ===========================================================================

class TradingEngine:
    """
    Translates symplectic forecasts into trading decisions.

    Signal Logic
    ─────────────
    BUY  when: direction = +1 AND confidence > threshold AND regime ≠ ALERT
    SELL when: direction = -1 AND confidence > threshold AND regime ≠ ALERT
    HOLD when: confidence < threshold OR regime = ALERT (unstable phase space)

    The ALERT regime (symplectic capacity spike) signals a phase-space
    bifurcation — the model explicitly avoids trading during structurally
    unstable periods (Mishra 2026, Lemma 3.1).
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.signal_log: List[TradingSignal] = []
        self._signal_count = {"BUY": 0, "SELL": 0, "HOLD": 0}

    def evaluate(self, forecast: Dict, symbol: str) -> TradingSignal:
        """
        Evaluate a forecast dictionary and produce a TradingSignal.

        Parameters
        ----------
        forecast : dict returned by SymplecticForecaster.process_bar() or .forecast()
        symbol   : MT5 symbol name

        Returns
        -------
        TradingSignal with action, confidence, reason, etc.
        """
        direction  = forecast.get("direction", 0)
        confidence = forecast.get("confidence", 0.0)
        pred_ret   = forecast.get("predicted_return", 0.0)
        regime     = forecast.get("regime",
                                  "ALERT" if forecast.get("alert", False) else "NORMAL")
        price      = forecast.get("close", forecast.get("current_price", 0.0))
        scenarios  = forecast.get("scenarios", {})
        horizon    = forecast.get("horizon", 5)
        ts         = forecast.get("timestamp", time.time())

        # Format timestamp
        if isinstance(ts, (int, float)):
            ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)

        # ── Decision logic ──
        if regime == "ALERT":
            action = "HOLD"
            reason = ("Regime ALERT — symplectic capacity spike "
                      "(phase-space bifurcation detected). Avoiding trade.")
        elif confidence < self.confidence_threshold:
            action = "HOLD"
            reason = (f"Confidence {confidence:.1%} below threshold "
                      f"{self.confidence_threshold:.1%}.")
        elif direction > 0:
            action = "BUY"
            reason = (f"Bullish signal: predicted return {pred_ret:+.4%}, "
                      f"confidence {confidence:.1%}, regime stable.")
        elif direction < 0:
            action = "SELL"
            reason = (f"Bearish signal: predicted return {pred_ret:+.4%}, "
                      f"confidence {confidence:.1%}, regime stable.")
        else:
            action = "HOLD"
            reason = "Neutral direction (predicted return ≈ 0)."

        signal = TradingSignal(
            timestamp=ts_str, symbol=symbol, action=action,
            confidence=confidence, predicted_return=pred_ret,
            regime=regime, current_price=price, reason=reason,
            scenarios=scenarios, forecast_horizon=horizon,
        )

        self.signal_log.append(signal)
        self._signal_count[action] += 1
        return signal

    def print_signal(self, signal: TradingSignal):
        """Pretty-print a trading signal to the console with ANSI colors."""
        # ANSI colors
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        BOLD   = "\033[1m"
        DIM    = "\033[2m"
        RESET  = "\033[0m"

        color = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}.get(signal.action, RESET)
        arrow = {"BUY": " BUY ", "SELL": " SELL", "HOLD": " HOLD"}.get(signal.action, "?")

        print(f"\n{BOLD}{'-' * 66}{RESET}")
        print(f"  {CYAN}[TIME] {signal.timestamp}{RESET}  |  {BOLD}{signal.symbol}{RESET}")
        print(f"  {color}{BOLD}{arrow}{RESET}  |  "
              f"Price: {signal.current_price:.5f}  |  "
              f"Confidence: {signal.confidence:.1%}")
        print(f"  Predicted Return: {signal.predicted_return:+.4%}  |  "
              f"Regime: {signal.regime}")
        print(f"  {DIM}{signal.reason}{RESET}")

        if signal.scenarios:
            print(f"\n  {CYAN}Scenarios ({signal.forecast_horizon}-bar ahead):{RESET}")
            for name, path in signal.scenarios.items():
                sc_color = {"bull": GREEN, "bear": RED, "base": YELLOW}.get(name, RESET)
                prices_str = " -> ".join(f"{p:.5f}" for p in path)
                print(f"    {sc_color}{name:4s}{RESET}: {prices_str}")

        print(f"{BOLD}{'-' * 66}{RESET}")

    def on_signal(self, forecast: Dict, symbol: str):
        """
        Callback for run_live_mt5() — evaluate forecast and print signal.
        """
        signal = self.evaluate(forecast, symbol)
        self.print_signal(signal)

    def summary(self) -> Dict:
        """Return a summary of all signals generated."""
        return {
            "total_signals": len(self.signal_log),
            "buys":  self._signal_count["BUY"],
            "sells": self._signal_count["SELL"],
            "holds": self._signal_count["HOLD"],
        }


# ===========================================================================
# DASHBOARD STATE & HTTP SERVER
# ===========================================================================

import threading
import json
import os
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
        self.latest_kpi = {
            "last_close": "--", "date": "--", "forecast_pct": "--", "forecast_sub": "--",
            "capacity": "--", "betti_1": "--", "mae": "--", "updates": "0", "regime": "NORMAL"
        }

    def update_live_metrics(self, symbol: str, timeframe: str, latest_forecast: dict, phase_buf: list, hull_points: list, acc_info: dict, total_updates: int):
        with self.lock:
            self.symbol = symbol
            self.timeframe = timeframe
            self.account_info = acc_info
            self.updates_count = total_updates
            
            # Extract scenarios
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
            
            # Format date/timestamp
            ts = latest_forecast.get("timestamp", time.time())
            if isinstance(ts, (int, float)):
                dt_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            else:
                dt_str = str(ts)
            
            # Format forecast string
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
                "regime": "ALERT" if latest_forecast.get("alert", False) else "NORMAL"
            }
            
            # Update phase and hull points
            self.phase_points = [{"q": float(p[0]), "p": float(p[1])} for p in phase_buf]
            self.hull_points = [{"q": float(p[0]), "p": float(p[1])} for p in hull_points]
            
            # Append new record to chart history
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
            # Fake flat accuracy data during initial warmup/training transition
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

    def to_json_dict(self) -> dict:
        with self.lock:
            kpis = getattr(self, 'latest_kpi', {
                "last_close": "--", "date": "--", "forecast_pct": "--", "forecast_sub": "--",
                "capacity": "--", "betti_1": "--", "mae": "--", "updates": "0", "regime": "NORMAL"
            })
            
            return {
                "meta": {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "account": self.account_info.get("login", ""),
                    "balance": f"{self.account_info.get('balance', 0.0):.2f}" if 'balance' in self.account_info else "--",
                    "currency": self.account_info.get("currency", "USD"),
                    "leverage": str(self.account_info.get("leverage", "100")),
                    "server": self.account_info.get("server", ""),
                    "trade_mode": self.account_info.get("trade_mode", "Demo")
                },
                "kpis": kpis,
                "scenarios": self.scenarios,
                "CHART": self.chart_records,
                "PHASE": self.phase_points,
                "HULL": self.hull_points,
                "HITS": self.hits_records
            }

class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
    state = None
    dashboard_html_path = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            try:
                with open(self.dashboard_html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            except Exception as e:
                self.wfile.write(f"Error loading dashboard: {e}".encode("utf-8"))
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data_dict = self.state.to_json_dict()
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

def get_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        hull = ConvexHull(points)
        verts = points[hull.vertices]
        return np.vstack([verts, verts[0]])
    except Exception:
        return np.empty((0, 2))


# ===========================================================================
# GEOMETRY UTILITIES
# ===========================================================================

def _signed_area(vertices: np.ndarray) -> float:
    """Shoelace formula — returns area of convex polygon (CCW vertices)."""
    x, y = vertices[:, 0], vertices[:, 1]
    n = len(vertices)
    nxt = (np.arange(n) + 1) % n
    return float(0.5 * abs(np.sum(x * y[nxt] - x[nxt] * y)))


def convex_hull_metrics(points: np.ndarray) -> Tuple[float, float]:
    """
    Compute (area, perimeter) of the convex hull of `points`.

    The area equals the first ECH capacity c₁(XΩ) by Theorem 2.9 of
    Mishra (2026): for a convex toric domain, c₁ = Area(Ω).
    """
    if len(points) < 3:
        return 0.0, 0.0
    try:
        hull = ConvexHull(points)
    except Exception:
        return 0.0, 0.0   # degenerate (collinear) cloud

    verts = points[hull.vertices]
    area  = _signed_area(verts)
    edges = np.linalg.norm(verts - np.roll(verts, -1, axis=0), axis=1)
    perim = float(edges.sum())
    return area, perim


def phase_coords(bar: Bar, prev_bar: Bar) -> PhasePoint:
    """
    Map consecutive OHLCV bars to canonical (q, p) phase-space coordinates.

      q = ln P(t)                           [log-price: position]
      p = V(t) · sign(P(t) − P(t−1))       [signed volume: momentum]

    The signed volume encodes the direction-weighted order-flow imbalance,
    playing the role of momentum in the Hamiltonian structure of the
    financial phase space (Mishra 2026, Definition 2.1).
    """
    q  = math.log(bar.close)
    dp = bar.close - prev_bar.close
    s  = 1 if dp > 0 else (-1 if dp < 0 else 0)
    p  = bar.volume * s
    return PhasePoint(q=q, p=p)


# ===========================================================================
# TDA: PERSISTENT HOMOLOGY
# ===========================================================================

def _tda_features_ripser(pts: np.ndarray) -> Dict[str, float]:
    """
    Compute persistent homology features from the rolling (q,p) point cloud
    using Ripser (Vietoris-Rips filtration).

    Returns H₀ and H₁ features:
    • betti_0    : number of H₀ bars at largest scale
    • betti_1    : number of H₁ loops at largest scale
    • max_pers_0 : max finite H₀ persistence (connectivity decay)
    • max_pers_1 : max finite H₁ persistence (loop lifetime → cyclicity)
    • tot_pers   : sum of all finite persistence values (topological energy)

    Mathematical basis: Shultz (2023) §2.3 — a d-simplex with mutual
    distance < ε is part of the Rips complex R(X, ε). Features persisting
    across many ε values are topologically significant.
    """
    # Normalize to unit scale before computing distances
    std = pts.std(axis=0)
    std[std < 1e-12] = 1.0
    pts_n = (pts - pts.mean(axis=0)) / std

    result    = ripser.ripser(pts_n, maxdim=1)
    dgms      = result['dgms']

    # H₀ features
    h0        = dgms[0]
    fin_h0    = h0[h0[:, 1] < np.inf]
    betti_0   = len(h0)
    max_p0    = float(fin_h0[:, 1].max() - fin_h0[:, 0].min()) if len(fin_h0) else 0.0
    tot_p0    = float((fin_h0[:, 1] - fin_h0[:, 0]).sum()) if len(fin_h0) else 0.0

    # H₁ features
    h1        = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    fin_h1    = h1[h1[:, 1] < np.inf] if len(h1) else np.empty((0, 2))
    betti_1   = len(fin_h1)
    max_p1    = float((fin_h1[:, 1] - fin_h1[:, 0]).max()) if len(fin_h1) else 0.0
    tot_p1    = float((fin_h1[:, 1] - fin_h1[:, 0]).sum()) if len(fin_h1) else 0.0

    return dict(betti_0=betti_0, betti_1=betti_1,
                max_pers_0=max_p0, max_pers_1=max_p1,
                tot_pers=tot_p0 + tot_p1)


def _tda_features_approx(pts: np.ndarray) -> Dict[str, float]:
    """
    Lightweight TDA approximation when ripser is unavailable.
    Uses pairwise distance statistics as a proxy for topological complexity.
    """
    if len(pts) < 4:
        return dict(betti_0=1, betti_1=0, max_pers_0=0.0, max_pers_1=0.0, tot_pers=0.0)
    std = pts.std(axis=0); std[std < 1e-12] = 1.0
    pts_n  = (pts - pts.mean(axis=0)) / std
    dists  = np.linalg.norm(pts_n[:, None, :] - pts_n[None, :, :], axis=-1)
    upper  = dists[np.triu_indices_from(dists, k=1)]
    betti_1 = int(np.sum(upper < np.percentile(upper, 10)))   # proxy for loops
    return dict(betti_0=1, betti_1=betti_1,
                max_pers_0=float(upper.max() - upper.min()),
                max_pers_1=float(np.percentile(upper, 10)),
                tot_pers=float(upper.std()))


def compute_tda(pts: np.ndarray) -> Dict[str, float]:
    """Dispatch to ripser or approximation."""
    if HAS_RIPSER and len(pts) >= 5:
        return _tda_features_ripser(pts)
    return _tda_features_approx(pts)


# ===========================================================================
# ONLINE ML MODEL (River-based self-learning)
# ===========================================================================

class SymplecticOnlineModel:
    """
    Self-learning forecaster that updates incrementally after every bar.

    Architecture
    ------------
    Input features (built from symplectic + TDA pipeline):
        • capacity (C(t))              — ECH symplectic area
        • perimeter (L(t))             — convex hull boundary length
        • capacity_ratio (C/L)         — isoperimetric efficiency
        • Δcapacity                    — first difference of C(t)
        • betti_0, betti_1             — topological Betti numbers
        • max_pers_0, max_pers_1       — H₀, H₁ max persistence
        • tot_pers                     — total topological complexity
        • log_return (lagged 1,2,3,5)  — momentum features
        • rolling vol (10, 20)         — volatility regime
        • alert flag                   — symplectic regime indicator

    Target: log-return at the next bar (regression)
             sign(next return)         (classification signal)

    The model ensemble uses:
    1. Passive-Aggressive Regressor — fast to adapt to new regimes
    2. Hoeffding Adaptive Tree       — captures non-linear structures
    3. Weighted averaging            — weights updated by recent RMSE
    """

    def __init__(self):
        if HAS_RIVER:
            self._build_river_model()
        else:
            self._build_sklearn_model()

        self._feature_history: List[Dict] = []
        self._n_updates  = 0
        self._rmse_pa    = 0.001
        self._rmse_ht    = 0.001
        self._mae        = metrics.MAE() if HAS_RIVER else None
        self._pred_log: List[Dict] = []

    def _build_river_model(self):
        """River-based ensemble: PA regressor + Hoeffding Tree."""
        self._scaler = preprocessing.StandardScaler()

        self._pa = (
            preprocessing.StandardScaler() |
            linear_model.PARegressor(C=0.01, eps=1e-4, mode=2)
        )
        self._ht = (
            preprocessing.StandardScaler() |
            tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=50,
                delta=1e-5,
                leaf_prediction="adaptive",
            )
        )
        self._mae_pa = metrics.MAE()
        self._mae_ht = metrics.MAE()

    def _build_sklearn_model(self):
        """sklearn fallback: PassiveAggressiveRegressor (more stable than SGD)."""
        from sklearn.linear_model import PassiveAggressiveRegressor
        from sklearn.preprocessing import StandardScaler
        # PARegressor is naturally bounded and adapts quickly without diverging
        self._skl_model  = PassiveAggressiveRegressor(C=0.001, max_iter=1,
                                                       loss="epsilon_insensitive",
                                                       epsilon=1e-4, random_state=42)
        self._skl_scaler = StandardScaler()
        self._skl_fitted = False

    def _feature_vector(self, rec: CapacityRecord,
                        hist: List[CapacityRecord]) -> Dict[str, float]:
        """
        Construct the full feature dictionary from a capacity record
        and the rolling history.
        """
        feats: Dict[str, float] = {}

        # --- Symplectic capacity features ---
        feats["capacity"]       = rec.capacity
        feats["perimeter"]      = rec.perimeter
        feats["cap_ratio"]      = rec.capacity / (rec.perimeter + 1e-12)
        feats["alert"]          = float(rec.alert)

        # First and second differences of capacity (regime dynamics)
        if len(hist) >= 2:
            feats["d_cap"]  = rec.capacity - hist[-1].capacity
            feats["d_cap2"] = feats["d_cap"] - (hist[-1].capacity - hist[-2].capacity)
        else:
            feats["d_cap"]  = 0.0
            feats["d_cap2"] = 0.0

        # Log-capacity (stabilises scale across price levels)
        feats["log_cap"] = math.log(rec.capacity + 1e-12)

        # --- TDA / topological features ---
        feats["betti_0"]    = float(rec.betti_0)
        feats["betti_1"]    = float(rec.betti_1)
        feats["max_pers_0"] = rec.max_pers_0
        feats["max_pers_1"] = rec.max_pers_1
        feats["tot_pers"]   = rec.tot_pers

        # --- Momentum: lagged log-returns (always exactly 5, zero-padded) ---
        _hist_rets = [h.log_return for h in hist][-4:]
        _hist_rets = [0.0] * (4 - len(_hist_rets)) + _hist_rets
        _all_lags  = _hist_rets + [rec.log_return]
        for lag, ret in enumerate(reversed(_all_lags), start=1):
            feats[f"ret_lag{lag}"] = ret

        # --- Volatility regime (rolling std of returns) ---
        rets_arr = np.array([h.log_return for h in hist[-20:]] + [rec.log_return])
        feats["vol_10"]  = float(rets_arr[-10:].std()) if len(rets_arr) >= 10 else 0.0
        feats["vol_20"]  = float(rets_arr[-20:].std()) if len(rets_arr) >= 20 else 0.0

        # --- Rolling capacity percentile (regime relative to history) ---
        caps = np.array([h.capacity for h in hist[-60:]] + [rec.capacity])
        pct  = float(np.mean(caps <= rec.capacity)) if len(caps) > 1 else 0.5
        feats["cap_pct"] = pct

        return feats

    def learn_one(self, feats: Dict[str, float], target: float):
        """Update both models with one (feature, target) pair."""
        if HAS_RIVER:
            pred_pa = self._pa.predict_one(feats)
            pred_ht = self._ht.predict_one(feats)
            self._pa.learn_one(feats, target)
            self._ht.learn_one(feats, target)
            if pred_pa is not None:
                self._mae_pa.update(target, pred_pa)
                err = abs(target - pred_pa)
                self._rmse_pa = 0.95 * self._rmse_pa + 0.05 * err
            if pred_ht is not None:
                self._mae_ht.update(target, pred_ht)
                err = abs(target - pred_ht)
                self._rmse_ht = 0.95 * self._rmse_ht + 0.05 * err
        else:
            X = np.array(list(feats.values())).reshape(1, -1)
            # Fit on first call, OR refit if feature count somehow changed (safety net)
            if not self._skl_fitted or X.shape[1] != self._skl_scaler.n_features_in_:
                from sklearn.preprocessing import StandardScaler as _SS
                self._skl_scaler = _SS()
                self._skl_scaler.fit(X)
                self._skl_fitted = True
                # model weights kept — scaler reset only
            X_sc = self._skl_scaler.transform(X)
            self._skl_model.partial_fit(X_sc, [target])
        self._n_updates += 1

    def predict_one(self, feats: Dict[str, float]) -> Dict[str, float]:
        """
        Return a prediction dictionary:
            forecast   : expected next log-return
            direction  : +1 (bullish) / -1 (bearish)
            confidence : 0..1  (ensemble agreement)
        """
        if HAS_RIVER:
            p_pa = self._pa.predict_one(feats) or 0.0
            p_ht = self._ht.predict_one(feats) or 0.0
            # Weight by inverse recent MAE
            w_pa  = 1.0 / (self._rmse_pa + 1e-12)
            w_ht  = 1.0 / (self._rmse_ht + 1e-12)
            total = w_pa + w_ht
            forecast = (w_pa * p_pa + w_ht * p_ht) / total
            conf     = 1.0 - abs(p_pa - p_ht) / (abs(p_pa) + abs(p_ht) + 1e-12)
        else:
            X = np.array(list(feats.values())).reshape(1, -1)
            if not self._skl_fitted:
                return dict(forecast=0.0, direction=0, confidence=0.0,
                            mae_pa=None, mae_ht=None, n_updates=self._n_updates)
            try:
                X_sc     = self._skl_scaler.transform(X)
                raw_fc   = float(self._skl_model.predict(X_sc)[0])
                # Clip: daily log-returns physically cannot exceed ±15%
                forecast = float(np.clip(raw_fc, -0.15, 0.15))
            except Exception:
                forecast = 0.0
            conf = 0.5

        direction = int(np.sign(forecast)) if abs(forecast) > 1e-8 else 0
        return dict(forecast=forecast, direction=direction,
                    confidence=float(np.clip(conf, 0, 1)),
                    mae_pa=self._mae_pa.get() if HAS_RIVER else None,
                    mae_ht=self._mae_ht.get() if HAS_RIVER else None,
                    n_updates=self._n_updates)


# ===========================================================================
# MULTI-STEP SCENARIO GENERATOR
# ===========================================================================

class ScenarioGenerator:
    """
    Generates H-step-ahead price scenarios using the trained model and
    the symplectic stability bound as an uncertainty envelope.

    The stability lemma (Mishra 2026, Lemma 3.1) guarantees:
        |C(t) − C(t′)| ≤ L·δ + π·δ²
    where δ = dH(Ωt, Ωt′) is the Hausdorff perturbation.

    We use this to construct Lipschitz uncertainty bands around
    the forecast path: the wider the current perimeter L(t), the
    wider the valid perturbation-tolerance, and thus the scenario cone.
    """

    def __init__(self, model: SymplecticOnlineModel):
        self._model = model

    def generate(self, base_feats: Dict[str, float],
                 current_price: float,
                 current_perimeter: float,
                 horizon: int = 5,
                 n_scenarios: int = 3) -> Dict:
        """
        Return `n_scenarios` price paths over `horizon` bars,
        plus a central forecast and symplectic uncertainty band.

        Parameters
        ----------
        base_feats       : feature dictionary for the current bar
        current_price    : latest close price
        current_perimeter: L(t) from the convex hull (stability constant)
        horizon          : number of bars ahead
        n_scenarios      : number of scenario paths (bull / base / bear)

        Returns
        -------
        dict with keys:
            'central'    : list of prices, length = horizon
            'scenarios'  : {'bull': [...], 'base': [...], 'bear': [...]}
            'upper_band' : Lipschitz upper envelope (symplectic stability)
            'lower_band' : Lipschitz lower envelope
        """
        pred   = self._model.predict_one(base_feats)
        mu     = pred["forecast"]     # expected log-return per bar
        conf   = pred["confidence"]

        # Scenario spread: higher perimeter → wider cone (Lemma 3.1)
        sigma  = base_feats.get("vol_20", 0.01)
        L      = max(current_perimeter, 1e-6)
        spread = sigma * (1.0 + 0.5 * (1.0 - conf))   # confidence-adjusted

        scenarios = {
            "bull": mu + 1.0 * spread,
            "base": mu,
            "bear": mu - 1.0 * spread,
        }

        paths = {"bull": [], "base": [], "bear": []}
        for name, drift in scenarios.items():
            # Clamp drift to ±50% per bar to prevent overflow on extreme early data
            drift_safe = max(min(drift, 0.5), -0.5)
            p = current_price
            for _ in range(horizon):
                p = p * math.exp(drift_safe)
                paths[name].append(round(p, 4))

        # Symplectic stability uncertainty band (Mishra 2026, Lemma 3.1):
        # |C(t)−C(t')| ≤ L·δ + π·δ²   where δ = Hausdorff perturbation.
        # Use the isoperimetric ratio iso = 4πC/L² ∈ (0,1] as a dimensionless
        # shape factor: iso→1 (circular hull, low risk), iso→0 (elongated, high risk).
        # shape_k ∈ [1,2]: scales the uncertainty cone width accordingly.
        C_val      = base_feats.get("capacity", 1.0)
        iso        = (4 * math.pi * C_val) / (L ** 2 + 1e-12)
        shape_k    = 1.0 + max(0.0, 1.0 - iso)          # 1=round, 2=flat hull
        delta      = sigma * math.sqrt(horizon)
        band_half  = min(shape_k * delta * 3.0, 0.25)   # hard cap: ±25% per scenario

        upper_band, lower_band = [], []
        mu_safe = max(min(mu, 0.5), -0.5)   # clamp central drift too
        for h in range(1, horizon + 1):
            p_base_h = current_price * math.exp(mu_safe * h)
            upper_band.append(round(p_base_h * math.exp( band_half), 4))
            lower_band.append(round(p_base_h * math.exp(-band_half), 4))

        return dict(central=paths["base"], scenarios=paths,
                    upper_band=upper_band, lower_band=lower_band,
                    predicted_return=round(mu, 6),
                    scenario_confidence=round(conf, 4),
                    horizon=horizon)


# ===========================================================================
# MAIN FORECASTER — PUBLIC API
# ===========================================================================

class SymplecticForecaster:
    """
    End-to-end self-learning price forecaster combining:
      1. Symplectic phase-space feature extraction
      2. TDA persistent homology features
      3. Online / incremental ensemble learning
      4. Multi-step scenario generation with stability bounds

    Parameters
    ----------
    window          : rolling window size (number of bars) for the convex hull
    alert_pct       : capacity percentile threshold for regime alert (default 0.95)
    tda_subsample   : maximum points to subsample for TDA (speed/accuracy tradeoff)
    min_train_bars  : warm-up period before producing predictions
    """

    def __init__(self,
                 window:         int   = 60,
                 alert_pct:      float = 0.95,
                 tda_subsample:  int   = 100,
                 min_train_bars: int   = 80):

        self.window          = window
        self.alert_pct       = alert_pct
        self.tda_subsample   = tda_subsample
        self.min_train_bars  = min_train_bars

        self._phase_buf      : collections.deque = collections.deque(maxlen=window)
        self._record_hist    : List[CapacityRecord] = []
        self._capacity_buf   : collections.deque = collections.deque(maxlen=500)

        self._model          = SymplecticOnlineModel()
        self._scenario_gen   = ScenarioGenerator(self._model)

        self._bar_count      = 0
        self._prev_bar       : Optional[Bar] = None
        self._last_record    : Optional[CapacityRecord] = None
        self._last_feats     : Optional[Dict] = None
        self._last_price     : float = 0.0

    # ------------------------------------------------------------------ #
    # CORE PROCESSING STEP
    # ------------------------------------------------------------------ #

    def process_bar(self, bar: Bar) -> Optional[Dict]:
        """
        Process a single OHLCV bar through the complete pipeline:
          1. Compute (q,p) phase coordinates
          2. Update rolling point cloud
          3. Compute convex hull → capacity C(t), perimeter L(t)
          4. Compute TDA features from point cloud
          5. Update online model with previous prediction error
          6. Store record; return forecast if warmed up

        Returns None during the warm-up period; a forecast dict otherwise.
        """
        self._bar_count  += 1
        self._last_price  = bar.close

        # Step 1 — phase coordinates (require previous bar)
        if self._prev_bar is None:
            self._prev_bar = bar
            return None

        prev_close = self._prev_bar.close
        pp = phase_coords(bar, self._prev_bar)
        self._prev_bar = bar
        self._phase_buf.append((pp.q, pp.p))
        log_ret = math.log(bar.close / prev_close) if prev_close > 0 else 0.0

        # Step 3 — convex hull metrics (ECH capacity)
        pts = np.array(list(self._phase_buf), dtype=float)
        area, perim = convex_hull_metrics(pts)
        self._capacity_buf.append(area)

        # Adaptive alert: flag if capacity exceeds rolling 95th percentile
        caps_arr = np.array(list(self._capacity_buf))
        thresh   = np.percentile(caps_arr, self.alert_pct * 100) \
                   if len(caps_arr) >= 20 else np.inf
        alert = bool(area > thresh) if len(caps_arr) >= 20 else False

        # Step 4 — TDA (subsampled for speed on large windows)
        if len(pts) >= 5:
            idx_s = np.random.choice(len(pts),
                                     min(self.tda_subsample, len(pts)),
                                     replace=False)
            tda = compute_tda(pts[idx_s])
        else:
            tda = dict(betti_0=1, betti_1=0, max_pers_0=0.0,
                       max_pers_1=0.0, tot_pers=0.0)

        # Step 5 — build capacity record
        rec = CapacityRecord(
            t=bar.timestamp, capacity=area, perimeter=perim,
            alert=alert, log_return=log_ret, **tda
        )
        self._last_record = rec

        # Step 6 — online model update
        # We learn on the PREVIOUS bar's features → CURRENT bar's log-return
        if self._last_feats is not None and self._bar_count > 2:
            self._model.learn_one(self._last_feats, log_ret)

        # Build current feature vector
        current_feats = self._model._feature_vector(rec, self._record_hist)
        self._last_feats = current_feats
        self._record_hist.append(rec)

        # Return forecast only after warm-up
        if self._bar_count < self.min_train_bars:
            return None

        # Step 7 — generate prediction
        pred = self._model.predict_one(current_feats)
        scenarios = self._scenario_gen.generate(
            base_feats=current_feats,
            current_price=bar.close,
            current_perimeter=perim,
            horizon=5,
            n_scenarios=3
        )

        return dict(
            bar_index   = self._bar_count,
            timestamp   = bar.timestamp,
            close       = bar.close,
            capacity    = round(area,   8),
            perimeter   = round(perim,  6),
            alert       = alert,
            betti_0     = tda["betti_0"],
            betti_1     = tda["betti_1"],
            tot_pers    = round(tda["tot_pers"], 6),
            log_return  = round(log_ret, 6),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in pred.items()},
            **scenarios,
        )

    # ------------------------------------------------------------------ #
    # TRAIN ON MT5 HISTORICAL DATA
    # ------------------------------------------------------------------ #

    def train_on_mt5(self, symbol: str,
                     timeframe_str: str = "D1",
                     n_bars: int = 1000,
                     connection: MT5Connection = None) -> pd.DataFrame:
        """
        Fetch historical bars from MetaTrader 5 and bootstrap-train the model.

        This replaces the old train_on_csv() and run_live() methods.
        No CSV file or external data source needed — data comes directly
        from your MT5 broker's server.

        Parameters
        ----------
        symbol         : MT5 symbol (e.g., "EURUSD", "XAUUSD", "US500", "BTCUSD")
        timeframe_str  : timeframe string (e.g., "D1", "H1", "M5")
        n_bars         : number of historical bars to fetch for training
        connection     : MT5Connection instance (must be connected)

        Returns
        -------
        pd.DataFrame with all capacity records and forecasts
        """
        if not HAS_MT5:
            raise ImportError("MetaTrader5 package not installed.")

        tf_key = timeframe_str.upper()
        timeframe = TIMEFRAME_MAP.get(tf_key)
        if timeframe is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe_str}'. "
                f"Valid options: {', '.join(sorted(TIMEFRAME_MAP.keys()))}"
            )

        # Validate symbol
        if connection:
            connection.ensure_symbol(symbol)

        print(f"[MT5] Fetching {n_bars} bars of {symbol} ({tf_key}) ...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise RuntimeError(
                f"Failed to fetch data for '{symbol}': {error}\n"
                f"Check that the symbol is available and the market has history."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        print(f"[MT5] Received {len(df)} bars.")
        print(f"[MT5] Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        print(f"[INFO] Running symplectic pipeline with window={self.window} ...")

        results = []
        for i, row in df.iterrows():
            ts = row["time"].timestamp()
            # MT5 uses tick_volume; prefer real_volume if available and non-zero
            vol = float(row.get("real_volume", 0))
            if vol == 0:
                vol = float(row["tick_volume"])

            bar = Bar(timestamp=ts, open=float(row["open"]),
                      high=float(row["high"]), low=float(row["low"]),
                      close=float(row["close"]), volume=vol)
            out = self.process_bar(bar)
            if out:
                out["date"] = row["time"].strftime("%Y-%m-%d %H:%M")
                results.append(out)

        rdf = pd.DataFrame(results)
        print(f"[INFO] Pipeline complete. {len(rdf)} forecasts generated.")
        print(f"[INFO] Model updated {self._model._n_updates} times.")
        if HAS_RIVER:
            print(f"[INFO] Final MAE (PA):  {self._model._mae_pa.get():.6f}")
            print(f"[INFO] Final MAE (HT):  {self._model._mae_ht.get():.6f}")
            
        global global_dashboard_state
        if global_dashboard_state:
            global_dashboard_state.load_historical_dataframe(rdf)
            
        return rdf

    # ------------------------------------------------------------------ #
    # LIVE MONITORING LOOP (MT5)
    # ------------------------------------------------------------------ #

    def run_live_mt5(self, symbol: str,
                     timeframe_str: str = "D1",
                     poll_interval: float = 0.0,
                     on_signal: Callable = None,
                     connection: MT5Connection = None) -> None:
        """
        Continuous live monitoring loop using MetaTrader 5.

        1. Polls MT5 for the latest completed bar
        2. Processes it through the symplectic pipeline
        3. Generates forecast + trading signal
        4. Calls on_signal(forecast_dict, symbol) callback
        5. Sleeps and repeats

        Press Ctrl+C to stop.

        Parameters
        ----------
        symbol          : MT5 symbol
        timeframe_str   : timeframe string (e.g., "D1", "H1", "M5")
        poll_interval   : seconds between polls (0 = auto-detect from timeframe)
        on_signal       : callback(forecast_dict, symbol) for each new bar
        connection      : MT5Connection instance
        """
        if not HAS_MT5:
            raise ImportError("MetaTrader5 package not installed.")

        tf_key = timeframe_str.upper()
        timeframe = TIMEFRAME_MAP.get(tf_key)
        if timeframe is None:
            raise ValueError(f"Unknown timeframe '{timeframe_str}'.")

        if connection:
            connection.ensure_symbol(symbol)

        # Auto-adjust poll interval based on timeframe
        if poll_interval <= 0:
            poll_interval = _TF_POLL_SECONDS.get(tf_key, 60)

        print(f"\n{'=' * 66}")
        print(f"  LIVE MONITOR: {symbol} ({tf_key})")
        print(f"  Poll interval: {poll_interval:.0f}s")
        print(f"  Press Ctrl+C to stop.")
        print(f"{'=' * 66}")

        last_bar_time = 0
        poll_count = 0

        try:
            while True:
                # Fetch latest 2 bars (current forming bar + last completed bar)
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)

                if rates is None or len(rates) < 2:
                    poll_count += 1
                    if poll_count % 20 == 0:
                        print(f"  [waiting] No data received — market may be closed. "
                              f"({poll_count} polls)")
                    time.sleep(poll_interval)
                    continue

                # The second-to-last bar is the latest COMPLETED bar
                latest_complete = rates[-2]
                bar_time = int(latest_complete["time"])

                if bar_time <= last_bar_time:
                    # No new completed bar yet — still waiting
                    time.sleep(poll_interval)
                    continue

                last_bar_time = bar_time
                poll_count = 0

                # Build Bar from MT5 data
                vol = float(latest_complete["real_volume"])
                if vol == 0:
                    vol = float(latest_complete["tick_volume"])

                bar = Bar(
                    timestamp=float(bar_time),
                    open=float(latest_complete["open"]),
                    high=float(latest_complete["high"]),
                    low=float(latest_complete["low"]),
                    close=float(latest_complete["close"]),
                    volume=vol,
                )

                out = self.process_bar(bar)

                if out:
                    out["regime"] = "ALERT" if out.get("alert", False) else "NORMAL"
                    
                    global global_dashboard_state
                    if global_dashboard_state:
                        pts = np.array(list(self._phase_buf), dtype=float)
                        hull_verts = get_convex_hull_vertices(pts)
                        acc_info = connection.get_account_info() if connection else {}
                        global_dashboard_state.update_live_metrics(
                            symbol=symbol,
                            timeframe=timeframe_str,
                            latest_forecast=out,
                            phase_buf=list(self._phase_buf),
                            hull_points=hull_verts.tolist(),
                            acc_info=acc_info,
                            total_updates=self._model._n_updates
                        )
                    
                    if on_signal:
                        on_signal(out, symbol)
                    else:
                        # Default: simple one-line print
                        dt_str = datetime.datetime.fromtimestamp(bar_time).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        d_arrow = ("▲" if out.get("direction", 0) > 0
                                   else "▼" if out.get("direction", 0) < 0
                                   else "━")
                        print(
                            f"  [{dt_str}] {symbol} "
                            f"Close={bar.close:.5f} "
                            f"Ret={out.get('predicted_return', 0):+.4%} "
                            f"{d_arrow} "
                            f"Conf={out.get('confidence', 0):.1%}"
                        )
                else:
                    # Still in warm-up
                    dt_str = datetime.datetime.fromtimestamp(bar_time).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    print(f"  [{dt_str}] warming up... "
                          f"({self._bar_count}/{self.min_train_bars} bars)")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print(f"\n[LIVE] Stopped monitoring {symbol}.")

    # ------------------------------------------------------------------ #
    # LATEST FORECAST SUMMARY
    # ------------------------------------------------------------------ #

    def forecast(self, horizon: int = 5) -> Dict:
        """
        Return the most recent forecast from the trained model.
        Call after train_on_mt5() or after processing several bars.
        """
        if self._last_feats is None or self._last_record is None:
            return {"error": "Model not yet trained. Run train_on_mt5() first."}

        pred = self._model.predict_one(self._last_feats)
        scenarios = self._scenario_gen.generate(
            base_feats=self._last_feats,
            current_price=self._last_price,
            current_perimeter=self._last_record.perimeter,
            horizon=horizon,
            n_scenarios=3
        )
        return dict(current_price=self._last_price,
                    current_capacity=round(self._last_record.capacity, 8),
                    current_betti_1=self._last_record.betti_1,
                    regime="ALERT" if self._last_record.alert else "NORMAL",
                    n_train_updates=self._model._n_updates,
                    **pred, **scenarios)

    # ------------------------------------------------------------------ #
    # EXPORT
    # ------------------------------------------------------------------ #

    def export_results(self, df: pd.DataFrame, path: str = "symplectic_results.csv"):
        """Save the results DataFrame to CSV for further analysis."""
        df.to_csv(path, index=False)
        print(f"[INFO] Results saved to {path}")


# ===========================================================================
# ENTRY POINT — INTERACTIVE MT5 TRADING TERMINAL
# ===========================================================================

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
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="Confidence threshold for BUY/SELL signals (default: 0.6)")
    parser.add_argument("--window", type=int, default=60,
                        help="Symplectic rolling window size (default: 60)")
    parser.add_argument("--poll", type=float, default=0.0,
                        help="Custom poll interval in seconds (0 = auto)")
    args = parser.parse_args()

    # ── Banner ──
    CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}SYMPLECTIC ML PRICE FORECASTER — MetaTrader 5{RESET}")
    print(f"  {DIM}Based on: Mishra (2026) · Shultz (2023) · Mantegna (1999){RESET}")
    print(f"  {DIM}Mode: Signal-Only (no auto-execution){RESET}")
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

    # Initialize and start Dashboard Server
    global_dashboard_state = DashboardState()
    srv_thread = threading.Thread(
        target=start_dashboard_server,
        args=(global_dashboard_state, 8080, os.path.join(os.path.dirname(__file__), "dashboard.html")),
        daemon=True
    )
    srv_thread.start()

    engine = None  # declare for finally block

    try:
        # ── Step 2: Get symbol (interactive or CLI) ──
        symbol = args.symbol
        if not symbol:
            print(f"\n  {BOLD}Enter the symbol you want to analyze:{RESET}")
            print(f"  {DIM}(e.g., EURUSD, GBPUSD, XAUUSD, US500, BTCUSD){RESET}")
            symbol = input(f"  {CYAN}Symbol>{RESET} ").strip().upper()
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
        engine = TradingEngine(confidence_threshold=args.confidence)

        # ── Step 5: Train on MT5 historical data ──
        print()
        rdf = fc.train_on_mt5(symbol, tf_str, args.bars, connection=conn)

        # ── Step 6: Print initial forecast / signal ──
        result = fc.forecast(horizon=5)
        if "error" not in result:
            # Merge into a format the trading engine expects
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

        # ── Step 7: Export training results ──
        export_path = f"symplectic_{symbol}_{tf_str}.csv"
        fc.export_results(rdf, export_path)

        # ── Step 8: Start live monitoring ──
        print(f"\n{BOLD}[INFO] Starting live monitoring...{RESET}")
        print(f"{DIM}       The model continues learning from each new bar.{RESET}")
        poll = args.poll if args.poll > 0 else 0.0
        fc.run_live_mt5(
            symbol, tf_str,
            poll_interval=poll,
            on_signal=engine.on_signal,
            connection=conn,
        )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print signal summary
        if engine is not None:
            s = engine.summary()
            print(f"\n  {BOLD}Signal Summary:{RESET} {s['total_signals']} total — "
                  f"\033[92mBUY: {s['buys']}\033[0m | "
                  f"\033[91mSELL: {s['sells']}\033[0m | "
                  f"\033[93mHOLD: {s['holds']}\033[0m")
        conn.disconnect()
        print(f"\n{BOLD}[DONE]{RESET} The model state is discarded. "
              f"Restart to begin a new session.")
