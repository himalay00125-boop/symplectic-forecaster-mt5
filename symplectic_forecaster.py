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
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple, Optional, Dict, List, Tuple, Callable, Any

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

    def hold_diagnostics(self) -> Dict[str, int]:
        """Count why HOLD signals were issued."""
        buckets = {
            "low_confidence": 0,
            "alert_regime": 0,
            "neutral_direction": 0,
            "other": 0,
        }
        for sig in self.signal_log:
            if sig.action != "HOLD":
                continue
            reason = sig.reason.lower()
            if "alert" in reason or "bifurcation" in reason:
                buckets["alert_regime"] += 1
            elif "confidence" in reason:
                buckets["low_confidence"] += 1
            elif "neutral" in reason:
                buckets["neutral_direction"] += 1
            else:
                buckets["other"] += 1
        return buckets


# ===========================================================================
# RISK MANAGEMENT & AUTO-EXECUTION
# ===========================================================================

@dataclass
class RiskConfig:
    """Risk parameters for automated trade execution."""
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    reward_risk_ratio: float = 2.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    trailing_atr_multiplier: float = 1.0
    max_positions: int = 1
    magic_number: int = 20260611
    min_stop_pips: float = 10.0
    use_stability_bands: bool = True
    allow_live: bool = False


@dataclass
class TradeResult:
    success: bool
    action: str
    message: str
    ticket: int = 0
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0


class RiskManager:
    """Position sizing, daily loss limits, and trade permission checks."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._session_start_equity: Optional[float] = None
        self._session_date: Optional[datetime.date] = None
        self._trading_halted = False
        self._halt_reason = ""

    def reset_session(self):
        """Reset daily tracking (call at bot startup or new trading day)."""
        acc = mt5.account_info()
        if acc:
            self._session_start_equity = acc.equity
            self._session_date = datetime.date.today()
            self._trading_halted = False
            self._halt_reason = ""

    def _roll_session_if_new_day(self):
        today = datetime.date.today()
        if self._session_date != today:
            self.reset_session()

    def can_trade(self) -> Tuple[bool, str]:
        """Return (allowed, reason)."""
        if not HAS_MT5:
            return False, "MT5 not available"
        if self._trading_halted:
            return False, self._halt_reason

        self._roll_session_if_new_day()
        acc = mt5.account_info()
        if acc is None:
            return False, "No account info"

        if self._session_start_equity is None:
            self.reset_session()

        if self.config.max_daily_loss_pct > 0 and self._session_start_equity > 0:
            daily_loss_pct = (
                (self._session_start_equity - acc.equity)
                / self._session_start_equity * 100.0
            )
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self._trading_halted = True
                self._halt_reason = (
                    f"Daily loss limit hit: {daily_loss_pct:.2f}% "
                    f"(max {self.config.max_daily_loss_pct:.1f}%)"
                )
                return False, self._halt_reason

        return True, "OK"

    def calculate_lot_size(self, symbol: str, stop_distance: float) -> float:
        """Size position so stop loss risks `risk_per_trade_pct` of equity."""
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        if acc is None or sym is None or stop_distance <= 0:
            return sym.volume_min if sym else 0.01

        risk_amount = acc.equity * (self.config.risk_per_trade_pct / 100.0)
        tick_value = sym.trade_tick_value
        tick_size = sym.trade_tick_size
        if tick_size <= 0 or tick_value <= 0:
            return sym.volume_min

        value_per_price_unit = tick_value / tick_size
        loss_per_lot = stop_distance * value_per_price_unit
        if loss_per_lot <= 0:
            return sym.volume_min

        lots = risk_amount / loss_per_lot
        step = sym.volume_step
        lots = math.floor(lots / step) * step
        lots = max(sym.volume_min, min(sym.volume_max, lots))
        return round(lots, 2)

    def compute_atr(self, symbol: str, timeframe: int, period: int) -> float:
        """Average True Range from recent MT5 bars."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, period + 1)
        if rates is None or len(rates) < 2:
            return 0.0

        trs = []
        for i in range(1, len(rates)):
            high = float(rates[i]["high"])
            low = float(rates[i]["low"])
            prev_close = float(rates[i - 1]["close"])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0


class MT5TradeExecutor:
    """Places and manages orders via the MT5 Python API."""

    def __init__(self, risk_config: RiskConfig, connection: MT5Connection = None):
        if not HAS_MT5:
            raise ImportError("MetaTrader5 package not installed.")
        self.risk = RiskManager(risk_config)
        self.config = risk_config
        self.connection = connection
        self.trade_log: List[TradeResult] = []
        self._trade_count = {"OPEN": 0, "CLOSE": 0, "MODIFY": 0, "SKIP": 0, "ERROR": 0}
        self.risk.reset_session()

    def _is_live_account(self) -> bool:
        acc = mt5.account_info()
        return acc is not None and acc.trade_mode == 2

    def _check_live_permission(self) -> Tuple[bool, str]:
        if self._is_live_account() and not self.config.allow_live:
            return False, (
                "Live account detected. Pass --allow-live to enable real-money trading."
            )
        return True, "OK"

    def _filling_mode(self, symbol: str) -> int:
        sym = mt5.symbol_info(symbol)
        if sym is None:
            return mt5.ORDER_FILLING_IOC
        filling = sym.filling_mode
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        if filling & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def get_positions(self, symbol: str) -> List:
        """Open positions for this bot's magic number."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.config.magic_number]

    def _normalize_price(self, symbol: str, price: float) -> float:
        sym = mt5.symbol_info(symbol)
        if sym is None:
            return price
        return round(price, sym.digits)

    def _min_stop_distance(self, symbol: str) -> float:
        sym = mt5.symbol_info(symbol)
        if sym is None:
            return 0.0
        stops_level = sym.trade_stops_level
        point = sym.point
        min_pips_dist = self.config.min_stop_pips * point * (
            10 if sym.digits in (3, 5) else 1
        )
        return max(stops_level * point, min_pips_dist)

    def compute_sl_tp(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        forecast: Dict,
        timeframe: int,
    ) -> Tuple[float, float, float]:
        """
        Compute stop-loss, take-profit, and stop distance.

        Uses symplectic stability bands when available, otherwise ATR.
        """
        sym = mt5.symbol_info(symbol)
        if sym is None:
            return 0.0, 0.0, 0.0

        min_dist = self._min_stop_distance(symbol)
        sl_dist = 0.0

        if self.config.use_stability_bands and forecast:
            lower = forecast.get("lower_band") or []
            upper = forecast.get("upper_band") or []
            if direction == "BUY" and lower:
                sl = float(lower[0])
                sl_dist = max(entry_price - sl, min_dist)
            elif direction == "SELL" and upper:
                sl = float(upper[0])
                sl_dist = max(sl - entry_price, min_dist)
            else:
                sl_dist = 0.0

        if sl_dist <= 0:
            atr = self.risk.compute_atr(symbol, timeframe, self.config.atr_period)
            sl_dist = max(atr * self.config.atr_sl_multiplier, min_dist)

        if direction == "BUY":
            sl = self._normalize_price(symbol, entry_price - sl_dist)
            tp = self._normalize_price(
                symbol, entry_price + sl_dist * self.config.reward_risk_ratio
            )
        else:
            sl = self._normalize_price(symbol, entry_price + sl_dist)
            tp = self._normalize_price(
                symbol, entry_price - sl_dist * self.config.reward_risk_ratio
            )
        return sl, tp, sl_dist

    def open_position(
        self,
        symbol: str,
        direction: str,
        forecast: Dict,
        timeframe: int,
    ) -> TradeResult:
        """Open a market order with risk-based lot sizing."""
        allowed, reason = self.risk.can_trade()
        if not allowed:
            self._trade_count["SKIP"] += 1
            return TradeResult(False, "SKIP", reason)

        live_ok, live_msg = self._check_live_permission()
        if not live_ok:
            self._trade_count["SKIP"] += 1
            return TradeResult(False, "SKIP", live_msg)

        positions = self.get_positions(symbol)
        if len(positions) >= self.config.max_positions:
            self._trade_count["SKIP"] += 1
            return TradeResult(
                False, "SKIP",
                f"Max positions ({self.config.max_positions}) already open."
            )

        tick = mt5.symbol_info_tick(symbol)
        sym = mt5.symbol_info(symbol)
        if tick is None or sym is None:
            self._trade_count["ERROR"] += 1
            return TradeResult(False, "ERROR", "No tick/symbol info")

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        sl, tp, sl_dist = self.compute_sl_tp(symbol, direction, price, forecast, timeframe)
        volume = self.risk.calculate_lot_size(symbol, sl_dist)
        if volume < sym.volume_min:
            self._trade_count["SKIP"] += 1
            return TradeResult(
                False, "SKIP",
                f"Calculated lot {volume} below minimum {sym.volume_min}"
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": "symplectic_bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        result = mt5.order_send(request)
        if result is None:
            self._trade_count["ERROR"] += 1
            return TradeResult(False, "ERROR", f"order_send failed: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self._trade_count["ERROR"] += 1
            return TradeResult(
                False, "ERROR",
                f"Order rejected: {result.retcode} — {result.comment}"
            )

        self._trade_count["OPEN"] += 1
        trade = TradeResult(
            True, "OPEN",
            f"{direction} {volume} lots @ {result.price:.5f} | SL={sl:.5f} TP={tp:.5f}",
            ticket=result.order, volume=volume, price=result.price, sl=sl, tp=tp,
        )
        self.trade_log.append(trade)
        return trade

    def close_position(self, position) -> TradeResult:
        """Close an open position by ticket."""
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            self._trade_count["ERROR"] += 1
            return TradeResult(False, "ERROR", "No tick for close")

        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": "symplectic_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(position.symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self._trade_count["ERROR"] += 1
            err = result.comment if result else str(mt5.last_error())
            return TradeResult(False, "ERROR", f"Close failed: {err}")

        self._trade_count["CLOSE"] += 1
        trade = TradeResult(
            True, "CLOSE",
            f"Closed #{position.ticket} {position.volume} lots @ {result.price:.5f}",
            ticket=position.ticket, volume=position.volume, price=result.price,
        )
        self.trade_log.append(trade)
        return trade

    def close_all(self, symbol: str) -> List[TradeResult]:
        """Close all bot positions on symbol."""
        results = []
        for pos in self.get_positions(symbol):
            results.append(self.close_position(pos))
        return results

    def modify_sl(self, position, new_sl: float) -> TradeResult:
        """Modify stop-loss on an open position."""
        new_sl = self._normalize_price(position.symbol, new_sl)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp,
            "magic": self.config.magic_number,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self._trade_count["ERROR"] += 1
            err = result.comment if result else str(mt5.last_error())
            return TradeResult(False, "ERROR", f"SL modify failed: {err}")

        self._trade_count["MODIFY"] += 1
        trade = TradeResult(
            True, "MODIFY",
            f"#{position.ticket} SL -> {new_sl:.5f}",
            ticket=position.ticket, sl=new_sl,
        )
        self.trade_log.append(trade)
        return trade

    def manage_trailing_stops(self, symbol: str, timeframe: int):
        """Trail stop-loss on open positions using ATR distance."""
        atr = self.risk.compute_atr(symbol, timeframe, self.config.atr_period)
        if atr <= 0:
            return

        trail_dist = atr * self.config.trailing_atr_multiplier
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return

        for pos in self.get_positions(symbol):
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = tick.bid - trail_dist
                if pos.sl == 0 or new_sl > pos.sl + mt5.symbol_info(symbol).point:
                    if new_sl < tick.bid:
                        self.modify_sl(pos, new_sl)
            else:
                new_sl = tick.ask + trail_dist
                if pos.sl == 0 or new_sl < pos.sl - mt5.symbol_info(symbol).point:
                    if new_sl > tick.ask:
                        self.modify_sl(pos, new_sl)

    def summary(self) -> Dict:
        return dict(self._trade_count, total_trades=len(self.trade_log))


class AutoTradingEngine(TradingEngine):
    """
    Extends TradingEngine with MT5 order execution.

    On each new bar:
      BUY  → close shorts, open long (if risk allows)
      SELL → close longs, open short (if risk allows)
      HOLD → manage trailing stops only
    """

    def __init__(
        self,
        executor: MT5TradeExecutor,
        confidence_threshold: float = 0.6,
        timeframe: int = None,
    ):
        super().__init__(confidence_threshold)
        self.executor = executor
        self.timeframe = timeframe

    def on_signal(self, forecast: Dict, symbol: str):
        """Evaluate signal, print it, and execute trades when appropriate."""
        signal = self.evaluate(forecast, symbol)
        self.print_signal(signal)

        if self.timeframe is None:
            return

        # Always manage trailing stops on open positions
        self.executor.manage_trailing_stops(symbol, self.timeframe)

        if signal.action == "HOLD":
            return

        positions = self.executor.get_positions(symbol)
        longs = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
        shorts = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]

        if signal.action == "BUY":
            for pos in shorts:
                result = self.executor.close_position(pos)
                self._print_trade(result)
            if not longs:
                result = self.executor.open_position(
                    symbol, "BUY", forecast, self.timeframe
                )
                self._print_trade(result)

        elif signal.action == "SELL":
            for pos in longs:
                result = self.executor.close_position(pos)
                self._print_trade(result)
            if not shorts:
                result = self.executor.open_position(
                    symbol, "SELL", forecast, self.timeframe
                )
                self._print_trade(result)

    def on_poll(self, symbol: str):
        """Called between bars to update trailing stops."""
        if self.timeframe is not None:
            self.executor.manage_trailing_stops(symbol, self.timeframe)

    def _print_trade(self, result: TradeResult):
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        color = GREEN if result.success else (YELLOW if result.action == "SKIP" else RED)
        print(f"  {color}{BOLD}[TRADE] {result.action}{RESET} — {result.message}")

    def trade_summary(self) -> Dict:
        return {
            "signals": self.summary(),
            "trades": self.executor.summary(),
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

    @staticmethod
    def _compute_confidence(
        forecast: float,
        feats: Dict[str, float],
        p_pa: float = 0.0,
        p_ht: float = 0.0,
    ) -> float:
        """
        Confidence score tuned for forex-scale log-returns.

        Old formula (ensemble agreement only) collapsed toward 0 when PA and HT
        predicted tiny returns — even when they agreed. This blends:
          • ensemble agreement (River only)
          • forecast magnitude vs rolling volatility (signal strength)
        """
        vol = max(feats.get("vol_20", 0.0), feats.get("vol_10", 0.0), 1e-6)

        if HAS_RIVER:
            scale = max(abs(p_pa), abs(p_ht), vol * 0.5, 1e-8)
            agreement = 1.0 - min(1.0, abs(p_pa - p_ht) / scale)
            if np.sign(p_pa) != np.sign(p_ht) and abs(p_pa) > vol * 0.05 and abs(p_ht) > vol * 0.05:
                agreement *= 0.4
        else:
            agreement = 0.55

        magnitude = min(1.0, abs(forecast) / (vol * 1.2 + 1e-12))
        conf = 0.30 * agreement + 0.70 * magnitude
        return float(np.clip(conf, 0, 1))

    def predict_one(self, feats: Dict[str, float]) -> Dict[str, float]:
        """
        Return a prediction dictionary:
            forecast   : expected next log-return
            direction  : +1 (bullish) / -1 (bearish)
            confidence : 0..1  (magnitude + ensemble agreement)
        """
        p_pa = 0.0
        p_ht = 0.0
        if HAS_RIVER:
            p_pa = self._pa.predict_one(feats) or 0.0
            p_ht = self._ht.predict_one(feats) or 0.0
            w_pa  = 1.0 / (self._rmse_pa + 1e-12)
            w_ht  = 1.0 / (self._rmse_ht + 1e-12)
            total = w_pa + w_ht
            forecast = (w_pa * p_pa + w_ht * p_ht) / total
        else:
            X = np.array(list(feats.values())).reshape(1, -1)
            if not self._skl_fitted:
                return dict(forecast=0.0, direction=0, confidence=0.0,
                            mae_pa=None, mae_ht=None, n_updates=self._n_updates)
            try:
                X_sc     = self._skl_scaler.transform(X)
                raw_fc   = float(self._skl_model.predict(X_sc)[0])
                forecast = float(np.clip(raw_fc, -0.15, 0.15))
            except Exception:
                forecast = 0.0

        vol = max(feats.get("vol_20", 0.0), feats.get("vol_10", 0.0), 1e-6)
        if abs(forecast) < vol * 0.05:
            direction = 0
        else:
            direction = int(np.sign(forecast))

        conf = self._compute_confidence(forecast, feats, p_pa, p_ht)
        return dict(forecast=forecast, direction=direction,
                    confidence=conf,
                    mae_pa=self._mae_pa.get() if HAS_RIVER else None,
                    mae_ht=self._mae_ht.get() if HAS_RIVER else None,
                    n_updates=self._n_updates)

    def export_state(self) -> Dict[str, Any]:
        """Serialize model weights and metrics for persistence."""
        state: Dict[str, Any] = {
            "backend": "river" if HAS_RIVER else "sklearn",
            "n_updates": self._n_updates,
            "rmse_pa": self._rmse_pa,
            "rmse_ht": self._rmse_ht,
        }
        if HAS_RIVER:
            state["pa"] = self._pa
            state["ht"] = self._ht
            state["mae_pa"] = self._mae_pa
            state["mae_ht"] = self._mae_ht
        else:
            state["skl_model"] = self._skl_model
            state["skl_scaler"] = self._skl_scaler
            state["skl_fitted"] = self._skl_fitted
        return state

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restore model from a previously exported state dict."""
        saved_backend = state.get("backend")
        current_backend = "river" if HAS_RIVER else "sklearn"
        if saved_backend != current_backend:
            raise ValueError(
                f"State backend '{saved_backend}' does not match current "
                f"environment '{current_backend}'. Install matching deps "
                f"(river vs sklearn-only)."
            )
        self._n_updates = state.get("n_updates", 0)
        self._rmse_pa = state.get("rmse_pa", 0.001)
        self._rmse_ht = state.get("rmse_ht", 0.001)
        if HAS_RIVER:
            self._pa = state["pa"]
            self._ht = state["ht"]
            self._mae_pa = state["mae_pa"]
            self._mae_ht = state["mae_ht"]
        else:
            self._skl_model = state["skl_model"]
            self._skl_scaler = state["skl_scaler"]
            self._skl_fitted = state.get("skl_fitted", True)


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
        self._freeze_learning: bool = False

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
        if (not self._freeze_learning
                and self._last_feats is not None and self._bar_count > 2):
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
                     on_poll: Callable = None,
                     connection: MT5Connection = None) -> None:
        """
        Continuous live monitoring loop using MetaTrader 5.

        1. Polls MT5 for the latest completed bar
        2. Processes it through the symplectic pipeline
        3. Generates forecast + trading signal
        4. Calls on_signal(forecast_dict, symbol) callback
        5. Calls on_poll(symbol) each cycle (e.g. trailing stops)
        6. Sleeps and repeats

        Press Ctrl+C to stop.

        Parameters
        ----------
        symbol          : MT5 symbol
        timeframe_str   : timeframe string (e.g., "D1", "H1", "M5")
        poll_interval   : seconds between polls (0 = auto-detect from timeframe)
        on_signal       : callback(forecast_dict, symbol) for each new bar
        on_poll         : callback(symbol) each poll cycle (between bars)
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
                    if on_poll:
                        on_poll(symbol)
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
    # MODEL STATE PERSISTENCE
    # ------------------------------------------------------------------ #

    STATE_VERSION = 1
    _MAX_RECORD_HIST = 500

    @staticmethod
    def default_state_path(symbol: str, timeframe: str,
                           state_dir: str = "states") -> Path:
        """Default pickle path: states/EURUSD_H1.pkl"""
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        return Path(state_dir) / f"{symbol.upper()}_{timeframe.upper()}.pkl"

    @staticmethod
    def _bar_to_dict(bar: Optional[Bar]) -> Optional[Dict]:
        if bar is None:
            return None
        return dict(
            timestamp=bar.timestamp, open=bar.open, high=bar.high,
            low=bar.low, close=bar.close, volume=bar.volume,
        )

    @staticmethod
    def _bar_from_dict(d: Optional[Dict]) -> Optional[Bar]:
        if d is None:
            return None
        return Bar(**d)

    @staticmethod
    def _rec_to_dict(rec: Optional[CapacityRecord]) -> Optional[Dict]:
        return asdict(rec) if rec else None

    @staticmethod
    def _rec_from_dict(d: Optional[Dict]) -> Optional[CapacityRecord]:
        return CapacityRecord(**d) if d else None

    def export_state(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Export full forecaster state (buffers + model) to a dict."""
        return {
            "version": self.STATE_VERSION,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "config": {
                "window": self.window,
                "alert_pct": self.alert_pct,
                "tda_subsample": self.tda_subsample,
                "min_train_bars": self.min_train_bars,
            },
            "bar_count": self._bar_count,
            "last_price": self._last_price,
            "phase_buf": list(self._phase_buf),
            "capacity_buf": list(self._capacity_buf),
            "record_hist": [asdict(r) for r in self._record_hist[-self._MAX_RECORD_HIST:]],
            "prev_bar": self._bar_to_dict(self._prev_bar),
            "last_record": self._rec_to_dict(self._last_record),
            "last_feats": self._last_feats,
            "model": self._model.export_state(),
        }

    def import_state(self, state: Dict[str, Any],
                     symbol: str = None, timeframe: str = None) -> None:
        """Restore forecaster from a previously saved state dict."""
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported state version {state.get('version')}. "
                f"Expected {self.STATE_VERSION}."
            )
        if symbol and state.get("symbol") != symbol.upper():
            raise ValueError(
                f"State symbol {state.get('symbol')} != requested {symbol.upper()}"
            )
        if timeframe and state.get("timeframe") != timeframe.upper():
            raise ValueError(
                f"State timeframe {state.get('timeframe')} != requested {timeframe.upper()}"
            )

        cfg = state["config"]
        self.window = cfg["window"]
        self.alert_pct = cfg["alert_pct"]
        self.tda_subsample = cfg["tda_subsample"]
        self.min_train_bars = cfg["min_train_bars"]

        self._phase_buf = collections.deque(state["phase_buf"], maxlen=self.window)
        self._capacity_buf = collections.deque(state["capacity_buf"], maxlen=500)
        self._record_hist = [CapacityRecord(**d) for d in state["record_hist"]]
        self._bar_count = state["bar_count"]
        self._last_price = state["last_price"]
        self._prev_bar = self._bar_from_dict(state.get("prev_bar"))
        self._last_record = self._rec_from_dict(state.get("last_record"))
        self._last_feats = state.get("last_feats")
        self._model.import_state(state["model"])

    def save_state(self, path: str, symbol: str, timeframe: str) -> str:
        """Persist forecaster state to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_state(symbol, timeframe)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[STATE] Saved model state -> {path}")
        print(f"[STATE] Updates: {self._model._n_updates} | "
              f"Bars processed: {self._bar_count}")
        return str(path)

    def load_state(self, path: str, symbol: str = None,
                   timeframe: str = None) -> Dict[str, Any]:
        """Load forecaster state from a pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.import_state(payload, symbol=symbol, timeframe=timeframe)
        print(f"[STATE] Loaded model state <- {path}")
        print(f"[STATE] Saved at: {payload.get('saved_at', 'unknown')}")
        print(f"[STATE] Updates: {self._model._n_updates} | "
              f"Bars processed: {self._bar_count}")
        return payload

    # ------------------------------------------------------------------ #
    # BACKTEST
    # ------------------------------------------------------------------ #

    def collect_bar_forecasts(
        self,
        bars: List[Bar],
        freeze_model: bool = True,
    ) -> List[Tuple[int, Bar, Optional[Dict]]]:
        """Process bars once and cache forecasts (for optimizer / replay)."""
        cache: List[Tuple[int, Bar, Optional[Dict]]] = []
        prev_freeze = self._freeze_learning
        self._freeze_learning = freeze_model
        try:
            for i, bar in enumerate(bars):
                out = self.process_bar(bar)
                if out is not None:
                    out["regime"] = "ALERT" if out.get("alert", False) else "NORMAL"
                cache.append((i, bar, out))
        finally:
            self._freeze_learning = prev_freeze
        return cache

    def run_backtest(
        self,
        symbol: str,
        timeframe_str: str,
        bars: List[Bar],
        engine: TradingEngine,
        risk_config: RiskConfig,
        initial_balance: float = 10000.0,
        spread_pips: float = 1.0,
        freeze_model: bool = False,
        symbol_spec: "SymbolSpec" = None,
        forecast_cache: List[Tuple[int, Bar, Optional[Dict]]] = None,
    ) -> "BacktestResult":
        """
        Walk-forward backtest over a list of bars with simulated fills.

        Processes each bar through the symplectic pipeline, evaluates signals,
        and simulates SL/TP/trailing-stop execution without placing real orders.
        """
        if forecast_cache is None:
            forecast_cache = self.collect_bar_forecasts(bars, freeze_model=freeze_model)
        return simulate_backtest_from_cache(
            forecast_cache=forecast_cache,
            bars=bars,
            symbol=symbol,
            engine=engine,
            risk_config=risk_config,
            initial_balance=initial_balance,
            spread_pips=spread_pips,
            symbol_spec=symbol_spec,
        )

    # ------------------------------------------------------------------ #
    # EXPORT
    # ------------------------------------------------------------------ #

    def export_results(self, df: pd.DataFrame, path: str = "symplectic_results.csv"):
        """Save the results DataFrame to CSV for further analysis."""
        df.to_csv(path, index=False)
        print(f"[INFO] Results saved to {path}")


# ===========================================================================
# BACKTEST SIMULATOR
# ===========================================================================

@dataclass
class SymbolSpec:
    """Cached symbol properties for offline lot/price calculations."""
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    tick_value: float
    tick_size: float
    trade_stops_level: int

    @classmethod
    def from_mt5(cls, symbol: str) -> "SymbolSpec":
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Cannot load symbol info for {symbol}")
        return cls(
            point=info.point,
            digits=info.digits,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            tick_value=info.trade_tick_value,
            tick_size=info.trade_tick_size,
            trade_stops_level=info.trade_stops_level,
        )


@dataclass
class SimPosition:
    direction: str
    entry_price: float
    volume: float
    sl: float
    tp: float
    entry_time: float
    entry_bar_idx: int


@dataclass
class ClosedTrade:
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    entry_time: float
    exit_time: float
    exit_reason: str


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    signal_summary: Dict
    trades: List[ClosedTrade]
    equity_curve: List[Tuple[float, float]]
    hold_diagnostics: Dict[str, int] = field(default_factory=dict)
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)


def simulate_backtest_from_cache(
    forecast_cache: List[Tuple[int, Bar, Optional[Dict]]],
    bars: List[Bar],
    symbol: str,
    engine: TradingEngine,
    risk_config: RiskConfig,
    initial_balance: float,
    spread_pips: float,
    symbol_spec: SymbolSpec,
) -> BacktestResult:
    """Run simulation on pre-computed bar forecasts (fast param sweeps)."""
    sim = BacktestSimulator(
        symbol=symbol,
        initial_balance=initial_balance,
        risk_config=risk_config,
        spread_pips=spread_pips,
        symbol_spec=symbol_spec,
    )
    engine.signal_log.clear()
    engine._signal_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
    confidences: List[float] = []

    for i, bar, out in forecast_cache:
        sim.update_trailing(bar, bars, i, risk_config)

        if sim.position is not None:
            exit_reason = sim.check_exit(bar)
            if exit_reason:
                sim.close_position(bar, exit_reason)

        if out is None:
            sim.record_equity(bar.timestamp, bar.close)
            continue

        confidences.append(out.get("confidence", 0.0))
        signal = engine.evaluate(out, symbol)

        if signal.action in ("BUY", "SELL"):
            if sim.position is not None:
                if (
                    (signal.action == "BUY" and sim.position.direction == "SELL")
                    or (signal.action == "SELL" and sim.position.direction == "BUY")
                ):
                    sim.close_position(bar, "REVERSE")
            if sim.position is None:
                sim.open_position(
                    direction=signal.action,
                    bar=bar,
                    bars=bars,
                    bar_idx=i,
                    forecast=out,
                    risk_config=risk_config,
                )

        sim.record_equity(bar.timestamp, bar.close)

    if sim.position is not None and bars:
        sim.close_position(bars[-1], "END")

    result = sim.build_result(engine)
    result.hold_diagnostics = engine.hold_diagnostics()
    if confidences:
        arr = np.array(confidences)
        result.confidence_stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "above_threshold": float(np.mean(arr >= engine.confidence_threshold)),
        }
    return result


def _spread_price(spec: SymbolSpec, spread_pips: float) -> float:
    pip_mult = 10 if spec.digits in (3, 5) else 1
    return spread_pips * spec.point * pip_mult


def _compute_atr_from_bars(bars: List[Bar], end_idx: int, period: int) -> float:
    if end_idx < 1:
        return 0.0
    start = max(1, end_idx - period + 1)
    trs = []
    for i in range(start, end_idx + 1):
        high, low = bars[i].high, bars[i].low
        prev_close = bars[i - 1].close
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return float(np.mean(trs)) if trs else 0.0


def _compute_sl_tp_backtest(
    direction: str,
    entry_price: float,
    forecast: Dict,
    bars: List[Bar],
    bar_idx: int,
    risk_config: RiskConfig,
    spec: SymbolSpec,
) -> Tuple[float, float, float]:
    """Mirror MT5TradeExecutor SL/TP logic using bar history for ATR."""
    min_dist = max(
        spec.trade_stops_level * spec.point,
        risk_config.min_stop_pips * spec.point * (10 if spec.digits in (3, 5) else 1),
    )
    sl_dist = 0.0

    if risk_config.use_stability_bands and forecast:
        lower = forecast.get("lower_band") or []
        upper = forecast.get("upper_band") or []
        if direction == "BUY" and lower:
            sl_dist = max(entry_price - float(lower[0]), min_dist)
        elif direction == "SELL" and upper:
            sl_dist = max(float(upper[0]) - entry_price, min_dist)

    if sl_dist <= 0:
        atr = _compute_atr_from_bars(bars, bar_idx, risk_config.atr_period)
        sl_dist = max(atr * risk_config.atr_sl_multiplier, min_dist)

    if direction == "BUY":
        sl = round(entry_price - sl_dist, spec.digits)
        tp = round(entry_price + sl_dist * risk_config.reward_risk_ratio, spec.digits)
    else:
        sl = round(entry_price + sl_dist, spec.digits)
        tp = round(entry_price - sl_dist * risk_config.reward_risk_ratio, spec.digits)
    return sl, tp, sl_dist


def _calc_lot_backtest(
    equity: float,
    stop_distance: float,
    risk_config: RiskConfig,
    spec: SymbolSpec,
) -> float:
    if stop_distance <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
        return spec.volume_min
    risk_amount = equity * (risk_config.risk_per_trade_pct / 100.0)
    value_per_unit = spec.tick_value / spec.tick_size
    loss_per_lot = stop_distance * value_per_unit
    if loss_per_lot <= 0:
        return spec.volume_min
    lots = risk_amount / loss_per_lot
    lots = math.floor(lots / spec.volume_step) * spec.volume_step
    return round(max(spec.volume_min, min(spec.volume_max, lots)), 2)


class BacktestSimulator:
    """Simulates order fills, SL/TP, and equity tracking."""

    def __init__(
        self,
        symbol: str,
        initial_balance: float,
        risk_config: RiskConfig,
        spread_pips: float,
        symbol_spec: SymbolSpec,
    ):
        self.symbol = symbol
        self.balance = initial_balance
        self.equity = initial_balance
        self.risk_config = risk_config
        self.spread_pips = spread_pips
        self.spec = symbol_spec
        self.position: Optional[SimPosition] = None
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[float, float]] = []
        self._session_start_equity = initial_balance
        self._trading_halted = False

    def _pip_value(self, volume: float) -> float:
        if self.spec.tick_size <= 0:
            return 1.0
        return self.spec.tick_value / self.spec.tick_size * volume

    def _check_daily_loss(self) -> bool:
        if self.risk_config.max_daily_loss_pct <= 0:
            return True
        loss_pct = (self._session_start_equity - self.equity) / self._session_start_equity * 100
        if loss_pct >= self.risk_config.max_daily_loss_pct:
            self._trading_halted = True
            return False
        return True

    def open_position(
        self,
        direction: str,
        bar: Bar,
        bars: List[Bar],
        bar_idx: int,
        forecast: Dict,
        risk_config: RiskConfig,
    ):
        if self._trading_halted or not self._check_daily_loss():
            return

        spread = _spread_price(self.spec, self.spread_pips)
        if direction == "BUY":
            entry = bar.close + spread / 2
        else:
            entry = bar.close - spread / 2

        sl, tp, sl_dist = _compute_sl_tp_backtest(
            direction, entry, forecast, bars, bar_idx, risk_config, self.spec,
        )
        volume = _calc_lot_backtest(self.equity, sl_dist, risk_config, self.spec)
        if volume < self.spec.volume_min:
            return

        self.position = SimPosition(
            direction=direction,
            entry_price=entry,
            volume=volume,
            sl=sl,
            tp=tp,
            entry_time=bar.timestamp,
            entry_bar_idx=bar_idx,
        )

    def check_exit(self, bar: Bar) -> Optional[str]:
        if self.position is None:
            return None
        pos = self.position
        if pos.direction == "BUY":
            if bar.low <= pos.sl:
                return "SL"
            if bar.high >= pos.tp:
                return "TP"
        else:
            if bar.high >= pos.sl:
                return "SL"
            if bar.low <= pos.tp:
                return "TP"
        return None

    def _exit_price(self, bar: Bar, reason: str) -> float:
        pos = self.position
        spread = _spread_price(self.spec, self.spread_pips)
        if reason == "SL":
            raw = pos.sl
        elif reason == "TP":
            raw = pos.tp
        else:
            raw = bar.close
        if pos.direction == "BUY":
            return raw - spread / 2
        return raw + spread / 2

    def close_position(self, bar: Bar, reason: str):
        if self.position is None:
            return
        pos = self.position
        exit_price = self._exit_price(bar, reason)
        price_diff = exit_price - pos.entry_price
        if pos.direction == "SELL":
            price_diff = -price_diff
        pnl = price_diff * self._pip_value(pos.volume)
        self.balance += pnl
        self.equity = self.balance
        self.closed_trades.append(ClosedTrade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            volume=pos.volume,
            pnl=pnl,
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            exit_reason=reason,
        ))
        self.position = None

    def update_trailing(
        self,
        bar: Bar,
        bars: List[Bar],
        bar_idx: int,
        risk_config: RiskConfig,
    ):
        if self.position is None:
            return
        atr = _compute_atr_from_bars(bars, bar_idx, risk_config.atr_period)
        if atr <= 0:
            return
        trail = atr * risk_config.trailing_atr_multiplier
        pos = self.position
        if pos.direction == "BUY":
            new_sl = round(bar.close - trail, self.spec.digits)
            if new_sl > pos.sl and new_sl < bar.close:
                pos.sl = new_sl
        else:
            new_sl = round(bar.close + trail, self.spec.digits)
            if (pos.sl == 0 or new_sl < pos.sl) and new_sl > bar.close:
                pos.sl = new_sl

    def record_equity(self, timestamp: float, mark_price: float):
        unrealized = 0.0
        if self.position is not None:
            diff = mark_price - self.position.entry_price
            if self.position.direction == "SELL":
                diff = -diff
            unrealized = diff * self._pip_value(self.position.volume)
        self.equity_curve.append((timestamp, self.balance + unrealized))

    def build_result(self, engine: TradingEngine) -> BacktestResult:
        equities = [e for _, e in self.equity_curve] or [self.balance]
        returns = np.diff(equities) / np.array(equities[:-1]) if len(equities) > 1 else []
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        pnls = [t.pnl for t in self.closed_trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        sharpe = (
            float(np.mean(returns) / (np.std(returns) + 1e-12) * np.sqrt(252))
            if len(returns) > 1 else 0.0
        )
        ret_pct = (self.balance - self._session_start_equity) / self._session_start_equity * 100

        return BacktestResult(
            initial_balance=self._session_start_equity,
            final_balance=self.balance,
            total_return_pct=ret_pct,
            total_trades=len(self.closed_trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(pnls) * 100 if pnls else 0.0,
            profit_factor=pf,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            signal_summary=engine.summary(),
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
        )


def _mt5_rates_to_bars(rates) -> List[Bar]:
    bars = []
    for row in rates:
        vol = float(row["real_volume"])
        if vol == 0:
            vol = float(row["tick_volume"])
        bars.append(Bar(
            timestamp=float(row["time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=vol,
        ))
    return bars


def print_signal_diagnostics(result: BacktestResult, confidence_threshold: float):
    """Explain why HOLD dominated or what blocked trades."""
    DIM = "\033[2m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"  {BOLD}Signal Diagnostics{RESET}")
    if result.confidence_stats:
        cs = result.confidence_stats
        print(f"    Confidence range : {cs['min']:.1%} - {cs['max']:.1%} "
              f"(mean {cs['mean']:.1%}, median {cs['median']:.1%})")
        print(f"    Above threshold  : {cs['above_threshold']:.1%} of bars "
              f"(threshold {confidence_threshold:.1%})")
    if result.hold_diagnostics:
        hd = result.hold_diagnostics
        total_holds = sum(hd.values())
        if total_holds:
            print(f"    HOLD breakdown   :")
            labels = {
                "low_confidence": "Low confidence (below threshold)",
                "alert_regime": "ALERT regime (capacity spike)",
                "neutral_direction": "Neutral forecast (move too small)",
                "other": "Other",
            }
            for key, count in hd.items():
                if count:
                    pct = count / total_holds * 100
                    print(f"      {labels.get(key, key):36s} {count:4d} ({pct:.0f}%)")
    print(f"  {DIM}Tip: if 'low_confidence' dominates, lower --confidence "
          f"(try 0.25-0.40). If 'alert_regime' dominates, market was unstable "
          f"or raise --window for smoother capacity.{RESET}")
    print()


def print_backtest_report(
    result: BacktestResult,
    symbol: str,
    timeframe: str,
    confidence_threshold: float = 0.6,
):
    """Pretty-print backtest summary."""
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    ret_color = GREEN if result.total_return_pct >= 0 else RED
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    title = f"BACKTEST REPORT — {symbol} ({timeframe})"
    if result.params:
        p = result.params
        title += f"  conf={p.get('confidence', confidence_threshold):.2f}"
    print(f"  {CYAN}{BOLD}{title}{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")
    print(f"  Initial balance : ${result.initial_balance:,.2f}")
    print(f"  Final balance   : ${result.final_balance:,.2f}")
    print(f"  Total return    : {ret_color}{result.total_return_pct:+.2f}%{RESET}")
    print(f"  Max drawdown    : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe (ann.)   : {result.sharpe_ratio:.2f}")
    print(f"  Trades          : {result.total_trades} "
          f"({result.wins}W / {result.losses}L)")
    print(f"  Win rate        : {result.win_rate:.1f}%")
    print(f"  Profit factor   : {result.profit_factor:.2f}")
    s = result.signal_summary
    print(f"  Signals         : {s['total_signals']} total — "
          f"BUY: {s['buys']} | SELL: {s['sells']} | HOLD: {s['holds']}")
    print_signal_diagnostics(result, confidence_threshold)
    print(f"{BOLD}{'=' * 66}{RESET}\n")


def save_equity_chart(
    result: BacktestResult,
    path: str,
    symbol: str,
    timeframe: str,
) -> bool:
    """Save equity curve PNG. Returns False if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[CHART] matplotlib not installed — skip chart. pip install matplotlib")
        return False

    if not result.equity_curve:
        return False

    times = [datetime.datetime.fromtimestamp(t) for t, _ in result.equity_curve]
    equities = [e for _, e in result.equity_curve]

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#333")

    color = "#2ecc71" if result.total_return_pct >= 0 else "#e74c3c"
    ax_eq.plot(times, equities, color=color, linewidth=1.8, label="Equity")
    ax_eq.axhline(result.initial_balance, color="#666", linestyle="--",
                  linewidth=0.8, label="Initial")
    ax_eq.fill_between(times, result.initial_balance, equities,
                       where=[e >= result.initial_balance for e in equities],
                       alpha=0.15, color="#2ecc71")
    ax_eq.fill_between(times, result.initial_balance, equities,
                       where=[e < result.initial_balance for e in equities],
                       alpha=0.15, color="#e74c3c")
    ax_eq.set_ylabel("Balance ($)", color="#ccc")
    ax_eq.set_title(
        f"Equity Curve — {symbol} {timeframe}  "
        f"({result.total_return_pct:+.2f}% | {result.total_trades} trades)",
        color="#eee", fontsize=11,
    )
    ax_eq.legend(facecolor="#1a1d27", edgecolor="#333", labelcolor="#ccc")
    ax_eq.grid(True, alpha=0.2, color="#444")

    peak = equities[0]
    drawdowns = []
    for eq in equities:
        peak = max(peak, eq)
        drawdowns.append(-(peak - eq) / peak * 100 if peak > 0 else 0.0)
    ax_dd.fill_between(times, 0, drawdowns, color="#e74c3c", alpha=0.5)
    ax_dd.set_ylabel("DD %", color="#ccc")
    ax_dd.set_xlabel("Date", color="#ccc")
    ax_dd.grid(True, alpha=0.2, color="#444")
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20)

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[CHART] Equity curve saved -> {path}")
    return True


def _parse_float_list(value: str, default: List[float]) -> List[float]:
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _optimizer_score(result: BacktestResult) -> float:
    """Rank parameter combos: reward return + Sharpe, penalize drawdown / few trades."""
    if result.total_trades < 2:
        return -1000.0 + result.total_return_pct
    return (
        result.total_return_pct
        - result.max_drawdown_pct * 0.4
        + result.sharpe_ratio * 8.0
        + min(result.win_rate, 80) * 0.05
    )


def run_optimizer_cli(
    fc: SymplecticForecaster,
    symbol: str,
    tf_str: str,
    train_bars: int,
    test_bars: int,
    connection: MT5Connection,
    confidence_grid: List[float],
    risk_grid: List[float],
    reward_risk_grid: List[float],
    initial_balance: float = 10000.0,
    spread_pips: float = 1.0,
    export_path: str = "optimize_results.csv",
    chart_path: str = None,
) -> Tuple[BacktestResult, pd.DataFrame]:
    """
    Walk-forward grid search over confidence / risk / reward-risk.

    Trains once, caches test-period forecasts, then replays simulation
    for each parameter combination.
    """
    tf_mt5 = TIMEFRAME_MAP[tf_str]
    total_bars = train_bars + test_bars
    connection.ensure_symbol(symbol)

    print(f"[OPTIMIZE] Fetching {total_bars} bars of {symbol} ({tf_str}) ...")
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, total_bars)
    if rates is None or len(rates) < train_bars + 50:
        raise RuntimeError("Insufficient historical data for optimization.")

    all_bars = _mt5_rates_to_bars(rates)
    train_slice = all_bars[:train_bars]
    test_slice = all_bars[train_bars:train_bars + test_bars]

    if fc._bar_count < fc.min_train_bars:
        print(f"[OPTIMIZE] Training on {len(train_slice)} bars ...")
        for bar in train_slice:
            fc.process_bar(bar)

    print(f"[OPTIMIZE] Caching forecasts for {len(test_slice)} test bars ...")
    forecast_cache = fc.collect_bar_forecasts(test_slice, freeze_model=True)
    spec = SymbolSpec.from_mt5(symbol)

    combos = [
        (conf, risk, rr)
        for conf in confidence_grid
        for risk in risk_grid
        for rr in reward_risk_grid
    ]
    print(f"[OPTIMIZE] Sweeping {len(combos)} parameter combinations ...")

    rows = []
    best_result: Optional[BacktestResult] = None
    best_score = -float("inf")

    for conf, risk, rr in combos:
        engine = TradingEngine(confidence_threshold=conf)
        risk_cfg = RiskConfig(
            risk_per_trade_pct=risk,
            reward_risk_ratio=rr,
        )
        result = simulate_backtest_from_cache(
            forecast_cache=forecast_cache,
            bars=test_slice,
            symbol=symbol,
            engine=engine,
            risk_config=risk_cfg,
            initial_balance=initial_balance,
            spread_pips=spread_pips,
            symbol_spec=spec,
        )
        result.params = {"confidence": conf, "risk_pct": risk, "reward_risk": rr}
        score = _optimizer_score(result)
        rows.append({
            "confidence": conf,
            "risk_pct": risk,
            "reward_risk": rr,
            "score": round(score, 3),
            "return_pct": round(result.total_return_pct, 3),
            "max_dd_pct": round(result.max_drawdown_pct, 3),
            "sharpe": round(result.sharpe_ratio, 3),
            "trades": result.total_trades,
            "win_rate": round(result.win_rate, 1),
            "buys": result.signal_summary["buys"],
            "sells": result.signal_summary["sells"],
            "holds": result.signal_summary["holds"],
        })
        if score > best_score:
            best_score = score
            best_result = result

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(export_path, index=False)
    print(f"[OPTIMIZE] Results saved -> {export_path}")

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}TOP 5 PARAMETER COMBINATIONS{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}")
    print(f"  {'conf':>5} {'risk%':>6} {'R:R':>5} {'score':>7} "
          f"{'ret%':>7} {'dd%':>6} {'shrp':>5} {'trds':>5}")
    for _, row in df.head(5).iterrows():
        print(f"  {row['confidence']:5.2f} {row['risk_pct']:6.1f} "
              f"{row['reward_risk']:5.1f} {row['score']:7.1f} "
              f"{row['return_pct']:+7.2f} {row['max_dd_pct']:6.1f} "
              f"{row['sharpe']:5.2f} {int(row['trades']):5d}")
    print(f"{BOLD}{'=' * 66}{RESET}")

    if best_result:
        print(f"\n[OPTIMIZE] Best combo:")
        print_backtest_report(
            best_result, symbol, tf_str,
            confidence_threshold=best_result.params.get("confidence", 0.4),
        )
        if chart_path:
            save_equity_chart(best_result, chart_path, symbol, tf_str)

    return best_result, df


def run_backtest_cli(
    fc: SymplecticForecaster,
    symbol: str,
    tf_str: str,
    train_bars: int,
    test_bars: int,
    engine: TradingEngine,
    risk_config: RiskConfig,
    connection: MT5Connection,
    initial_balance: float = 10000.0,
    spread_pips: float = 1.0,
    freeze_model: bool = False,
    export_path: str = None,
) -> BacktestResult:
    """Fetch data, train (or use loaded state), and run out-of-sample backtest."""
    tf_mt5 = TIMEFRAME_MAP[tf_str]
    total_bars = train_bars + test_bars

    connection.ensure_symbol(symbol)
    print(f"[BACKTEST] Fetching {total_bars} bars of {symbol} ({tf_str}) ...")
    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, total_bars)
    if rates is None or len(rates) < train_bars + 50:
        raise RuntimeError(
            f"Insufficient data: got {len(rates) if rates is not None else 0} bars, "
            f"need at least {train_bars + 50}."
        )

    all_bars = _mt5_rates_to_bars(rates)
    # MT5 returns oldest-first; split train / test
    train_slice = all_bars[:train_bars]
    test_slice = all_bars[train_bars:train_bars + test_bars]

    if fc._bar_count < fc.min_train_bars:
        if train_bars <= 0:
            raise RuntimeError(
                "Model is not trained. Use --train-bars N or --load-state <file.pkl>."
            )
        print(f"[BACKTEST] Training on {len(train_slice)} bars ...")
        for bar in train_slice:
            fc.process_bar(bar)
        print(f"[BACKTEST] Training complete. Model updates: {fc._model._n_updates}")
    else:
        print(f"[BACKTEST] Using trained/loaded state ({fc._model._n_updates} updates). "
              f"Skipping training pass.")

    spec = SymbolSpec.from_mt5(symbol)
    print(f"[BACKTEST] Simulating {len(test_slice)} out-of-sample bars "
          f"(spread={spread_pips} pips, balance=${initial_balance:,.0f}) ...")

    result = fc.run_backtest(
        symbol=symbol,
        timeframe_str=tf_str,
        bars=test_slice,
        engine=engine,
        risk_config=risk_config,
        initial_balance=initial_balance,
        spread_pips=spread_pips,
        freeze_model=freeze_model,
        symbol_spec=spec,
    )

    result.params = {
        "confidence": engine.confidence_threshold,
        "risk_pct": risk_config.risk_per_trade_pct,
        "reward_risk": risk_config.reward_risk_ratio,
    }
    print_backtest_report(result, symbol, tf_str, engine.confidence_threshold)

    chart_path = export_path.replace(".csv", "_equity.png") if export_path else None
    if chart_path:
        save_equity_chart(result, chart_path, symbol, tf_str)

    if export_path:
        rows = []
        for t in result.trades:
            rows.append({
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "volume": t.volume,
                "pnl": t.pnl,
                "entry_time": datetime.datetime.fromtimestamp(t.entry_time),
                "exit_time": datetime.datetime.fromtimestamp(t.exit_time),
                "exit_reason": t.exit_reason,
            })
        pd.DataFrame(rows).to_csv(export_path, index=False)
        print(f"[BACKTEST] Trade log saved -> {export_path}")

    return result


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
    parser.add_argument("--trailing-atr", type=float, default=1.0,
                        help="ATR multiplier for trailing stop (default: 1.0)")
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
            executor = MT5TradeExecutor(risk_cfg, connection=conn)
            engine = AutoTradingEngine(
                executor, confidence_threshold=args.confidence, timeframe=tf_mt5,
            )
        else:
            engine = TradingEngine(confidence_threshold=args.confidence)

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
            fc.load_state(args.load_state, symbol=symbol, timeframe=tf_str)
            state_loaded = True
        elif (state_path and Path(state_path).exists()
              and not args.backtest and not args.load_state):
            try:
                fc.load_state(state_path, symbol=symbol, timeframe=tf_str)
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
                fc.save_state(save_path, symbol, tf_str)
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
                    fc.save_state(save_path, symbol, tf_str)
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
