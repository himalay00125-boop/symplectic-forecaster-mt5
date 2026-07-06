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
from core.config import *
from models.forecaster import *

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
        
        min_lot_to_enforce = max(0.01, self.spec.volume_min)
        if volume < min_lot_to_enforce:
            volume = min_lot_to_enforce
            risk_amount = self.equity * (risk_config.risk_per_trade_pct / 100.0)
            if self.spec.tick_size > 0 and self.spec.tick_value > 0:
                value_per_unit = self.spec.tick_value / self.spec.tick_size
                new_sl_dist = risk_amount / (volume * value_per_unit)
                
                min_dist = max(
                    self.spec.trade_stops_level * self.spec.point,
                    risk_config.min_stop_pips * self.spec.point * (10 if self.spec.digits in (3, 5) else 1),
                )
                new_sl_dist = max(new_sl_dist, min_dist)
                
                if sl_dist > 0:
                    original_rr = abs(tp - entry) / sl_dist
                else:
                    original_rr = risk_config.reward_risk_ratio
                
                sl_dist = new_sl_dist
                
                if direction == "BUY":
                    sl = round(entry - sl_dist, self.spec.digits)
                    tp = round(entry + sl_dist * original_rr, self.spec.digits)
                else:
                    sl = round(entry + sl_dist, self.spec.digits)
                    tp = round(entry - sl_dist * original_rr, self.spec.digits)
        
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

    times = [datetime.fromtimestamp(t) for t, _ in result.equity_curve]
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
                "entry_time": datetime.fromtimestamp(t.entry_time),
                "exit_time": datetime.fromtimestamp(t.exit_time),
                "exit_reason": t.exit_reason,
            })
        pd.DataFrame(rows).to_csv(export_path, index=False)
        print(f"[BACKTEST] Trade log saved -> {export_path}")

    return result

