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
from execution.connection import *
from execution.risk import *
from analytics.trade_analytics import TradeAnalytics

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
        pred_ret   = forecast.get("predicted_return", forecast.get("forecast", 0.0))
        regime     = forecast.get("regime",
                                  "ALERT" if forecast.get("alert", False) else "NORMAL")
        price      = forecast.get("close", forecast.get("current_price", 0.0))
        scenarios  = forecast.get("scenarios", {})
        horizon    = forecast.get("horizon", 5)
        ts         = forecast.get("timestamp", time.time())

        # Format timestamp
        if isinstance(ts, (int, float)):
            ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
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

class MT5TradeExecutor:
    """Places and manages orders via the MT5 Python API."""

    def __init__(self, risk_config: RiskConfig, connection: MT5Connection = None, forecaster = None):
        if not HAS_MT5:
            raise ImportError("MetaTrader5 package not installed.")
        self.risk = RiskManager(risk_config)
        self.config = risk_config
        self.connection = connection
        self.forecaster = forecaster
        self.trade_log: List[TradeResult] = []
        self.active_trade_features: Dict[int, Dict] = {}
        self.trade_records: List[TradeRecord] = []  # Full trade records for learning
        self._trade_count = {"OPEN": 0, "CLOSE": 0, "MODIFY": 0, "SKIP": 0, "ERROR": 0}
        self._position_hwm: Dict[int, float] = {}  # High water mark per position ticket
        self.risk.reset_session()
        self._last_deals_check: float = 0.0  # Timestamp of last deal history check

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
        analytics: 'TradeAnalytics' = None,
    ) -> Tuple[float, float, float]:
        """
        Compute stop-loss, take-profit, and stop distance.

        Uses symplectic stability bands when available, otherwise ATR.
        Adaptively scales SL by:
          - Entry volatility ratio (wider SL in high-vol conditions)
          - SL-hit streak correction (widen if SL keeps getting hit)
        """
        sym = mt5.symbol_info(symbol)
        if sym is None:
            return 0.0, 0.0, 0.0

        min_dist = self._min_stop_distance(symbol)
        sl_dist = 0.0
        atr_sl_mult = self.config.atr_sl_multiplier
        rr_ratio = self.config.reward_risk_ratio

        # ── Adaptive SL multiplier from analytics ──
        if analytics is not None:
            atr_sl_mult = analytics.suggest_sl_multiplier_adjustment(atr_sl_mult)

        # ── Dynamic Reward:Risk based on Confidence ──
        conf = forecast.get("confidence", 0.0) if forecast else 0.0
        if conf > 0.8:
            rr_ratio *= 1.5
        elif conf < 0.65:
            rr_ratio *= 0.8

        # ── Volatility-based scaling ──
        # Scale SL by how volatile current conditions are vs average
        feats = forecast.get("features", {}) if forecast else {}
        vol_20 = feats.get("vol_20", 0.0)
        vol_10 = feats.get("vol_10", 0.0)
        if vol_20 > 0 and vol_10 > 0:
            vol_ratio = vol_10 / vol_20  # Short-term vs longer-term vol
            if vol_ratio > 1.2:
                atr_sl_mult *= max(1.0, vol_ratio)  # Widen SL in high vol
                rr_ratio *= 0.8  # Tighter TP (take profit faster in chaos)
                if analytics is not None:
                    rr_ratio = analytics.suggest_rr_adjustment(rr_ratio, vol_ratio)

        # ── Cap RR Ratio ──
        # Ensure we never risk more than our reward (TP must be >= SL)
        rr_ratio = max(1.0, rr_ratio)

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
            sl_dist = max(atr * atr_sl_mult, min_dist)

        # ── Hard ATR Cap ──
        atr = self.risk.compute_atr(symbol, timeframe, self.config.atr_period)
        if atr > 0:
            max_dist = atr * getattr(self.config, 'max_atr_sl_multiplier', 3.5)
            if sl_dist > max_dist:
                sl_dist = max_dist

        # Guarantee minimum stop distance to prevent MT5 invalid stops
        sl_dist = max(sl_dist, min_dist)

        if direction == "BUY":
            sl = self._normalize_price(symbol, entry_price - sl_dist)
            tp = self._normalize_price(
                symbol, entry_price + sl_dist * rr_ratio
            )
        else:
            sl = self._normalize_price(symbol, entry_price + sl_dist)
            tp = self._normalize_price(
                symbol, entry_price - sl_dist * rr_ratio
            )
        return sl, tp, sl_dist

    def open_position(
        self,
        symbol: str,
        direction: str,
        forecast: Dict,
        timeframe: int,
        analytics: 'TradeAnalytics' = None,
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
        
        # Correlation-based portfolio risk check
        if not self.check_correlation_risk(symbol, direction):
            self._trade_count["SKIP"] += 1
            return TradeResult(
                False, "SKIP",
                f"Correlation risk too high for {symbol} {direction}."
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

        sl, tp, sl_dist = self.compute_sl_tp(symbol, direction, price, forecast, timeframe, analytics=analytics)
        confidence = forecast.get("confidence", 0.5) if forecast else 0.5
        interval_width = forecast.get("pred_interval_width", 0.0) if forecast else 0.0
        volume = self.risk.calculate_lot_size(symbol, sl_dist, confidence=confidence, pred_interval_width=interval_width)
        
        min_lot_to_enforce = max(0.01, sym.volume_min if sym else 0.01)
        if volume < min_lot_to_enforce:
            volume = min_lot_to_enforce
            acc = mt5.account_info()
            if acc is not None and sym is not None:
                risk_amount = acc.equity * (self.config.risk_per_trade_pct / 100.0)
                tick_value = sym.trade_tick_value
                tick_size = sym.trade_tick_size
                if tick_size > 0 and tick_value > 0:
                    value_per_price_unit = tick_value / tick_size
                    new_sl_dist = risk_amount / (volume * value_per_price_unit)
                    
                    min_dist = self._min_stop_distance(symbol)
                    new_sl_dist = max(new_sl_dist, min_dist)
                    
                    if sl_dist > 0:
                        original_rr = abs(tp - price) / sl_dist
                    else:
                        original_rr = self.config.reward_risk_ratio
                    
                    sl_dist = new_sl_dist
                    
                    if direction == "BUY":
                        sl = self._normalize_price(symbol, price - sl_dist)
                        tp = self._normalize_price(symbol, price + sl_dist * original_rr)
                    else:
                        sl = self._normalize_price(symbol, price + sl_dist)
                        tp = self._normalize_price(symbol, price - sl_dist * original_rr)
                    
                    print(f"[{symbol}] Force min lot {volume:.2f}: adjusted SL dist to {sl_dist:.5f} to maintain risk.")
        
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
        entry_feats = forecast.get("features", {})
        self.active_trade_features[result.order] = entry_feats
        
        # Create comprehensive TradeRecord for learning
        trade_record = TradeRecord(
            ticket=result.order,
            symbol=symbol,
            entry_time=time.time(),
            entry_bar_idx=self.forecaster._bar_count if self.forecaster else 0,
            entry_price=result.price,
            direction=direction,
            volume=volume,
            sl=sl,
            tp=tp,
            entry_confidence=forecast.get("confidence", 0.0),
            entry_predicted_return=forecast.get("predicted_return", forecast.get("forecast", 0.0)),
            entry_regime="ALERT" if forecast.get("alert", False) else "NORMAL",
            entry_capacity=forecast.get("capacity", 0.0),
            entry_perimeter=forecast.get("perimeter", 0.0),
            entry_betti_0=forecast.get("betti_0", 0),
            entry_betti_1=forecast.get("betti_1", 0),
            entry_tot_pers=forecast.get("tot_pers", 0.0),
            entry_vol_10=entry_feats.get("vol_10", 0.0),
            entry_vol_20=entry_feats.get("vol_20", 0.0),
            entry_features=dict(entry_feats),
        )
        self.trade_records.append(trade_record)
        
        trade = TradeResult(
            True, "OPEN",
            f"{direction} {volume} lots @ {result.price:.5f} | SL={sl:.5f} TP={tp:.5f}",
            ticket=result.order, volume=volume, price=result.price, sl=sl, tp=tp,
            entry_features=entry_feats
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
        
        sym = mt5.symbol_info(position.symbol)
        realized_pnl = 0.0
        if sym and sym.trade_tick_size > 0:
            price_diff = result.price - position.price_open
            if position.type == mt5.POSITION_TYPE_SELL:
                price_diff = -price_diff
            points = price_diff / sym.trade_tick_size
            realized_pnl = points * sym.trade_tick_value * position.volume
        
        realized_return = (result.price - position.price_open) / position.price_open if position.price_open > 0 else 0.0
        if position.type == mt5.POSITION_TYPE_SELL:
            realized_return = -realized_return
        
        entry_feats = self.active_trade_features.pop(position.ticket, {})
        hwm = self._position_hwm.pop(position.ticket, None)  # Clean up high-water-mark
        
        # Determine exit reason by comparing close price with SL/TP
        exit_reason = "MANUAL"
        if position.sl > 0 and position.tp > 0:
            if position.type == mt5.POSITION_TYPE_BUY:
                if result.price <= position.sl + mt5.symbol_info(position.symbol).point:
                    exit_reason = "SL"
                elif result.price >= position.tp - mt5.symbol_info(position.symbol).point:
                    exit_reason = "TP"
            else:
                if result.price >= position.sl - mt5.symbol_info(position.symbol).point:
                    exit_reason = "SL"
                elif result.price <= position.tp + mt5.symbol_info(position.symbol).point:
                    exit_reason = "TP"
        
        # Approximate MFE/MAE (Max Favorable/Adverse Excursion)
        # Since we don't have tick data, use SL/TP distances and high-water-mark
        entry_price = position.price_open
        if position.type == mt5.POSITION_TYPE_BUY:
            sl_dist = entry_price - position.sl if position.sl > 0 else 0
            tp_dist = position.tp - entry_price if position.tp > 0 else 0
            # MAE: worst case was at least SL distance if hit SL, else min of SL and actual drawdown
            if exit_reason == "SL":
                mae = sl_dist
            elif hwm is not None and hwm > entry_price:
                mae = max(0.0, entry_price - min(hwm, entry_price))  # Approximate from HWM
            else:
                mae = max(0.0, entry_price - result.price) if result.price < entry_price else 0.0
            # MFE: best case was at least TP distance if hit TP, else high-water-mark
            if exit_reason == "TP":
                mfe = tp_dist
            elif hwm is not None and hwm > entry_price:
                mfe = hwm - entry_price
            else:
                mfe = max(0.0, result.price - entry_price) if result.price > entry_price else 0.0
        else:  # SELL
            sl_dist = position.sl - entry_price if position.sl > 0 else 0
            tp_dist = entry_price - position.tp if position.tp > 0 else 0
            if exit_reason == "SL":
                mae = sl_dist
            elif hwm is not None and hwm < entry_price:
                mae = max(0.0, max(hwm, entry_price) - entry_price)  # Approximate
            else:
                mae = max(0.0, result.price - entry_price) if result.price > entry_price else 0.0
            if exit_reason == "TP":
                mfe = tp_dist
            elif hwm is not None and hwm < entry_price:
                mfe = entry_price - hwm
            else:
                mfe = max(0.0, entry_price - result.price) if result.price < entry_price else 0.0
        
        # Update the TradeRecord with exit details
        for tr in reversed(self.trade_records):
            if tr.ticket == position.ticket and not tr.is_closed():
                tr.exit_time = time.time()
                tr.exit_price = result.price
                tr.exit_reason = exit_reason
                tr.realized_pnl = realized_pnl
                tr.holding_bars = int((tr.exit_time - tr.entry_time) / 
                                       (self.risk.config.atr_period * 60)) if hasattr(self.risk, 'config') else 0
                tr.max_favorable_excursion = mfe
                tr.max_adverse_excursion = mae
                break
        
        trade = TradeResult(
            True, "CLOSE",
            f"Closed #{position.ticket} {position.volume} lots @ {result.price:.5f} | PnL: ${realized_pnl:.2f} [{exit_reason}]",
            ticket=position.ticket, volume=position.volume, price=result.price,
            entry_features=entry_feats, realized_pnl=realized_pnl, realized_return=realized_return
        )
        self.trade_log.append(trade)
        return trade

    def close_all(self, symbol: str) -> List[TradeResult]:
        """Close all bot positions on symbol."""
        results = []
        for pos in self.get_positions(symbol):
            results.append(self.close_position(pos))
        return results

    def check_correlation_risk(self, symbol: str, direction: str, max_correlation: float = 0.7) -> bool:
        """
        Check if opening a position in `symbol` with `direction` would create
        excessive correlated exposure with existing positions.
        
        Returns True if safe to open, False if correlation risk is too high.
        """
        if not HAS_MT5:
            return True
        
        # Get current open positions
        positions = self.get_positions(symbol)  # Same symbol only for now
        # TODO: Add cross-symbol correlation check by fetching rates for other symbols
        
        # For now, just check same symbol (don't add to same direction if already maxed)
        # The max_positions config already handles this
        
        # Future enhancement: fetch correlation matrix from MT5 or compute from rates
        # corr = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
        # For other symbols in portfolio, compute correlation and check
        
        return True

    def check_closed_trades(self, symbol: str) -> List[TradeRecord]:
        """Check MT5 deal history for positions closed by SL/TP (not by our code).
        
        Returns list of newly closed TradeRecords.
        """
        if not HAS_MT5:
            return []
        
        now = time.time()
        # Only check every 30 seconds to avoid API rate limits
        if now - self._last_deals_check < 30:
            return []
        self._last_deals_check = now
        
        newly_closed = []
        
        # Get all deals for our magic number in the last check period
        from_date = datetime.fromtimestamp(self._last_deals_check - 30)
        to_date = datetime.fromtimestamp(now + 10)
        deals = mt5.history_deals_get(from_date, to_date, group=f"*,magic={self.config.magic_number}")
        
        if deals is None:
            return []
        
        # Get current open position tickets
        open_tickets = {p.ticket for p in self.get_positions(symbol)}
        
        # Find deals that closed positions we were tracking but are no longer open
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT or deal.entry == mt5.DEAL_ENTRY_INOUT:
                ticket = deal.position_id
                if ticket not in open_tickets:
                    # This position was closed - find our TradeRecord
                    for tr in reversed(self.trade_records):
                        if tr.ticket == ticket and not tr.is_closed():
                            # Determine exit reason
                            exit_reason = "MANUAL"
                            if deal.price <= 0:
                                continue
                            
                            # We need to find the original position to get SL/TP
                            # Since position is closed, we infer from deal profit
                            if deal.profit < -abs(deal.commission + deal.swap):
                                exit_reason = "SL"
                            elif deal.profit > abs(deal.commission + deal.swap):
                                exit_reason = "TP"
                            else:
                                exit_reason = "REVERSE"
                            
                            tr.exit_time = deal.time
                            tr.exit_price = deal.price
                            tr.exit_reason = exit_reason
                            # Use deal.profit directly (already in account currency, no lot size assumption)
                            tr.realized_pnl = deal.profit
                            tr.holding_bars = 0  # Would need bar data to compute accurately
                            
                            # Approximate MFE/MAE from deal profit/commission/swap
                            # deal.profit is already in account currency
                            if deal.profit < 0:
                                # Loss trade: MAE ≈ |profit| + commissions + swaps
                                tr.max_adverse_excursion = abs(deal.profit) + abs(deal.commission) + abs(deal.swap)
                                tr.max_favorable_excursion = 0.0
                            else:
                                # Win trade: MFE ≈ profit + commissions + swaps
                                tr.max_favorable_excursion = deal.profit + abs(deal.commission) + abs(deal.swap)
                                tr.max_adverse_excursion = 0.0
                            
                            newly_closed.append(tr)
                            break
        
        return newly_closed

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

    def modify_tp(self, position, new_tp: float) -> TradeResult:
        """Modify take-profit on an open position."""
        new_tp = self._normalize_price(position.symbol, new_tp)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": position.sl,
            "tp": new_tp,
            "magic": self.config.magic_number,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeResult(False, "MODIFY_TP", "TP modify failed")
        trade = TradeResult(True, "MODIFY_TP", f"#{position.ticket} TP -> {new_tp:.5f}")
        self.trade_log.append(trade)
        return trade

    def manage_dynamic_targets(self, symbol: str, forecast: Dict):
        """Bring TP closer and tighten SL if predictive confidence drops significantly."""
        conf = forecast.get("confidence", 0.0) if forecast else 0.0
        if conf < 0.65:
            tick = mt5.symbol_info_tick(symbol)
            if not tick: return
            for pos in self.get_positions(symbol):
                is_buy = (pos.type == mt5.POSITION_TYPE_BUY)
                current_price = tick.bid if is_buy else tick.ask
                min_dist = self._min_stop_distance(symbol)
                
                if is_buy and pos.tp > current_price:
                    # Cut remaining distance to TP in half
                    new_tp = current_price + (pos.tp - current_price) * 0.5
                    # Never drag TP below breakeven
                    new_tp = max(new_tp, pos.price_open + min_dist)
                    
                    # Symmetrically tighten SL to preserve Risk/Reward
                    if pos.sl > 0 and pos.sl < current_price:
                        new_sl = current_price - (current_price - pos.sl) * 0.5
                        new_sl = max(new_sl, pos.sl) # Only trail up
                        if new_sl > pos.sl + min_dist:
                            self.modify_sl(pos, new_sl)
                            
                    if new_tp < pos.tp:
                        self.modify_tp(pos, new_tp)
                        
                elif not is_buy and pos.tp > 0 and pos.tp < current_price:
                    new_tp = current_price - (current_price - pos.tp) * 0.5
                    # Never drag TP above breakeven
                    new_tp = min(new_tp, pos.price_open - min_dist)
                    
                    # Symmetrically tighten SL to preserve Risk/Reward
                    if pos.sl > 0 and pos.sl > current_price:
                        new_sl = current_price + (pos.sl - current_price) * 0.5
                        new_sl = min(new_sl, pos.sl) # Only trail down
                        if new_sl < pos.sl - min_dist:
                            self.modify_sl(pos, new_sl)
                            
                    if new_tp > pos.tp:
                        self.modify_tp(pos, new_tp)

    def manage_trailing_stops(self, symbol: str, timeframe: int):
        """Trail stop-loss on open positions using ATR distance.
        
        Key behaviors:
        1. Don't trail until trade is in profit by trail_activation_atr × ATR (gives breathing room)
        2. Once activated, move to breakeven first, then trail from the high-water-mark
        3. Uses trailing_atr_multiplier (default 2.0) for the trail distance
        """
        atr = self.risk.compute_atr(symbol, timeframe, self.config.atr_period)
        if atr <= 0:
            return

        trail_dist = atr * self.config.trailing_atr_multiplier
        activation_dist = atr * self.config.trail_activation_atr
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        
        point = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.00001

        for pos in self.get_positions(symbol):
            is_buy = (pos.type == mt5.POSITION_TYPE_BUY)
            current_price = tick.bid if is_buy else tick.ask
            
            # Calculate how far in profit the trade is
            if is_buy:
                profit_dist = current_price - pos.price_open
            else:
                profit_dist = pos.price_open - current_price
            
            # Phase 1: Don't touch the stop until trade is sufficiently in profit
            if profit_dist < activation_dist:
                continue
            
            # Phase 2: Track high-water mark (best price since entry)
            hwm = self._position_hwm.get(pos.ticket, pos.price_open)
            if is_buy:
                if current_price > hwm:
                    hwm = current_price
                    self._position_hwm[pos.ticket] = hwm
                # Trail from the high-water mark, not current price
                new_sl = hwm - trail_dist
                # Ensure we never move SL backwards
                if pos.sl == 0 or new_sl > pos.sl + point:
                    if new_sl < current_price:  # SL must be below current price
                        self.modify_sl(pos, new_sl)
            else:
                if current_price < hwm:
                    hwm = current_price
                    self._position_hwm[pos.ticket] = hwm
                new_sl = hwm + trail_dist
                if pos.sl == 0 or new_sl < pos.sl - point:
                    if new_sl > current_price:  # SL must be above current price
                        self.modify_sl(pos, new_sl)

    def summary(self) -> Dict:
        return dict(self._trade_count, total_trades=len(self.trade_log))

class AutoTradingEngine(TradingEngine):
    """
    Extends TradingEngine with MT5 order execution and adaptive intelligence.

    On each new bar:
      BUY  → close shorts, open long (if risk allows)
      SELL → close longs, open short (if risk allows)
      HOLD → manage trailing stops only
    
    Adaptive behaviors:
      - Dynamic confidence threshold based on rolling win rate
      - Trade-weighted learning (big losses get more update passes)
      - Feature attribution on losing trades
      - SL multiplier self-correction on repeated SL hits
    """

    def __init__(
        self,
        executor: MT5TradeExecutor,
        confidence_threshold: float = 0.6,
        timeframe: int = None,
        forecaster = None
    ):
        super().__init__(confidence_threshold)
        self.executor = executor
        self.timeframe = timeframe
        self.forecaster = forecaster
        self.analytics = TradeAnalytics()
        self._base_confidence = confidence_threshold  # The original user-set value

    def on_signal(self, forecast: Dict, symbol: str):
        """Evaluate signal, print it, and execute trades when appropriate."""
        # Update regime tracking
        is_alert = forecast.get("alert", False)
        self.analytics.update_regime_bar(is_alert)

        # Apply dynamic confidence threshold
        self.confidence_threshold = self.analytics.suggest_confidence_adjustment(
            self._base_confidence
        )

        signal = self.evaluate(forecast, symbol)
        self.print_signal(signal)

        # Expectancy-based filter: stop trading if rolling expectancy is negative
        expectancy = self.analytics.get_expectancy()
        if expectancy < 0 and len(self.analytics._closed_trades) >= 20:
            signal.action = "HOLD"
            signal.reason = f"Negative expectancy ({expectancy:.4f}) — filtering trades"
            self._print_trade(TradeResult(True, "HOLD", signal.reason))

        if self.timeframe is None:
            return

        # Check for trades closed by SL/TP (market-driven closes)
        self._process_closed_trades(symbol)
        
        # Always manage trailing stops on open positions
        self.executor.manage_trailing_stops(symbol, self.timeframe)
        # Removed dynamic target management to keep trades simple

        # News blackout filter
        if getattr(self, '_news_filter_enabled', False):
            is_blackout, news_reason = is_news_blackout(symbol)
            if is_blackout:
                YELLOW = "\033[93m"
                RESET = "\033[0m"
                print(f"  {YELLOW}[NEWS BLACKOUT] {news_reason} — skipping trade{RESET}")
                return

        # Time-of-day filter: avoid low-liquidity hours (00:00-06:00 UTC and 22:00-23:59 UTC)
        if getattr(self, '_time_filter_enabled', True):
            import datetime as _dt
            hour = _dt.datetime.utcnow().hour
            if hour < 6 or hour >= 22:  # 00:00-05:59 and 22:00-23:59 UTC
                signal.action = "HOLD"
                signal.reason = f"Low liquidity hour filter (UTC {hour:02d}:00)"
                return

        if signal.action == "HOLD":
            return

        positions = self.executor.get_positions(symbol)
        longs = [p for p in positions if p.type == mt5.POSITION_TYPE_BUY]
        shorts = [p for p in positions if p.type == mt5.POSITION_TYPE_SELL]

        if signal.action == "BUY":
            for pos in shorts:
                result = self.executor.close_position(pos)
                if result.success and result.entry_features:
                    self._learn_from_close(result, exit_reason="REVERSE")
                self._print_trade(result)
                # Record reversal close in analytics
                if result.success:
                    closed_records = [tr for tr in self.executor.trade_records if tr.ticket == pos.ticket and tr.is_closed()]
                    for tr in closed_records:
                        self.analytics.record_trade(tr)
            if not longs:
                result = self.executor.open_position(
                    symbol, "BUY", forecast, self.timeframe, analytics=self.analytics
                )
                self._print_trade(result)

        elif signal.action == "SELL":
            for pos in longs:
                result = self.executor.close_position(pos)
                if result.success and result.entry_features:
                    self._learn_from_close(result, exit_reason="REVERSE")
                self._print_trade(result)
                # Record reversal close in analytics
                if result.success:
                    closed_records = [tr for tr in self.executor.trade_records if tr.ticket == pos.ticket and tr.is_closed()]
                    for tr in closed_records:
                        self.analytics.record_trade(tr)
            if not shorts:
                result = self.executor.open_position(
                    symbol, "SELL", forecast, self.timeframe, analytics=self.analytics
                )
                self._print_trade(result)

    def _learn_from_close(self, result: TradeResult, exit_reason: str = ""):
        """Trade-weighted learning: penalize big losses harder (up to 3× updates)."""
        if not self.forecaster or not self.forecaster._model:
            return
        if not result.entry_features:
            return
        
        pnl = result.realized_pnl
        avg_loss = self.analytics.get_avg_loss()
        
        # Use percentage return for model training, not absolute money
        target = result.realized_return
        
        # Determine how many learning passes to run
        n_updates = 1
        if pnl < 0 and avg_loss > 0:
            # Big losses get more update passes (capped at 3)
            n_updates = min(3, int(abs(pnl) / avg_loss) + 1)
        
        try:
            for _ in range(n_updates):
                self.forecaster._model.learn_one(result.entry_features, target)
            
            tag = f"[×{n_updates} Learned]" if n_updates > 1 else "[Learned]"
            result.message += f" {tag}"
        except Exception:
            pass

    def _process_closed_trades(self, symbol: str):
        """Check for and process trades closed by market (SL/TP hits)."""
        if not self.executor or not self.forecaster or not self.forecaster._model:
            return
        
        closed_records = self.executor.check_closed_trades(symbol)
        for tr in closed_records:
            # Trade-weighted learning
            if tr.entry_features:
                pnl = tr.realized_pnl
                
                # Counterfactual Learning & Fix:
                # Model expects next-bar log_return as target, not cumulative trade PnL!
                # We need to find the actual next-bar return that occurred after trade entry.
                target = 0.0
                
                # Look up the actual next-bar return from forecaster's record history
                # The trade was opened at entry_bar_idx, so the next bar is entry_bar_idx + 1
                if tr.entry_bar_idx > 0 and tr.entry_bar_idx < len(self.forecaster._record_hist):
                    # The next bar after entry
                    next_bar_idx = tr.entry_bar_idx
                    if next_bar_idx < len(self.forecaster._record_hist):
                        next_record = self.forecaster._record_hist[next_bar_idx]
                        target = next_record.log_return
                        
                        # Counterfactual boost: if SL hit, explicitly amplify the actual opposite return
                        if tr.exit_reason == "SL":
                            target = target * 2.0  # Amplify to heavily penalize the blindspot
                    else:
                        # Fallback: use the actual next-bar return from forecaster's current state
                        # This happens when the trade was opened very recently
                        target = 0.0
                
                try:
                    # Offload to AI engine if using PyTorch/RL
                    if hasattr(self.forecaster, 'ai_actor') and self.forecaster.ai_actor:
                        # In V5 Ray Actor, counterfactual learning would be handled here
                        print(f"  [AI-LEARN] Trade #{tr.ticket} closed via {tr.exit_reason}: "
                              f"PnL={pnl:.4f} — Counterfactual stored for offline RL.")
                    else:
                        self.forecaster._model.learn_one(tr.entry_features, target)
                        print(f"  [LEARN] Trade #{tr.ticket} closed via {tr.exit_reason}: "
                              f"PnL={pnl:.4f} — Counterfactual target={target:.5f}")
                except Exception as e:
                    print(f"  [WARN] Failed to learn from trade #{tr.ticket}: {e}")
            
            # Record in analytics and analyze
            self.analytics.record_trade(tr)
            self._analyze_trade_failure(tr)
            
            # Print confidence adjustment if it changed
            new_conf = self.analytics.suggest_confidence_adjustment(self._base_confidence)
            if abs(new_conf - self._base_confidence) > 0.005:
                print(f"  [CONFIDENCE ADJUSTED] {self._base_confidence:.1%} → {new_conf:.1%} "
                      f"(rolling 10-trade WR: {self.analytics.get_rolling_win_rate(10):.0%})")

    def _analyze_trade_failure(self, trade: TradeRecord):
        """Analyze why a trade failed, classify failure mode, print feature attribution."""
        if trade.is_win():
            return  # Only analyze losing trades
        
        # ── Failure Mode Classification ──
        failure_mode = "UNKNOWN"
        if trade.exit_reason == "SL":
            if trade.max_adverse_excursion > abs(trade.realized_pnl) * 1.5:
                failure_mode = "VOLATILITY_SPIKE"
            elif trade.entry_regime == "ALERT":
                failure_mode = "REGIME_CHANGE"
            elif trade.entry_confidence > 0.7:
                failure_mode = "HIGH_CONFIDENCE_ERROR"
            else:
                failure_mode = "DIRECTION_ERROR"
        elif trade.exit_reason == "TP":
            failure_mode = "EARLY_EXIT"
        elif trade.exit_reason == "REVERSE":
            failure_mode = "REVERSAL_SIGNAL"
        
        trade.failure_mode = failure_mode
        
        # ── Feature Attribution (lightweight) ──
        attribution_str = ""
        try:
            if HAS_RIVER and hasattr(self.forecaster._model, '_pa'):
                # For River: inspect PA regressor weights via prediction contribution
                feats = trade.entry_features
                if feats:
                    # Compute signed contribution: feature_value for each feature
                    # Sort by absolute value to find top contributors
                    sorted_feats = sorted(feats.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                    attribution_str = " | Top features: " + ", ".join(
                        f"{k}={v:+.3f}" for k, v in sorted_feats
                    )
            elif not HAS_RIVER and hasattr(self.forecaster._model, '_skl_model'):
                # For sklearn: inspect SGDRegressor coefficients
                model = self.forecaster._model._skl_model
                if hasattr(model, 'coef_') and trade.entry_features:
                    feat_names = list(trade.entry_features.keys())
                    coefs = model.coef_
                    if len(coefs) == len(feat_names):
                        feat_vals = list(trade.entry_features.values())
                        contributions = [(feat_names[i], coefs[i] * feat_vals[i]) 
                                        for i in range(len(feat_names))]
                        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                        top3 = contributions[:3]
                        attribution_str = " | Top contributors: " + ", ".join(
                            f"{k}={v:+.4f}" for k, v in top3
                        )
        except Exception:
            pass
        
        print(f"  [ANALYSIS] Trade #{trade.ticket} ({trade.direction}): "
              f"Failure={failure_mode} | Conf={trade.entry_confidence:.1%} | "
              f"Regime={trade.entry_regime}{attribution_str}")

    def on_poll(self, symbol: str):
        """Called between bars to update trailing stops and check closed trades."""
        if self.timeframe is not None:
            self.executor.manage_trailing_stops(symbol, self.timeframe)
            self._process_closed_trades(symbol)

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
            "analytics": self.analytics.get_summary(),
        }

