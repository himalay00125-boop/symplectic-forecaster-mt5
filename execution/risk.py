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


from core.config import RiskConfig
from core.types import *

class RiskManager:
    """Position sizing, daily loss limits, and trade permission checks."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self._session_start_equity: Optional[float] = None
        self._session_date: Optional[date] = None
        self._trading_halted = False
        self._halt_reason = ""

    def reset_session(self):
        """Reset daily tracking (call at bot startup or new trading day)."""
        acc = mt5.account_info()
        if acc:
            self._session_start_equity = acc.equity
            self._session_date = date.today()
            self._trading_halted = False
            self._halt_reason = ""

    def _roll_session_if_new_day(self):
        today = date.today()
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

    def calculate_lot_size(self, symbol: str, stop_distance: float, confidence: float = 0.5, pred_interval_width: float = 0.0) -> float:
        """Size position so stop loss risks `risk_per_trade_pct` of equity.
        
        Conviction-based scaling:
        - confidence 0.5 → 0.5x base size (cautious)
        - confidence 0.7 → 1.0x base size (normal)
        - confidence 0.9 → 1.5x base size (high conviction)
        
        Conformal Prediction Scaling:
        - Wide intervals (high uncertainty) reduce position size.
        - Narrow intervals (high certainty) increase position size.
        """
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        if acc is None or sym is None or stop_distance <= 0:
            return sym.volume_min if sym else 0.01

        # Conviction scaling: 0.5 + confidence (maps 0.5->1.0, 0.7->1.2, 0.9->1.4)
        conviction_mult = 0.5 + confidence
        
        # Conformal interval scaling
        if pred_interval_width > 0:
            # Baseline typical interval is around 0.005 for forex
            baseline_width = 0.005
            uncertainty_scale = baseline_width / max(pred_interval_width, 1e-6)
            # Combine scalings
            conviction_mult *= uncertainty_scale
            
        conviction_mult = max(0.3, min(conviction_mult, 2.0))  # Clamp 0.3x to 2.0x
        
        risk_amount = acc.equity * (self.config.risk_per_trade_pct / 100.0) * conviction_mult
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
        lots = min(sym.volume_max, lots)
        return round(lots, 2)
    
    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute Kelly fraction for position sizing.
        
        Kelly % = (b * p - q) / b
        where b = avg_win / avg_loss (reward:risk ratio)
              p = win_rate
              q = 1 - p (loss rate)
        
        Returns quarter-Kelly capped at 2% for safety.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self.config.risk_per_trade_pct / 100.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly = max(0.0, kelly)  # No negative Kelly
        quarter_kelly = kelly * 0.25  # Quarter-Kelly for safety
        
        # Cap at 2% of equity per trade
        return min(quarter_kelly, 0.02)

    def calculate_lot_size_kelly(self, symbol: str, stop_distance: float, 
                                  win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Size position using Kelly Criterion based on historical performance."""
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        if acc is None or sym is None or stop_distance <= 0:
            return sym.volume_min if sym else 0.01

        kelly_risk_pct = self.kelly_fraction(win_rate, avg_win, avg_loss)
        risk_amount = acc.equity * kelly_risk_pct
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
        lots = min(sym.volume_max, lots)
        return round(lots, 2)

    def calculate_lot_size_uncertainty(self, symbol: str, stop_distance: float,
                                        confidence: float, interval_width: float,
                                        vol_20: float) -> float:
        """
        Size position accounting for prediction uncertainty.
        
        Higher uncertainty (wider intervals) → smaller position.
        Uses conformal interval width relative to volatility.
        """
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        if acc is None or sym is None or stop_distance <= 0:
            return sym.volume_min if sym else 0.01

        # Base conviction from confidence
        conviction_mult = 0.5 + confidence
        conviction_mult = max(0.3, min(conviction_mult, 2.0))
        
        # Uncertainty penalty: wider interval relative to vol = less certain
        if vol_20 > 0 and interval_width > 0:
            uncertainty_ratio = interval_width / (vol_20 * 2.0)  # Normalize by 2x vol
            uncertainty_penalty = min(1.0, uncertainty_ratio)  # Cap at 1.0
            conviction_mult *= (1.0 - 0.5 * uncertainty_penalty)  # Up to 50% reduction
        
        conviction_mult = max(0.2, conviction_mult)  # Floor at 0.2x
        
        risk_amount = acc.equity * (self.config.risk_per_trade_pct / 100.0) * conviction_mult
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
        lots = min(sym.volume_max, lots)
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

