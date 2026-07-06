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
    p: float

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
    entry_features: Dict = field(default_factory=dict)
    realized_pnl: float = 0.0
    realized_return: float = 0.0

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
class TradeRecord:
    """Complete trade record with entry context for learning from outcomes."""
    # Trade identification
    ticket: int
    symbol: str
    
    # Entry context (captured at trade open)
    entry_time: float
    entry_bar_idx: int
    entry_price: float
    direction: str  # "BUY" or "SELL"
    volume: float
    sl: float
    tp: float
    
    # Model state at entry
    entry_confidence: float
    entry_predicted_return: float
    entry_regime: str
    entry_capacity: float
    entry_perimeter: float
    entry_betti_0: int
    entry_betti_1: int
    entry_tot_pers: float
    entry_vol_10: float
    entry_vol_20: float
    entry_features: Dict  # Full feature vector
    
    # Exit details (filled when trade closes)
    exit_time: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""  # "SL", "TP", "REVERSE", "MANUAL", "END"
    realized_pnl: float = 0.0
    
    # Analysis fields (computed after close)
    holding_bars: int = 0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    failure_mode: str = ""                # Classified failure type
    
    def is_closed(self) -> bool:
        return self.exit_time > 0
    
    def is_win(self) -> bool:
        return self.realized_pnl > 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TradeRecord":
        return cls(**d)

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

