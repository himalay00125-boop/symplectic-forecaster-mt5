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

class TradeAnalytics:
    """Rolling analytics computed from closed TradeRecords.
    
    Tracks per-regime accuracy, per-confidence-band accuracy,
    SL-hit rates, and provides adaptive recommendations for
    confidence threshold and SL/TP sizing.
    """

    def __init__(self):
        self._closed_trades: List[TradeRecord] = []
        self._sl_hit_count = 0
        self._tp_hit_count = 0
        self._regime_stats: Dict[str, Dict] = {
            "NORMAL": {"wins": 0, "total": 0},
            "ALERT":  {"wins": 0, "total": 0},
        }
        self._confidence_bands: Dict[str, Dict] = {
            "low":  {"wins": 0, "total": 0},   # 0.0 – 0.50
            "mid":  {"wins": 0, "total": 0},   # 0.50 – 0.65
            "high": {"wins": 0, "total": 0},   # 0.65+
        }
        self._recent_sl_streak = 0
        self._bars_since_alert = 999  # Bars since last ALERT regime
        
        # Feature importance tracking for pruning harmful features
        self._feature_harm_scores: Dict[str, float] = {}  # feature -> cumulative harm score
        self._feature_help_scores: Dict[str, float] = {}  # feature -> cumulative help score

    def record_trade(self, trade: TradeRecord):
        """Ingest a closed trade and update all rolling statistics."""
        self._closed_trades.append(trade)
        is_win = trade.is_win()

        # Regime stats
        regime = trade.entry_regime or "NORMAL"
        if regime not in self._regime_stats:
            self._regime_stats[regime] = {"wins": 0, "total": 0}
        self._regime_stats[regime]["total"] += 1
        if is_win:
            self._regime_stats[regime]["wins"] += 1

        # Confidence band stats
        conf = trade.entry_confidence
        if conf < 0.50:
            band = "low"
        elif conf < 0.65:
            band = "mid"
        else:
            band = "high"
        self._confidence_bands[band]["total"] += 1
        if is_win:
            self._confidence_bands[band]["wins"] += 1

        # SL/TP hit tracking
        if trade.exit_reason == "SL":
            self._sl_hit_count += 1
            self._recent_sl_streak += 1
        elif trade.exit_reason == "TP":
            self._tp_hit_count += 1
            self._recent_sl_streak = 0
        else:
            self._recent_sl_streak = 0
        
        # Feature importance tracking (pruning harmful features)
        if trade.entry_features:
            for feat, val in trade.entry_features.items():
                if is_win:
                    # Winning trade: features with large absolute values that aligned with direction get credit
                    self._feature_help_scores[feat] = self._feature_help_scores.get(feat, 0) + abs(val)
                else:
                    # Losing trade: features with large absolute values that misled get penalty
                    self._feature_harm_scores[feat] = self._feature_harm_scores.get(feat, 0) + abs(val)

    def update_regime_bar(self, is_alert: bool):
        """Called every bar to track how many bars since last ALERT."""
        if is_alert:
            self._bars_since_alert = 0
        else:
            self._bars_since_alert += 1

    def get_rolling_win_rate(self, n: int = 10) -> float:
        """Win rate of last N closed trades."""
        recent = self._closed_trades[-n:]
        if not recent:
            return 0.5
        return sum(1 for t in recent if t.is_win()) / len(recent)

    def get_avg_loss(self) -> float:
        """Average absolute loss across all losing trades."""
        losses = [abs(t.realized_pnl) for t in self._closed_trades if not t.is_win()]
        return sum(losses) / len(losses) if losses else 0.01
    
    def get_expectancy(self) -> float:
        """Compute rolling expectancy (expected value per trade in account currency).
        
        Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        Positive = profitable strategy, Negative = losing strategy.
        """
        if not self._closed_trades:
            return 0.0
        
        n = len(self._closed_trades)
        wins = [t.realized_pnl for t in self._closed_trades if t.is_win()]
        losses = [t.realized_pnl for t in self._closed_trades if not t.is_win()]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / n
        loss_rate = len(losses) / n
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    def get_harmful_features(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features most associated with losing trades."""
        sorted_harm = sorted(self._feature_harm_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_harm[:top_n]
    
    def get_helpful_features(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features most associated with winning trades."""
        sorted_help = sorted(self._feature_help_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_help[:top_n]

    def suggest_confidence_adjustment(self, base_threshold: float) -> float:
        """Suggest adjusted confidence threshold based on recent performance.
        
        - If winning (>60% last 10): lower threshold (be bolder)
        - If losing (<40% last 10): raise threshold (be pickier)
        - After ALERT clears: temporarily boost threshold for 3 bars
        """
        adjusted = base_threshold
        n_closed = len(self._closed_trades)
        
        if n_closed >= 10:
            recent_wr = self.get_rolling_win_rate(10)
            if recent_wr > 0.60:
                adjusted = max(0.25, adjusted - 0.03)
            elif recent_wr < 0.40:
                adjusted = min(0.70, adjusted + 0.05)
        
        # Post-ALERT caution: boost threshold for 3 bars after alert clears
        if self._bars_since_alert <= 3:
            adjusted = min(0.75, adjusted + 0.10)
        
        return adjusted

    def suggest_sl_multiplier_adjustment(self, base_mult: float) -> float:
        """Widen SL if the last 5 trades all hit SL (self-correcting)."""
        if self._recent_sl_streak >= 5:
            return base_mult * 1.20  # Widen by 20%
        return base_mult

    def suggest_rr_adjustment(self, base_rr: float, entry_vol_ratio: float) -> float:
        """Tighten R:R in high-volatility entries."""
        if entry_vol_ratio > 1.2:
            return base_rr * 0.8  # Tighter TP in high vol
        return base_rr

    def get_summary(self) -> Dict:
        """Return a summary dict for logging/dashboard."""
        n = len(self._closed_trades)
        return {
            "total_analyzed": n,
            "sl_hits": self._sl_hit_count,
            "tp_hits": self._tp_hit_count,
            "sl_streak": self._recent_sl_streak,
            "bars_since_alert": self._bars_since_alert,
            "regime_stats": {k: v for k, v in self._regime_stats.items()},
            "conf_bands": {k: v for k, v in self._confidence_bands.items()},
            "rolling_10_wr": round(self.get_rolling_win_rate(10) * 100, 1) if n > 0 else 0.0,
        }

