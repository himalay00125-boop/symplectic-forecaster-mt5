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

def get_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        hull = ConvexHull(points)
        verts = points[hull.vertices]
        return np.vstack([verts, verts[0]])
    except Exception:
        return np.empty((0, 2))

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

def detect_fair_value_gaps(bars: List[Bar], lookback: int = 20) -> List[Dict]:
    """Detect Fair Value Gaps (FVG) in recent price action.

    An FVG forms when bar[i-1].high < bar[i+1].low (bullish) or
    bar[i-1].low > bar[i+1].high (bearish), leaving an unfilled gap.
    Returns list of gap dicts with direction, top, bottom, midpoint.
    """
    gaps = []
    start = max(0, len(bars) - lookback)
    for i in range(start + 1, len(bars) - 1):
        # Bullish FVG: gap up (candle 1 high < candle 3 low)
        if bars[i - 1].high < bars[i + 1].low:
            gaps.append({
                "direction": "bullish",
                "top": bars[i + 1].low,
                "bottom": bars[i - 1].high,
                "midpoint": (bars[i + 1].low + bars[i - 1].high) / 2,
                "size": bars[i + 1].low - bars[i - 1].high,
                "bar_index": i,
            })
        # Bearish FVG: gap down (candle 1 low > candle 3 high)
        if bars[i - 1].low > bars[i + 1].high:
            gaps.append({
                "direction": "bearish",
                "top": bars[i - 1].low,
                "bottom": bars[i + 1].high,
                "midpoint": (bars[i - 1].low + bars[i + 1].high) / 2,
                "size": bars[i - 1].low - bars[i + 1].high,
                "bar_index": i,
            })
    return gaps

def compute_session_levels(bars: List[Bar]) -> Dict[str, float]:
    """Compute ICT session reference levels from recent bars.

    Identifies:
    - Midnight Open: the open price of the bar closest to 00:00 UTC
    - Asian Session High/Low: high/low between 00:00-08:00 UTC
    - London Open: open of bar closest to 08:00 UTC

    Falls back to simple statistical levels if session data is sparse.
    """
    import datetime as _dt
    levels = {
        "midnight_open": 0.0,
        "asian_high": 0.0,
        "asian_low": float('inf'),
        "london_open": 0.0,
    }

    if not bars:
        levels["asian_low"] = 0.0
        return levels

    # Work backwards through bars to find session levels from today/yesterday
    asian_bars = []
    for bar in reversed(bars[-96:]):  # Up to 96 bars back (covers 24h on M15)
        try:
            bar_dt = _dt.datetime.utcfromtimestamp(bar.timestamp)
        except (OSError, ValueError):
            continue
        hour = bar_dt.hour

        # Midnight open: closest bar to 00:00 UTC
        if hour == 0 and levels["midnight_open"] == 0.0:
            levels["midnight_open"] = bar.open

        # Asian session: 00:00 - 08:00 UTC
        if 0 <= hour < 8:
            asian_bars.append(bar)

        # London open: closest bar to 08:00 UTC
        if hour == 8 and levels["london_open"] == 0.0:
            levels["london_open"] = bar.open

    if asian_bars:
        levels["asian_high"] = max(b.high for b in asian_bars)
        levels["asian_low"] = min(b.low for b in asian_bars)
    else:
        levels["asian_high"] = bars[-1].high
        levels["asian_low"] = bars[-1].low

    if levels["midnight_open"] == 0.0:
        levels["midnight_open"] = bars[-1].open
    if levels["london_open"] == 0.0:
        levels["london_open"] = bars[-1].open

    return levels

def compute_mtf_betti(symbol: str, base_tf_str: str, bars_count: int = 200) -> Dict[str, float]:
    """Compute Betti numbers on higher timeframes for topological alignment.

    Given a base timeframe (e.g., M15), also computes TDA features on H1 and H4.
    Returns a dict of higher-TF Betti numbers and an alignment score.
    """
    if not HAS_MT5:
        return {"mtf_alignment": 0.5, "h1_betti1": 0, "h4_betti1": 0}

    MTF_PAIRS = {
        "M1": ["M15", "H1"], "M5": ["M30", "H1"], "M15": ["H1", "H4"],
        "M30": ["H1", "H4"], "H1": ["H4", "D1"], "H4": ["D1", "W1"],
        "D1": ["W1", "MN1"],
    }

    higher_tfs = MTF_PAIRS.get(base_tf_str.upper(), [])
    if not higher_tfs:
        return {"mtf_alignment": 0.5, "h1_betti1": 0, "h4_betti1": 0}

    results = {"mtf_alignment": 0.5}
    directions = []

    for i, htf_str in enumerate(higher_tfs):
        htf = TIMEFRAME_MAP.get(htf_str)
        if htf is None:
            continue
        try:
            rates = mt5.copy_rates_from_pos(symbol, htf, 1, min(bars_count, 100))
            if rates is None or len(rates) < 10:
                continue

            # Build phase space for higher TF
            htf_bars = []
            for r in rates:
                vol = float(r["real_volume"]) if r["real_volume"] > 0 else float(r["tick_volume"])
                htf_bars.append(Bar(
                    timestamp=float(r["time"]), open=float(r["open"]),
                    high=float(r["high"]), low=float(r["low"]),
                    close=float(r["close"]), volume=vol
                ))

            # Compute phase coords and TDA
            phase_pts = []
            for j in range(1, len(htf_bars)):
                pp = phase_coords(htf_bars[j], htf_bars[j-1])
                phase_pts.append((pp.q, pp.p))

            if len(phase_pts) >= 5:
                pts_arr = np.array(phase_pts, dtype=float)
                idx_s = np.random.choice(len(pts_arr), min(80, len(pts_arr)), replace=False)
                tda = compute_tda(pts_arr[idx_s])

                key_prefix = f"htf{i+1}"
                results[f"{key_prefix}_betti1"] = float(tda["betti_1"])
                results[f"{key_prefix}_tot_pers"] = float(tda["tot_pers"])

                # Infer direction from recent returns on higher TF
                recent_rets = [math.log(htf_bars[k].close / htf_bars[k-1].close)
                               for k in range(max(1, len(htf_bars)-5), len(htf_bars))
                               if htf_bars[k-1].close > 0]
                avg_ret = sum(recent_rets) / len(recent_rets) if recent_rets else 0
                directions.append(1 if avg_ret > 0 else -1 if avg_ret < 0 else 0)
        except Exception:
            continue

    # Alignment: all higher TFs agree on direction = 1.0, disagree = 0.0
    if len(directions) >= 2:
        if all(d == directions[0] and d != 0 for d in directions):
            results["mtf_alignment"] = 1.0
        elif any(d != directions[0] for d in directions):
            results["mtf_alignment"] = 0.0
        else:
            results["mtf_alignment"] = 0.5

    # Flatten keys for backward compat
    results.setdefault("h1_betti1", results.get("htf1_betti1", 0))
    results.setdefault("h4_betti1", results.get("htf2_betti1", 0))

    return results

