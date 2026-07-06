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
from models.base import *
from models.meta import *
from models.validation import *
from analytics.trade_analytics import *
from execution.executor import *

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
                 min_train_bars: int   = 80,
                 shared_nlp_agent=None):

        global _GLOBAL_RAY_READY, _GLOBAL_MAMBA_MODEL, _GLOBAL_NLP_AGENT, _RAY_INIT_ATTEMPTED

        self.window          = window
        self.alert_pct       = alert_pct
        self.tda_subsample   = tda_subsample
        self.min_train_bars  = min_train_bars

        self._phase_buf      : collections.deque = collections.deque(maxlen=window)
        self._record_hist    : List[CapacityRecord] = []
        self._capacity_buf   : collections.deque = collections.deque(maxlen=500)

        self._model          = BatchedLearner(RegimeAwareModel(), batch_size=32)
        self._scenario_gen   = ScenarioGenerator(self._model)

        # Ray Distributed Architecture for AI Engine (singleton init)
        self.ai_actor = None
        if HAS_AI_ENGINE:
            with _INIT_LOCK:
                if not _RAY_INIT_ATTEMPTED:
                    _RAY_INIT_ATTEMPTED = True
                    try:
                        if not ray.is_initialized():
                            ray.init(ignore_reinit_error=True, logging_level="ERROR")
                        _GLOBAL_RAY_READY = True
                    except Exception:
                        _GLOBAL_RAY_READY = False
                        print("[RAY] Ray unavailable on Windows — using local Mamba.")

            if _GLOBAL_RAY_READY:
                try:
                    self.ai_actor = AIEngineActor.remote()
                except Exception:
                    pass

            if self.ai_actor is None:
                # Reuse a single fallback Mamba model across all symbols
                with _INIT_LOCK:
                    if _GLOBAL_MAMBA_MODEL is None:
                        from ai_engine import SymplecticSTGCN_KAN
                        _GLOBAL_MAMBA_MODEL = SymplecticSTGCN_KAN(
                            input_dim=16, hidden_dim=64, num_layers=2
                        )
                        print("[AI ENGINE] Loaded local Mamba model (shared).")
                self._fallback_mamba = _GLOBAL_MAMBA_MODEL
                self.ai_actor = "LOCAL_MAMBA_FALLBACK"

        # NLP Agent — reuse shared instance if provided
        if shared_nlp_agent is not None:
            self.nlp_agent = shared_nlp_agent
        else:
            self.nlp_agent = None
            if HAS_NLP:
                with _INIT_LOCK:
                    if _GLOBAL_NLP_AGENT is None:
                        _GLOBAL_NLP_AGENT = FundamentalAgent()
                self.nlp_agent = _GLOBAL_NLP_AGENT


        self.causal_adj_matrix = None

        self._bar_count      = 0
        self._prev_bar       : Optional[Bar] = None
        self._last_record    : Optional[CapacityRecord] = None
        self._last_feats     : Optional[Dict] = None
        self._last_price     : float = 0.0
        self._freeze_learning: bool = False
        self._bar_buf: collections.deque = collections.deque(maxlen=200)
        self._mtf_symbol: str = ""
        self._mtf_tf_str: str = ""

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
        self._bar_buf.append(bar)
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

        # --- Attach ICT features to record for the feature vector ---
        bar_list = list(self._bar_buf)
        fvgs = detect_fair_value_gaps(bar_list)
        session = compute_session_levels(bar_list)

        rec._fvg_count = len(fvgs)
        if fvgs and bar.close > 0:
            nearest = min(fvgs, key=lambda g: abs(bar.close - g["midpoint"]))
            rec._fvg_nearest_dist = (bar.close - nearest["midpoint"]) / bar.close
        else:
            rec._fvg_nearest_dist = 0.0

        if session["midnight_open"] > 0:
            rec._midnight_dist = (bar.close - session["midnight_open"]) / bar.close
        else:
            rec._midnight_dist = 0.0

        asian_range = session["asian_high"] - session["asian_low"]
        if asian_range > 0:
            rec._asian_range_pct = (bar.close - session["asian_low"]) / asian_range
        else:
            rec._asian_range_pct = 0.5

        # Multi-timeframe alignment (only if we know the symbol/TF)
        if self._mtf_symbol and self._mtf_tf_str:
            try:
                mtf = compute_mtf_betti(self._mtf_symbol, self._mtf_tf_str)
                rec._mtf_alignment = mtf.get("mtf_alignment", 0.5)
                rec._htf1_betti1 = mtf.get("h1_betti1", 0)
                rec._htf2_betti1 = mtf.get("h4_betti1", 0)
            except Exception:
                rec._mtf_alignment = 0.5
                rec._htf1_betti1 = 0
                rec._htf2_betti1 = 0
        else:
            rec._mtf_alignment = 0.5
            rec._htf1_betti1 = 0
            rec._htf2_betti1 = 0

        # Step 6 — online model update (Multi-Horizon Loss)
        if not hasattr(self, '_mh_buf'):
            self._mh_buf = []

        # Queue previous bar's features for future returns BEFORE adding current return
        # This ensures the features don't miss the immediate next bar's return.
        if self._last_feats is not None and self._bar_count > 2:
            self._mh_buf.append({"feats": self._last_feats, "returns": []})

        # Add current bar's log_return to pending features
        for item in self._mh_buf:
            item["returns"].append(log_ret)

        # When the oldest item has accumulated 3 forward returns, compute weighted target
        if self._mh_buf and len(self._mh_buf[0]["returns"]) >= 3:
            ready_item = self._mh_buf.pop(0)
            if not self._freeze_learning:
                rets = ready_item["returns"]
                
                # Push to AI Process for True Multi-Horizon Training
                if self.ai_actor == "LOCAL_MAMBA_FALLBACK":
                    import torch
                    feat_keys = sorted(ready_item["feats"].keys())
                    feats_val = [ready_item["feats"][k] for k in feat_keys]
                    feats_t = torch.tensor(feats_val, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    if feats_t.shape[-1] < 16:
                        pad = torch.zeros(1, 1, 1, 16 - feats_t.shape[-1])
                        feats_t = torch.cat([feats_t, pad], dim=-1)
                    elif feats_t.shape[-1] > 16:
                        feats_t = feats_t[:, :, :, :16]
                    adj_t = torch.eye(1).unsqueeze(0)
                    if not hasattr(self, "_fallback_optim"):
                        import torch.optim as optim
                        import torch.nn as nn
                        self._fallback_optim = optim.Adam(self._fallback_mamba.parameters(), lr=0.001)
                        self._fallback_loss = nn.MSELoss()

                    self._fallback_optim.zero_grad()
                    out = self._fallback_mamba(feats_t, adj_t)
                    weighted_target = 0.5 * rets[0] + 0.3 * rets[1] + 0.2 * rets[2]
                    target_t = torch.tensor([[weighted_target]], dtype=torch.float32)
                    loss = self._fallback_loss(out, target_t)
                    loss.backward()
                    self._fallback_optim.step()
                elif self.ai_actor:
                    # Ray actor trains online during inference via continuous learning
                    pass
                else:
                    # Fallback to River
                    weighted_target = 0.5 * rets[0] + 0.3 * rets[1] + 0.2 * rets[2]
                    self._model.learn_one(ready_item["feats"], weighted_target)

        # Build current feature vector
        current_feats = self._model._feature_vector(rec, self._record_hist)
        
        # Inject NLP Sentiment if available
        if self.nlp_agent:
            sym_clean = self._mtf_symbol if self._mtf_symbol else "EURUSD"
            current_feats["sentiment"] = self.nlp_agent.get_sentiment(sym_clean)
        else:
            current_feats["sentiment"] = 0.0
            
        # Ensure deterministic feature ordering for AI engine
        self._feature_keys = sorted(current_feats.keys())
        
        self._last_feats = current_feats
        self._record_hist.append(rec)
        
        # Causal Graph Discovery (Run every 100 bars)
        if HAS_CAUSAL and len(self._record_hist) >= 100 and self._bar_count % 100 == 0:
            # Build a simple mock historical matrix for the single symbol (we need multi-symbol for real causal discovery)
            # In a real environment, we'd gather X from all symbols here.
            X_hist = np.array([r.capacity for r in self._record_hist[-100:]]).reshape(-1, 1)
            try:
                self.causal_adj_matrix = learn_causal_graph(X_hist)
            except Exception as e:
                print(f"[CAUSAL ERROR] {e}")

        # Return forecast only after warm-up
        if self._bar_count < self.min_train_bars:
            return None

        # Step 7 — generate prediction
        if self.ai_actor == "LOCAL_MAMBA_FALLBACK":
            import torch
            # Use deterministic feature ordering
            feat_values = [current_feats.get(k, 0.0) for k in self._feature_keys]
            feats_t = torch.tensor(feat_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # Pad features to 16 since model expects input_dim=16
            if feats_t.shape[-1] < 16:
                pad = torch.zeros(1, 1, 1, 16 - feats_t.shape[-1])
                feats_t = torch.cat([feats_t, pad], dim=-1)
            elif feats_t.shape[-1] > 16:
                feats_t = feats_t[:, :, :, :16]
            
            adj_t = torch.eye(1).unsqueeze(0)
            with torch.no_grad():
                out = self._fallback_mamba(feats_t, adj_t)
                forecast_val = out[0, 0].item()
            
            pred = {
                "forecast": forecast_val,
                "direction": 1 if forecast_val > 0 else -1,
                "confidence": 0.6,
                "pred_interval_width": 0.001,
                "pred_interval_lower": forecast_val - 0.0005,
                "pred_interval_upper": forecast_val + 0.0005,
                "regime_used": "AI_ENGINE_V5_MAMBA"
            }
        elif self.ai_actor:
            try:
                # Wait for AI engine (PyTorch on Ray)
                adj_list = self.causal_adj_matrix.tolist() if self.causal_adj_matrix is not None else None
                # Use deterministic feature ordering for Ray actor
                feats_ordered = {k: current_feats.get(k, 0.0) for k in self._feature_keys}
                future_resp = self.ai_actor.process_features.remote(self._bar_count, feats_ordered, adj_list)
                resp = ray.get(future_resp, timeout=5.0)
                pred = {
                    "forecast": resp["forecast"],
                    "direction": resp["direction"],
                    "confidence": resp.get("confidence", 0.0),
                    "pred_interval_width": resp["pred_interval_width"],
                    "pred_interval_lower": resp["forecast"] - resp["pred_interval_width"]/2,
                    "pred_interval_upper": resp["forecast"] + resp["pred_interval_width"]/2,
                    "regime_used": "AI_ENGINE_V4"
                }
            except Exception as e:
                print(f"[RAY ERROR] {e}")
                # Fallback if AI crashes
                pred = self._model.predict_one(current_feats)
        else:
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
            features    = current_feats,
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

        self._mtf_symbol = symbol
        self._mtf_tf_str = tf_key

        print(f"[MT5] Fetching {n_bars} historical completed bars of {symbol} ({tf_key}) ...")
        # Fetch from position 1 to exclude the current forming bar (which is incomplete)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n_bars)

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
                     connection: MT5Connection = None,
                     dashboard_state = None) -> None:
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

        self._mtf_symbol = symbol
        self._mtf_tf_str = tf_key

        # Auto-adjust poll interval based on timeframe
        if poll_interval <= 0:
            poll_interval = _TF_POLL_SECONDS.get(tf_key, 60)

        print(f"\n{'=' * 66}")
        print(f"  LIVE MONITOR: {symbol} ({tf_key})")
        print(f"  Poll interval: {poll_interval:.0f}s")
        print(f"  Press Ctrl+C to stop.")
        print(f"{'=' * 66}")

        # Initialize last_bar_time to the last processed historical bar to prevent duplicate processing
        last_bar_time = int(self._prev_bar.timestamp) if hasattr(self, '_prev_bar') and self._prev_bar else 0
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
                    poll_count += 1
                    if poll_count % 60 == 0:  # Print every ~60 polls
                        print(f"  [{symbol}] Waiting for next bar to complete... (Market might be closed or slow)")
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
                    if dashboard_state:
                        pts = np.array(list(self._phase_buf), dtype=float)
                        hull_verts = get_convex_hull_vertices(pts)
                        acc_info = connection.get_account_info() if connection else {}
                        dashboard_state.update_live_metrics(
                            symbol=symbol,
                            timeframe=timeframe_str,
                            latest_forecast=out,
                            phase_buf=list(self._phase_buf),
                            hull_points=hull_verts.tolist(),
                            acc_info=acc_info,
                            total_updates=self._model._n_updates
                        )
                    else:
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
                        dt_str = datetime.fromtimestamp(bar_time).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        d_arrow = ("▲" if out.get("direction", 0) > 0
                                   else "▼" if out.get("direction", 0) < 0
                                   else "━")
                        print(
                            f"  [{dt_str}] {symbol} "
                            f"Close={bar.close:.5f} "
                            f"Ret={out.get('predicted_return', out.get('forecast', 0)):+.4%} "
                            f"{d_arrow} "
                            f"Conf={out.get('confidence', 0):.1%}"
                        )
                else:
                    # Still in warm-up
                    dt_str = datetime.fromtimestamp(bar_time).strftime(
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

    def export_state(self, symbol: str, timeframe: str, executor: "MT5TradeExecutor" = None) -> Dict[str, Any]:
        """Export full forecaster state (buffers + model + trade history) to a dict."""
        trade_records_data = []
        if executor and executor.trade_records:
            # Keep last 1000 trade records
            for tr in executor.trade_records[-1000:]:
                trade_records_data.append(tr.to_dict())
        
        return {
            "version": self.STATE_VERSION,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "saved_at": datetime.utcnow().isoformat() + "Z",
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
            "trade_records": trade_records_data,
        }

    def import_state(self, state: Dict[str, Any],
                     symbol: str = None, timeframe: str = None,
                     executor: "MT5TradeExecutor" = None) -> None:
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
        
        # Restore trade records to executor if provided
        if executor and "trade_records" in state:
            executor.trade_records = [TradeRecord.from_dict(d) for d in state["trade_records"]]
            print(f"[STATE] Restored {len(executor.trade_records)} trade records")

    def save_state(self, path: str, symbol: str, timeframe: str,
                   executor: "MT5TradeExecutor" = None) -> str:
        """Persist forecaster state to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_state(symbol, timeframe, executor=executor)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[STATE] Saved model state -> {path}")
        print(f"[STATE] Updates: {self._model._n_updates} | "
              f"Bars processed: {self._bar_count}")
        if executor:
            print(f"[STATE] Trade records saved: {len(executor.trade_records)}")
        return str(path)

    def load_state(self, path: str, symbol: str = None,
                   timeframe: str = None, executor: "MT5TradeExecutor" = None) -> Dict[str, Any]:
        """Load forecaster state from a pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.import_state(payload, symbol=symbol, timeframe=timeframe, executor=executor)
        print(f"[STATE] Loaded model state <- {path}")
        print(f"[STATE] Saved at: {payload.get('saved_at', 'unknown')}")
        print(f"[STATE] Updates: {self._model._n_updates} | "
              f"Bars processed: {self._bar_count}")
        if executor and "trade_records" in payload:
            print(f"[STATE] Trade records loaded: {len(payload['trade_records'])}")
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

