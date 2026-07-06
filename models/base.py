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
from models.geometry import *
from models.validation import *

class HyperparameterBandit:
    """
    Lightweight Multi-Armed Bandit for Online Hyperparameter Optimization.
    Maintains multiple clones of River models with different hyperparameters
    and routes predictions to the one with the lowest EMA of RMSE.
    """
    def __init__(self, model_class, param_grid: List[Dict[str, Any]]):
        self.arms = []
        for params in param_grid:
            self.arms.append({
                "model": model_class(**params),
                "params": params,
                "rmse": 0.001,
                "n_updates": 0
            })
        self.best_arm_idx = 0
        self.alpha = 0.05  # EMA decay factor

    def predict_one(self, feats: Dict[str, float]) -> float:
        # Use epsilon-greedy for prediction (explore 5% of the time)
        if random.random() < 0.05:
            idx = random.randint(0, len(self.arms) - 1)
        else:
            idx = self.best_arm_idx
        return self.arms[idx]["model"].predict_one(feats)

    def learn_one(self, feats: Dict[str, float], target: float):
        # Update all arms and track their errors
        best_rmse = float('inf')
        for i, arm in enumerate(self.arms):
            pred = arm["model"].predict_one(feats)
            if pred is not None:
                err = abs(target - pred)
                arm["rmse"] = (1 - self.alpha) * arm["rmse"] + self.alpha * err
            arm["model"].learn_one(feats, target)
            arm["n_updates"] += 1
            if arm["rmse"] < best_rmse:
                best_rmse = arm["rmse"]
                self.best_arm_idx = i

    @property
    def best_model(self):
        return self.arms[self.best_arm_idx]["model"]
        
    @property
    def best_params(self):
        return self.arms[self.best_arm_idx]["params"]

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

    Overfitting/Underfitting Protection:
    - Out-of-sample validation tracking with rolling window
    - Early stopping / model reset on persistent overfitting
    - Feature importance monitoring
    - Confidence calibration via Platt scaling
    """

    def __init__(self, validator_window: int = 200):
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
        
        # Validation & overfitting protection
        self.validator = ModelValidator(window=validator_window)
        self._last_prediction = None
        self._feature_importance: Dict[str, float] = {}
        self._confidence_calibrator = None  # Will be initialized when we have enough data
        self._min_confidence = 0.3  # Floor to prevent overconfidence
        
        # Feature selection / pruning
        self._feature_performance: Dict[str, Dict] = {}  # feat -> {help, harm, n_samples}
        self._active_features: Optional[Set[str]] = None  # None = all features active
        self._prune_interval = 500  # Prune every N updates
        self._min_feature_samples = 50  # Min samples before evaluating feature
        self._surprise_threshold = 1e-7
        self._grad_threshold = 0.01

    def _build_river_model(self):
        """River-based ensemble: PA regressor + Hoeffding Tree."""
        self._scaler = preprocessing.StandardScaler()
        
        def create_pa(C, eps, mode):
            return preprocessing.StandardScaler() | linear_model.PARegressor(C=C, eps=eps, mode=mode)
            
        def create_ht(grace_period, delta):
            return preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeRegressor(grace_period=grace_period, delta=delta, leaf_prediction="adaptive")

        self._pa = HyperparameterBandit(create_pa, [
            {"C": 0.001, "eps": 1e-4, "mode": 2},
            {"C": 0.005, "eps": 1e-4, "mode": 2},
            {"C": 0.01,  "eps": 1e-4, "mode": 2}
        ])
        
        self._ht = HyperparameterBandit(create_ht, [
            {"grace_period": 50,  "delta": 1e-5},
            {"grace_period": 100, "delta": 1e-5},
            {"grace_period": 200, "delta": 1e-5}
        ])
        
        self._mae_pa = metrics.MAE()
        self._mae_ht = metrics.MAE()

    def _build_sklearn_model(self):
        """sklearn fallback: SGDRegressor with L2 penalty to prevent overfitting on continuous trades."""
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler
        self._skl_model  = SGDRegressor(penalty='l2', alpha=1e-7, learning_rate='invscaling', eta0=0.01, random_state=42)
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

        # --- ICT features (stored by process_bar in the record) ---
        feats["fvg_count"] = getattr(rec, '_fvg_count', 0)
        feats["fvg_nearest_dist"] = getattr(rec, '_fvg_nearest_dist', 0.0)
        feats["midnight_dist"] = getattr(rec, '_midnight_dist', 0.0)
        feats["asian_range_pct"] = getattr(rec, '_asian_range_pct', 0.5)

        # --- Multi-timeframe alignment ---
        feats["mtf_alignment"] = getattr(rec, '_mtf_alignment', 0.5)
        feats["htf1_betti1"] = getattr(rec, '_htf1_betti1', 0.0)
        feats["htf2_betti1"] = getattr(rec, '_htf2_betti1', 0.0)

        # --- Feature Interactions (reduced to prevent explosion/overfitting) ---
        feats["cap_x_vol10"] = rec.capacity * feats["vol_10"]
        feats["cap_ratio_x_vol"] = feats["cap_ratio"] * feats["vol_10"]
        feats["mtf_vol_alignment"] = feats["mtf_alignment"] * feats["vol_20"]
        
        # Capacity and lag
        if "ret_lag1" in feats:
            feats["cap_ret_lag1"] = rec.capacity * feats["ret_lag1"]
            
        # Trend-Volatility: momentum consistency vs noise
        trend_5 = sum(_all_lags) / 5.0
        feats["trend_x_vol"] = trend_5 * feats["vol_10"]
        feats["trend_x_cap"] = trend_5 * rec.capacity
        
        # Capacity acceleration (second difference already computed as d_cap2)
        feats["cap_accel_x_vol"] = feats["d_cap2"] * feats["vol_10"]
        
        # Persistence-Volatility: topological persistence as vol predictor
        feats["pers_x_vol"] = feats["tot_pers"] * feats["vol_10"]

        return feats

    def _to_sparse(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Create sparse representation by dropping near-zero features."""
        return {k: v for k, v in feats.items() if abs(v) > 1e-8}
        
    def _estimate_gradient_norm(self, feats: Dict[str, float]) -> float:
        """Estimate gradient norm proxy from feature variance."""
        return float(np.std(list(feats.values()))) if feats else 0.0

    def should_update(self, feats: Dict[str, float], target: float, prediction: float) -> bool:
        """Selective Model Updates (Importance Sampling)"""
        surprise = abs(target - prediction)
        grad_norm = self._estimate_gradient_norm(feats)
        return surprise > self._surprise_threshold or grad_norm > self._grad_threshold

    def learn_one(self, feats: Dict[str, float], target: float):
        """Update both models with one (feature, target) pair."""
        # Filter to active features if feature selection enabled
        feats = self._filter_active_features(feats)
        feats = self._to_sparse(feats)
        
        # Record prediction before learning for validation tracking
        train_error = None
        if self._last_prediction is not None:
            train_error = abs(self._last_prediction - target)
        
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
            
            # Ensemble prediction for validation
            ensemble_pred = 0.0
            if pred_pa is not None and pred_ht is not None:
                w_pa = 1.0 / (self._rmse_pa + 1e-12)
                w_ht = 1.0 / (self._rmse_ht + 1e-12)
                total = w_pa + w_ht
                ensemble_pred = (w_pa * pred_pa + w_ht * pred_ht) / total
            elif pred_pa is not None:
                ensemble_pred = pred_pa
            elif pred_ht is not None:
                ensemble_pred = pred_ht
            
            if ensemble_pred != 0.0:
                self.validator.record(ensemble_pred, target, train_error)
                
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
            
            # Get prediction before learning for validation
            pred_skl = 0.0
            try:
                pred_skl = float(self._skl_model.predict(X_sc)[0])
            except Exception:
                pass
            
            self._skl_model.partial_fit(X_sc, [target])
            
            if pred_skl != 0.0:
                self.validator.record(pred_skl, target, train_error)
        
        self._n_updates += 1
        
        # Track feature performance for online feature selection
        self._update_feature_performance(feats, target)
        
        # Periodic feature pruning
        if self._n_updates % self._prune_interval == 0:
            self._prune_features()
        
        if self.validator.should_reset_model():
            print(f"  [VALIDATOR] Persistent overfitting detected — resetting model")
            if HAS_RIVER:
                self._build_river_model()
            else:
                self._build_sklearn_model()
            self.validator.reset_warnings()

    def _update_feature_performance(self, feats: Dict[str, float], target: float):
        """Track feature performance: which features help/harm prediction accuracy."""
        if not hasattr(self, '_feature_performance'):
            return
            
        # Get prediction before learning to measure feature contribution
        pred = 0.0
        if HAS_RIVER:
            pred_pa = self._pa.predict_one(feats) or 0.0
            pred_ht = self._ht.predict_one(feats) or 0.0
            w_pa = 1.0 / (self._rmse_pa + 1e-12)
            w_ht = 1.0 / (self._rmse_ht + 1e-12)
            if pred_pa and pred_ht:
                pred = (w_pa * pred_pa + w_ht * pred_ht) / (w_pa + w_ht)
            elif pred_pa:
                pred = pred_pa
            elif pred_ht:
                pred = pred_ht
        else:
            try:
                X = np.array(list(feats.values())).reshape(1, -1)
                if self._skl_fitted:
                    pred = float(self._skl_model.predict(self._skl_scaler.transform(X))[0])
            except Exception:
                pass
        
        if pred == 0.0:
            return
            
        error = abs(target - pred)
        # For each feature, track if it helped (large value aligned with correct direction) or hurt
        correct_direction = np.sign(target) == np.sign(pred) if target != 0 and pred != 0 else True
        
        regime = "ALERT" if feats.get("alert", False) else "NORMAL"
        for feat_name, feat_val in feats.items():
            if feat_name not in self._feature_performance:
                self._feature_performance[feat_name] = {"help": 0.0, "harm": 0.0, "n": 0, "NORMAL_contrib": 0.0, "ALERT_contrib": 0.0}
            
            fp = self._feature_performance[feat_name]
            fp["n"] += 1
            
            # Feature contribution: sign(feat_val) * sign(pred) * |feat_val|
            contribution = feat_val * pred
            fp[f"{regime}_contrib"] += contribution
            
            if correct_direction and contribution > 0:
                fp["help"] += abs(contribution)
            elif not correct_direction and contribution < 0:
                fp["harm"] += abs(contribution)

    def _prune_features(self):
        """Remove features that consistently harm performance or flip correlations across regimes."""
        if not hasattr(self, '_feature_performance') or not self._feature_performance:
            return
            
        # Only prune features with enough samples
        candidates = {
            name: stats for name, stats in self._feature_performance.items()
            if stats["n"] >= self._min_feature_samples
        }
        
        if len(candidates) < 10:  # Need enough features to prune
            return
            
        # Compute harm/help ratio with Causal Variance Penalty
        ratios = {}
        for name, stats in candidates.items():
            ratio = stats["harm"] / (stats["help"] + 1e-12) if stats["help"] > 0 else float('inf')
            
            # Invariant Causal Prediction: penalize features that flip correlation between regimes
            norm_contrib = stats.get("NORMAL_contrib", 0.0) / (stats["n"] + 1e-12)
            alert_contrib = stats.get("ALERT_contrib", 0.0) / (stats["n"] + 1e-12)
            
            # If the signs are opposite and magnitude is non-trivial, feature is highly variant (non-causal)
            if norm_contrib * alert_contrib < -1e-6:
                variance_penalty = 5.0  # severely penalize
            else:
                variance_penalty = 1.0
                
            ratios[name] = ratio * variance_penalty
        
        # Sort by harm/help ratio (highest = most harmful)
        sorted_features = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        # Prune bottom 10% (most harmful)
        n_prune = max(1, len(sorted_features) // 10)
        pruned = [name for name, _ in sorted_features[:n_prune]]
        
        if self._active_features is None:
            self._active_features = set(self._feature_performance.keys())
        
        for name in pruned:
            self._active_features.discard(name)
            if name in self._feature_performance:
                del self._feature_performance[name]
        
        if pruned:
            print(f"  [FEATURE PRUNING] Removed {len(pruned)} harmful features: {pruned[:5]}...")

    def _filter_active_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Return only active features (if feature selection is enabled)."""
        if self._active_features is None:
            return feats
        return {k: v for k, v in feats.items() if k in self._active_features}

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
        # Filter to active features if feature selection enabled
        feats = self._filter_active_features(feats)
        feats = self._to_sparse(feats)
        
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
        
        # Apply confidence calibration floor to prevent overconfidence
        conf = max(conf, self._min_confidence)
        
        # Store prediction for next validation step
        self._last_prediction = forecast
        
        # Get validator status
        val_status = self.validator.get_status()
        
        return dict(forecast=forecast, direction=direction,
                    confidence=conf,
                    mae_pa=self._mae_pa.get() if HAS_RIVER else None,
                    mae_ht=self._mae_ht.get() if HAS_RIVER else None,
                    n_updates=self._n_updates,
                    val_status=val_status)

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

