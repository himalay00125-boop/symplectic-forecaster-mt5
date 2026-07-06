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

class ModelValidator:
    """Out-of-sample validation monitor to detect overfitting/underfitting."""
    
    def __init__(self, window: int = 100):
        self.window = window
        self._predictions: List[Tuple[float, float]] = []  # (pred, actual)
        self._val_errors: List[float] = []
        self._train_errors: List[float] = []
        self._overfit_warnings = 0
        self._underfit_warnings = 0
        self.best_val_rmse = float('inf')
        self.patience = 50
        self.patience_counter = 0
    
    def record(self, prediction: float, actual: float, train_error: float = None):
        """Record a prediction and its actual outcome for validation tracking."""
        self._predictions.append((prediction, actual))
        if len(self._predictions) > self.window:
            self._predictions.pop(0)
        
        # Compute validation error (out-of-sample)
        if len(self._predictions) >= 10:
            recent = self._predictions[-self.window:]
            val_rmse = math.sqrt(sum((p - a)**2 for p, a in recent) / len(recent))
            self._val_errors.append(val_rmse)
            if len(self._val_errors) > 200:
                self._val_errors.pop(0)
            
            self._check_overfitting(train_error)
            self._check_underfitting()
    
    def _check_overfitting(self, train_error: float = None):
        """Detect overfitting: validation error rising while training error falls."""
        # Append train_error first (fixes first-value-discard bug)
        if train_error is not None:
            self._train_errors.append(train_error)
            if len(self._train_errors) > 200:
                self._train_errors.pop(0)
        
        # Overfitting signal: validation RMSE increasing while training RMSE decreasing
        if len(self._val_errors) >= 20 and len(self._train_errors) >= 20:
            recent_val = np.mean(self._val_errors[-10:])
            recent_train = np.mean(self._train_errors[-10:])
            older_val = np.mean(self._val_errors[-20:-10])
            older_train = np.mean(self._train_errors[-20:-10])
            
            if recent_val > older_val * 1.05 and recent_train < older_train * 0.95:
                self._overfit_warnings += 1
                if self._overfit_warnings >= 3:
                    print(f"  [VALIDATOR] Overfitting detected (val RMSE rising, train RMSE falling)")
        
        return None
    
    def _check_underfitting(self):
        """Detect underfitting: both validation and training errors high."""
        if len(self._val_errors) >= 20:
            recent_val = np.mean(self._val_errors[-10:])
            # Adaptive threshold: use 2x the average recent validation error baseline
            # or minimum 0.08 for very quiet markets
            baseline = np.mean(self._val_errors[:10]) if len(self._val_errors) >= 10 else 0.04
            threshold = max(0.08, baseline * 2.0)
            if recent_val > threshold:
                self._underfit_warnings += 1
                if self._underfit_warnings >= 5:
                    print(f"  [VALIDATOR] Underfitting detected (val RMSE={recent_val:.4f} > threshold={threshold:.4f})")
        return None
    
    def get_status(self) -> Dict:
        """Return current validation status."""
        status = "HEALTHY"
        if self._overfit_warnings >= 3:
            status = "OVERFITTING"
        elif self._underfit_warnings >= 5:
            status = "UNDERFITTING"
        
        current_val = self._val_errors[-1] if self._val_errors else None
        current_train = self._train_errors[-1] if self._train_errors else None
        
        return {
            "status": status,
            "val_rmse": round(current_val, 6) if current_val else None,
            "train_rmse": round(current_train, 6) if current_train else None,
            "overfit_warnings": self._overfit_warnings,
            "underfit_warnings": self._underfit_warnings,
            "n_samples": len(self._predictions)
        }
    
    def should_reset_model(self) -> bool:
        """Return True if model should be reset due to persistent overfitting."""
        return self._overfit_warnings >= 10
    
    def reset_warnings(self):
        self._overfit_warnings = 0
        self._underfit_warnings = 0

class AdversarialValidator:
    """
    Detects distribution shift between training and recent data.
    
    Trains a binary classifier to distinguish old vs recent samples.
    """
    
    def __init__(self, window: int = 2000, check_interval: int = 2000):
        self.window = window
        self.check_interval = check_interval
        self._feature_buffer: List[Dict] = []
        self._shift_detected = False
        self._shift_auc = 0.0
        self._n_checks = 0
    
    def record(self, feats: Dict[str, float]):
        """Record feature vector for shift detection."""
        self._feature_buffer.append(dict(feats))
        if len(self._feature_buffer) > self.window:
            self._feature_buffer.pop(0)
    
    def check_shift(self) -> Dict:
        """Run adversarial validation check."""
        self._n_checks += 1
        if len(self._feature_buffer) < self.window // 2:
            return {"shift_detected": False, "auc": 0.0, "reason": "insufficient_data"}
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score
            
            # Split into old (40%), gap (20%), and recent (40%) to prevent temporal data leakage
            n = len(self._feature_buffer)
            idx1 = int(n * 0.4)
            idx2 = int(n * 0.6)
            old_feats = self._feature_buffer[:idx1]
            recent_feats = self._feature_buffer[idx2:]
            
            # Convert to arrays
            all_keys = set()
            for f in old_feats + recent_feats:
                all_keys.update(f.keys())
            all_keys = sorted(all_keys)
            
            X_old = np.array([[f.get(k, 0.0) for k in all_keys] for f in old_feats])
            X_recent = np.array([[f.get(k, 0.0) for k in all_keys] for f in recent_feats])
            
            # Remove constant columns
            std_old = X_old.std(axis=0)
            std_recent = X_recent.std(axis=0)
            valid_cols = (std_old > 1e-8) | (std_recent > 1e-8)
            
            if not valid_cols.any():
                return {"shift_detected": False, "auc": 0.0, "reason": "no_variance"}
            
            X_old = X_old[:, valid_cols]
            X_recent = X_recent[:, valid_cols]
            
            # Standardize
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_combined = np.vstack([X_old, X_recent])
            X_scaled = scaler.fit_transform(X_combined)
            
            X_old = X_scaled[:len(X_old)]
            X_recent = X_scaled[len(X_old):]
            
            # Labels: 0 = old, 1 = recent
            y = np.hstack([np.zeros(len(X_old)), np.ones(len(X_recent))])
            X = np.vstack([X_old, X_recent])
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train classifier
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X_train, y_train)
            
            # AUC on test set
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            
            self._shift_auc = auc
            shift_detected = auc > 0.75
            self._shift_detected = shift_detected
            
            return {
                "shift_detected": shift_detected,
                "auc": round(auc, 4),
                "n_old": len(X_old),
                "n_recent": len(X_recent),
                "n_features": X.shape[1]
            }
        except Exception as e:
            return {"shift_detected": False, "auc": 0.0, "error": str(e)}
    
    def should_reset(self) -> bool:
        return self._shift_detected

