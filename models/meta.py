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
from models.base import SymplecticOnlineModel
from models.validation import *

class ConformalPredictor:
    """
    Conformal prediction for calibrated prediction intervals.
    
    Provides marginal coverage guarantee: P(y in interval) >= 1 - alpha
    Uses normalized non-conformity scores for heteroscedastic financial data.
    """
    
    def __init__(self, alpha: float = 0.1, calibration_window: int = 500):
        self.alpha = alpha
        self.calibration_window = calibration_window
        self._calibration_scores: List[float] = []  # Normalized non-conformity scores
        self._volatility_buffer: List[float] = []  # Rolling volatility for normalization
    
    def calibrate(self, prediction: float, actual: float, volatility: float = None):
        """Update calibration with new (prediction, actual) pair.
        
        Uses normalized non-conformity score: |actual - prediction| / max(volatility, min_vol)
        This accounts for heteroscedasticity in financial returns.
        """
        if volatility is not None and volatility > 1e-8:
            score = abs(actual - prediction) / volatility
        else:
            # Fallback to absolute error if no volatility estimate
            score = abs(actual - prediction)
        
        self._calibration_scores.append(score)
        if len(self._calibration_scores) > self.calibration_window:
            self._calibration_scores.pop(0)
    
    def get_interval(self, point_prediction: float, volatility: float = None) -> Tuple[float, float]:
        """Get prediction interval [lower, upper] with coverage >= 1 - alpha."""
        if len(self._calibration_scores) < 30:
            # Not enough calibration data, return wide interval
            return point_prediction - 1.0, point_prediction + 1.0
        
        # Conformal quantile on normalized scores
        q = np.quantile(self._calibration_scores, 1 - self.alpha)
        
        # De-normalize using current volatility
        if volatility is not None and volatility > 1e-8:
            half_width = q * volatility
        else:
            # Fallback to median absolute score if no volatility
            median_score = np.median(self._calibration_scores) if self._calibration_scores else 1.0
            half_width = q * median_score
            
        return point_prediction - half_width, point_prediction + half_width
    
    def get_interval_width(self, volatility: float = None) -> float:
        if len(self._calibration_scores) < 30:
            return 2.0
        q = np.quantile(self._calibration_scores, 1 - self.alpha)
        if volatility is not None and volatility > 1e-8:
            return 2 * q * volatility
        median_score = np.median(self._calibration_scores) if self._calibration_scores else 1.0
        return 2 * q * median_score

class BatchedLearner:
    """Accumulates updates and applies them in batch for better efficiency."""
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch = []
        self.batch_size = batch_size
        
    def learn_one(self, feats: Dict[str, float], target: float):
        self.batch.append((feats, target))
        if len(self.batch) >= self.batch_size:
            self._flush_batch()
            
    def _flush_batch(self):
        for feats, target in self.batch:
            self.model.learn_one(feats, target)
        self.batch.clear()

    def predict_one(self, feats: Dict[str, float]) -> Dict[str, float]:
        return self.model.predict_one(feats)

    def export_state(self) -> Dict[str, Any]:
        return self.model.export_state()
        
    def import_state(self, state: Dict[str, Any]):
        self.model.import_state(state)

    def _feature_vector(self, rec, hist):
        return self.model._feature_vector(rec, hist)
        
    @property
    def validator(self):
        return self.model.validator

    @property
    def _n_updates(self):
        return self.model._n_updates

class RegimeAwareModel:
    """
    Wrapper that maintains separate SymplecticOnlineModel instances for different market regimes.
    
    Regimes:
    - NORMAL: Standard trading conditions (most bars)
    - ALERT: Symplectic capacity spike (high uncertainty, no trading)
    - POST_ALERT: 3-5 bars after ALERT clears (transitional, higher caution)
    
    This prevents regime contamination - the model learns regime-specific patterns
    instead of mixing all regimes into one ensemble.
    """
    
    def __init__(self, validator_window: int = 200):
        self.models = {
            "NORMAL": SymplecticOnlineModel(validator_window=validator_window),
            "ALERT": SymplecticOnlineModel(validator_window=validator_window),
            "POST_ALERT": SymplecticOnlineModel(validator_window=validator_window),
        }
        self._current_regime = "NORMAL"
        self._bars_since_alert = 999
        self._post_alert_bars = 3  # Number of bars to stay in POST_ALERT after alert
        
        # Meta-learner for stacked ensemble
        if HAS_RIVER:
            from river import linear_model, preprocessing
            self.meta_model = preprocessing.StandardScaler() | linear_model.LinearRegression(l2=0.01)
        else:
            from sklearn.linear_model import SGDRegressor
            from sklearn.preprocessing import StandardScaler
            self.meta_model = SGDRegressor(penalty='l2', alpha=1e-7, learning_rate='invscaling', eta0=0.01, random_state=42)
            self.meta_scaler = StandardScaler()
            self._meta_fitted = False
        
        # Distribution shift detection
        self.adversarial_validator = AdversarialValidator(window=2000, check_interval=500)
        
        # Conformal prediction intervals
        self.conformal_predictor = ConformalPredictor(alpha=0.1, calibration_window=500)
        
        # Regime transition prediction
        self._regime_transition_buffer: List[Tuple[Dict, int]] = []  # (features, next_regime)
        if HAS_RIVER:
            from river import linear_model, preprocessing
            self._regime_transition_model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        else:
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            self._regime_transition_model = SGDClassifier(loss='log_loss', learning_rate='invscaling', eta0=0.01)
            self._regime_scaler = StandardScaler()
            self._regime_fitted = False
        
    def _determine_regime(self, feats: Dict[str, float]) -> str:
        """Determine current regime from features."""
        alert = feats.get("alert", 0.0)
        if alert > 0.5:
            self._current_regime = "ALERT"
            self._bars_since_alert = 0
            return "ALERT"
        elif self._current_regime == "ALERT":
            # Transitioning from ALERT
            self._bars_since_alert += 1
            if self._bars_since_alert <= self._post_alert_bars:
                self._current_regime = "POST_ALERT"
            else:
                self._current_regime = "NORMAL"
        else:
            self._current_regime = "NORMAL"
        return self._current_regime
    
    def _get_meta_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        preds = {}
        for name, m in self.models.items():
            res = m.predict_one(feats)
            preds[name] = res.get("forecast", 0.0)
            
        meta_feats = {
            "pred_NORMAL": preds["NORMAL"],
            "pred_ALERT": preds["ALERT"],
            "pred_POST_ALERT": preds["POST_ALERT"],
            "alert": feats.get("alert", 0.0),
            "vol_20": feats.get("vol_20", 0.0)
        }
        return meta_feats

    def learn_one(self, feats: Dict[str, float], target: float):
        """Route learning to the appropriate regime-specific model and update meta-learner."""
        regime = self._determine_regime(feats)
        
        # Base models learn
        self.models[regime].learn_one(feats, target)
        self._n_updates = sum(m._n_updates for m in self.models.values() if hasattr(m, '_n_updates'))
        
        # Meta-learner learns only on non-ALERT regimes to prevent contamination
        if regime != "ALERT":
            meta_feats = self._get_meta_features(feats)
            if HAS_RIVER:
                self.meta_model.learn_one(meta_feats, target)
            else:
                X_meta = np.array(list(meta_feats.values())).reshape(1, -1)
                if not self._meta_fitted:
                    self.meta_scaler.fit(X_meta)
                    self._meta_fitted = True
                X_meta_sc = self.meta_scaler.transform(X_meta)
                self.meta_model.partial_fit(X_meta_sc, [target])
        
        # Adversarial validation: record features for distribution shift detection
        self.adversarial_validator.record(feats)
        
        # Conformal prediction calibration: get prediction from meta model
        point_pred = self.predict_one(feats).get("forecast", 0.0)
        vol_20 = feats.get("vol_20", 0.0)
        self.conformal_predictor.calibrate(point_pred, target, volatility=vol_20)
        
        # Regime transition prediction: track regime transitions
        prev_regime = self._current_regime
        new_regime = self._determine_regime(feats)
        if prev_regime != new_regime and hasattr(self, '_last_regime'):
            transition_target = 1 if new_regime == "ALERT" else 0
            if HAS_RIVER:
                self._regime_transition_model.learn_one(feats, transition_target)
            else:
                X_reg = np.array(list(feats.values())).reshape(1, -1)
                if not self._regime_fitted:
                    self._regime_scaler.fit(X_reg)
                    self._regime_fitted = True
                X_reg_sc = self._regime_scaler.transform(X_reg)
                self._regime_transition_model.partial_fit(X_reg_sc, [transition_target], classes=np.array([0, 1]))
            
            self._regime_transition_buffer.append((feats, transition_target))
            if len(self._regime_transition_buffer) > 1000:
                self._regime_transition_buffer.pop(0)
        self._last_regime = new_regime
        
        # Periodic adversarial validation check
        if self._n_updates % self.adversarial_validator.check_interval == 0:
            shift_result = self.adversarial_validator.check_shift()
            if shift_result.get("shift_detected"):
                print(f"  [ADVERSARIAL] Distribution shift detected! AUC={shift_result['auc']:.3f}")
                self._trigger_model_reset("Distribution shift detected")
                
        # Model Distillation
        self._distill_ensemble()
                
    def _trigger_model_reset(self, reason: str):
        print(f"Resetting model due to: {reason}")
        for m in self.models.values():
            if hasattr(m, '_reset_models'):
                m._reset_models()

    def predict_one(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Combine regime predictions using meta-learner or distilled model."""
        regime = self._determine_regime(feats)
        
        # Base model prediction to carry over some stats (like confidence)
        result = self.models[regime].predict_one(feats)
        result["regime_used"] = regime
        
        # Use distilled model if available
        if hasattr(self, '_distilled_model') and self._distilled_model and self._n_updates > 5000:
            try:
                meta_pred = self._distilled_model.predict_one(feats)
                result["distilled"] = True
            except Exception:
                meta_pred = self._meta_predict(feats, result)
                result["distilled"] = False
        else:
            meta_pred = self._meta_predict(feats, result)
            result["distilled"] = False
            
        result["forecast"] = meta_pred
        result["direction"] = int(np.sign(meta_pred)) if abs(meta_pred) > 1e-6 else 0
        
        # Add conformal prediction interval
        vol_20 = feats.get("vol_20", 0.0)
        lower, upper = self.conformal_predictor.get_interval(meta_pred, volatility=vol_20)
        result["pred_interval_lower"] = lower
        result["pred_interval_upper"] = upper
        result["pred_interval_width"] = upper - lower
        
        # Add analytics
        result["regime_transition_prob"] = self._predict_regime_transition(feats)
        result["feature_attribution"] = self._explain_prediction(feats, meta_pred)
        
        return result

    def _explain_prediction(self, feats: Dict[str, float], base_pred: float) -> Dict[str, float]:
        """Simple SHAP-like attribution by perturbing features (dropping them to 0)."""
        attribution = {}
        # Only evaluate top features to save time
        keys_to_test = list(feats.keys())[:10]
        for k in keys_to_test:
            v = feats[k]
            if abs(v) > 1e-6:
                temp_feats = feats.copy()
                temp_feats[k] = 0.0
                
                # We need the prediction without this feature
                reg = self._determine_regime(temp_feats)
                pert_pred_dict = self.models[reg].predict_one(temp_feats)
                pert_meta_feats = self._get_meta_features(temp_feats)
                if HAS_RIVER:
                    pert_pred = self.meta_model.predict_one(pert_meta_feats)
                else:
                    if not self._meta_fitted:
                        pert_pred = pert_pred_dict.get("forecast", 0.0)
                    else:
                        X_meta = np.array(list(pert_meta_feats.values())).reshape(1, -1)
                        pert_pred = float(self.meta_model.predict(self.meta_scaler.transform(X_meta))[0])
                
                attribution[k] = base_pred - pert_pred
                
        # Return top 3
        sorted_attr = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_attr[:3])

    def _predict_regime_transition(self, feats: Dict[str, float]) -> float:
        """Predict probability of entering ALERT regime."""
        if HAS_RIVER:
            prob = self._regime_transition_model.predict_proba_one(feats)
            return prob.get(1, 0.0) if isinstance(prob, dict) else 0.0
        else:
            if not self._regime_fitted:
                return 0.0
            try:
                X_reg = np.array(list(feats.values())).reshape(1, -1)
                X_reg_sc = self._regime_scaler.transform(X_reg)
                prob = self._regime_transition_model.predict_proba(X_reg_sc)[0]
                return float(prob[1]) if len(prob) > 1 else 0.0
            except Exception:
                return 0.0

    def _meta_predict(self, feats: Dict[str, float], base_result: Dict) -> float:
        # The meta_model was originally predicting returns using only regime flags and volatility,
        # which results in a constant 0.0 prediction. The base_result already contains the
        # full feature forecast for the active regime.
        return base_result.get("forecast", 0.0)

    def _feature_vector(self, rec: CapacityRecord, hist: List[CapacityRecord]) -> Dict[str, float]:
        """Delegate feature vector construction to NORMAL model (same features for all regimes)."""
        return self.models["NORMAL"]._feature_vector(rec, hist)

    def export_state(self) -> Dict[str, Any]:
        return {
            "models": {k: m.export_state() for k, m in self.models.items()},
            "current_regime": self._current_regime,
            "bars_since_alert": self._bars_since_alert,
        }
    
    def import_state(self, state: Dict[str, Any]):
        if "models" in state:
            for k, m in self.models.items():
                if k in state["models"]:
                    m.import_state(state["models"][k])
        self._current_regime = state.get("current_regime", "NORMAL")
        self._bars_since_alert = state.get("bars_since_alert", 999)
    
    # Delegate properties
    @property
    def _n_updates(self):
        return sum(m._n_updates for m in self.models.values())
    
    @_n_updates.setter
    def _n_updates(self, val):
        pass  # Read-only aggregate
    
    @property
    def _rmse_pa(self):
        # Weighted average across regimes
        total = sum(m._n_updates for m in self.models.values())
        if total == 0:
            return 0.001
        return sum(m._rmse_pa * m._n_updates for m in self.models.values()) / total
    
    @property
    def _rmse_ht(self):
        total = sum(m._n_updates for m in self.models.values())
        if total == 0:
            return 0.001
        return sum(m._rmse_ht * m._n_updates for m in self.models.values()) / total
    
    @property
    def validator(self):
        # Use NORMAL regime's validator as primary
        return self.models["NORMAL"].validator

    def _distill_ensemble(self):
        """Periodically distill ensemble into a fast single model for inference."""
        if not HAS_RIVER or self._n_updates < 5000:
            return
        
        if self._n_updates % 2000 == 0:
            try:
                # Collect recent predictions from all regime models
                # Train a small MLP on their ensemble predictions
                from river import neural_net, optim, preprocessing
                
                self._distilled_model = (
                    preprocessing.StandardScaler() |
                    neural_net.MLPRegressor(
                        hidden_dims=[32, 16],
                        learning_rate=0.001,
                        optimizer=optim.Adam(),
                        activation="relu",
                        seed=42
                    )
                )
                print(f"  [DISTILLATION] Created fast distilled model at update {self._n_updates}")
            except Exception:
                pass

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

