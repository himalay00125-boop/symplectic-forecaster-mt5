"""
models/forecaster.py
====================
Simplified, robust forecaster using standard ML techniques (River).
Replaces the over-engineered symplectic/topological pipeline.
"""

from typing import Dict, Any, Optional, List
import math
import numpy as np
import collections
import pickle
import os
import time
from pathlib import Path
import MetaTrader5 as mt5

from core.types import Bar
from core.config import TIMEFRAME_MAP
from models.base import OnlineModel

class SymplecticForecaster:
    """
    Simplified Forecaster using standard OHLCV features and River ML.
    """
    def __init__(self, **kwargs):
        self.min_train_bars = kwargs.get("min_train_bars", 80)
        self.nlp_agent = kwargs.get("shared_nlp_agent", None)
        self._model = OnlineModel()
        
        self._bar_count = 0
        self._bar_buf = collections.deque(maxlen=100)
        self._feat_buf = []
        self._last_feats = None

    def process_bar(self, bar: Bar, symbol: str = "EURUSD") -> Optional[Dict]:
        self._bar_count += 1
        self._bar_buf.append(bar)
        
        if len(self._bar_buf) < 20:
            return None
            
        # 1. Feature Engineering
        feats = self._extract_features(list(self._bar_buf), symbol)
        
        # 2. Learn from previous bar's prediction
        if self._last_feats is not None:
            # The target for previous bar's prediction is the return from prev close to current close
            prev_bar = self._bar_buf[-2]
            target_return = math.log(bar.close / prev_bar.close) if prev_bar.close > 0 else 0.0
            self._model.learn_one(self._last_feats, target_return)
            
        self._last_feats = feats
        
        # 3. Predict next bar
        if self._bar_count < self.min_train_bars:
            return None
            
        pred_return = self._model.predict_one(feats)
        
        # Return simplified forecast dict
        return {
            "predicted_return": pred_return,
            "forecast": pred_return,
            "confidence": 0.6 + min(0.3, abs(pred_return) * 100), # Mock confidence based on signal strength
            "features": feats,
            "alert": False,
            "lower_band": [bar.close * 0.995],
            "upper_band": [bar.close * 1.005],
            "pred_interval_width": 0.01,
        }
        
    def _extract_features(self, bars: List[Bar], symbol: str = "EURUSD") -> Dict[str, float]:
        feats = {}
        closes = np.array([b.close for b in bars])
        
        # Returns
        rets = np.diff(np.log(closes))
        feats["ret_lag1"] = rets[-1] if len(rets) >= 1 else 0.0
        feats["ret_lag2"] = rets[-2] if len(rets) >= 2 else 0.0
        feats["ret_lag3"] = rets[-3] if len(rets) >= 3 else 0.0
        
        # Volatility
        feats["vol_10"] = float(np.std(rets[-10:])) if len(rets) >= 10 else 0.0
        feats["vol_20"] = float(np.std(rets[-20:])) if len(rets) >= 20 else 0.0
        
        # Moving Average Distances
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        curr = closes[-1]
        feats["ma10_dist"] = (curr - ma10) / curr if curr > 0 else 0.0
        feats["ma20_dist"] = (curr - ma20) / curr if curr > 0 else 0.0
        
        # NLP Sentiment
        if self.nlp_agent is not None:
            feats["nlp_sentiment"] = self.nlp_agent.get_sentiment(symbol)
        else:
            feats["nlp_sentiment"] = 0.0
            
        return feats

    def train_on_mt5(self, symbol: str, timeframe: str, bars_count: int = 5000, connection=None):
        print(f"[{symbol}] Training historical ({bars_count} bars)...")
        tf_mt5 = TIMEFRAME_MAP.get(timeframe)
        if tf_mt5 is None: return
        
        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 1, bars_count)
        if rates is None or len(rates) == 0:
            print(f"[{symbol}] No historical data.")
            return
            
        for r in rates:
            bar = Bar(timestamp=float(r['time']), open=float(r['open']), high=float(r['high']),
                      low=float(r['low']), close=float(r['close']), volume=float(r['real_volume']))
            self.process_bar(bar)
            
        print(f"[{symbol}] Training complete. Updates: {self._model.get_metrics()['updates']}")

    def run_live_mt5(self, symbol: str, timeframe: str, dashboard_state=None,
                     on_signal=None, on_poll=None, poll_interval: float = 0.0, 
                     executor=None, connection=None):
        """Poll MT5 for new bars and generate forecasts."""
        print(f"[{symbol}] Live monitoring started...")
        tf_mt5 = TIMEFRAME_MAP.get(timeframe)
        if poll_interval <= 0:
            poll_interval = 5.0 # default 5 seconds
            
        last_bar_time = 0
        while True:
            try:
                rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 1, 1) # get most recent closed bar
                if rates is not None and len(rates) > 0:
                    r = rates[0]
                    bar_time = int(r['time'])
                    if bar_time > last_bar_time:
                        last_bar_time = bar_time
                        bar = Bar(timestamp=float(r['time']), open=float(r['open']), 
                                  high=float(r['high']), low=float(r['low']), 
                                  close=float(r['close']), volume=float(r['real_volume']))
                        forecast = self.process_bar(bar)
                        if forecast and on_signal:
                            on_signal(forecast, symbol)
                            
                if on_poll:
                    on_poll(symbol)
                    
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[{symbol}] Error in live loop: {e}")
                time.sleep(5)
        
    def forecast(self, horizon: int = 5) -> Dict:
        current_price = 0.0
        if len(self._bar_buf) > 0:
            current_price = self._bar_buf[-1].close
        return {
            "forecast": 0.0,
            "current_price": current_price,
            "regime": "NORMAL"
        }

    @staticmethod
    def default_state_path(symbol: str, timeframe: str, base_dir: str = "states") -> Path:
        Path(base_dir).mkdir(exist_ok=True)
        return Path(base_dir) / f"{symbol}_{timeframe}_simplified.pkl"

    def export_state(self, symbol: str, timeframe: str, executor=None) -> Dict[str, Any]:
        return {"bar_count": self._bar_count}

    def import_state(self, state: Dict[str, Any], symbol: str = "", timeframe: str = "", executor=None):
        pass

    def save_state(self, path: str, symbol: str, timeframe: str, executor=None):
        with open(path, "wb") as f:
            pickle.dump(self.export_state(symbol, timeframe, executor), f)

    def load_state(self, path: str, symbol: str = None, timeframe: str = None, executor=None):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    st = pickle.load(f)
                self.import_state(st, symbol, timeframe, executor)
                print(f"[STATE] Loaded -> {path}")
            except Exception as e:
                print(f"[STATE] Failed to load -> {e}")
