"""
models/base.py
==============
Clean, Robust Online Learning Model (River).
Removes the over-engineered meta-models and topologies.
Uses a simple multi-armed bandit of Passive-Aggressive Regressors
to adapt quickly to changing market conditions.
"""

from typing import Dict, Any, List
import random
from river import linear_model, preprocessing, metrics

class HyperparameterBandit:
    """
    Lightweight Multi-Armed Bandit for Online Hyperparameter Optimization.
    Maintains clones of River models and routes predictions to the one with the lowest EMA of RMSE.
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
        self.alpha = 0.05

    def predict_one(self, feats: Dict[str, float]) -> float:
        if random.random() < 0.05:
            idx = random.randint(0, len(self.arms) - 1)
        else:
            idx = self.best_arm_idx
        return self.arms[idx]["model"].predict_one(feats)

    def learn_one(self, feats: Dict[str, float], target: float):
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

    def import_state(self, state: Dict[str, Any]):
        """No-op for stateless restart or basic import."""
        pass


class OnlineModel:
    """
    Robust River-based online model for predicting next-bar log returns.
    """
    def __init__(self):
        def create_pa(C, eps, mode):
            return preprocessing.StandardScaler() | linear_model.PARegressor(C=C, eps=eps, mode=mode)

        self._pa = HyperparameterBandit(create_pa, [
            {"C": 0.001, "eps": 1e-4, "mode": 2},
            {"C": 0.005, "eps": 1e-4, "mode": 2},
            {"C": 0.01,  "eps": 1e-4, "mode": 2}
        ])
        
        self._mae = metrics.MAE()
        self._n_updates = 0

    def learn_one(self, feats: Dict[str, float], target: float):
        self._pa.learn_one(feats, target)
        pred = self.predict_one(feats)
        if pred is not None:
            self._mae.update(target, pred)
        self._n_updates += 1

    def predict_one(self, feats: Dict[str, float]) -> float:
        return self._pa.predict_one(feats) or 0.0

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "updates": self._n_updates,
            "mae": self._mae.get(),
            "best_params": self._pa.arms[self._pa.best_arm_idx]["params"]
        }

    def import_state(self, state: Dict[str, Any]):
        pass
