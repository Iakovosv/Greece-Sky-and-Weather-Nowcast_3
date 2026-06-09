from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class OnlineState:
    pop: Dict[int, Tuple[np.ndarray, float]]
    q: Dict[int, Dict[float, Tuple[np.ndarray, float]]]
    trained_samples: int = 0

def sgd_logistic_step(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, lr: float, l2: float) -> Tuple[np.ndarray, float]:
    n = max(1, len(y))
    z = X @ w + b
    p = sigmoid(z)
    gw = (X.T @ (p - y)) / n + (l2 / n) * w
    gb = float(np.mean(p - y))
    return w - lr * gw, b - lr * gb

def sgd_quantile_step(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, q: float, lr: float, l2: float) -> Tuple[np.ndarray, float]:
    n = max(1, len(y))
    pred = X @ w + b
    e = y - pred
    grad = np.where(e >= 0, -q, 1 - q)
    gw = (X.T @ grad) / n + (l2 / n) * w
    gb = float(np.mean(grad))
    return w - lr * gw, b - lr * gb

def apply_training(state: OnlineState, batch: Dict[int, Dict[str, Any]], lr_pop: float=0.02, lr_q: float=0.02, l2: float=5.0, pos_weight: float=10.0) -> Dict[str, Any]:
    """batch[h] = {'X': [x...], 'amt': [mm...], 'pop': [0/1...]} with x already normalized"""
    info = {"updated": 0, "by_horizon": {}}
    for h, data in batch.items():
        X = np.asarray(data["X"], dtype=float)
        if X.size == 0:
            continue
        y_pop = np.asarray(data["pop"], dtype=float)
        y_amt = np.asarray(data["amt"], dtype=float)
        
        # Compute sample weights for class imbalance
        sample_weights = np.ones_like(y_pop, dtype=float)
        sample_weights[y_pop > 0.5] = pos_weight
        
        # Weighted SGD for logistic
        w, b = state.pop[h]
        n = max(1, len(y_pop))
        z = X @ w + b
        p = sigmoid(z)
        grad_p = sample_weights * (p - y_pop)
        gw = (X.T @ grad_p) / n + (l2 / n) * w
        gb = float(np.mean(grad_p))
        w = w - lr_pop * gw
        b = b - lr_pop * gb
        state.pop[h] = (w, b)
        
        # quantiles on log1p
        y_log = np.log1p(np.maximum(0.0, y_amt))
        for q in (0.1, 0.5, 0.9):
            wq, bq = state.q[h][q]
            wq, bq = sgd_quantile_step(wq, bq, X, y_log, q=q, lr=lr_q, l2=l2)
            state.q[h][q] = (wq, bq)
        state.trained_samples += int(len(y_pop))
        info["updated"] += int(len(y_pop))
        info["by_horizon"][str(h)] = int(len(y_pop))
    return info