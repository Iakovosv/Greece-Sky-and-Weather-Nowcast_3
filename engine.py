# model/engine.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from ml.base import BaseModel
from ml.online import OnlineState, apply_training
from ml.persistence import load_state, atomic_save

HORIZONS = (30, 60, 120, 360)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid_stable(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def _expm1_mm_stable(y: float, max_mm: float) -> float:
    y = float(np.clip(y, -20.0, 6.0))
    mm = float(np.expm1(y))
    if not np.isfinite(mm):
        mm = 0.0
    return float(_clamp(mm, 0.0, max_mm))


class ModelEngine:
    """Base model + lightweight local adaptation (safe for RPi4)."""

    def __init__(self, logger=None):
        self.log = logger

        self.base = BaseModel.load()
        self.features = self.base.features
        self.base_version = self.base.version

        self.trained_samples = 0
        self.last_training_info: Dict[str, Any] = {"updated": 0}
        self.last_training_ts: str | None = None
        self.training_count_total: int = 0

        pop: Dict[int, Any] = {}
        q: Dict[int, Any] = {}

        for h in (60, 120, 360):
            bm = self.base.pop_models[str(h)]
            pop[h] = (np.array(bm["w"], dtype=float), float(bm["b"]))

            am = self.base.amount_models[str(h)]
            q[h] = {}
            for qq in (0.1, 0.5, 0.9):
                coeff = am[str(qq)]
                q[h][qq] = (np.array(coeff["w"], dtype=float), float(coeff["b"]))

        self.state = OnlineState(pop=pop, q=q, trained_samples=0)

        st = load_state()
        if st:
            try:
                self._apply_persisted(st)
            except Exception as e:
                if self.log:
                    self.log.warning(f"Failed to load station state: {e}")

        self.trained_samples = int(getattr(self.state, "trained_samples", 0))

        if self.log:
            self.log.info(
                f"Loaded base model version={self.base_version} features={len(self.features)}"
            )

    def _apply_persisted(self, st: Dict[str, Any]) -> None:
        if st.get("base_version") != self.base_version:
            if self.log:
                self.log.warning(
                    f"Station state base_version mismatch: "
                    f"{st.get('base_version')} != {self.base_version} (ignored)"
                )
            return

        for h_str, wb in st.get("pop", {}).items():
            h = int(h_str)
            w = np.array(wb.get("w", []), dtype=float)
            b = float(wb.get("b", 0.0))
            if h in self.state.pop and len(w) == len(self.features):
                self.state.pop[h] = (w, b)

        for h_str, qs in st.get("q", {}).items():
            h = int(h_str)
            if h not in self.state.q:
                continue
            for q_str, wb in qs.items():
                qq = float(q_str)
                w = np.array(wb.get("w", []), dtype=float)
                b = float(wb.get("b", 0.0))
                if qq in self.state.q[h] and len(w) == len(self.features):
                    self.state.q[h][qq] = (w, b)

        self.state.trained_samples = int(st.get("trained_samples", 0))
        self.training_count_total = int(st.get("training_count_total", 0) or 0)
        self.last_training_ts = st.get("last_training_ts")

    def save_state(self) -> None:
        obj = {
            "base_version": self.base_version,
            "trained_samples": int(self.state.trained_samples),
            "training_count_total": int(self.training_count_total),
            "last_training_ts": self.last_training_ts,
            "pop": {},
            "q": {},
        }

        for h, (w, b) in self.state.pop.items():
            obj["pop"][str(h)] = {"w": w.tolist(), "b": float(b)}

        for h, qs in self.state.q.items():
            obj["q"][str(h)] = {}
            for qq, (w, b) in qs.items():
                obj["q"][str(h)][str(qq)] = {"w": w.tolist(), "b": float(b)}

        atomic_save(obj)

    def infer(self, buffer: Any) -> Dict[str, float]:
        last = buffer.last() if hasattr(buffer, "last") else None
        if not last:
            return (
                {f"pop_{h}m": 0.0 for h in HORIZONS}
                | {f"p{p}_{h}m": 0.0 for h in HORIZONS for p in (10, 50, 90)}
            )

        x = self.base.featurize(last)
        out: Dict[str, float] = {}

        p60 = self._infer_pop_local(x, 60)
        p120 = self._infer_pop_local(x, 120)
        p360 = self._infer_pop_local(x, 360)

        out["pop_60m"] = round(100.0 * p60, 1)
        out["pop_30m"] = round(100.0 * _clamp(p60 * 0.85, 0.0, 1.0), 1)
        out["pop_120m"] = round(100.0 * p120, 1)
        out["pop_360m"] = round(100.0 * p360, 1)

        max_mm = {60: 50.0, 120: 80.0, 360: 120.0}

        q10_60, q50_60, q90_60 = self._infer_q_local(x, 60, max_mm=max_mm[60])
        q10_120, q50_120, q90_120 = self._infer_q_local(x, 120, max_mm=max_mm[120])
        q10_360, q50_360, q90_360 = self._infer_q_local(x, 360, max_mm=max_mm[360])

        def gate(q: float, p: float) -> float:
            g = float(_clamp((p - 0.02) / 0.08, 0.0, 1.0))
            return q * g

        q10_60, q50_60, q90_60 = gate(q10_60, p60), gate(q50_60, p60), gate(q90_60, p60)
        q10_120, q50_120, q90_120 = gate(q10_120, p120), gate(q50_120, p120), gate(q90_120, p120)
        q10_360, q50_360, q90_360 = gate(q10_360, p360), gate(q50_360, p360), gate(q90_360, p360)

        out.update(
            {
                "p10_60m": round(q10_60, 3),
                "p50_60m": round(q50_60, 3),
                "p90_60m": round(q90_60, 3),
                "p10_30m": round(_clamp(q10_60 * 0.7, 0.0, 30.0), 3),
                "p50_30m": round(_clamp(q50_60 * 0.7, 0.0, 30.0), 3),
                "p90_30m": round(_clamp(q90_60 * 0.7, 0.0, 30.0), 3),
                "p10_120m": round(q10_120, 3),
                "p50_120m": round(q50_120, 3),
                "p90_120m": round(q90_120, 3),
                "p10_360m": round(q10_360, 3),
                "p50_360m": round(q50_360, 3),
                "p90_360m": round(q90_360, 3),
            }
        )

        return out

    def _infer_pop_local(self, x, h):
        w, b = self.state.pop[h]
        raw_logit = float(x @ w + b)

        bm = self.base.pop_models[str(h)]
        a = float(bm.get("platt_a", 1.0))
        c = float(bm.get("platt_b", 0.0))

        z = a * raw_logit + c
        
        # Calibration for class imbalance (38:1)
        z = z + np.log(2.5)
        
        p = _sigmoid_stable(z)
        if not np.isfinite(p):
            p = 0.0
        return float(_clamp(p, 0.0, 1.0))

    def _infer_q_local(self, x, h, max_mm) -> Tuple[float, float, float]:
        qs: Dict[float, float] = {}

        for qq, (w, b) in self.state.q[h].items():
            yhat = float(x @ w + b)
            if not np.isfinite(yhat):
                yhat = 0.0
            qs[qq] = _expm1_mm_stable(yhat, max_mm=max_mm)

        q10, q50, q90 = qs.get(0.1, 0.0), qs.get(0.5, 0.0), qs.get(0.9, 0.0)

        q10 = min(q10, q50, q90)
        q90 = max(q10, q50, q90)
        q50 = max(q10, min(q50, q90))

        return q10, q50, q90

    def train_from_buffer(
        self,
        buffer_rows: List[Dict[str, Any]],
        horizons_min=(60, 120, 360),
        threshold_mm=0.1,
        pos_weight=10.0,
    ) -> Dict[str, Any]:
        batch: Dict[int, Any] = {}

        for h in horizons_min:
            if len(buffer_rows) < h + 2:
                continue

            snap = buffer_rows[-(h + 1)]
            window = buffer_rows[-h:]

            amt = float(sum(float(r.get("rain_1m_mm", 0.0) or 0.0) for r in window))
            pop = 1.0 if amt > threshold_mm else 0.0

            x = self.base.featurize(snap)

            batch.setdefault(h, {"X": [], "amt": [], "pop": []})
            batch[h]["X"].append(x.tolist())
            batch[h]["amt"].append(amt)
            batch[h]["pop"].append(pop)

        if not batch:
            return {"updated": 0, "reason": "not_enough_history"}

        info = apply_training(self.state, batch, lr_pop=0.01, lr_q=0.01, l2=8.0, pos_weight=pos_weight)

        updated = int(info.get("updated", 0) or 0)
        if updated > 0:
            self.training_count_total += 1
            from datetime import datetime as _dt
            self.last_training_ts = _dt.utcnow().isoformat()

        self.last_training_info = info
        self.trained_samples = self.state.trained_samples
        return info