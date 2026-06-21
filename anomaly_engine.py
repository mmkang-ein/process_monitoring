"""
공정제어 상시모니터링 시스템 — 이상탐지 엔진
Z-score / IQR / Isolation Forest / SPC(X-bar, R chart) / Cp·Cpk
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

# ── SPC 관리도 상수 (서브그룹 크기별) ────────────────────────────
SPC_CONSTANTS = {
    2:  {"A2": 1.880, "D3": 0.000, "D4": 3.267, "d2": 1.128},
    3:  {"A2": 1.023, "D3": 0.000, "D4": 2.575, "d2": 1.693},
    4:  {"A2": 0.729, "D3": 0.000, "D4": 2.282, "d2": 2.059},
    5:  {"A2": 0.577, "D3": 0.000, "D4": 2.115, "d2": 2.326},
    6:  {"A2": 0.483, "D3": 0.000, "D4": 2.004, "d2": 2.534},
    7:  {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704},
    8:  {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847},
    9:  {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078},
}

PROCESS_FEATURES = ["temperature", "pressure", "speed",
                    "thickness_dev", "defect_rate", "vibration"]


class ProcessAnomalyEngine:
    """공정 이상탐지 통합 엔진"""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state  = random_state
        self._scalers: Dict[str, StandardScaler] = {}
        self._isoforests: Dict[str, IsolationForest] = {}

    def _get_iso(self, key: str) -> IsolationForest:
        if key not in self._isoforests:
            self._isoforests[key] = IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._scalers[key] = StandardScaler()
        return self._isoforests[key], self._scalers[key]

    # ── Z-Score ──────────────────────────────────────────────────
    def detect_zscore(self, series: pd.Series,
                      threshold: float = 3.0) -> pd.DataFrame:
        filled = series.fillna(series.median())
        z      = np.abs(stats.zscore(filled))
        return pd.DataFrame({
            "timestamp":  series.index if hasattr(series.index, 'dtype') else range(len(series)),
            "value":      series.values,
            "z_score":    z.round(4),
            "is_anomaly": z > threshold,
            "method":     "Z-Score",
        }).reset_index(drop=True)

    # ── IQR ──────────────────────────────────────────────────────
    def detect_iqr(self, series: pd.Series,
                   multiplier: float = 1.5) -> pd.DataFrame:
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        return pd.DataFrame({
            "timestamp":   series.index if hasattr(series.index, 'dtype') else range(len(series)),
            "value":       series.values,
            "lower_bound": round(lo, 4),
            "upper_bound": round(hi, 4),
            "is_anomaly":  ((series < lo) | (series > hi)).values,
            "method":      "IQR",
        }).reset_index(drop=True)

    # ── Isolation Forest ──────────────────────────────────────────
    def detect_isolation_forest(self, df: pd.DataFrame,
                                features: List[str],
                                key: str = "default") -> pd.DataFrame:
        iso, scaler = self._get_iso(key)
        X        = df[features].fillna(0).values
        X_scaled = scaler.fit_transform(X)
        preds    = iso.fit_predict(X_scaled)
        scores   = iso.score_samples(X_scaled)
        result   = df[["timestamp"] + features].copy()
        result["anomaly_score"] = scores.round(4)
        result["is_anomaly"]    = preds == -1
        result["method"]        = "IsolationForest"
        return result.reset_index(drop=True)

    # ── SPC X-bar / R 관리도 ─────────────────────────────────────
    def spc_xbar_r(self, series: pd.Series,
                   subgroup_size: int = 5) -> Dict:
        """
        X-bar/R 관리도 계산
        UCL_x = X̄̄ + A₂·R̄  /  LCL_x = X̄̄ - A₂·R̄
        UCL_r = D₄·R̄       /  LCL_r = D₃·R̄
        """
        n_sub = max(2, min(subgroup_size, 10))
        data  = series.dropna().values
        n_full = (len(data) // n_sub) * n_sub
        if n_full < n_sub * 2:
            return {"error": "데이터 부족 (최소 서브그룹 2개 이상 필요)"}
        data   = data[:n_full]
        groups = data.reshape(-1, n_sub)

        x_bars = groups.mean(axis=1)
        r_bars = groups.max(axis=1) - groups.min(axis=1)

        Xdbar = x_bars.mean()
        Rbar  = r_bars.mean()

        c   = SPC_CONSTANTS[n_sub]
        A2, D3, D4 = c["A2"], c["D3"], c["D4"]

        UCL_x = Xdbar + A2 * Rbar
        LCL_x = Xdbar - A2 * Rbar
        UCL_r = D4 * Rbar
        LCL_r = D3 * Rbar

        ooc_x = (x_bars > UCL_x) | (x_bars < LCL_x)
        ooc_r = (r_bars > UCL_r) | (r_bars < LCL_r)

        return {
            "x_bars":   x_bars,
            "r_bars":   r_bars,
            "Xdbar":    round(float(Xdbar), 4),
            "Rbar":     round(float(Rbar),  4),
            "UCL_x":    round(float(UCL_x), 4),
            "LCL_x":    round(float(LCL_x), 4),
            "UCL_r":    round(float(UCL_r), 4),
            "LCL_r":    round(float(LCL_r), 4),
            "OOC_x":    ooc_x,
            "OOC_r":    ooc_r,
            "n_ooc_x":  int(ooc_x.sum()),
            "n_ooc_r":  int(ooc_r.sum()),
            "n_groups": len(x_bars),
            "subgroup_size": n_sub,
        }

    # ── Cp / Cpk 공정능력지수 ─────────────────────────────────────
    def process_capability(self, series: pd.Series,
                           usl: float, lsl: float) -> Dict:
        """
        Cp  = (USL − LSL) / (6σ)
        Cpk = min[(USL − μ)/(3σ), (μ − LSL)/(3σ)]
        """
        data  = series.dropna()
        mu    = float(data.mean())
        sigma = float(data.std(ddof=1))

        if sigma < 1e-10:
            Cp = Cpu = Cpl = Cpk = float("inf")
        else:
            Cp  = (usl - lsl)  / (6 * sigma)
            Cpu = (usl - mu)   / (3 * sigma)
            Cpl = (mu  - lsl)  / (3 * sigma)
            Cpk = min(Cpu, Cpl)

        if Cpk == float("inf") or Cpk >= 1.67:
            grade, grade_color = "매우 우수", "#10b981"
        elif Cpk >= 1.33:
            grade, grade_color = "우수",     "#3b82f6"
        elif Cpk >= 1.00:
            grade, grade_color = "보통",     "#f59e0b"
        else:
            grade, grade_color = "불량",     "#ef4444"

        within = float(((data >= lsl) & (data <= usl)).mean() * 100)

        return {
            "Cp":   round(Cp,  3),
            "Cpk":  round(Cpk, 3),
            "Cpu":  round(Cpu, 3),
            "Cpl":  round(Cpl, 3),
            "mean": round(mu,  4),
            "std":  round(sigma, 4),
            "USL":  usl,
            "LSL":  lsl,
            "grade":       grade,
            "grade_color": grade_color,
            "within_spec_pct": round(within, 2),
        }

    # ── 라인별 통합 실행 ──────────────────────────────────────────
    def run_all(self, df: pd.DataFrame, line: str) -> Dict:
        sub = df[df["line"] == line].copy().reset_index(drop=True)
        sub.index = sub["timestamp"]

        return {
            "zscore_temp":      self.detect_zscore(sub["temperature"]),
            "zscore_vib":       self.detect_zscore(sub["vibration"]),
            "iqr_pressure":     self.detect_iqr(sub["pressure"]),
            "iqr_defect":       self.detect_iqr(sub["defect_rate"]),
            "iso_forest":       self.detect_isolation_forest(
                                    sub, PROCESS_FEATURES, key=line),
            "spc_temp":         self.spc_xbar_r(sub["temperature"]),
            "spc_thickness":    self.spc_xbar_r(sub["thickness_dev"]),
            "spc_vibration":    self.spc_xbar_r(sub["vibration"]),
        }


def calc_risk_score(df_line: pd.DataFrame, cfg: dict) -> float:
    """
    설비 위험점수 계산 (0~10)
    - 이상 비율, 평균 진동, 결함률, 온도 이탈 종합
    """
    n = len(df_line)
    if n == 0:
        return 0.0

    anom_ratio  = df_line["is_anomaly"].mean()
    vib_ratio   = (df_line["vibration"] / cfg["vibration_ucl"]).clip(0, 2).mean()
    defect_ratio= (df_line["defect_rate"] / cfg["defect_limit"]).clip(0, 2).mean()
    temp_out    = (
        ((df_line["temperature"] > cfg["temp_ucl"]) |
         (df_line["temperature"] < cfg["temp_lcl"])).mean()
    )

    score = (anom_ratio * 3.5 + vib_ratio * 2.5 +
             defect_ratio * 2.5 + temp_out * 1.5)
    return round(min(float(score) * 10, 10.0), 2)
