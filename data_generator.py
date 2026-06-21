"""
공정제어 상시모니터링 시스템 — 샘플 데이터 생성기
철강/금속 압연·코일 생산라인 시계열 데이터 (이상패턴 ~15% 내장)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── 라인별 공정 기준값 ─────────────────────────────────────────
LINE_CONFIG = {
    "A라인": {
        "temp_nominal": 900, "temp_ucl": 950, "temp_lcl": 850,
        "pressure_nominal": 280, "pressure_ucl": 315, "pressure_lcl": 245,
        "speed_nominal": 45.0, "speed_ucl": 54, "speed_lcl": 36,
        "thickness_nominal": 3.00, "thickness_tol": 0.08,
        "defect_nominal": 0.80, "defect_limit": 2.0,
        "vibration_nominal": 12.0, "vibration_ucl": 20.0,
        "color": "#00d4ff",
    },
    "B라인": {
        "temp_nominal": 880, "temp_ucl": 930, "temp_lcl": 840,
        "pressure_nominal": 320, "pressure_ucl": 360, "pressure_lcl": 285,
        "speed_nominal": 38.0, "speed_ucl": 47, "speed_lcl": 30,
        "thickness_nominal": 4.50, "thickness_tol": 0.10,
        "defect_nominal": 1.00, "defect_limit": 2.5,
        "vibration_nominal": 14.0, "vibration_ucl": 24.0,
        "color": "#7c3aed",
    },
    "C라인": {
        "temp_nominal": 920, "temp_ucl": 960, "temp_lcl": 870,
        "pressure_nominal": 260, "pressure_ucl": 295, "pressure_lcl": 228,
        "speed_nominal": 55.0, "speed_ucl": 65, "speed_lcl": 45,
        "thickness_nominal": 2.00, "thickness_tol": 0.05,
        "defect_nominal": 0.60, "defect_limit": 1.8,
        "vibration_nominal": 10.0, "vibration_ucl": 18.0,
        "color": "#10b981",
    },
}

VARIABLE_LABELS = {
    "temperature":    "압연 온도(℃)",
    "pressure":       "압연 압력(ton)",
    "speed":          "라인 속도(m/min)",
    "thickness_dev":  "두께 편차(mm)",
    "defect_rate":    "표면 결함률(%)",
    "vibration":      "진동(Hz)",
}


def generate_process_data(days: int = 30, freq_min: int = 10,
                           seed: int = 42) -> pd.DataFrame:
    """
    공정 시계열 데이터 생성
    - 3개 라인(A/B/C), 10분 간격
    - 이상 패턴 ~15% 내장
    """
    np.random.seed(seed)
    end_dt   = datetime.now().replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    timestamps = pd.date_range(start_dt, end_dt, freq=f"{freq_min}min")
    n = len(timestamps)

    records = []
    for line, cfg in LINE_CONFIG.items():
        t_sigma  = (cfg["temp_ucl"]      - cfg["temp_lcl"])      * 0.035
        p_sigma  = (cfg["pressure_ucl"]  - cfg["pressure_lcl"])  * 0.035
        s_sigma  = (cfg["speed_ucl"]     - cfg["speed_lcl"])     * 0.035
        th_sigma = cfg["thickness_tol"] * 0.25
        d_sigma  = cfg["defect_nominal"] * 0.18
        v_sigma  = cfg["vibration_ucl"]  * 0.04

        # 기저 시계열 (완만한 사인파 drift 포함)
        t_base  = np.array([cfg["temp_nominal"]     + t_sigma  * np.random.randn()
                             + 4 * np.sin(2 * np.pi * i / (6 * 24)) for i in range(n)])
        p_base  = np.array([cfg["pressure_nominal"] + p_sigma  * np.random.randn() for _ in range(n)])
        s_base  = np.array([cfg["speed_nominal"]    + s_sigma  * np.random.randn() for _ in range(n)])
        th_base = np.array([cfg["thickness_nominal"]+ th_sigma * np.random.randn() for _ in range(n)])
        d_base  = np.abs(np.array([cfg["defect_nominal"] + d_sigma * np.random.randn() for _ in range(n)]))
        v_base  = np.array([cfg["vibration_nominal"]+ v_sigma  * np.random.randn() for _ in range(n)])

        anomaly_flags = np.zeros(n, dtype=bool)
        anomaly_types = np.array(["정상"] * n)

        budget = int(n * 0.15)

        # 패턴 1: 온도 스파이크 (급격한 상승/하강)
        n_spk = max(2, n // 180)
        spike_pos = np.random.choice(n - 4, size=n_spk, replace=False)
        for pos in spike_pos:
            if budget <= 0: break
            span = min(3, n - pos)
            t_base[pos:pos+span] += np.random.uniform(60, 130)
            anomaly_flags[pos:pos+span] = True
            anomaly_types[pos:pos+span] = "온도스파이크"
            budget -= span

        # 패턴 2: 온도 점진적 드리프트
        if budget > 20:
            d_start = np.random.randint(n // 4, n // 2)
            d_len   = min(n // 8, n - d_start, budget)
            t_base[d_start:d_start+d_len] += np.linspace(0, 45, d_len)
            anomaly_flags[d_start:d_start+d_len] = True
            anomaly_types[d_start:d_start+d_len] = "온도드리프트"
            budget -= d_len

        # 패턴 3: 압력 급변동
        n_pspk = max(2, n // 220)
        p_spike_pos = np.random.choice(n - 3, size=n_pspk, replace=False)
        for pos in p_spike_pos:
            if budget <= 0: break
            span = min(2, n - pos)
            sign = np.random.choice([-1, 1])
            p_base[pos:pos+span] += sign * np.random.uniform(40, 90)
            anomaly_flags[pos:pos+span] = True
            anomaly_types[pos:pos+span] = "압력급변동"
            budget -= span

        # 패턴 4: 라인 속도 정지/급가속
        if budget > 5:
            st_pos = np.random.randint(n // 3, 2 * n // 3)
            st_len = min(max(2, n // 280), n - st_pos, budget)
            s_base[st_pos:st_pos+st_len] = np.random.uniform(0, 4)
            anomaly_flags[st_pos:st_pos+st_len] = True
            anomaly_types[st_pos:st_pos+st_len] = "속도이상(정지)"
            budget -= st_len
            # 재기동 후 급가속
            acc_pos = st_pos + st_len
            acc_len = min(3, n - acc_pos, budget)
            if acc_len > 0:
                s_base[acc_pos:acc_pos+acc_len] = cfg["speed_nominal"] * 1.3
                anomaly_flags[acc_pos:acc_pos+acc_len] = True
                anomaly_types[acc_pos:acc_pos+acc_len] = "속도이상(급가속)"
                budget -= acc_len

        # 패턴 5: 두께 공차 이탈
        n_th = max(2, n // 200)
        th_pos = np.random.choice(n - 6, size=n_th, replace=False)
        for pos in th_pos:
            if budget <= 0: break
            span = min(5, n - pos)
            direction = np.random.choice([-1, 1])
            th_base[pos:pos+span] += (
                direction * np.random.uniform(cfg["thickness_tol"] * 1.8,
                                              cfg["thickness_tol"] * 3.5)
            )
            anomaly_flags[pos:pos+span] = True
            anomaly_types[pos:pos+span] = "두께공차이탈"
            budget -= span

        # 패턴 6: 결함률 급증
        n_dspk = max(2, n // 280)
        d_spike_pos = np.random.choice(n - 7, size=n_dspk, replace=False)
        for pos in d_spike_pos:
            if budget <= 0: break
            span = min(6, n - pos)
            d_base[pos:pos+span] = np.random.uniform(
                cfg["defect_limit"] * 1.3, cfg["defect_limit"] * 2.8)
            anomaly_flags[pos:pos+span] = True
            anomaly_types[pos:pos+span] = "결함률급증"
            budget -= span

        # 패턴 7: 진동 베어링 이상 (점진 증가 → 스파이크)
        if budget > 15:
            v_start = np.random.randint(2 * n // 3, n - n // 6)
            v_len   = min(n // 9, n - v_start, budget)
            ramp    = np.linspace(6, 30, v_len) + 2 * np.random.randn(v_len)
            v_base[v_start:v_start+v_len] += ramp
            anomaly_flags[v_start:v_start+v_len] = True
            anomaly_types[v_start:v_start+v_len] = "진동이상(베어링)"
            budget -= v_len

        # 두께편차 → 절대값 편차
        th_dev = np.abs(th_base - cfg["thickness_nominal"])

        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp":    ts,
                "line":         line,
                "temperature":  round(float(t_base[i]), 1),
                "pressure":     round(float(p_base[i]), 1),
                "speed":        round(float(s_base[i]), 1),
                "thickness_dev":round(float(th_dev[i]), 4),
                "defect_rate":  round(max(0.0, float(d_base[i])), 3),
                "vibration":    round(float(v_base[i]), 2),
                "is_anomaly":   bool(anomaly_flags[i]),
                "anomaly_type": str(anomaly_types[i]),
            })

    df = (pd.DataFrame(records)
          .sort_values(["timestamp", "line"])
          .reset_index(drop=True))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """각 라인 최신 데이터 1건"""
    return df.groupby("line").last().reset_index()


def get_recent(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    cutoff = df["timestamp"].max() - timedelta(hours=hours)
    return df[df["timestamp"] >= cutoff].copy()


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    anom = df[df["is_anomaly"]].copy()
    return (
        anom.groupby(["line", "anomaly_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["line", "count"], ascending=[True, False])
    )


def get_hourly_stats(df: pd.DataFrame, line: str, col: str) -> pd.DataFrame:
    sub = df[df["line"] == line].copy()
    sub = sub.set_index("timestamp")
    return (sub[col]
            .resample("1h")
            .agg(["mean", "std", "min", "max"])
            .reset_index())
