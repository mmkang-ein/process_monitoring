"""
============================================================
공정제어 상시모니터링 시스템 v1.0
철강/금속 압연·코일 생산라인 — 데모 버전
실행: streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data_generator import (
    generate_process_data, get_latest_snapshot,
    get_recent, get_anomaly_summary, LINE_CONFIG, VARIABLE_LABELS,
)
from anomaly_engine import ProcessAnomalyEngine, calc_risk_score, PROCESS_FEATURES

# ══════════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="공정제어 상시모니터링",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 글로벌 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background: #070b14; }
[data-testid="stSidebar"] { background: #0d1422 !important; border-right: 1px solid #1e3a5c; }
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; color: #e8f0ff; }

/* ── KPI 카드 ── */
.kpi-card {
    background: linear-gradient(135deg, #0f1e38 0%, #142040 100%);
    border: 1px solid #1e3a5c;
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 30px; font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.15; color: #ffffff;
}
.kpi-label { font-size: 11px; color: #7a9cc0; margin-top: 4px; letter-spacing: 0.5px; }
.kpi-delta { font-size: 12px; margin-top: 4px; }

/* ── 섹션 카드 ── */
.pm-card {
    background: linear-gradient(135deg, #112038 0%, #152640 100%);
    border: 1px solid #1e3a5c;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative; overflow: hidden;
    box-shadow: 0 2px 14px rgba(0,0,0,0.45);
}
.pm-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.pm-card.cyan::before   { background: linear-gradient(90deg,#00d4ff,#0097b2); }
.pm-card.purple::before { background: linear-gradient(90deg,#7c3aed,#4f46e5); }
.pm-card.green::before  { background: linear-gradient(90deg,#10b981,#059669); }
.pm-card.amber::before  { background: linear-gradient(90deg,#f59e0b,#d97706); }
.pm-card.red::before    { background: linear-gradient(90deg,#ef4444,#dc2626); }

/* ── 배지 ── */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
}
.badge-critical { background:rgba(239,68,68,0.25);  color:#ff7070; border:1px solid rgba(239,68,68,0.4); }
.badge-warning  { background:rgba(245,158,11,0.25); color:#fbbf24; border:1px solid rgba(245,158,11,0.4); }
.badge-normal   { background:rgba(16,185,129,0.25); color:#34d399; border:1px solid rgba(16,185,129,0.4); }
.badge-info     { background:rgba(59,130,246,0.25); color:#93c5fd; border:1px solid rgba(59,130,246,0.4); }

/* ── 도움말 박스 ── */
.help-box {
    background:#0f1e35; border:1px solid #1e3a5c; border-radius:10px;
    padding:16px 20px; margin:10px 0;
}
.help-title { font-size:13px; font-weight:700; color:#00d4ff; margin-bottom:6px; }
.help-body  { font-size:13px; color:#c8d8f0; line-height:1.7; }

/* ── 섹션 헤더 ── */
.section-header {
    font-size: 18px; font-weight: 700; color: #e8f0ff;
    border-left: 4px solid #00d4ff; padding-left: 12px;
    margin: 20px 0 14px 0;
}

/* 사이드바 타이틀 */
.sidebar-title {
    font-size: 13px; font-weight: 700; color: #7a9cc0;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px;
}

/* Plotly 배경 투명 */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 데이터 로딩 (캐시)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="데이터 생성 중…")
def load_data():
    return generate_process_data(days=30, freq_min=10, seed=42)


@st.cache_resource
def get_engine():
    return ProcessAnomalyEngine(contamination=0.05)


def plotly_dark_layout(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a1628",
        font=dict(family="Noto Sans KR", color="#c8d8f0", size=12),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="#1a2d4a", zerolinecolor="#1a2d4a"),
        yaxis=dict(gridcolor="#1a2d4a", zerolinecolor="#1a2d4a"),
    )
    return fig


def status_badge(value, ucl, lcl=None, warn_ratio=0.9):
    """값/한계 기준 상태 뱃지 HTML 반환"""
    if lcl is not None and (value > ucl or value < lcl):
        return '<span class="badge badge-critical">이상</span>'
    elif value > ucl * warn_ratio:
        return '<span class="badge badge-warning">주의</span>'
    else:
        return '<span class="badge badge-normal">정상</span>'


# ══════════════════════════════════════════════════════════════════
# 메뉴 1: 라인별 현황
# ══════════════════════════════════════════════════════════════════
def show_line_status(df: pd.DataFrame):
    st.markdown('<div class="section-header">🏭 라인별 현황</div>', unsafe_allow_html=True)

    snap = get_latest_snapshot(df)
    recent_24h = get_recent(df, hours=24)

    # ── KPI 요약 (전체) ──────────────────────────────────────────
    total_anomalies = recent_24h["is_anomaly"].sum()
    total_records   = len(recent_24h)
    anomaly_pct     = total_anomalies / total_records * 100 if total_records > 0 else 0
    active_lines    = df["line"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#00d4ff">{active_lines}</div>
            <div class="kpi-label">가동 라인 수</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#f59e0b">{total_anomalies:,}</div>
            <div class="kpi-label">이상 감지 (24h)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#ef4444">{anomaly_pct:.1f}%</div>
            <div class="kpi-label">이상 비율 (24h)</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_defect = recent_24h["defect_rate"].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#10b981">{avg_defect:.2f}%</div>
            <div class="kpi-label">평균 결함률 (24h)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 라인별 상세 카드 ─────────────────────────────────────────
    for _, row in snap.iterrows():
        line = row["line"]
        cfg  = LINE_CONFIG[line]
        line_recent = recent_24h[recent_24h["line"] == line]
        anom_cnt    = line_recent["is_anomaly"].sum()
        anom_rate   = anom_cnt / len(line_recent) * 100 if len(line_recent) > 0 else 0

        color_class = {"A라인": "cyan", "B라인": "purple", "C라인": "green"}[line]
        badge_temp  = status_badge(row["temperature"], cfg["temp_ucl"], cfg["temp_lcl"])
        badge_press = status_badge(row["pressure"],    cfg["pressure_ucl"], cfg["pressure_lcl"])
        badge_vib   = status_badge(row["vibration"],   cfg["vibration_ucl"])
        badge_def   = status_badge(row["defect_rate"], cfg["defect_limit"])

        st.markdown(f"""
        <div class="pm-card {color_class}">
          <b style="font-size:17px;color:#ffffff">{line}</b>
          &nbsp;&nbsp;<span style="font-size:12px;color:#7a9cc0">
            최근 데이터: {row['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(row['timestamp'], 'strftime') else row['timestamp']}
          </span>
          &nbsp;&nbsp;<span class="badge {'badge-critical' if anom_rate > 20 else 'badge-warning' if anom_rate > 5 else 'badge-normal'}">
            24h 이상 {anom_cnt}건({anom_rate:.1f}%)
          </span>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metrics = [
            (c1, "압연 온도(℃)",     f"{row['temperature']:.1f}",  badge_temp,  cfg["temp_ucl"]),
            (c2, "압연 압력(ton)",    f"{row['pressure']:.1f}",     badge_press, cfg["pressure_ucl"]),
            (c3, "라인 속도(m/min)", f"{row['speed']:.1f}",        '<span class="badge badge-normal">정상</span>', None),
            (c4, "두께 편차(mm)",    f"{row['thickness_dev']:.4f}", '<span class="badge badge-normal">정상</span>', None),
            (c5, "결함률(%)",        f"{row['defect_rate']:.3f}",   badge_def,   cfg["defect_limit"]),
            (c6, "진동(Hz)",         f"{row['vibration']:.2f}",     badge_vib,   cfg["vibration_ucl"]),
        ]
        for col, label, val, badge, _ in metrics:
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-value" style="font-size:22px">{val}</div>
                    <div class="kpi-label">{label}</div>
                    <div style="margin-top:6px">{badge}</div>
                </div>""", unsafe_allow_html=True)

    # ── 라인별 이상 빈도 비교 차트 ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📊 라인별 이상 유형 분포 (최근 7일)</div>',
                unsafe_allow_html=True)

    recent_7d = get_recent(df, hours=168)
    summ = get_anomaly_summary(recent_7d)
    if not summ.empty:
        fig = px.bar(summ, x="anomaly_type", y="count", color="line",
                     barmode="group",
                     color_discrete_map={"A라인":"#00d4ff","B라인":"#7c3aed","C라인":"#10b981"},
                     labels={"anomaly_type":"이상 유형","count":"발생 횟수","line":"라인"})
        plotly_dark_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # ── 최근 24h 온도 비교 추이 ──────────────────────────────────
    st.markdown('<div class="section-header">🌡️ 압연 온도 추이 비교 (최근 24h)</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    colors = {"A라인":"#00d4ff","B라인":"#7c3aed","C라인":"#10b981"}
    for line in df["line"].unique():
        sub = recent_24h[recent_24h["line"] == line]
        fig2.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["temperature"],
            mode="lines", name=line,
            line=dict(color=colors[line], width=1.5),
        ))
    plotly_dark_layout(fig2, height=300)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 메뉴 2: 변수별 추이
# ══════════════════════════════════════════════════════════════════
def show_variable_trends(df: pd.DataFrame):
    st.markdown('<div class="section-header">📈 변수별 추이 분석</div>', unsafe_allow_html=True)

    col_sel, col_var, col_period = st.columns([1, 1, 1])
    with col_sel:
        line_sel = st.selectbox("라인 선택", list(LINE_CONFIG.keys()))
    with col_var:
        var_sel = st.selectbox("변수 선택",
                               list(VARIABLE_LABELS.keys()),
                               format_func=lambda x: VARIABLE_LABELS[x])
    with col_period:
        period_map = {"1시간": 1, "6시간": 6, "24시간": 24, "3일": 72, "7일": 168, "30일": 720}
        period_sel = st.selectbox("기간", list(period_map.keys()), index=3)
    hours = period_map[period_sel]

    cfg  = LINE_CONFIG[line_sel]
    sub  = get_recent(df[df["line"] == line_sel], hours=hours)

    if sub.empty:
        st.warning("해당 기간 데이터가 없습니다.")
        return

    # ── 통계 KPI ─────────────────────────────────────────────────
    vals = sub[var_sel]
    c1, c2, c3, c4, c5 = st.columns(5)
    stats_items = [
        (c1, "현재값",  f"{vals.iloc[-1]:.3f}"),
        (c2, "평균",    f"{vals.mean():.3f}"),
        (c3, "표준편차",f"{vals.std():.3f}"),
        (c4, "최솟값",  f"{vals.min():.3f}"),
        (c5, "최댓값",  f"{vals.max():.3f}"),
    ]
    for col, label, val in stats_items:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="font-size:22px">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── 시계열 차트 (이상 포인트 마킹) ──────────────────────────
    normal = sub[~sub["is_anomaly"]]
    anom   = sub[sub["is_anomaly"] & (sub["anomaly_type"] != "정상")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal[var_sel],
        mode="lines", name="정상",
        line=dict(color="#3b82f6", width=1.5),
    ))
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom[var_sel],
            mode="markers", name="이상",
            marker=dict(color="#ef4444", size=6, symbol="x"),
            hovertemplate="<b>%{x}</b><br>" + VARIABLE_LABELS[var_sel] + ": %{y}<extra></extra>",
        ))

    # UCL/LCL 라인
    ucl_map = {
        "temperature":   (cfg["temp_ucl"],     cfg["temp_lcl"]),
        "pressure":      (cfg["pressure_ucl"], cfg["pressure_lcl"]),
        "speed":         (cfg["speed_ucl"],    cfg["speed_lcl"]),
        "thickness_dev": (cfg["thickness_tol"]*2, 0),
        "defect_rate":   (cfg["defect_limit"], 0),
        "vibration":     (cfg["vibration_ucl"], 0),
    }
    if var_sel in ucl_map:
        ucl_v, lcl_v = ucl_map[var_sel]
        fig.add_hline(y=ucl_v, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"UCL {ucl_v}", annotation_position="top right")
        if lcl_v > 0:
            fig.add_hline(y=lcl_v, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"LCL {lcl_v}", annotation_position="bottom right")

    fig.update_layout(title=f"{line_sel} · {VARIABLE_LABELS[var_sel]} 추이 ({period_sel})")
    plotly_dark_layout(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── 분포 히스토그램 ──────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig_h = px.histogram(sub, x=var_sel, nbins=40,
                             color_discrete_sequence=["#3b82f6"],
                             title=f"{VARIABLE_LABELS[var_sel]} 분포",
                             labels={var_sel: VARIABLE_LABELS[var_sel]})
        plotly_dark_layout(fig_h, height=300)
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        # Box plot per anomaly_type
        sub_box = sub.copy()
        sub_box["상태"] = sub_box["is_anomaly"].map({True: "이상", False: "정상"})
        fig_b = px.box(sub_box, x="상태", y=var_sel,
                       color="상태",
                       color_discrete_map={"정상":"#3b82f6","이상":"#ef4444"},
                       title=f"정상 vs 이상 분포",
                       labels={var_sel: VARIABLE_LABELS[var_sel]})
        plotly_dark_layout(fig_b, height=300)
        st.plotly_chart(fig_b, use_container_width=True)

    # ── 이상 구간 테이블 ─────────────────────────────────────────
    if not anom.empty:
        st.markdown("**⚠️ 이상 감지 구간 상세**")
        disp = anom[["timestamp","line","anomaly_type",var_sel]].copy()
        disp.columns = ["시각","라인","이상유형", VARIABLE_LABELS[var_sel]]
        st.dataframe(disp.tail(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# 메뉴 3: 이상탐지 알림
# ══════════════════════════════════════════════════════════════════
def show_anomaly_alerts(df: pd.DataFrame, engine: ProcessAnomalyEngine):
    st.markdown('<div class="section-header">🚨 이상탐지 알림</div>', unsafe_allow_html=True)

    # 필터
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        line_filter = st.multiselect("라인 필터", list(LINE_CONFIG.keys()),
                                      default=list(LINE_CONFIG.keys()))
    with c2:
        period_h = st.selectbox("기간", {"24시간": 24, "3일": 72, "7일": 168}.keys())
        hour_map = {"24시간": 24, "3일": 72, "7일": 168}
        hours = hour_map[period_h]
    with c3:
        method_sel = st.selectbox("탐지 방법", ["통합(라벨)", "Z-Score", "IQR", "Isolation Forest"])

    recent = get_recent(df, hours=hours)
    recent = recent[recent["line"].isin(line_filter)]

    if method_sel == "통합(라벨)":
        anom_df = recent[recent["is_anomaly"]].copy()
    else:
        # 엔진 실행
        method_map = {"Z-Score":"zscore_temp","IQR":"iqr_pressure","Isolation Forest":"iso_forest"}
        key = method_map[method_sel]
        results_all = []
        for line in line_filter:
            res = engine.run_all(recent[recent["line"] == line], line)
            det = res.get(key, pd.DataFrame())
            if not det.empty and "is_anomaly" in det.columns:
                idxs = det[det["is_anomaly"]].index.tolist()
                sub_line = recent[recent["line"] == line]
                results_all.append(sub_line.iloc[[i for i in idxs if i < len(sub_line)]])
        anom_df = pd.concat(results_all) if results_all else pd.DataFrame()

    # ── KPI ──────────────────────────────────────────────────────
    n_total = len(recent)
    n_anom  = len(anom_df) if not anom_df.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#f59e0b">{n_anom:,}</div>
            <div class="kpi-label">이상 감지 건수</div></div>""", unsafe_allow_html=True)
    with c2:
        pct = n_anom / n_total * 100 if n_total else 0
        color = "#ef4444" if pct > 20 else "#f59e0b" if pct > 10 else "#10b981"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{pct:.1f}%</div>
            <div class="kpi-label">이상 비율</div></div>""", unsafe_allow_html=True)
    with c3:
        n_types = anom_df["anomaly_type"].nunique() if not anom_df.empty and "anomaly_type" in anom_df else 0
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#7c3aed">{n_types}</div>
            <div class="kpi-label">이상 유형 수</div></div>""", unsafe_allow_html=True)
    with c4:
        worst_line = (anom_df.groupby("line").size().idxmax()
                      if not anom_df.empty and "line" in anom_df else "—")
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="font-size:22px;color:#ef4444">{worst_line}</div>
            <div class="kpi-label">최다 이상 라인</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if anom_df.empty:
        st.success("선택 기간/라인에서 이상이 감지되지 않았습니다.")
        return

    # ── 이상 유형별 타임라인 ─────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**⏱️ 이상 감지 타임라인**")
        fig = px.scatter(anom_df, x="timestamp", y="line",
                         color="anomaly_type",
                         hover_data=["temperature","pressure","vibration","defect_rate"],
                         labels={"timestamp":"시각","line":"라인","anomaly_type":"이상 유형"})
        plotly_dark_layout(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**📊 이상 유형 비율**")
        type_cnt = anom_df["anomaly_type"].value_counts()
        fig_p = px.pie(values=type_cnt.values, names=type_cnt.index,
                       color_discrete_sequence=px.colors.qualitative.Set2)
        plotly_dark_layout(fig_p, height=280)
        st.plotly_chart(fig_p, use_container_width=True)

    # ── 이상 알림 테이블 ─────────────────────────────────────────
    st.markdown("**📋 이상 알림 목록 (최근 100건)**")

    def severity(row):
        if row["anomaly_type"] in ["온도스파이크","결함률급증","진동이상(베어링)"]:
            return "긴급"
        elif row["anomaly_type"] in ["온도드리프트","압력급변동","두께공차이탈"]:
            return "경고"
        else:
            return "주의"

    disp = anom_df.tail(100).copy()
    disp["심각도"] = disp.apply(severity, axis=1)
    disp = disp[["timestamp","line","anomaly_type","심각도",
                 "temperature","pressure","vibration","defect_rate"]].copy()
    disp.columns = ["시각","라인","이상유형","심각도","온도(℃)","압력(ton)","진동(Hz)","결함률(%)"]

    def style_severity(val):
        colors = {"긴급": "background-color:#3d0000;color:#ff7070",
                  "경고": "background-color:#3d2d00;color:#fbbf24",
                  "주의": "background-color:#00203d;color:#93c5fd"}
        return colors.get(val, "")

    st.dataframe(
        disp.style.applymap(style_severity, subset=["심각도"]),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════
# 메뉴 4: 설비별 위험평가
# ══════════════════════════════════════════════════════════════════
def show_equipment_risk(df: pd.DataFrame):
    st.markdown('<div class="section-header">⚠️ 설비별 위험평가</div>', unsafe_allow_html=True)

    period_h = st.selectbox("평가 기간", {"최근 24h": 24, "최근 7일": 168, "최근 30일": 720}.keys())
    hour_map = {"최근 24h": 24, "최근 7일": 168, "최근 30일": 720}
    hours = hour_map[period_h]
    recent = get_recent(df, hours=hours)

    # ── 라인별 위험점수 ──────────────────────────────────────────
    risk_rows = []
    equip_list = ["압연기(메인롤)", "권취기", "구동모터", "냉각장치", "가열로"]
    np.random.seed(99)
    for line, cfg in LINE_CONFIG.items():
        sub = recent[recent["line"] == line]
        base_score = calc_risk_score(sub, cfg)
        for equip in equip_list:
            # 설비별 미세 변동 (시뮬)
            eq_score = round(min(10.0, max(0.0,
                base_score + np.random.uniform(-1.5, 2.0))), 2)
            if eq_score >= 7:   level, color = "HIGH",   "#ef4444"
            elif eq_score >= 4: level, color = "MEDIUM", "#f59e0b"
            else:               level, color = "LOW",    "#10b981"
            risk_rows.append({
                "라인": line, "설비": equip,
                "위험점수": eq_score, "위험등급": level,
                "color": color,
            })

    risk_df = pd.DataFrame(risk_rows)

    # ── 히트맵 ──────────────────────────────────────────────────
    pivot = risk_df.pivot(index="설비", columns="라인", values="위험점수")
    fig_h = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#10b981"],[0.4,"#f59e0b"],[1,"#ef4444"]],
        zmin=0, zmax=10,
        text=[[f"{v:.1f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="위험점수"),
    ))
    fig_h.update_layout(title="설비 × 라인 위험점수 히트맵")
    plotly_dark_layout(fig_h, height=360)
    st.plotly_chart(fig_h, use_container_width=True)

    # ── 위험 등급별 테이블 ───────────────────────────────────────
    st.markdown("**📋 설비 위험평가 상세**")

    def risk_badge(level):
        map_ = {"HIGH": "badge-critical", "MEDIUM": "badge-warning", "LOW": "badge-normal"}
        return f'<span class="badge {map_.get(level,"")}">{"긴급" if level=="HIGH" else "경고" if level=="MEDIUM" else "정상"}</span>'

    risk_df["등급배지"] = risk_df["위험등급"].apply(risk_badge)

    st.dataframe(
        risk_df[["라인","설비","위험점수","위험등급"]]
        .sort_values("위험점수", ascending=False),
        use_container_width=True, hide_index=True,
    )

    # ── 라인별 종합 위험 추이 (시뮬) ────────────────────────────
    st.markdown("---")
    st.markdown("**📉 위험점수 일별 추이**")
    days_range = pd.date_range(df["timestamp"].max() - timedelta(days=29),
                               df["timestamp"].max(), freq="1D")
    np.random.seed(77)
    fig_t = go.Figure()
    line_colors = {"A라인":"#00d4ff","B라인":"#7c3aed","C라인":"#10b981"}
    for line, cfg in LINE_CONFIG.items():
        sub = df[df["line"] == line]
        daily_scores = []
        for day in days_range:
            day_data = sub[(sub["timestamp"] >= day) &
                           (sub["timestamp"] < day + timedelta(days=1))]
            daily_scores.append(calc_risk_score(day_data, cfg) if len(day_data) > 0 else 0)
        fig_t.add_trace(go.Scatter(
            x=days_range, y=daily_scores,
            mode="lines+markers", name=line,
            line=dict(color=line_colors[line], width=2),
            marker=dict(size=5),
        ))
    fig_t.add_hline(y=7, line_dash="dash", line_color="#ef4444",
                    annotation_text="HIGH 임계치(7)", annotation_position="top right")
    fig_t.add_hline(y=4, line_dash="dash", line_color="#f59e0b",
                    annotation_text="MEDIUM 임계치(4)", annotation_position="top right")
    plotly_dark_layout(fig_t, height=320)
    st.plotly_chart(fig_t, use_container_width=True)

    # ── 조치 우선순위 ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔧 즉시 조치 필요 항목 (HIGH)**")
    high_risk = risk_df[risk_df["위험등급"] == "HIGH"][["라인","설비","위험점수"]].copy()
    if high_risk.empty:
        st.success("현재 즉시 조치가 필요한 항목이 없습니다.")
    else:
        high_risk["권장조치"] = "즉시 점검 및 예방 정비 실시"
        high_risk["담당팀"]   = "설비유지보수팀"
        st.dataframe(high_risk.sort_values("위험점수", ascending=False),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# 메뉴 5: SPC 관리도
# ══════════════════════════════════════════════════════════════════
def show_spc_charts(df: pd.DataFrame, engine: ProcessAnomalyEngine):
    st.markdown('<div class="section-header">📐 SPC 관리도 / Cp·Cpk</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        line_sel = st.selectbox("라인", list(LINE_CONFIG.keys()), key="spc_line")
    with c2:
        var_opt = {
            "압연 온도(℃)": ("temperature", "temp_ucl", "temp_lcl"),
            "두께 편차(mm)": ("thickness_dev", "thickness_tol", 0),
            "진동(Hz)":      ("vibration", "vibration_ucl", 0),
            "압연 압력(ton)":("pressure", "pressure_ucl", "pressure_lcl"),
        }
        var_disp = st.selectbox("변수", list(var_opt.keys()), key="spc_var")
    with c3:
        n_sub = st.selectbox("서브그룹 크기", [3, 4, 5, 6, 7, 8], index=2, key="spc_n")

    cfg = LINE_CONFIG[line_sel]
    col_name, ucl_key, lcl_key = var_opt[var_disp]
    usl = cfg[ucl_key] if isinstance(ucl_key, str) else ucl_key
    lsl = cfg[lcl_key] if isinstance(lcl_key, str) and lcl_key else (lcl_key if lcl_key else 0.0)

    sub = df[df["line"] == line_sel].copy()
    series = sub[col_name].dropna()

    spc = engine.spc_xbar_r(series, subgroup_size=n_sub)
    if "error" in spc:
        st.warning(spc["error"])
        return

    cap = engine.process_capability(series, usl=float(usl), lsl=float(lsl))

    # ── Cp/Cpk KPI ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cap_items = [
        (c1, "Cp",   f"{cap['Cp']:.3f}",  cap["grade_color"]),
        (c2, "Cpk",  f"{cap['Cpk']:.3f}", cap["grade_color"]),
        (c3, "Cpu",  f"{cap['Cpu']:.3f}", "#c8d8f0"),
        (c4, "Cpl",  f"{cap['Cpl']:.3f}", "#c8d8f0"),
        (c5, "규격 내 비율", f"{cap['within_spec_pct']:.1f}%", "#10b981"),
    ]
    for col, label, val, color in cap_items:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{color};font-size:24px">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    grade_badge_map = {
        "매우 우수": "badge-normal", "우수": "badge-info",
        "보통": "badge-warning",    "불량": "badge-critical"
    }
    st.markdown(
        f"공정능력 판정: <span class='badge {grade_badge_map.get(cap['grade'],\"\")}'>"
        f"{cap['grade']}</span>　"
        f"평균={cap['mean']:.3f}　표준편차={cap['std']:.3f}　"
        f"USL={cap['USL']}　LSL={cap['LSL']}",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── X-bar 관리도 ─────────────────────────────────────────────
    x_idx = list(range(1, len(spc["x_bars"]) + 1))
    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(
        x=x_idx, y=spc["x_bars"], mode="lines+markers", name="X̄",
        line=dict(color="#3b82f6", width=1.5), marker=dict(size=5),
    ))
    # OOC 포인트
    ooc_idx = [i+1 for i, v in enumerate(spc["OOC_x"]) if v]
    ooc_val = [spc["x_bars"][i] for i, v in enumerate(spc["OOC_x"]) if v]
    if ooc_idx:
        fig_x.add_trace(go.Scatter(
            x=ooc_idx, y=ooc_val, mode="markers", name="OOC",
            marker=dict(color="#ef4444", size=9, symbol="circle"),
        ))
    fig_x.add_hline(y=spc["UCL_x"], line_dash="dash", line_color="#ef4444",
                    annotation_text=f"UCL={spc['UCL_x']:.2f}")
    fig_x.add_hline(y=spc["Xdbar"], line_dash="solid", line_color="#10b981",
                    annotation_text=f"CL={spc['Xdbar']:.2f}")
    fig_x.add_hline(y=spc["LCL_x"], line_dash="dash", line_color="#f59e0b",
                    annotation_text=f"LCL={spc['LCL_x']:.2f}")
    fig_x.update_layout(
        title=f"X-bar 관리도 — {line_sel} {var_disp}　"
              f"(OOC: {spc['n_ooc_x']}/{spc['n_groups']}그룹)",
    )
    plotly_dark_layout(fig_x, height=320)
    st.plotly_chart(fig_x, use_container_width=True)

    # ── R 관리도 ─────────────────────────────────────────────────
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=x_idx, y=spc["r_bars"], mode="lines+markers", name="R",
        line=dict(color="#7c3aed", width=1.5), marker=dict(size=5),
    ))
    ooc_r_idx = [i+1 for i, v in enumerate(spc["OOC_r"]) if v]
    ooc_r_val = [spc["r_bars"][i] for i, v in enumerate(spc["OOC_r"]) if v]
    if ooc_r_idx:
        fig_r.add_trace(go.Scatter(
            x=ooc_r_idx, y=ooc_r_val, mode="markers", name="OOC(R)",
            marker=dict(color="#ef4444", size=9, symbol="circle"),
        ))
    fig_r.add_hline(y=spc["UCL_r"], line_dash="dash", line_color="#ef4444",
                    annotation_text=f"UCL_R={spc['UCL_r']:.2f}")
    fig_r.add_hline(y=spc["Rbar"], line_dash="solid", line_color="#10b981",
                    annotation_text=f"R̄={spc['Rbar']:.2f}")
    if spc["LCL_r"] > 0:
        fig_r.add_hline(y=spc["LCL_r"], line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"LCL_R={spc['LCL_r']:.2f}")
    fig_r.update_layout(
        title=f"R 관리도 — {line_sel} {var_disp}　"
              f"(OOC: {spc['n_ooc_r']}/{spc['n_groups']}그룹)",
    )
    plotly_dark_layout(fig_r, height=320)
    st.plotly_chart(fig_r, use_container_width=True)

    # ── 설명 박스 ─────────────────────────────────────────────────
    with st.expander("📖 SPC 관리도 해석 기준"):
        st.markdown("""
| 지수 | 기준 | 판정 |
|------|------|------|
| Cpk ≥ 1.67 | 규격 여유 충분 | 매우 우수 |
| 1.33 ≤ Cpk < 1.67 | 양호 | 우수 |
| 1.00 ≤ Cpk < 1.33 | 최소 허용 수준 | 보통 |
| Cpk < 1.00 | 공정 개선 필요 | 불량 |

- **OOC(Out of Control)**: 관리 한계선 이탈 포인트 → 특수 원인 조사 필요
- **UCL/LCL**: 3σ 기준 상·하 관리 한계 (규격 한계와 다름)
- **서브그룹 크기 n=5** 기준 A₂=0.577 적용
        """)


# ══════════════════════════════════════════════════════════════════
# 메뉴 6: 결함 트렌드
# ══════════════════════════════════════════════════════════════════
def show_defect_trends(df: pd.DataFrame):
    st.markdown('<div class="section-header">🔍 결함 트렌드</div>', unsafe_allow_html=True)

    period_h = st.selectbox("분석 기간",
                             {"24시간": 24, "7일": 168, "30일": 720}.keys(),
                             index=1, key="defect_period")
    hour_map = {"24시간": 24, "7일": 168, "30일": 720}
    hours = hour_map[period_h]
    recent = get_recent(df, hours=hours)

    # ── KPI ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_d = recent["defect_rate"].mean()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#f59e0b">{avg_d:.3f}%</div>
            <div class="kpi-label">평균 결함률</div></div>""", unsafe_allow_html=True)
    with c2:
        max_d = recent["defect_rate"].max()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#ef4444">{max_d:.3f}%</div>
            <div class="kpi-label">최대 결함률</div></div>""", unsafe_allow_html=True)
    with c3:
        defect_events = (recent["anomaly_type"] == "결함률급증").sum()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#ef4444">{defect_events}</div>
            <div class="kpi-label">결함률 급증 이벤트</div></div>""", unsafe_allow_html=True)
    with c4:
        best_line = recent.groupby("line")["defect_rate"].mean().idxmin()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="font-size:20px;color:#10b981">{best_line}</div>
            <div class="kpi-label">최저 결함 라인</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 라인별 결함률 추이 ───────────────────────────────────────
    colors = {"A라인":"#00d4ff","B라인":"#7c3aed","C라인":"#10b981"}
    fig = go.Figure()
    for line, cfg in LINE_CONFIG.items():
        sub = recent[recent["line"] == line]
        # 1시간 rolling 평균
        sub_r = sub.set_index("timestamp")["defect_rate"].resample("1h").mean().reset_index()
        fig.add_trace(go.Scatter(
            x=sub_r["timestamp"], y=sub_r["defect_rate"],
            mode="lines", name=line,
            line=dict(color=colors[line], width=2),
        ))
        # 관리 한계선
        fig.add_hline(y=cfg["defect_limit"],
                      line_dash="dot", line_color=colors[line],
                      opacity=0.5,
                      annotation_text=f"{line} 한계({cfg['defect_limit']}%)")
    fig.update_layout(title="라인별 결함률 추이 (1시간 평균)")
    plotly_dark_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)

    # ── 결함률 × 온도 상관 산점도 ────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🌡️ 온도 vs 결함률 상관**")
        fig_s = px.scatter(recent, x="temperature", y="defect_rate",
                           color="line",
                           color_discrete_map=colors,
                           opacity=0.5, trendline="lowess",
                           labels={"temperature":"압연 온도(℃)","defect_rate":"결함률(%)"})
        plotly_dark_layout(fig_s, height=300)
        st.plotly_chart(fig_s, use_container_width=True)
    with c2:
        st.markdown("**📦 라인별 결함률 분포**")
        fig_b = px.box(recent, x="line", y="defect_rate",
                       color="line", color_discrete_map=colors,
                       labels={"line":"라인","defect_rate":"결함률(%)"})
        plotly_dark_layout(fig_b, height=300)
        st.plotly_chart(fig_b, use_container_width=True)

    # ── 결함 급증 이벤트 목록 ────────────────────────────────────
    st.markdown("---")
    st.markdown("**⚠️ 결함률 급증 이벤트 목록**")
    defect_events_df = recent[recent["anomaly_type"] == "결함률급증"][
        ["timestamp","line","defect_rate","temperature","vibration"]
    ].copy()
    if defect_events_df.empty:
        st.info("선택 기간 내 결함률 급증 이벤트가 없습니다.")
    else:
        defect_events_df.columns = ["시각","라인","결함률(%)","온도(℃)","진동(Hz)"]
        st.dataframe(defect_events_df.sort_values("시각", ascending=False).head(50),
                     use_container_width=True, hide_index=True)

    # ── 일별 결함률 히트맵 ──────────────────────────────────────
    st.markdown("---")
    st.markdown("**📅 일별 결함률 히트맵**")
    recent2 = recent.copy()
    recent2["date"] = recent2["timestamp"].dt.date
    recent2["hour"] = recent2["timestamp"].dt.hour
    pivot_d = recent2.groupby(["date","hour"])["defect_rate"].mean().unstack(fill_value=0)
    if not pivot_d.empty:
        fig_hm = go.Figure(go.Heatmap(
            z=pivot_d.values,
            x=[f"{h}시" for h in pivot_d.columns],
            y=[str(d) for d in pivot_d.index],
            colorscale=[[0,"#0f2d40"],[0.5,"#f59e0b"],[1,"#ef4444"]],
            colorbar=dict(title="결함률(%)"),
        ))
        fig_hm.update_layout(title="날짜 × 시간대 결함률 히트맵 (전 라인 평균)")
        plotly_dark_layout(fig_hm, height=min(400, max(200, len(pivot_d) * 18)))
        st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 메뉴 7: 데이터 수집 (센서 연동 관리)
# ══════════════════════════════════════════════════════════════════
def show_data_collection(df: pd.DataFrame):
    st.markdown('<div class="section-header">🔌 데이터 수집 / 센서 연동 관리</div>',
                unsafe_allow_html=True)

    # ── 센서 현황 테이블 (시뮬) ──────────────────────────────────
    np.random.seed(55)
    sensor_data = []
    sensor_names = {
        "temperature": "열전대(K-Type)",
        "pressure":    "유압 압력 트랜스듀서",
        "speed":       "로터리 인코더",
        "thickness_dev": "레이저 두께 측정기",
        "defect_rate": "표면 비전 카메라",
        "vibration":   "가속도계(MEMS)",
    }
    for line in LINE_CONFIG.keys():
        for var, sensor in sensor_names.items():
            total = len(df[df["line"] == line])
            missing = int(total * np.random.uniform(0.0, 0.005))
            quality = round((1 - missing / max(1, total)) * 100, 2)
            last_upd = df[df["line"] == line]["timestamp"].max()
            status = "정상" if quality >= 99.0 else ("주의" if quality >= 97.0 else "점검필요")
            sensor_data.append({
                "라인": line,
                "변수": VARIABLE_LABELS[var],
                "센서 종류": sensor,
                "총 수집건": total,
                "결측건": missing,
                "데이터 품질(%)": quality,
                "마지막 수신": str(last_upd)[:16] if hasattr(last_upd, "strftime") else str(last_upd)[:16],
                "상태": status,
            })
    sensor_df = pd.DataFrame(sensor_data)

    # ── KPI ──────────────────────────────────────────────────────
    total_sensors = len(sensor_df)
    ok_cnt  = (sensor_df["상태"] == "정상").sum()
    warn_cnt= (sensor_df["상태"] == "주의").sum()
    err_cnt = (sensor_df["상태"] == "점검필요").sum()
    avg_q   = sensor_df["데이터 품질(%)"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#10b981">{ok_cnt}/{total_sensors}</div>
            <div class="kpi-label">정상 센서</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#f59e0b">{warn_cnt}</div>
            <div class="kpi-label">주의 센서</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#ef4444">{err_cnt}</div>
            <div class="kpi-label">점검 필요 센서</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#3b82f6">{avg_q:.2f}%</div>
            <div class="kpi-label">평균 데이터 품질</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 센서 현황 테이블 ─────────────────────────────────────────
    def style_status(val):
        colors = {"정상":  "background-color:#002d1a;color:#34d399",
                  "주의":  "background-color:#3d2d00;color:#fbbf24",
                  "점검필요":"background-color:#3d0000;color:#ff7070"}
        return colors.get(val, "")

    st.markdown("**📡 센서별 수집 현황**")
    st.dataframe(
        sensor_df.style.applymap(style_status, subset=["상태"]),
        use_container_width=True, hide_index=True,
    )

    # ── 데이터 수집 간격 분포 ────────────────────────────────────
    st.markdown("---")
    st.markdown("**⏱️ 수집 간격 분포 (분)**")
    for line in LINE_CONFIG.keys():
        sub = df[df["line"] == line]["timestamp"].sort_values()
        gaps = sub.diff().dt.total_seconds().div(60).dropna()
        if not gaps.empty:
            fig_g = px.histogram(gaps, nbins=20,
                                 title=f"{line} 수집 간격 분포",
                                 labels={"value":"수집 간격(분)"},
                                 color_discrete_sequence=[LINE_CONFIG[line]["color"]])
            plotly_dark_layout(fig_g, height=220)
            st.plotly_chart(fig_g, use_container_width=True)

    # ── 연동 구성 안내 ────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔧 실제 센서 연동 확장 방법 (운영 단계 안내)"):
        st.markdown("""
**1단계 (현재 — 데모)**: 샘플 시계열 데이터로 시스템 검증
**2단계 (파일럿)**: OPC-UA 또는 MQTT → FastAPI 수집 서버 → 실시간 스트리밍
**3단계 (운영)**: PLC/SCADA 직접 연동, InfluxDB/TimescaleDB 시계열 DB 저장
**4단계 (고도화)**: 엣지 컴퓨팅 전처리 + 클라우드 분석 파이프라인

```python
# 예시: MQTT 구독 코드 (운영 단계)
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    db.insert(payload)   # TimescaleDB 저장

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt-broker", 1883)
client.subscribe("factory/line/+/sensor/#")
client.loop_forever()
```
        """)


# ══════════════════════════════════════════════════════════════════
# 메뉴 8: 도움말
# ══════════════════════════════════════════════════════════════════
def show_help():
    st.markdown('<div class="section-header">❓ 도움말</div>', unsafe_allow_html=True)

    help_tabs = st.tabs([
        "라인별 현황", "변수별 추이", "이상탐지 알림",
        "설비별 위험평가", "SPC 관리도", "결함 트렌드",
        "데이터 수집", "시스템 개요",
    ])

    # ── 헬퍼: 3단 도움말 박스 ────────────────────────────────────
    def help_block(usage, criteria, action):
        st.markdown(f"""
<div class="help-box">
  <div class="help-title">📌 활용 방법</div>
  <div class="help-body">{usage}</div>
</div>
<div class="help-box">
  <div class="help-title">📊 판단 기준</div>
  <div class="help-body">{criteria}</div>
</div>
<div class="help-box">
  <div class="help-title">🔧 실무 적용</div>
  <div class="help-body">{action}</div>
</div>
""", unsafe_allow_html=True)

    # ── 탭 1: 라인별 현황 ─────────────────────────────────────────
    with help_tabs[0]:
        help_block(
            usage="""이 화면은 A·B·C 압연라인의 실시간 공정 상태를 한눈에 보여줍니다.
상단 KPI(가동 라인 수, 24h 이상 건수, 이상 비율, 평균 결함률)로 전체 공장 상태를 즉시 파악하고,
각 라인 카드에서 6개 공정변수의 현재값과 정상/주의/이상 배지를 확인합니다.
하단 차트는 라인 간 이상 유형 빈도와 최근 24h 온도 추이를 비교합니다.""",
            criteria="""· <b>배지 기준</b>: 이상(빨강) = UCL/LCL 이탈 / 주의(노랑) = UCL의 90% 초과 / 정상(초록) = 관리 한계 이내<br>
· <b>이상 비율</b>: 24h 기준 5% 미만 정상, 5~20% 주의, 20% 초과 긴급<br>
· <b>온도</b>: A라인 850~950℃, B라인 840~930℃, C라인 870~960℃ 정상 범위<br>
· <b>결함률</b>: A라인 2.0%, B라인 2.5%, C라인 1.8% 이하 유지 목표""",
            action="""1. 매 교대조 시작 시 라인별 현황 화면 확인 → 이상(빨강) 배지 라인 우선 점검<br>
2. 이상 비율 20% 초과 라인 → <b>이상탐지 알림</b> 메뉴로 이동하여 구체적 원인 확인<br>
3. 온도 드리프트 패턴 발견 시 → <b>SPC 관리도</b>로 이동하여 Cpk 점수 확인 후 공정 조정<br>
4. 차트에서 특정 이상 유형이 반복 증가 추세 → <b>설비별 위험평가</b>에서 해당 설비 점검 우선순위 확인""",
        )

    # ── 탭 2: 변수별 추이 ─────────────────────────────────────────
    with help_tabs[1]:
        help_block(
            usage="""특정 라인의 단일 공정 변수(온도, 압력, 속도, 두께편차, 결함률, 진동)를 선택해
시계열 추이와 이상 구간을 상세히 분석하는 화면입니다.
기간(1시간~30일)을 자유롭게 변경하며 분포 히스토그램과 정상/이상 박스플롯도 함께 제공합니다.""",
            criteria="""· <b>빨간 X 마커</b>: 이상 탐지 포인트 (라벨 기준)<br>
· <b>UCL 빨간 점선</b>: 관리 상한 (이탈 시 즉시 확인)<br>
· <b>LCL 노란 점선</b>: 관리 하한 (속도·온도처럼 하한도 중요한 변수에만 표시)<br>
· <b>히스토그램</b>: 분포가 한쪽으로 치우치면 공정 편향(Offset) 발생 징후<br>
· <b>박스플롯</b>: 이상 박스가 정상 박스 대비 크게 이탈 → 이상 변동 폭이 큼""",
            action="""1. <b>기간 1시간</b>: 현재 발생 중인 이상의 원인 변수 실시간 추적<br>
2. <b>기간 24시간</b>: 교대조 단위 공정 안정성 확인 (일일 보고서 참고 자료)<br>
3. <b>기간 7일</b>: 주별 변동 패턴·드리프트 확인 (주간 미팅 준비)<br>
4. <b>기간 30일</b>: 월간 공정 추세·계절성 분석 (월간 공정 개선 회의)<br>
5. 이상 구간 테이블에서 특정 시간대 이상 집중 확인 → 해당 시간대 작업일지와 교차 검토""",
        )

    # ── 탭 3: 이상탐지 알림 ──────────────────────────────────────
    with help_tabs[2]:
        help_block(
            usage="""Z-Score, IQR, Isolation Forest 3가지 탐지 방법 중 선택하거나
통합 라벨 기준으로 이상 이벤트를 목록화합니다.
타임라인 산점도로 언제·어느 라인에서 이상이 집중됐는지 확인하고,
심각도(긴급/경고/주의) 필터로 우선순위를 정할 수 있습니다.""",
            criteria="""· <b>긴급</b>(빨강): 온도스파이크, 결함률급증, 진동이상(베어링) → 즉시 라인 정지 검토<br>
· <b>경고</b>(노랑): 온도드리프트, 압력급변동, 두께공차이탈 → 30분 내 원인 분석<br>
· <b>주의</b>(파랑): 속도이상, 기타 패턴 → 교대 내 확인<br>
· <b>이상 비율 20% 초과</b>: 설비 결함 또는 원자재 이상 가능성<br>
· <b>Isolation Forest</b>: 단일 변수가 아닌 다변량(6개 변수 동시) 복합 이상 탐지에 적합""",
            action="""1. 화면 진입 후 <b>라인 필터</b>로 관심 라인 선택 → <b>기간 24시간</b>으로 최신 현황 파악<br>
2. 타임라인에서 이상 군집 구간 클릭 → 해당 시각 원인 변수 확인<br>
3. 긴급 알림 발생 시: 현장 담당자 즉시 호출 → <b>변수별 추이</b>에서 원인 변수 특정<br>
4. 탐지 방법 변경으로 탐지 방식 간 비교 → '통합'과 'Isolation Forest' 교차 확인으로 복합 이상 확인<br>
5. 알림 목록 엑셀 캡처 후 일일 이상보고서 첨부""",
        )

    # ── 탭 4: 설비별 위험평가 ─────────────────────────────────────
    with help_tabs[3]:
        help_block(
            usage="""압연기, 권취기, 구동모터, 냉각장치, 가열로 5개 설비를 대상으로
각 라인별 위험점수(0~10)를 계산해 히트맵으로 표시합니다.
이상 비율, 진동 수준, 결함률, 온도 이탈을 종합한 복합 점수이며,
일별 추이 차트로 위험 변화 흐름을 파악할 수 있습니다.""",
            criteria="""· <b>HIGH (7 이상)</b>: 즉시 점검 및 예방 정비 — 생산 중단 위험<br>
· <b>MEDIUM (4~7)</b>: 계획 정비 주기 앞당기기 검토<br>
· <b>LOW (4 미만)</b>: 정기 점검 유지<br>
· <b>히트맵 색상</b>: 빨강(위험) → 노랑(주의) → 초록(안전)<br>
· 위험점수 = 이상비율×3.5 + 진동비율×2.5 + 결함률비율×2.5 + 온도이탈비율×1.5""",
            action="""1. <b>최근 24h 기준</b>: 현재 진행 중인 위험 파악 → 즉시 정비 대상 식별<br>
2. <b>7일 기준</b>: 주별 설비 상태 추이 → 주간 정비 계획 수립<br>
3. <b>30일 기준</b>: 월간 설비 건전성 평가 → 연간 정비 예산 근거 자료<br>
4. HIGH 등급 설비 목록 출력 → 설비팀에 즉시 공유 및 조치계획 수립<br>
5. 일별 추이에서 특정 설비 위험점수 상승 패턴 → 예측 정비(PdM) 스케줄 조정""",
        )

    # ── 탭 5: SPC 관리도 ─────────────────────────────────────────
    with help_tabs[4]:
        help_block(
            usage="""X-bar(평균) 관리도와 R(범위) 관리도를 표시하여 공정의 통계적 안정성을 판정합니다.
Cp/Cpk 공정능력지수로 규격 대비 공정 여유를 수치화하며,
서브그룹 크기(n=3~10) 변경으로 다양한 분석 조건을 적용할 수 있습니다.""",
            criteria="""· <b>Cpk ≥ 1.67</b>: 매우 우수 (6σ 수준, 불량률 0.1ppm 이하)<br>
· <b>1.33 ≤ Cpk < 1.67</b>: 우수 (63ppm 이하)<br>
· <b>1.00 ≤ Cpk < 1.33</b>: 보통 (최소 허용 수준, 2,700ppm)<br>
· <b>Cpk < 1.00</b>: 불량 (즉시 공정 개선 필요)<br>
· <b>OOC(Out of Control)</b>: 관리 한계선 이탈 포인트 → 특수 원인 조사<br>
· <b>8점 연속 중심선 한쪽</b>: 드리프트/편향 징후 (Nelson 규칙 적용 권장)""",
            action="""1. 변수 선택 → X-bar 차트에서 OOC 포인트 확인 → 해당 시간대 <b>변수별 추이</b>에서 원인 분석<br>
2. Cpk < 1.00: 공정 파라미터 조정(온도 설정값, 압력 조정) 후 재측정<br>
3. R 관리도 OOC: 변동 원인 분석 (작업자 변경, 원자재 로트 변경, 설비 마모)<br>
4. <b>월 단위</b>: 전월 대비 Cpk 추이 → 공정 개선 효과 검증<br>
5. <b>분기 단위</b>: 라인 간 Cpk 비교 → 표준화 작업 우선순위 결정<br>
6. <b>연간</b>: 연간 Cpk 목표 달성 여부 평가 → 다음 연도 품질 목표 수립""",
        )

    # ── 탭 6: 결함 트렌드 ─────────────────────────────────────────
    with help_tabs[5]:
        help_block(
            usage="""표면 결함률의 시계열 추이, 라인 간 비교, 온도와의 상관관계,
시간대별 발생 히트맵을 종합 제공합니다.
결함률 급증 이벤트 목록으로 고위험 구간을 빠르게 식별합니다.""",
            criteria="""· <b>결함률 한계</b>: A라인 2.0%, B라인 2.5%, C라인 1.8% 초과 → 즉시 대응<br>
· <b>히트맵</b>: 빨간 구간(높은 결함률) 시간대 → 조업 조건 재검토<br>
· <b>온도 상관 산점도</b>: 온도 편차와 결함률의 양의 상관 → 가열로 제어 정밀도 점검<br>
· <b>24h 기준</b>: 당일 품질 관리 (QC 리포트)<br>
· <b>7일 기준</b>: 주간 품질 경향 분석<br>
· <b>30일 기준</b>: 월간 품질 성과 평가 (고객사 보고)""",
            action="""1. 교대조 종료 전 24h 기준 결함 트렌드 확인 → QC 일일 보고서 작성<br>
2. 결함률 급증 이벤트 발생 시: 해당 라인의 <b>이상탐지 알림</b>에서 동시 발생 이상 확인<br>
3. 히트맵에서 특정 시간대(예: 오전 2~4시) 결함 집중 → 해당 시간대 작업자/원자재 이력 확인<br>
4. 온도-결함률 상관이 강한 라인 → 가열로 PID 파라미터 조정 검토<br>
5. <b>월간</b>: 전월 대비 결함률 개선/악화 원인 분석 → 공정 개선 보고서 작성""",
        )

    # ── 탭 7: 데이터 수집 ─────────────────────────────────────────
    with help_tabs[6]:
        help_block(
            usage="""각 라인·센서별 데이터 수집 현황(수집건수, 결측건, 품질%)을 모니터링합니다.
수집 간격 분포 차트로 데이터 연속성을 확인하고,
센서 이상이나 통신 장애를 조기에 감지합니다.""",
            criteria="""· <b>데이터 품질 99% 이상</b>: 정상<br>
· <b>97~99%</b>: 주의 → 결측 원인 파악<br>
· <b>97% 미만</b>: 점검 필요 → 센서 교체 또는 통신 점검<br>
· <b>수집 간격 분포</b>: 10분 간격(설정값) 이외 긴 간격 → 연결 단절 이력 확인<br>
· <b>마지막 수신 시각</b>이 10분 이상 경과 → 통신 장애 의심""",
            action="""1. 매일 1회 데이터 수집 화면 점검 → '점검필요' 센서 즉시 현장 확인<br>
2. 결측건 급증 센서 → 배선·커넥터 불량 또는 PLC 통신 오류 점검<br>
3. 수집 간격 분포에서 장시간 공백 구간 → 해당 시간 데이터는 분석에서 제외하고 보완 계획 수립<br>
4. 운영 단계 전환 시: OPC-UA/MQTT 연동 구성 후 실시간 데이터로 교체<br>
5. 품질 임계치 미달 센서 현황을 주간 설비 미팅 보고서에 포함""",
        )

    # ── 탭 8: 시스템 개요 ─────────────────────────────────────────
    with help_tabs[7]:
        st.markdown("""
<div class="pm-card cyan">
<b style="font-size:16px;color:#ffffff">공정제어 상시모니터링 시스템 v1.0</b><br>
<span style="color:#7a9cc0">철강/금속 압연·코일 생산라인 이상탐지 및 SPC 관리 플랫폼</span>
</div>
""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
**📊 탑재 분석 방법**
| 방법 | 설명 |
|------|------|
| Z-Score | 단변량 통계 이상치 (μ±3σ 기준) |
| IQR | 사분위 범위 기반 이상치 (Q1-1.5×IQR ~ Q3+1.5×IQR) |
| Isolation Forest | 다변량 복합 이상 (ML, 6개 변수 동시 적용) |
| X-bar 관리도 | 서브그룹 평균 추이 모니터링 |
| R 관리도 | 서브그룹 범위 변동 모니터링 |
| Cp / Cpk | 공정능력지수 (규격 대비 공정 여유) |

**🏭 적용 공정 변수**
- 압연 온도(℃) · 압연 압력(ton) · 라인 속도(m/min)
- 코일 두께 편차(mm) · 표면 결함률(%) · 진동(Hz)
            """)
        with c2:
            st.markdown("""
**🔧 시스템 구성**
```
process_monitoring/
├── app.py              # Streamlit 대시보드
├── data_generator.py   # 샘플 데이터 생성
├── anomaly_engine.py   # 이상탐지 엔진
├── requirements.txt    # 패키지 의존성
└── .streamlit/
    └── config.toml     # 배포 설정
```

**🚀 확장 로드맵**
1. **데모** (현재): 샘플 데이터, Streamlit Cloud 배포
2. **파일럿**: OPC-UA/MQTT 실시간 연동
3. **운영**: PLC 직접 연동, 알림(이메일/SMS)
4. **고도화**: 예측 정비(PdM), AI 불량 원인 분류

**📧 문의**: 시스템 연동·커스터마이징 필요 시 관리자 문의
            """)


# ══════════════════════════════════════════════════════════════════
# 사이드바 & 라우팅
# ══════════════════════════════════════════════════════════════════
def main():
    df     = load_data()
    engine = get_engine()

    # ── 사이드바 ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:20px 0 10px 0">
  <div style="font-size:36px">🏭</div>
  <div style="font-size:15px;font-weight:700;color:#ffffff;margin-top:6px">
    공정제어 상시모니터링
  </div>
  <div style="font-size:11px;color:#4a6a8a;margin-top:2px">
    철강/금속 압연·코일 라인 v1.0
  </div>
</div>
<hr style="border-color:#1e3a5c;margin:8px 0 16px 0">
""", unsafe_allow_html=True)

        # 데이터 기간 표시
        data_start = df["timestamp"].min().strftime("%Y-%m-%d")
        data_end   = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"""
<div style="background:#0a1628;border:1px solid #1e3a5c;border-radius:8px;padding:10px 14px;margin-bottom:16px">
  <div style="font-size:10px;color:#4a6a8a;margin-bottom:2px">데이터 기간</div>
  <div style="font-size:11px;color:#93b4d0">{data_start}</div>
  <div style="font-size:11px;color:#93b4d0">~ {data_end}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">메뉴</div>', unsafe_allow_html=True)
        menu = st.radio("", [
            "🏭 라인별 현황",
            "📈 변수별 추이",
            "🚨 이상탐지 알림",
            "⚠️ 설비별 위험평가",
            "📐 SPC 관리도",
            "🔍 결함 트렌드",
            "🔌 데이터 수집",
            "❓ 도움말",
        ], label_visibility="collapsed")

        st.markdown("---")

        # 라인별 최신 상태 요약
        st.markdown('<div class="sidebar-title">라인 상태</div>', unsafe_allow_html=True)
        snap = get_latest_snapshot(df)
        for _, row in snap.iterrows():
            line = row["line"]
            cfg  = LINE_CONFIG[line]
            is_anom = (row["temperature"] > cfg["temp_ucl"] or
                       row["temperature"] < cfg["temp_lcl"] or
                       row["vibration"] > cfg["vibration_ucl"] or
                       row["defect_rate"] > cfg["defect_limit"])
            dot = "🔴" if is_anom else "🟢"
            st.markdown(
                f"{dot} **{line}** — T:{row['temperature']:.0f}℃ "
                f"/ 결함:{row['defect_rate']:.2f}%"
            )

        st.markdown("---")
        st.markdown(f"""<div style="font-size:10px;color:#3a5a7a;text-align:center">
            총 {len(df):,}건 · 3개 라인<br>
            이상 {df['is_anomaly'].sum():,}건 ({df['is_anomaly'].mean()*100:.1f}%)
        </div>""", unsafe_allow_html=True)

    # ── 메뉴 라우팅 ──────────────────────────────────────────────
    if   "라인별 현황"    in menu: show_line_status(df)
    elif "변수별 추이"    in menu: show_variable_trends(df)
    elif "이상탐지 알림"  in menu: show_anomaly_alerts(df, engine)
    elif "설비별 위험평가" in menu: show_equipment_risk(df)
    elif "SPC 관리도"     in menu: show_spc_charts(df, engine)
    elif "결함 트렌드"    in menu: show_defect_trends(df)
    elif "데이터 수집"    in menu: show_data_collection(df)
    elif "도움말"         in menu: show_help()


if __name__ == "__main__":
    main()
