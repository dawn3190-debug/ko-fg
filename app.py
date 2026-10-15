"""
KOSPI Fear & Greed 대시보드 (Streamlit)
로컬 실행:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import FinanceDataReader as fdr

st.set_page_config(page_title="KOSPI Fear & Greed", page_icon="📈", layout="wide")

START = "2021-01-01"
WINDOW = 252

NAMES = {
    "momentum": "모멘텀 (125일선 괴리)",
    "volatility": "변동성 (역)",
    "rsi": "RSI(14)",
    "foreign_flow": "외국인 20일 순매수",
    "fx": "원/달러 환율 (역)",
}


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def pct_rank(s, window=WINDOW):
    return s.rolling(window).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)


@st.cache_data(ttl=60 * 60)  # 1시간마다 새로 받아옴
def build():
    kospi = fdr.DataReader("KS11", START)["Close"]
    comp = pd.DataFrame(index=kospi.index)
    comp["momentum"] = pct_rank(kospi / kospi.rolling(125).mean() - 1)
    vol = kospi.pct_change().rolling(20).std() * np.sqrt(252)
    comp["volatility"] = 100 - pct_rank(vol)
    comp["rsi"] = pct_rank(rsi(kospi))

    warn = None
    try:
        from pykrx import stock
        flow = stock.get_market_trading_value_by_date(
            START.replace("-", ""), pd.Timestamp.today().strftime("%Y%m%d"), "KOSPI"
        )["외국인합계"].reindex(kospi.index)
        comp["foreign_flow"] = pct_rank(flow.rolling(20).sum())
    except Exception as ex:
        warn = f"외국인 수급 데이터를 받지 못해 제외했습니다 ({ex})"

    fx = fdr.DataReader("USD/KRW", START)["Close"].reindex(kospi.index).ffill()
    comp["fx"] = 100 - pct_rank(fx.pct_change(20))

    comp["fear_greed"] = comp.mean(axis=1, skipna=True)
    comp["kospi"] = kospi
    return comp.dropna(subset=["fear_greed"]), warn


def label(v):
    if v < 25: return "극단적 공포", "#1f6feb"
    if v < 45: return "공포", "#58a6ff"
    if v <= 55: return "중립", "#8b949e"
    if v <= 75: return "탐욕", "#f0883e"
    return "극단적 탐욕", "#da3633"


df, warn = build()
last, prev = df.iloc[-1], df.iloc[-2]
text, color = label(last["fear_greed"])

st.title("KOSPI Fear & Greed Oscillator")
st.caption(f"기준일 {df.index[-1]:%Y-%m-%d} · 데이터는 1시간마다 갱신")
if warn:
    st.warning(warn)

c1, c2 = st.columns([1, 2])
with c1:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(last["fear_greed"], 1),
        delta={"reference": round(prev["fear_greed"], 1)},
        title={"text": text},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#dbe9ff"},
                {"range": [25, 45], "color": "#eef4ff"},
                {"range": [45, 55], "color": "#f2f2f2"},
                {"range": [55, 75], "color": "#fff1e6"},
                {"range": [75, 100], "color": "#ffe0e0"},
            ],
        },
    ))
    gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)
    st.metric("KOSPI", f"{last['kospi']:,.2f}",
              f"{(last['kospi'] / prev['kospi'] - 1) * 100:+.2f}%")

with c2:
    period = st.radio("기간", ["3개월", "6개월", "1년", "3년"], index=2, horizontal=True)
    days = {"3개월": 92, "6개월": 183, "1년": 365, "3년": 1095}[period]
    r = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=25, fillcolor="#1f6feb", opacity=0.08, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#da3633", opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=r.index, y=r["fear_greed"], name="Fear & Greed",
                             line=dict(color="#da3633")))
    fig.add_trace(go.Scatter(x=r.index, y=r["kospi"], name="KOSPI", yaxis="y2",
                             line=dict(color="gray"), opacity=0.6))
    fig.update_layout(
        height=380, margin=dict(t=20, b=20, l=10, r=10),
        yaxis=dict(range=[0, 100], title="F&G"),
        yaxis2=dict(overlaying="y", side="right", title="KOSPI", showgrid=False),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("구성 지표 (0 = 공포, 100 = 탐욕)")
parts = [k for k in NAMES if k in df.columns]
bar = go.Figure(go.Bar(
    x=[last[k] for k in parts], y=[NAMES[k] for k in parts], orientation="h",
    marker_color=[label(last[k])[1] for k in parts],
    text=[f"{last[k]:.0f}" for k in parts], textposition="outside",
))
bar.update_layout(height=280, xaxis=dict(range=[0, 105]), margin=dict(t=10, b=10, l=10, r=30))
st.plotly_chart(bar, use_container_width=True)

with st.expander("계산 방법"):
    st.markdown(
        "각 지표가 최근 252거래일 분포에서 차지하는 백분위를 0~100점으로 환산한 뒤 평균합니다. "
        "변동성과 환율은 높을수록 공포이므로 방향을 뒤집었습니다."
    )
