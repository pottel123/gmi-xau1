
import os
import io
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="GMI-XAU • Gold Daily Macro Index", layout="wide")

st.title("GMI-XAU • Gold Daily Macro Index")
st.caption("Índice diário que resume drivers macro para ouro (XAU) e estima a % de alta/baixa do dia. Agora com calendário de eventos e 'risk flag'.")

# ---------------------
# Settings
# ---------------------
LOOKBACK_YEARS = st.sidebar.slider("Histórico para calibrar (anos)", 2, 10, 5, 1)
USE_BREAKEVEN = st.sidebar.checkbox("Usar breakeven de inflação (T10YIE) se disponível", value=True)
TARGET_TICKER = st.sidebar.selectbox("Série alvo (ouro)", ["XAUUSD=X (spot)", "GC=F (futuro)"], index=0)
TARGET = "XAUUSD=X" if TARGET_TICKER.startswith("XAUUSD") else "GC=F"

st.sidebar.markdown("---")
st.sidebar.markdown("### Modos extras")
ENABLE_EVENTS = st.sidebar.checkbox("Mostrar calendário de eventos (14 dias)", value=True)
ENABLE_RISK_FLAG = st.sidebar.checkbox("Ativar 'Market Stress Flag'", value=True)

# Pesos base (informativo)
weights = {
    "Dollar (DXY)": -0.35,
    "Rates (10Y real proxy)": -0.35,
    "Risk (S&P500)": -0.12,
    "Risk (VIX)": 0.08,
    "Growth (Copper)": -0.10,
}
st.sidebar.markdown("### Pesos (intuídos)")
for k,v in weights.items():
    st.sidebar.write(f"{k}: {v:+.2f}")
st.sidebar.info("Obs.: O sinal final é obtido via regressão linear, não apenas pelos pesos.")

# ---------------------
# Economic events calendar (US) - lightweight, offline-friendly
# ---------------------
# Hardcoded schedules for 2025 (BRT-aware times are not applied; display local)
FOMC_2025 = [
    "2025-01-29", "2025-03-19", "2025-04-30", "2025-06-11",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]

# BLS CPI 2025 planned (approx official schedule)
CPI_2025 = [
    "2025-01-14","2025-02-13","2025-03-12","2025-04-10",
    "2025-05-14","2025-06-11","2025-07-10","2025-08-13",
    "2025-09-10","2025-10-15","2025-11-12","2025-12-10"
]

# BEA PCE 2025 planned (typical end-of-month pattern; approximate)
PCE_2025 = [
    "2025-01-31","2025-02-28","2025-03-28","2025-04-30",
    "2025-05-30","2025-06-27","2025-07-31","2025-08-29",
    "2025-09-26","2025-10-30","2025-11-26","2025-12-24"
]

def first_friday(year, month):
    d = dt.date(year, month, 1)
    while d.weekday() != 4:  # 0=Mon ... 4=Fri
        d += dt.timedelta(days=1)
    return d

def nfp_schedule(year=2025):
    return [first_friday(year, m).isoformat() for m in range(1,13)]

def economic_events_window(center_date=None, lookahead_days=14):
    if center_date is None:
        center_date = dt.date.today()
    start = center_date
    end = center_date + dt.timedelta(days=lookahead_days)
    rows = []
    # FOMC
    for s in FOMC_2025:
        d = dt.date.fromisoformat(s)
        if start <= d <= end:
            rows.append({"date": d, "event": "FOMC Rate Decision", "time_ET": "14:00", "note": "Dot plot quando trimestral"})
    # NFP
    for s in nfp_schedule(2025):
        d = dt.date.fromisoformat(s)
        if start <= d <= end:
            rows.append({"date": d, "event": "US Nonfarm Payrolls (NFP)", "time_ET": "08:30", "note": "Emprego"})
    # CPI
    for s in CPI_2025:
        d = dt.date.fromisoformat(s)
        if start <= d <= end:
            rows.append({"date": d, "event": "US CPI", "time_ET": "08:30", "note": "Inflação ao consumidor"})
    # PCE
    for s in PCE_2025:
        d = dt.date.fromisoformat(s)
        if start <= d <= end:
            rows.append({"date": d, "event": "US PCE", "time_ET": "08:30", "note": "Medida preferida do Fed"})
    if not rows:
        return pd.DataFrame(columns=["date","event","time_ET","note"])
    out = pd.DataFrame(rows).sort_values("date")
    return out

# ---------------------
# Data fetch helpers
# ---------------------
def fetch_hist(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

# Map for tickers
tickers = {
    "GOLD": TARGET,
    "DXY": "^DXY",
    "US10Y": "^TNX",
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "COPPER": "HG=F",
}

# Optional breakeven
if USE_BREAKEVEN:
    tickers["BE10Y"] = "T10YIE"

years = LOOKBACK_YEARS
start_date = (dt.date.today() - dt.timedelta(days=365*years+30)).isoformat()

with st.spinner("Baixando séries históricas..."):
    panel = fetch_hist(list(tickers.values()), start=start_date)
    panel.columns = list(tickers.keys())

df = panel.copy()

# Normalize ^TNX
if "US10Y" in df.columns:
    df["US10Y"] = df["US10Y"] / 100.0

rets = df.pct_change().dropna()

# Real yield proxy
if "BE10Y" in df.columns and df["BE10Y"].notna().sum() > 30:
    be = df["BE10Y"] / 100.0
    real_yield = df["US10Y"] - be
else:
    real_yield = df["US10Y"]

real_chg = real_yield.diff()

# Factors
factors = pd.DataFrame(index=rets.index)
factors["DXY_ret"] = rets["DXY"]
factors["US10Y_dchg"] = real_chg.loc[rets.index]
factors["SPX_ret"] = rets["SPX"]
factors["VIX_ret"] = rets["VIX"]
factors["COPPER_ret"] = rets["COPPER"]

gold_rets = rets["GOLD"].rename("GOLD_ret")

data = pd.concat([gold_rets, factors], axis=1).dropna()

col1, col2 = st.columns([2,1], gap="large")

with col1:
    st.subheader("Séries & Fatores")
    st.write("As séries abaixo são baixadas do Yahoo Finance via `yfinance`.")
    st.dataframe(data.tail(10))

with col2:
    if ENABLE_EVENTS:
        st.subheader("Próximos eventos (14 dias)")
        ev = economic_events_window()
        if ev.empty:
            st.write("Sem eventos-chave na janela.")
        else:
            st.table(ev)

# ---------------------
# Modeling
# ---------------------
st.subheader("Modelo de Sinal Diário")
st.caption("Regressão linear (OLS) de GOLD_ret contra fatores (janela móvel até o dia anterior).")

if data.shape[0] < 100:
    st.error("Histórico insuficiente para calibrar (precisa de ~100 dias com dados). Tente aumentar o lookback.")
    st.stop()

X = data[["DXY_ret","US10Y_dchg","SPX_ret","VIX_ret","COPPER_ret"]].iloc[:-1]
y = data["GOLD_ret"].iloc[:-1]

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

model = LinearRegression()
model.fit(X, y)

coefs = pd.Series(model.coef_, index=X.columns)
st.markdown("**Coeficientes estimados (OLS):**")
st.write(coefs.to_frame("beta"))

# Today's prediction
x0 = data[["DXY_ret","US10Y_dchg","SPX_ret","VIX_ret","COPPER_ret"]].iloc[-1:].copy()
signal_available = True
if x0.isna().any(axis=None):
    st.warning("Há NAs nos fatores do dia mais recente; sinal pode não estar disponível hoje.")
    signal_available = False

pred_pct = None
direction = None
if signal_available:
    pred = float(model.predict(x0)[0])
    pred_pct = 100*pred
    direction = "Alta" if pred_pct >= 0 else "Baixa"
    st.metric("Sinal diário previsto para o ouro", f"{direction} {pred_pct:+.2f}%")
    st.progress(min(max((pred_pct/2.5 + 0.5), 0), 1.0))

# ---------------------
# Market Stress Flag
# ---------------------
risk_score = None
risk_label = None
risk_details = {}
if ENABLE_RISK_FLAG:
    st.subheader("Market Stress Flag")
    # Use latest factor returns to form a quick stress score
    # thresholds chosen heuristically
    latest = data.iloc[-1]
    score = 0
    details = {}

    # VIX surge
    if latest["VIX_ret"] > 0.10:
        score += 2; details["VIX"] = "surge (>+10%)"
    elif latest["VIX_ret"] > 0.05:
        score += 1; details["VIX"] = "up (>+5%)"

    # SPX drop
    if latest["SPX_ret"] < -0.01:
        score += 1; details["SPX"] = "selloff (<-1%)"
    if latest["SPX_ret"] < -0.02:
        score += 1; details["SPX+"] = "hard selloff (<-2%)"

    # Dollar spike
    if latest["DXY_ret"] > 0.005:
        score += 1; details["DXY"] = "spike (>+0.5%)"

    # Rates move (absolute real change)
    if abs(latest["US10Y_dchg"]) > 0.0005:  # >5 bps
        score += 1; details["Real10y"] = "shock (>5bps abs)"

    # Copper drop (growth proxy)
    if latest["COPPER_ret"] < -0.01:
        score += 1; details["Copper"] = "drop (<-1%)"

    labels = {0:"Low",1:"Low",2:"Moderate",3:"Moderate",4:"High",5:"High",6:"Severe"}
    risk_score = score
    risk_label = labels.get(score, "Severe")
    risk_details = details

    st.metric("Stress (0-6)", f"{risk_score} • {risk_label}")
    if details:
        st.write("Contribuições:", ", ".join([f"{k}: {v}" for k,v in details.items()]))
    st.caption("Heurística baseada em movimentos do dia. Útil como proxy de 'risk-off'.")

# ---------------------
# Backtest informativo
# ---------------------
st.subheader("Backtest informativo")
bt = data.copy()
bt["pred"] = np.nan

X_all = data[["DXY_ret","US10Y_dchg","SPX_ret","VIX_ret","COPPER_ret"]]
y_all = data["GOLD_ret"]
for i in range(120, len(data)):
    Xi = X_all.iloc[:i]
    yi = y_all.iloc[:i]
    if Xi.notna().all(axis=1).sum() < 100:
        continue
    m = LinearRegression().fit(Xi, yi)
    bt.iloc[i, bt.columns.get_loc("pred")] = float(m.predict(X_all.iloc[i:i+1])[0])

bt["signal_ret"] = bt["pred"]
bt["gold_ret"] = bt["GOLD_ret"]
bt = bt.dropna()

if not bt.empty:
    gold_curve = (1 + bt["gold_ret"]).cumprod()
    sig_curve = (1 + bt["signal_ret"]).cumprod()
    st.line_chart(pd.DataFrame({"Ouro": gold_curve, "Sinal (modelo)": sig_curve}))

st.caption("Aviso: ferramenta educacional. Não é recomendação. Riscos de mercado existem.")

# ---------------------
# Reference levels
# ---------------------
st.subheader("Níveis de referência (último fechamento)")
latest_levels = df.iloc[-1][["GOLD","DXY","US10Y","SPX","VIX","COPPER"]].dropna().to_frame("Último")
st.table(latest_levels)

# ---------------------
# CSV Export
# ---------------------
st.subheader("Exportar resultado do dia")
today = dt.date.today().isoformat()
export = {
    "date": today,
    "target": TARGET,
    "signal_pct": None if pred_pct is None else round(pred_pct, 4),
    "signal_direction": direction,
    "risk_score": risk_score,
    "risk_label": risk_label,
    "factors": {
        "DXY_ret": None if signal_available is False else round(float(x0["DXY_ret"].values[0]), 6),
        "US10Y_dchg": None if signal_available is False else round(float(x0["US10Y_dchg"].values[0]), 6),
        "SPX_ret": None if signal_available is False else round(float(x0["SPX_ret"].values[0]), 6),
        "VIX_ret": None if signal_available is False else round(float(x0["VIX_ret"].values[0]), 6),
        "COPPER_ret": None if signal_available is False else round(float(x0["COPPER_ret"].values[0]), 6),
    },
}

csv_df = pd.json_normalize(export, sep="_")
st.dataframe(csv_df)

csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV do sinal de hoje", data=csv_bytes, file_name=f"gmi_xau_{today}.csv", mime="text/csv")

st.markdown("---")
st.caption("Calendário de eventos é aproximado para 2025 (FOMC, CPI, PCE e NFP). Idealmente, conecte uma API oficial (TradingEconomics, FMP, etc.) se quiser dados 'reais-time' de calendário e consenso/atual.")
