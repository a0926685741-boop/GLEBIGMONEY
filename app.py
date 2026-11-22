
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import SECTOR_POOL, DEFAULT_POOL, SECTOR_MAP, fetch_stock_data, fetch_chip_data, run_hmm_model, resample_ohlcv

st.set_page_config(page_title="台股 HMM 量化戰情室 v2.0", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container { padding: 1rem 1rem 2rem 1rem; }</style>", unsafe_allow_html=True)

st.title("📊 台股 HMM 量化多宮格戰情室 (v2.0 專屬股票池版)")
c1, c2, c3 = st.columns((2, 1.5, 1.5))
with c1: pool_choice = st.selectbox("📌 從專屬股票池選取 (共約100檔)", list(DEFAULT_POOL.keys()), format_func=lambda x: DEFAULT_POOL.get(x, x))
with c2: custom_code = st.text_input("🔍 或直接輸入台股代號 (例: 2887)", value="")
with c3: timeframe = st.radio("⏱️ K線週期切換", ("日線 (1D)", "週線 (1W)", "月線 (1M)"), horizontal=True)

target_ticker = f"{custom_code.strip()}.TW" if custom_code.strip() else pool_choice
ticker_num = target_ticker.split(".").pop(0)
raw_df = fetch_stock_data(target_ticker)
if raw_df.empty and custom_code.strip():
    target_ticker = f"{custom_code.strip()}.TWO"
    raw_df = fetch_stock_data(target_ticker)
if raw_df.empty:
    st.error(f"找不到 {target_ticker} 資料")
    st.stop()

chip_df, _ = fetch_chip_data(ticker_num, raw_df.index)
full_df = pd.concat((raw_df, chip_df), axis=1).dropna()
full_df = pd.concat((full_df, run_hmm_model(full_df)), axis=1)
plot_df = resample_ohlcv(full_df, timeframe).tail(120)

st.subheader(f"🚦 {DEFAULT_POOL.get(target_ticker, target_ticker)}｜HMM 隱藏狀態與進場雷達")
latest, prev_5 = full_df.iloc[-1], full_df.iloc[-6:-1]
s0, s1, s2 = latest['State_0_Base'], latest['State_1_Bull'], latest['State_2_Bear']
was_bear = prev_5['State_2_Bear'].max() > 60

if s1 > 70 and (was_bear or prev_5['State_0_Base'].max() > 70):
    sig, desc = "🟢 多頭啟動買進訊號 (狀態 1 > 70%)", "符合由空/盤整轉強條件，外資賣壓消退、波動結構轉多。"
elif s0 > 70 and was_bear:
    sig, desc = "🟡 跌勢趨緩・盤整打底中 (狀態 0 > 70%)", "已脫離狀態 2 空頭急跌，進入築底觀察期，等待狀態 1 突破 70%。"
elif s2 > 70:
    sig, desc = "🔴 空頭高風險期 (狀態 2 > 70%)・嚴禁接刀", "處於高波動下跌慣性中，建議空手觀望。"
else:
    sig, desc = "⚪ 狀態過渡觀察期", "各狀態機率尚未集中突破 70% 門檻，持續追蹤變化。"

m1, m2, m3, m4 = st.columns(4)
m1.metric("狀態 1：多頭啟動", f"{s1:.1f}%")
m2.metric("狀態 0：盤整打底", f"{s0:.1f}%")
m3.metric("狀態 2：空頭風險", f"{s2:.1f}%")
m4.metric("最新收盤價", f"{latest['Close']:.2f}", f"{latest['Return']*100:.2f}%")
st.info(f"**系統判讀：{sig}** — {desc}")

g1, g2 = st.columns(2)
with g1:
    st.markdown(f"**📈 技術線型與布林頻寬 ({timeframe})**")
    ft = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=(0.55, 0.2, 0.25))
    ft.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="K線", increasing_line_color='#EF5350', decreasing_line_color='#26A69A'), row=1, col=1)
    for ma, col in ((10, '#FFD54F'), (20, '#4FC3F7'), (40, '#BA68C8'), (60, '#FF8A65')):
        ft.add_trace(go.Scatter(x=plot_df.index, y=plot_df[f'MA{ma}'], name=f'MA{ma}', line=dict(width=1.2, color=col)), row=1, col=1)
    ft.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Up'], name='布林上軌', line=dict(width=1, dash='dot', color='gray')), row=1, col=1)
    ft.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Low'], name='布林下軌', line=dict(width=1, dash='dot', color='gray')), row=1, col=1)
    vc = list('#EF5350' if c >= o else '#26A69A' for c, o in zip(plot_df['Close'], plot_df['Open']))
    ft.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume']/1000, name='成交量(張)', marker_color=vc), row=2, col=1)
    ft.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Width'], name='布林頻寬', line=dict(color='#29B6F6', width=1.5), fill='tozeroy'), row=3, col=1)
    ft.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.05))
    st.plotly_chart(ft, use_container_width=True, config={'scrollZoom': True})

with g2:
    st.markdown("**🏦 籌碼追蹤 (近25日外資/投信 + 融資 + 千張大戶)**")
    cv = full_df.tail(60)
    fc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=(0.4, 0.3, 0.3))
    fc.add_trace(go.Bar(x=cv.index, y=cv['Foreign_Net'], name='外資', marker_color='#42A5F5'), row=1, col=1)
    fc.add_trace(go.Bar(x=cv.index, y=cv['Trust_Net'], name='投信', marker_color='#AB47BC'), row=1, col=1)
    if len(cv) >= 25: fc.add_vrect(x0=cv.index[-25], x1=cv.index[-1], fillcolor="yellow", opacity=0.1, line_width=0, row=1, col=1)
    fc.add_trace(go.Scatter(x=cv.index, y=cv['Margin_Bal'], name='融資餘額', line=dict(color='#FFA726', width=2)), row=2, col=1)
    fc.add_trace(go.Scatter(x=cv.index, y=cv['Major_1000_Pct'], name='千張大戶(%)', line=dict(color='#EF5350', width=2, shape='hv')), row=3, col=1)
    fc.update_layout(height=560, barmode='relative', margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fc, use_container_width=True, config={'scrollZoom': True})

st.divider()
b1, b2 = st.columns((1, 1.6))
with b1:
    st.subheader("🔥 六大族群今日強弱排行")
    sp = list()
    for sn, tks in SECTOR_MAP.items():
        rets = list(fetch_stock_data(t, "1mo")['Return'].iloc[-1]*100 for t in tks if not fetch_stock_data(t, "1mo").empty)
        sp.append({"族群分類": sn, "平均漲跌幅(%)": round(float(np.mean(rets)) if rets else 0.0, 2), "代表股": ", ".join(list(x.split('.').pop(0) for x in tks))})
    st.dataframe(pd.DataFrame(sp).sort_values("平均漲跌幅(%)", ascending=False), use_container_width=True, hide_index=True)

with b2:
    st.subheader("🎯 專屬股票池・分族群 HMM 狀態雷達")
    scan_sec = st.selectbox("選擇要掃描的族群", list(SECTOR_POOL.keys()))
    if st.button(f"🔄 掃描「{scan_sec}」全部成分股"):
        rows = list()
        target_dict = SECTOR_POOL.get(scan_sec, {})
        prog = st.progress(0)
        total = len(target_dict)
        for idx, (tk, lb) in enumerate(target_dict.items()):
            d = fetch_stock_data(tk)
            if not d.empty:
                cd, _ = fetch_chip_data(tk.split('.').pop(0), d.index)
                md = pd.concat((d, cd), axis=1).dropna()
                lp = run_hmm_model(md).iloc[-1]
                bb_low = "✅" if (md['BB_Width'].iloc[-1] <= md['BB_Width'].tail(120).quantile(0.2)) else "-"
                stt = "🟢 多頭啟動" if lp['State_1_Bull'] > 70 else ("🟡 盤整打底" if lp['State_0_Base'] > 70 else ("🔴 空頭風險" if lp['State_2_Bear'] > 70 else "⚪ 過渡期"))
                rows.append({"股票": lb, "收盤": round(md['Close'].iloc[-1], 2), "狀態": stt, "多頭%": f"{lp['State_1_Bull']:.0f}%", "盤整%": f"{lp['State_0_Base']:.0f}%", "25日外資": int(md['Foreign_Net'].tail(25).sum()), "25日投信": int(md['Trust_Net'].tail(25).sum()), "布林壓縮": bb_low})
            prog.progress((idx + 1) / total)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
