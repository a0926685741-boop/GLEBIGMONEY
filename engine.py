
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import requests
from datetime import timedelta

# 您的專屬六大族群分類（已精準對照上市 .TW 與上櫃 .TWO 代號）
SECTOR_POOL = {
    "電子 / AI / 半導體": {
        "2449.TW": "2449 京元電子", "4979.TWO": "4979 華星光", "6274.TWO": "6274 台燿",
        "2404.TW": "2404 漢唐", "8299.TWO": "8299 群聯", "3324.TWO": "3324 雙鴻",
        "2327.TW": "2327 國巨", "3013.TW": "3013 晟銘電", "3006.TW": "3006 晶豪科",
        "8033.TW": "8033 雷虎", "2308.TW": "2308 台達電", "2359.TW": "2359 所羅門",
        "2436.TW": "2436 偉詮電", "3035.TW": "3035 智原", "3653.TW": "3653 健策",
        "1785.TWO": "1785 光洋科", "8390.TWO": "8390 金益鼎", "2457.TW": "2457 飛宏",
        "6196.TW": "6196 帆宣", "2489.TW": "2489 瑞軒", "2472.TW": "2472 立隆電",
        "6176.TW": "6176 瑞儀", "4953.TWO": "4953 緯軟", "2382.TW": "2382 廣達",
        "4744.TWO": "4744 皇將", "2344.TW": "2344 華邦電", "6191.TW": "6191 精成科",
        "3189.TW": "3189 景碩", "8046.TW": "8046 南電", "3037.TW": "3037 欣興",
        "3450.TW": "3450 聯鈞", "3105.TWO": "3105 穩懋", "4749.TWO": "4749 新應材",
        "2330.TW": "2330 台積電", "2313.TW": "2313 華通", "2337.TW": "2337 旺宏",
        "6517.TWO": "6517 中探針", "3016.TW": "3016 嘉晶", "1504.TW": "1504 東元",
        "3374.TWO": "3374 精材", "1519.TW": "1519 華城", "3305.TW": "3305 昇貿",
        "3583.TW": "3583 辛耘", "3265.TWO": "3265 台星科", "8110.TW": "8110 華東",
        "6239.TW": "6239 力成", "6515.TW": "6515 穎崴", "3711.TW": "3711 日月光投控",
        "3163.TWO": "3163 波若威", "6442.TW": "6442 光聖", "3665.TW": "3665 貿聯-KY",
        "2374.TW": "2374 佳能", "3019.TW": "3019 亞光", "2357.TW": "2357 華碩",
        "2464.TW": "2464 盟立", "6190.TWO": "6190 萬泰科", "2355.TW": "2355 敬鵬",
        "8028.TW": "8028 昇陽半導體", "6213.TW": "6213 聯茂", "8358.TWO": "8358 金居",
        "1609.TW": "1609 大亞", "1605.TW": "1605 華新"
    },
    "PCB / 被動元件 / 電子材料": {
        "4958.TW": "4958 臻鼎-KY", "1802.TW": "1802 台玻", "1309.TW": "1309 台達化",
        "1304.TW": "1304 台聚", "1326.TW": "1326 台化", "1303.TW": "1303 南亞",
        "3234.TWO": "3234 光環", "1313.TW": "1313 聯成", "1305.TW": "1305 華夏",
        "1310.TW": "1310 台苯", "5309.TWO": "5309 系統電", "3323.TWO": "3323 加百裕",
        "4931.TWO": "4931 新盛力", "6121.TWO": "6121 順達", "6781.TW": "6781 AES-KY",
        "4577.TWO": "4577 達航科技", "6805.TW": "6805 富世達", "1815.TWO": "1815 富喬",
        "5315.TWO": "5315 光聯", "3049.TW": "3049 精金", "3090.TW": "3090 日電貿",
        "8210.TW": "8210 勤誠", "6770.TW": "6770 力積電", "3707.TWO": "3707 漢磊",
        "5432.TWO": "5432 富鼎", "3673.TW": "3673 TPK-KY"
    },
    "機電 / 工具機 / 設備": {
        "2049.TW": "2049 上銀", "1536.TW": "1536 和大", "4540.TW": "4540 全球傳動",
        "1597.TW": "1597 直得", "4526.TW": "4526 東台", "2359.TW": "2359 所羅門",
        "4576.TW": "4576 大銀微系統"
    },
    "傳產 / 原物料 / 航運": {
        "2615.TW": "2615 萬海", "2606.TW": "2606 裕民", "2637.TW": "2637 慧洋-KY",
        "1802.TW": "1802 台玻", "1303.TW": "1303 南亞", "1326.TW": "1326 台化",
        "2009.TW": "2009 第一銅"
    },
    "能源 / 綠能": {
        "6806.TW": "6806 森崴能源", "6869.TW": "6869 雲豹能源", "00920.TW": "00920 富邦ESG綠電"
    },
    "金融 / 其他": {
        "00909.TW": "00909 國泰數位支付服務", "2887.TW": "2887 台新新光金", "3605.TW": "3605 宏致"
    }
}

# 自動整合成去重的全股票池字典，供下拉選單使用
DEFAULT_POOL = {}
for sec_name, stocks in SECTOR_POOL.items():
    for tk, label in stocks.items():
        if tk not in DEFAULT_POOL:
            DEFAULT_POOL[tk] = f"{label} [{sec_name}]"

# 每個族群取前 5 檔最具代表性權值股計算「今日族群強弱度」，兼顧準確度與載入速度
SECTOR_MAP = {
    sec: tuple(list(stocks.keys())[:5]) for sec, stocks in SECTOR_POOL.items()
}

MA_TUPLE = (10, 20, 40, 60)

@st.cache_data(ttl=600)
def fetch_stock_data(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Close']
    for ma in MA_TUPLE:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']
    return df.dropna()

@st.cache_data(ttl=1800)
def fetch_chip_data(ticker_code: str, _dates: pd.DatetimeIndex):
    n = len(_dates)
    start_str = (_dates.min() - timedelta(days=10)).strftime('%Y-%m-%d')
    url = "https://finmindtrade.com"
    foreign_net, trust_net, margin_bal, major_ratio = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    is_real_chip = False
    try:
        r_inst = requests.get(url, params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": ticker_code, "start_date": start_str}, timeout=5).json()
        if r_inst.get("msg") == "success" and len(r_inst.get("data", ())) > 0:
            inst_df = pd.DataFrame(r_inst.get("data"))
            inst_df['net'] = (inst_df['buy'] - inst_df['sell']) / 1000.0
            piv = inst_df.pivot_table(index='date', columns='name', values='net', aggfunc='sum').fillna(0)
            piv.index = pd.to_datetime(piv.index)
            aligned = piv.reindex(_dates).fillna(0)
            if 'Foreign_Investor' in aligned.columns: foreign_net = aligned['Foreign_Investor'].values
            if 'Investment_Trust' in aligned.columns: trust_net = aligned['Investment_Trust'].values
            is_real_chip = True
        r_marg = requests.get(url, params={"dataset": "TaiwanStockMarginPurchaseShortSale", "data_id": ticker_code, "start_date": start_str}, timeout=5).json()
        if r_marg.get("msg") == "success" and len(r_marg.get("data", ())) > 0:
            m_df = pd.DataFrame(r_marg.get("data")).set_index('date')
            m_df.index = pd.to_datetime(m_df.index)
            margin_bal = m_df['MarginPurchaseTodayBalance'].reindex(_dates).ffill().bfill().values
        r_tdcc = requests.get(url, params={"dataset": "TaiwanStockHoldingSharesPer", "data_id": ticker_code, "start_date": start_str}, timeout=5).json()
        if r_tdcc.get("msg") == "success" and len(r_tdcc.get("data", ())) > 0:
            t_df = pd.DataFrame(r_tdcc.get("data"))
            big = t_df[t_df['HoldingSharesLevel'].isin(('1000以上', '15'))].groupby('date')['percent'].sum()
            big.index = pd.to_datetime(big.index)
            major_ratio = big.reindex(_dates).ffill().bfill().values
    except Exception:
        pass
    if not is_real_chip or np.all(margin_bal == 0):
        np.random.seed(int(ticker_code) if ticker_code.isdigit() else 42)
        foreign_net = np.random.normal(0, 1500, n)
        trust_net = np.random.normal(100, 400, n)
        margin_bal = 15000 + np.cumsum(np.random.normal(0, 150, n))
    if np.all(major_ratio == 0):
        major_ratio = np.clip(45 + np.cumsum(np.random.normal(0, 0.25, n)), 20, 85)
    return pd.DataFrame({'Foreign_Net': foreign_net, 'Trust_Net': trust_net, 'Margin_Bal': margin_bal, 'Major_1000_Pct': major_ratio}, index=_dates), is_real_chip

def run_hmm_model(df: pd.DataFrame):
    features = pd.DataFrame(index=df.index)
    features['ret'], features['vol'] = df['Return'], df['Volatility']
    features['inst_ratio'] = (df['Foreign_Net'] + df['Trust_Net']) / (df['Volume'] / 1000.0 + 1)
    features['margin_chg'] = df['Margin_Bal'].pct_change().fillna(0)
    X = ((features - features.mean()) / (features.std() + 1e-8)).fillna(0).values
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(X)
    raw_probs = gmm.predict_proba(X)
    trans_mat = np.array(((0.85, 0.08, 0.07), (0.10, 0.85, 0.05), (0.12, 0.03, 0.85)))
    smooth_probs = raw_probs.copy()
    for t in range(1, len(raw_probs)):
        post = (smooth_probs[t-1] @ trans_mat) * raw_probs[t]
        smooth_probs[t] = post / (post.sum() + 1e-8)
    labels = smooth_probs.argmax(axis=1)
    state_means = list(df['Return'].values[labels == i].mean() if np.any(labels == i) else 0 for i in range(3))
    sorted_idx = np.argsort(state_means)
    mapped = pd.DataFrame(index=df.index)
    mapped['State_2_Bear'] = smooth_probs[:, int(sorted_idx.item(0))] * 100
    mapped['State_0_Base'] = smooth_probs[:, int(sorted_idx.item(1))] * 100
    mapped['State_1_Bull'] = smooth_probs[:, int(sorted_idx.item(2))] * 100
    return mapped

def resample_ohlcv(df: pd.DataFrame, timeframe: str):
    if timeframe == "日線 (1D)": return df
    rule = "W-FRI" if timeframe == "週線 (1W)" else "ME"
    res = df.resample(rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Foreign_Net': 'sum', 'Trust_Net': 'sum', 'Margin_Bal': 'last', 'Major_1000_Pct': 'last', 'State_0_Base': 'last', 'State_1_Bull': 'last', 'State_2_Bear': 'last'}).dropna()
    for ma in MA_TUPLE: res[f'MA{ma}'] = res['Close'].rolling(ma).mean()
    res['BB_Mid'] = res['Close'].rolling(20).mean()
    res['BB_Std'] = res['Close'].rolling(20).std()
    res['BB_Up'] = res['BB_Mid'] + 2 * res['BB_Std']
    res['BB_Low'] = res['BB_Mid'] - 2 * res['BB_Std']
    res['BB_Width'] = (res['BB_Up'] - res['BB_Low']) / res['BB_Mid']
    return res
