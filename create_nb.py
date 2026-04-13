import json

with open('colab_lstm.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 1: 설치 (numpy 버전 고정 포함) ──────────────────────
nb['cells'][1]['source'] = [
    '!pip install -q "numpy==1.26.4" --force-reinstall\n',
    '!pip install -q ccxt pandas-ta tensorflow scikit-learn matplotlib\n',
    'print("설치 완료. Runtime -> Restart session 후 Cell 2부터 실행하세요.")'
]

# ── Cell 3: download_btc → Data Vision으로 교체 ───────────────
cell3_new = r"""import requests, zipfile, io, time
from pathlib import Path

DV_SPOT = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m"

def _fetch_zip(url):
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200: return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        return pd.read_csv(z.open(name), header=None)
    except Exception as e:
        print(f"    오류: {e}"); return None

def download_btc(years=range(2020, 2026)):
    cache = Path('btc_15m_all.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f"캐시 로드: {len(df):,}개")
        return df

    frames = []
    for year in years:
        for month in range(1, 13):
            url = f"{DV_SPOT}/BTCUSDT-15m-{year}-{month:02d}.zip"
            print(f"  {year}-{month:02d}...", end=' ', flush=True)
            df = _fetch_zip(url)
            if df is None: print("없음"); continue
            frames.append(df)
            print(f"{len(df):,}개")
            time.sleep(0.05)

    df = pd.concat(frames, ignore_index=True)
    df.columns = ['ts','open','high','low','close','volume',
                  'close_time','quote_vol','n_trades',
                  'taker_buy_vol','taker_buy_quote','ignore']
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df[['timestamp','open','high','low','close','volume']].drop_duplicates('timestamp')
    df = df.set_index('timestamp').sort_index().astype(float)
    df.to_csv(cache)
    print(f"\n저장 완료: {len(df):,}개")
    return df

print("BTC/USDT 15분봉 다운로드 (Binance Data Vision)...")
df_raw = download_btc()
print(f"기간: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}")
"""

# ── v2-1: 데이터 수집 (Data Vision + fapi) ───────────────────
cell_v2_1 = r"""import requests, zipfile, io, time
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

DV_SPOT = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m"
DV_FUND = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT"

def _fetch_zip(url):
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200: return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        return pd.read_csv(z.open(name), header=None)
    except Exception as e:
        print(f"    오류: {e}"); return None

# ── 1. OHLCV + 체결강도 (Data Vision) ────────────────────────
def download_ohlcv_taker(years=range(2020, 2026)):
    cache = Path('btc_ohlcv_taker_all.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f"  OHLCV: 캐시 로드 ({len(df):,}개)"); return df

    frames = []
    for year in years:
        for month in range(1, 13):
            url = f"{DV_SPOT}/BTCUSDT-15m-{year}-{month:02d}.zip"
            print(f"  {year}-{month:02d}...", end=' ', flush=True)
            df = _fetch_zip(url)
            if df is None: print("없음"); continue
            frames.append(df); print(f"{len(df):,}개")
            time.sleep(0.05)

    df = pd.concat(frames, ignore_index=True)
    df.columns = ['ts','open','high','low','close','volume',
                  'close_time','quote_vol','n_trades',
                  'taker_buy_vol','taker_buy_quote','ignore']
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df[['timestamp','open','high','low','close','volume','taker_buy_vol','n_trades']]
    df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index().astype(float)
    df.to_csv(cache); print(f"  저장 완료: {len(df):,}개"); return df

# ── 2. 펀딩비 (Data Vision) ───────────────────────────────────
def download_funding(years=range(2020, 2026)):
    cache = Path('btc_funding.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f"  펀딩비: 캐시 로드 ({len(df):,}개)"); return df

    frames = []
    for year in years:
        for month in range(1, 13):
            url = f"{DV_FUND}/BTCUSDT-fundingRate-{year}-{month:02d}.zip"
            print(f"  펀딩비 {year}-{month:02d}...", end=' ', flush=True)
            df = _fetch_zip(url)
            if df is None: print("없음"); continue
            frames.append(df); print(f"{len(df):,}개")
            time.sleep(0.05)

    df = pd.concat(frames, ignore_index=True)
    df.columns = ['calc_time','funding_interval_hours','fundingRate']
    # 헤더 행 제거 (일부 파일에 포함될 수 있음)
    df = df[df['calc_time'] != 'calc_time'].reset_index(drop=True)
    df['calc_time']    = pd.to_numeric(df['calc_time'], errors='coerce')
    df['fundingRate']  = pd.to_numeric(df['fundingRate'], errors='coerce')
    df = df.dropna(subset=['calc_time','fundingRate'])
    df['timestamp'] = pd.to_datetime(df['calc_time'].astype(int), unit='ms', utc=True)
    df = df[['timestamp','fundingRate']].drop_duplicates('timestamp')
    df = df.set_index('timestamp').sort_index()
    df.to_csv(cache); print(f"  저장 완료: {len(df):,}개"); return df

# ── 3. OI / L/S / Taker (fapi - 별도 서버, 지역제한 없음) ───
def download_futures_metric(endpoint, col, filename):
    from datetime import datetime, timezone
    cache = Path(filename)
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f"  {col}: 캐시 로드 ({len(df):,}개)"); return df
    print(f"  {col} 다운로드 중...", flush=True)
    rows = []; since = int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp()*1000)
    end   = int(datetime.now(timezone.utc).timestamp()*1000)
    while since < end:
        try:
            r = requests.get(f"https://fapi.binance.com/futures/data/{endpoint}",
                params={"symbol":"BTCUSDT","period":"15m","startTime":since,"limit":500},
                timeout=15).json()
        except Exception as e: print(f"재시도 {e}"); time.sleep(3); continue
        if not r or isinstance(r, dict): break
        rows.extend(r); since = r[-1]['timestamp'] + 1; time.sleep(0.15)
    if not rows: print(f"  {col}: 데이터 없음"); return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df[['timestamp', col]].drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df[col] = df[col].astype(float)
    df.to_csv(cache); print(f"  {col}: {len(df):,}개 저장"); return df

# ── 실행 ──────────────────────────────────────────────────────
print("="*55)
print("데이터 수집 시작 (최초 실행: 15~25분 소요)")
print("="*55)
print("\n[1/5] OHLCV + 체결강도 (Data Vision)...")
df_ohlcv = download_ohlcv_taker()
print("\n[2/5] 펀딩비 (Data Vision)...")
df_funding = download_funding()
print("\n[3/5] OI...")
df_oi = download_futures_metric('openInterestHist', 'sumOpenInterest', 'btc_oi.csv')
print("\n[4/5] 롱숏 비율...")
df_ls = download_futures_metric('globalLongShortAccountRatio', 'longShortRatio', 'btc_ls.csv')
print("\n[5/5] 테이커 비율...")
df_tk = download_futures_metric('takerlongshortRatio', 'buySellRatio', 'btc_taker.csv')
print(f"\n수집 완료:")
print(f"  OHLCV    : {len(df_ohlcv):,}개")
print(f"  펀딩비   : {len(df_funding):,}개")
print(f"  OI       : {len(df_oi):,}개")
print(f"  L/S      : {len(df_ls):,}개")
print(f"  Taker    : {len(df_tk):,}개")
"""

cell_v2_2 = r"""def make_features_v2(df_base):
    d = df_base.copy()
    d['rsi']       = ta.rsi(d['close'], length=14)
    d['ema9']      = ta.ema(d['close'], length=9)
    d['ema21']     = ta.ema(d['close'], length=21)
    d['atr']       = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['vol_ma']    = ta.sma(d['volume'], length=20)
    macd           = ta.macd(d['close'])
    d['macd_hist'] = macd.iloc[:, 2]
    bb             = ta.bbands(d['close'], length=20)
    d['bb_width']  = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    d['ret_1']     = d['close'].pct_change(1)
    d['ret_4']     = d['close'].pct_change(4)
    d['ret_16']    = d['close'].pct_change(16)
    d['hl_ratio']  = (d['high'] - d['low']) / d['close']
    d['ema_ratio'] = d['ema9'] / d['ema21'] - 1
    d['atr_pct']   = d['atr'] / d['close']
    d['vol_ratio'] = d['volume'] / (d['vol_ma'] + 1e-10)
    d['rsi_norm']  = d['rsi'] / 100

    # CVD
    d['taker_buy_vol'] = d['taker_buy_vol'].astype(float)
    d['delta']         = d['taker_buy_vol'] - (d['volume'] - d['taker_buy_vol'])
    d['delta_pct']     = d['delta'] / (d['volume'] + 1e-10)
    cvd_roll           = d['delta'].rolling(20).sum()
    d['cvd_norm']      = (cvd_roll - cvd_roll.rolling(100).mean()) / (cvd_roll.rolling(100).std() + 1e-10)
    d['delta_ma']      = d['delta_pct'].rolling(5).mean()

    # 펀딩비
    d = d.join(df_funding.reindex(d.index, method='ffill'))
    d['funding_norm']    = (d['fundingRate'].fillna(0) * 1000).clip(-5, 5)
    d['funding_extreme'] = (d['fundingRate'].fillna(0).abs() > 0.005).astype(float)

    # OI
    if len(df_oi) > 0:
        d = d.join(df_oi.reindex(d.index, method='ffill'))
        oi_ma = d['sumOpenInterest'].rolling(20).mean()
        d['oi_ratio']  = (d['sumOpenInterest'] / (oi_ma + 1e-10) - 1).fillna(0)
        d['oi_change'] = d['sumOpenInterest'].pct_change(4).fillna(0).clip(-0.1, 0.1)
    else:
        d['oi_ratio'] = 0.0; d['oi_change'] = 0.0

    # L/S 비율
    if len(df_ls) > 0:
        d = d.join(df_ls.reindex(d.index, method='ffill'))
        d['ls_norm'] = (d['longShortRatio'].fillna(1) - 1).clip(-1, 1)
    else:
        d['ls_norm'] = 0.0

    # 테이커 비율
    if len(df_tk) > 0:
        d = d.join(df_tk.reindex(d.index, method='ffill'))
        d['taker_norm'] = (d['buySellRatio'].fillna(1) - 1).clip(-1, 1)
    else:
        d['taker_norm'] = 0.0

    future_ret = d['close'].shift(-16) / d['close'] - 1
    d['target'] = (future_ret >= 0.003).astype(int)
    return d.dropna()

print("강화된 피처 엔지니어링 중...")
df_v2 = make_features_v2(df_ohlcv)

FEATURES_V2 = [
    'ret_1','ret_4','ret_16','hl_ratio','ema_ratio',
    'atr_pct','vol_ratio','rsi_norm','macd_hist','bb_width',
    'delta_pct','cvd_norm','delta_ma',
    'funding_norm','funding_extreme',
    'oi_ratio','oi_change',
    'ls_norm','taker_norm',
]
print(f"피처: {len(FEATURES_V2)}개 (기존 10 -> {len(FEATURES_V2)}개)")
print(f"데이터: {len(df_v2):,}개")
print(f"타겟 분포: 상승 {df_v2['target'].mean()*100:.1f}% / 하락 {(1-df_v2['target'].mean())*100:.1f}%")
"""

cell_v2_3 = r"""from sklearn.utils.class_weight import compute_class_weight

def make_sequences(df, features, seq_len=60):
    X, y = [], []
    vals = df[features].values
    tgt  = df['target'].values
    for i in range(seq_len, len(df)):
        X.append(vals[i-seq_len:i])
        y.append(tgt[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_lstm(seq_len, n_features):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    inp = Input(shape=(seq_len, n_features))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.2)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='LSTM')

def build_transformer(seq_len, n_features, num_heads=4, ff_dim=64):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                         MultiHeadAttention, GlobalAveragePooling1D, Add)
    inp = Input(shape=(seq_len, n_features))
    x = inp
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
        attn = Dropout(0.1)(attn)
        x    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
        ff   = Dense(ff_dim, activation='relu')(x)
        ff   = Dropout(0.1)(ff)
        ff   = Dense(n_features)(ff)
        x    = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='Transformer')

SEQ_LEN_V2 = 60
df_tr_v2 = df_v2[df_v2.index <= '2023-12-31']
df_va_v2 = df_v2[(df_v2.index > '2023-12-31') & (df_v2.index <= '2024-12-31')]
df_te_v2 = df_v2[df_v2.index > '2024-12-31']

X_tr_v2, y_tr_v2 = make_sequences(df_tr_v2, FEATURES_V2, SEQ_LEN_V2)
X_va_v2, y_va_v2 = make_sequences(df_va_v2, FEATURES_V2, SEQ_LEN_V2)
X_te_v2, y_te_v2 = make_sequences(df_te_v2, FEATURES_V2, SEQ_LEN_V2)

print(f"학습: {X_tr_v2.shape}  ({y_tr_v2.mean()*100:.1f}% 상승)")
print(f"검증: {X_va_v2.shape}  ({y_va_v2.mean()*100:.1f}% 상승)")
print(f"테스트: {X_te_v2.shape}  ({y_te_v2.mean()*100:.1f}% 상승)")

cw = compute_class_weight('balanced', classes=np.unique(y_tr_v2), y=y_tr_v2)
class_w_v2 = {0: float(cw[0]), 1: float(cw[1])}
print(f"클래스 가중치: {class_w_v2}")

lstm_v3 = build_lstm(SEQ_LEN_V2, len(FEATURES_V2))
lstm_v3.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
tf_v3 = build_transformer(SEQ_LEN_V2, len(FEATURES_V2))
tf_v3.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

cb_v3 = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                     monitor='val_auc', mode='max', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5,
                                         monitor='val_auc', mode='max', verbose=1),
]

print("\n" + "="*50 + "\nLSTM v3 학습 중...\n" + "="*50)
lstm_v3.fit(X_tr_v2, y_tr_v2, validation_data=(X_va_v2, y_va_v2),
            epochs=60, batch_size=256, class_weight=class_w_v2, callbacks=cb_v3, verbose=1)

print("\n" + "="*50 + "\nTransformer v3 학습 중...\n" + "="*50)
tf_v3.fit(X_tr_v2, y_tr_v2, validation_data=(X_va_v2, y_va_v2),
          epochs=60, batch_size=256, class_weight=class_w_v2, callbacks=cb_v3, verbose=1)

print("\n확률 분포 확인...")
X_all_v2, _ = make_sequences(df_v2, FEATURES_V2, SEQ_LEN_V2)
p_lstm_v3 = lstm_v3.predict(X_all_v2, verbose=0).flatten()
p_tf_v3   = tf_v3.predict(X_all_v2, verbose=0).flatten()
ts_v2      = df_v2.index[SEQ_LEN_V2:]
ai_lstm_v3 = dict(zip(ts_v2, p_lstm_v3))
ai_tf_v3   = dict(zip(ts_v2, p_tf_v3))

for name, p in [('LSTM v3', p_lstm_v3), ('Transformer v3', p_tf_v3)]:
    print(f"\n=== {name} ===")
    print(f"  범위: {p.min():.3f} ~ {p.max():.3f}  평균: {p.mean():.3f}")
    print(f"  0.40 이상: {(p>=0.40).mean()*100:.1f}%")
    print(f"  0.50 이상: {(p>=0.50).mean()*100:.1f}%")
    print(f"  0.60 이상: {(p>=0.60).mean()*100:.1f}%")

lstm_v3.save('lstm_v3.keras')
tf_v3.save('transformer_v3.keras')
print("\n모델 저장: lstm_v3.keras, transformer_v3.keras")
"""

# ── 노트북 재구성 ─────────────────────────────────────────────
# Cell 3 교체
lines = cell3_new.split('\n')
source = [line + '\n' for line in lines[:-1]]
if lines[-1]: source.append(lines[-1])
nb['cells'][3]['source'] = source

# 구분 마크다운 + v2 셀 3개
separator = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## v2: 강화 데이터 (CVD + 펀딩비 + OI + L/S) 재학습\n",
        "**Cell 1(설치) → Restart → Cell 2(import) → Cell v2-1 → v2-2 → v2-3 순서로 실행**\n",
        "- Cell 3~13은 건너뛰어도 됩니다."
    ]
}
nb['cells'] = nb['cells'][:14]  # 기존 14개만 유지
nb['cells'].append(separator)

for src in [cell_v2_1, cell_v2_2, cell_v2_3]:
    lines = src.split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]: source.append(lines[-1])
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

cell_v2_4 = r"""import pandas_ta as ta

UPBIT_FEE = 0.0005
MKT_LABEL = {
    2020:'상승 +306%', 2021:'강세  +76%',
    2022:'폭락  -63%', 2023:'강세+170%',
    2024:'강세+143%', 2025:'하락  -27%',
}

def make_daily_ma(df, period=200):
    daily = df['close'].resample('1D').last().dropna()
    ma    = daily.rolling(period).mean()
    def get(ts):
        i = ma.index.searchsorted(ts.normalize(), side='right') - 1
        return float(ma.iloc[i]) if i >= 0 else None
    return get

def make_4h(df):
    d4 = df.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna()
    d4['rsi']      = ta.rsi(d4['close'], length=14)
    d4['atr']      = ta.atr(d4['high'], d4['low'], d4['close'], length=14)
    d4['vol_ma']   = ta.sma(d4['volume'], length=20)
    d4['don_high'] = d4['high'].rolling(20).max().shift(1)
    d4 = d4.dropna()
    def get(ts):
        i = d4.index.searchsorted(ts, side='right') - 1
        return d4.iloc[i] if i >= 0 else None
    return get

def backtest_v2(df_year, ai_map=None, threshold=0.60):
    df_feat   = make_features_v2(df_year.copy()).dropna()
    get_ma200 = make_daily_ma(df_year)
    get_4h    = make_4h(df_year)
    krw = 1_000_000; btc = 0.0; pos = None; trades = []
    for i in range(1, len(df_feat)):
        row   = df_feat.iloc[i]
        price = row['close']
        ts    = df_feat.index[i]
        r4h   = get_4h(ts)
        if r4h is None or pd.isna(r4h['don_high']): continue
        if pos:
            new_stop = price - r4h['atr'] * 2
            if new_stop > pos['stop']: pos['stop'] = new_stop
            if price <= pos['stop']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
                continue
        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200): continue
        if price < ma200:
            if pos:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
            continue
        if not (price > r4h['don_high']
                and r4h['volume'] > r4h['vol_ma'] * 1.5
                and r4h['rsi'] < 75): continue
        ema_ok = row['rsi_norm'] * 100 < 65 and price > row['ema21']
        if not ema_ok: continue
        if ai_map is not None:
            if ai_map.get(ts, 0.0) < threshold: continue
        if pos is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            pos = {'price': price, 'stop': price - r4h['atr'] * 2}
            krw = 0.0
    if pos:
        lp = df_feat['close'].iloc[-1]
        r  = btc * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
        krw = r
    if not trades:
        return dict(ret=0, n=0, wr=0, pf=0, mdd=0)
    ret    = (krw - 1_000_000) / 1_000_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 99.0
    eq = pk = 1_000_000.0; mdd = 0.0
    for t in trades:
        eq += t['pnl']; pk = max(pk, eq)
        mdd = max(mdd, (pk - eq) / pk * 100)
    return dict(ret=ret, n=len(trades), wr=wr, pf=pf, mdd=mdd)

# AI 확률 계산
print("AI 확률 계산 중...")
X_all_v2, _ = make_sequences(df_v2, FEATURES_V2, SEQ_LEN_V2)
p_lstm = lstm_v3.predict(X_all_v2, verbose=0).flatten()
p_tf   = tf_v3.predict(X_all_v2, verbose=0).flatten()
ts_v2  = df_v2.index[SEQ_LEN_V2:]
ai_lstm = dict(zip(ts_v2, p_lstm))
ai_tf   = dict(zip(ts_v2, p_tf))

strategies = [
    ('전략A',        None,    0.60),
    ('A+LSTM(0.55)', ai_lstm, 0.55),
    ('A+LSTM(0.60)', ai_lstm, 0.60),
    ('A+TF(0.55)',   ai_tf,   0.55),
    ('A+TF(0.60)',   ai_tf,   0.60),
]

for sname, ai_map, thr in strategies:
    print(f'\n[{sname}]')
    print(f'  {"연도":>4}  {"시장":>11}  {"수익률":>8}  {"거래수":>5}  {"승률":>6}  {"MDD":>6}  {"B&H":>8}')
    print('  ' + '-'*60)
    cum = 1_000_000.0
    for year in range(2020, 2026):
        df_yr = df_ohlcv[(df_ohlcv.index >= f'{year}-01-01') & (df_ohlcv.index <= f'{year}-12-31')]
        if len(df_yr) < 500: continue
        r   = backtest_v2(df_yr, ai_map=ai_map, threshold=thr)
        bah = (df_yr['close'].iloc[-1] - df_yr['close'].iloc[0]) / df_yr['close'].iloc[0] * 100
        cum *= (1 + r['ret'] / 100)
        print(f'  {year}  {MKT_LABEL.get(year,""):>11}  {r["ret"]:>+7.1f}%'
              f'  {r["n"]:>5}  {r["wr"]:>5.1f}%  {r["mdd"]:>5.1f}%  {bah:>+7.1f}%')
    cr = (cum - 1_000_000) / 1_000_000 * 100
    print('  ' + '-'*60)
    print(f'  6년 누적: {cum:>10,.0f}원  ({cr:>+.1f}%)')
"""

cell_v2_5 = r"""# ── 전략A + 펀딩비 필터 + 5:1 손익비 필터 백테스트 ─────────
print("="*60)
print("전략A 개선: 펀딩비 필터 + 5:1 손익비 필터")
print("="*60)

def make_funding_map(df_funding):
    # 타임스탬프 -> 가장 최근 펀딩비 매핑
    return df_funding['fundingRate'].to_dict()

def make_4h_v2(df):
    # 100봉 Donchian 고점(장기 저항선) 추가
    d4 = df.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna()
    d4['rsi']       = ta.rsi(d4['close'], length=14)
    d4['atr']       = ta.atr(d4['high'], d4['low'], d4['close'], length=14)
    d4['vol_ma']    = ta.sma(d4['volume'], length=20)
    d4['don_high']  = d4['high'].rolling(20).max().shift(1)   # 진입 신호용
    d4['res_high']  = d4['high'].rolling(100).max().shift(1)  # 장기 저항선 (5:1 계산용)
    d4 = d4.dropna()
    def get(ts):
        i = d4.index.searchsorted(ts, side='right') - 1
        return d4.iloc[i] if i >= 0 else None
    return get

def backtest_v3(df_year, use_funding=False, use_rr=False):
    df_feat   = make_features_v2(df_year.copy()).dropna()
    get_ma200 = make_daily_ma(df_year)
    get_4h    = make_4h_v2(df_year)
    krw = 1_000_000; btc = 0.0; pos = None; trades = []

    for i in range(1, len(df_feat)):
        row   = df_feat.iloc[i]
        price = row['close']
        ts    = df_feat.index[i]
        r4h   = get_4h(ts)
        if r4h is None or pd.isna(r4h['don_high']): continue

        if pos:
            new_stop = price - r4h['atr'] * 2
            if new_stop > pos['stop']: pos['stop'] = new_stop
            if price <= pos['stop']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
                continue

        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200): continue
        if price < ma200:
            if pos:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
            continue

        if not (price > r4h['don_high']
                and r4h['volume'] > r4h['vol_ma'] * 1.5
                and r4h['rsi'] < 75): continue

        ema_ok = row['rsi_norm'] * 100 < 65 and price > row['ema21']
        if not ema_ok: continue

        # ── 펀딩비 필터 ──────────────────────────────────────
        if use_funding:
            funding = row.get('fundingRate', 0.0)
            if pd.isna(funding): funding = 0.0
            if funding > 0.0005:  # 0.05% = 연 200% 수준 과열
                continue

        # ── 5:1 손익비 필터 ──────────────────────────────────
        if use_rr:
            stop_dist   = r4h['atr'] * 2
            resistance  = r4h.get('res_high', price * 1.20)
            if pd.isna(resistance): resistance = price * 1.20
            target_dist = resistance - price
            if target_dist < stop_dist * 5:  # 5:1 미만이면 스킵
                continue

        if pos is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            pos = {'price': price, 'stop': price - r4h['atr'] * 2}
            krw = 0.0

    if pos:
        lp = df_feat['close'].iloc[-1]
        r  = btc * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
        krw = r

    if not trades:
        return dict(ret=0, n=0, wr=0, pf=0, mdd=0)
    ret    = (krw - 1_000_000) / 1_000_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 99.0
    eq = pk = 1_000_000.0; mdd = 0.0
    for t in trades:
        eq += t['pnl']; pk = max(pk, eq)
        mdd = max(mdd, (pk - eq) / pk * 100)
    return dict(ret=ret, n=len(trades), wr=wr, pf=pf, mdd=mdd)

# ── 연도별 비교 ───────────────────────────────────────────────
strategies_v3 = [
    ('전략A (기존)',          False, False),
    ('A + 펀딩비 필터',       True,  False),
    ('A + 5:1 손익비',        False, True),
    ('A + 펀딩비 + 5:1',      True,  True),
]

for sname, use_f, use_r in strategies_v3:
    print(f'\n[{sname}]')
    print(f'  {"연도":>4}  {"시장":>11}  {"수익률":>8}  {"거래수":>5}  {"승률":>6}  {"MDD":>6}  {"B&H":>8}')
    print('  ' + '-'*60)
    cum = 1_000_000.0
    for year in range(2020, 2026):
        df_yr = df_ohlcv[(df_ohlcv.index >= f'{year}-01-01') & (df_ohlcv.index <= f'{year}-12-31')]
        if len(df_yr) < 500: continue
        r   = backtest_v3(df_yr, use_funding=use_f, use_rr=use_r)
        bah = (df_yr['close'].iloc[-1] - df_yr['close'].iloc[0]) / df_yr['close'].iloc[0] * 100
        cum *= (1 + r['ret'] / 100)
        print(f'  {year}  {MKT_LABEL.get(year,""):>11}  {r["ret"]:>+7.1f}%'
              f'  {r["n"]:>5}  {r["wr"]:>5.1f}%  {r["mdd"]:>5.1f}%  {bah:>+7.1f}%')
    cr = (cum - 1_000_000) / 1_000_000 * 100
    print('  ' + '-'*60)
    print(f'  6년 누적: {cum:>10,.0f}원  ({cr:>+.1f}%)')
"""

for src in [cell_v2_4, cell_v2_5]:
    lines = src.split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines[-1]: source.append(lines[-1])
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

with open('colab_lstm_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"완료: {len(nb['cells'])}개 셀 -> colab_lstm_v2.ipynb")
