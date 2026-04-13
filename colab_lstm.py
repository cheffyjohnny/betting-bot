
# ============================================================
# BTC 가격 예측 AI + 전략A 통합 백테스트
# Google Colab에서 셀별로 순서대로 실행하세요
# ============================================================


# ── CELL 1: 설치 ─────────────────────────────────────────────
# Colab에서 실행:
# !pip install -q ccxt pandas-ta tensorflow


# ── CELL 2: 임포트 & GPU 확인 ────────────────────────────────
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt, time, warnings
from pathlib import Path
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU 사용 가능: {len(gpus)}개")
for g in gpus:
    print(f"  {g.name}")
if not gpus:
    print("  CPU로 실행됩니다 (느릴 수 있음)")


# ── CELL 3: 데이터 다운로드 ──────────────────────────────────
def download_btc(years=range(2020, 2026)):
    exchange = ccxt.binance({'enableRateLimit': True})
    frames = []

    for year in years:
        end_year = min(year + 1, 2026)
        start_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ts   = int(datetime(end_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if year == 2025:
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        cache = Path(f'btc_15m_{year}.csv')
        if cache.exists():
            df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
            print(f"  {year}: 캐시 로드 ({len(df):,}개)")
            frames.append(df)
            continue

        print(f"  {year}: 다운로드 중...", flush=True)
        rows = []; since = start_ts
        while since < end_ts:
            try:
                c = exchange.fetch_ohlcv('BTC/USDT', '15m', since=since, limit=1000)
            except Exception as e:
                print(f"    재시도... {e}"); time.sleep(3); continue
            if not c: break
            c = [x for x in c if x[0] < end_ts]
            if not c: break
            rows.extend(c)
            since = c[-1][0] + 1
            time.sleep(0.1)

        df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        df.to_csv(cache)
        print(f"  {year}: {len(df):,}개 저장")
        frames.append(df)

    return pd.concat(frames).sort_index()

print("BTC/USDT 15분봉 데이터 다운로드 (2020~2025)...")
df_raw = download_btc()
print(f"\n전체 데이터: {len(df_raw):,}개 캔들")
print(f"기간: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}")


# ── CELL 4: 피처 엔지니어링 ──────────────────────────────────
def make_features(df):
    d = df.copy()

    # 기술 지표
    d['rsi']       = ta.rsi(d['close'], length=14)
    d['ema9']      = ta.ema(d['close'], length=9)
    d['ema21']     = ta.ema(d['close'], length=21)
    d['ema50']     = ta.ema(d['close'], length=50)
    d['atr']       = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['vol_ma']    = ta.sma(d['volume'], length=20)
    macd           = ta.macd(d['close'])
    d['macd_hist'] = macd.iloc[:, 2]
    bb             = ta.bbands(d['close'], length=20)
    d['bb_width']  = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]

    # 정규화된 피처 (모델 입력값)
    d['ret_1']    = d['close'].pct_change(1)            # 1봉 수익률
    d['ret_4']    = d['close'].pct_change(4)            # 1시간 수익률
    d['ret_16']   = d['close'].pct_change(16)           # 4시간 수익률
    d['hl_ratio'] = (d['high'] - d['low']) / d['close'] # 고저 범위
    d['ema_ratio']= d['ema9'] / d['ema21'] - 1          # EMA 배열
    d['atr_pct']  = d['atr'] / d['close']               # 변동성
    d['vol_ratio']= d['volume'] / d['vol_ma']           # 거래량 비율
    d['rsi_norm'] = d['rsi'] / 100                      # RSI 정규화

    # 타겟: 16봉(4시간) 후 1% 이상 상승 여부
    future_ret    = d['close'].shift(-16) / d['close'] - 1
    d['target']   = (future_ret >= 0.01).astype(int)

    return d.dropna()

print("피처 엔지니어링 중...")
df = make_features(df_raw)

FEATURES = ['ret_1','ret_4','ret_16','hl_ratio','ema_ratio',
            'atr_pct','vol_ratio','rsi_norm','macd_hist','bb_width']

print(f"피처 수: {len(FEATURES)}개")
print(f"최종 데이터: {len(df):,}개")
print(f"타겟 분포: 상승 {df['target'].mean()*100:.1f}% / 하락 {(1-df['target'].mean())*100:.1f}%")


# ── CELL 5: 시퀀스 데이터셋 생성 ─────────────────────────────
SEQ_LEN = 60  # 60봉 = 15시간치 데이터로 예측

def make_sequences(df, features, seq_len=60):
    X, y = [], []
    vals = df[features].values
    tgt  = df['target'].values
    for i in range(seq_len, len(df)):
        X.append(vals[i-seq_len:i])
        y.append(tgt[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 시간순 분할 (데이터 누수 방지)
train_end = '2023-12-31'
val_end   = '2024-12-31'

df_train = df[df.index <= train_end]
df_val   = df[(df.index > train_end) & (df.index <= val_end)]
df_test  = df[df.index > val_end]

X_train, y_train = make_sequences(df_train, FEATURES, SEQ_LEN)
X_val,   y_val   = make_sequences(df_val,   FEATURES, SEQ_LEN)
X_test,  y_test  = make_sequences(df_test,  FEATURES, SEQ_LEN)

print(f"학습:    {X_train.shape}  ({y_train.mean()*100:.1f}% 상승)")
print(f"검증:    {X_val.shape}    ({y_val.mean()*100:.1f}% 상승)")
print(f"테스트:  {X_test.shape}   ({y_test.mean()*100:.1f}% 상승)")


# ── CELL 6: LSTM 모델 ────────────────────────────────────────
def build_lstm(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x   = LSTM(128, return_sequences=True)(inp)
    x   = Dropout(0.2)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.2)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out, name='LSTM')
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

lstm_model = build_lstm(SEQ_LEN, len(FEATURES))
lstm_model.summary()


# ── CELL 7: Transformer 모델 ─────────────────────────────────
def build_transformer(seq_len, n_features, num_heads=4, ff_dim=64):
    inp = Input(shape=(seq_len, n_features))

    # Transformer Block x2
    x = inp
    for _ in range(2):
        # Multi-Head Attention
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
        attn = Dropout(0.1)(attn)
        x    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
        # Feed Forward
        ff   = Dense(ff_dim, activation='relu')(x)
        ff   = Dropout(0.1)(ff)
        ff   = Dense(n_features)(ff)
        x    = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))

    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(0.2)(x)
    x   = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out, name='Transformer')
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

tf_model = build_transformer(SEQ_LEN, len(FEATURES))
tf_model.summary()


# ── CELL 8: 학습 ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
]

print("=" * 50)
print("LSTM 학습 중...")
print("=" * 50)
hist_lstm = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=256,
    callbacks=callbacks, verbose=1
)

print("\n" + "=" * 50)
print("Transformer 학습 중...")
print("=" * 50)
hist_tf = tf_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=256,
    callbacks=callbacks, verbose=1
)


# ── CELL 9: 평가 ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name):
    prob  = model.predict(X_test, verbose=0).flatten()
    pred  = (prob >= 0.55).astype(int)   # 임계값 0.55
    print(f"\n{'='*50}")
    print(f"[{name}] 테스트셋 평가 (2025년)")
    print(f"{'='*50}")
    print(classification_report(y_test, pred, target_names=['하락','상승']))
    cm = confusion_matrix(y_test, pred)
    print(f"혼동행렬:\n{cm}")
    acc = (pred == y_test).mean() * 100
    print(f"정확도: {acc:.1f}%")
    return prob

prob_lstm = evaluate_model(lstm_model, X_test, y_test, 'LSTM')
prob_tf   = evaluate_model(tf_model,   X_test, y_test, 'Transformer')


# ── CELL 10: 학습 곡선 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, hist, name in zip(axes, [hist_lstm, hist_tf], ['LSTM', 'Transformer']):
    ax.plot(hist.history['accuracy'],     label='학습 정확도')
    ax.plot(hist.history['val_accuracy'], label='검증 정확도')
    ax.set_title(f'{name} 학습 곡선')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("training_curves.png 저장됨")


# ── CELL 11: 전략A + AI 필터 백테스트 ────────────────────────
UPBIT_FEE = 0.0005

def make_daily_ma(df, period=200):
    daily = df['close'].resample('1D').last().dropna()
    ma    = daily.rolling(period).mean()
    def get(ts):
        day = ts.normalize() if hasattr(ts, 'normalize') else ts
        i   = ma.index.searchsorted(day, side='right') - 1
        return float(ma.iloc[i]) if i >= 0 else None
    return get

def make_4h(df):
    d4 = df.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna()
    d4['ema9']    = ta.ema(d4['close'], length=9)
    d4['ema21']   = ta.ema(d4['close'], length=21)
    d4['rsi']     = ta.rsi(d4['close'], length=14)
    d4['atr']     = ta.atr(d4['high'], d4['low'], d4['close'], length=14)
    d4['vol_ma']  = ta.sma(d4['volume'], length=20)
    d4['don_high']= d4['high'].rolling(20).max().shift(1)
    d4 = d4.dropna()
    def get(ts):
        i = d4.index.searchsorted(ts, side='right') - 1
        return d4.iloc[i] if i >= 0 else None
    return get

def backtest(df_15m, ai_probs=None, ai_threshold=0.60, label=''):
    """
    ai_probs: None이면 전략A만, 배열이면 AI 필터 추가
    """
    df = make_features(df_15m.copy())
    df = df.dropna()

    get_ma200 = make_daily_ma(df_15m)
    get_4h    = make_4h(df_15m)

    # AI 확률 인덱스 매핑
    if ai_probs is not None:
        ai_idx = {ts: p for ts, p in zip(df.index[SEQ_LEN:], ai_probs)}
    else:
        ai_idx = None

    krw = 1_000_000; btc = 0.0; pos = None; trades = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = row['close']
        ts    = df.index[i]
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

        # AI 필터 (선택적)
        if ai_idx is not None:
            ai_prob = ai_idx.get(ts, 0.0)
            if ai_prob < ai_threshold:
                continue

        if pos is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            pos = {'price': price, 'stop': price - r4h['atr'] * 2}
            krw = 0.0

    if pos:
        lp = df['close'].iloc[-1]
        r  = btc * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
        krw = r

    if not trades:
        return dict(ret=0, n=0, wr=0, pf=0, mdd=0, label=label)

    ret    = (krw - 1_000_000) / 1_000_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins)/len(trades)*100
    pf     = abs(sum(t['pnl'] for t in wins)/sum(t['pnl'] for t in losses)) if losses else 99.0
    eq=pk=1_000_000.0; mdd=0.0
    for t in trades:
        eq += t['pnl']; pk = max(pk,eq)
        mdd = max(mdd,(pk-eq)/pk*100)
    return dict(ret=ret, n=len(trades), wr=wr, pf=pf, mdd=mdd, label=label)


# 테스트 기간 데이터 (2025년)
print("\n백테스트 실행 중 (2025년 테스트셋)...")
df_test_raw = df_raw[df_raw.index > '2024-12-31'].copy()

results = {}

# 1) 전략A 단독
r = backtest(df_test_raw, ai_probs=None, label='전략A')
results['전략A'] = r

# 2) 전략A + LSTM (임계값 0.55, 0.60, 0.65)
for threshold in [0.55, 0.60, 0.65]:
    r = backtest(df_test_raw, ai_probs=prob_lstm,
                 ai_threshold=threshold, label=f'A+LSTM({threshold})')
    results[f'A+LSTM({threshold})'] = r

# 3) 전략A + Transformer
r = backtest(df_test_raw, ai_probs=prob_tf,
             ai_threshold=0.60, label='A+Transformer(0.60)')
results['A+Transformer'] = r


# ── CELL 12: 최종 결과 출력 ──────────────────────────────────
print("\n" + "=" * 65)
print("  2025년 테스트셋 백테스트 결과 비교")
print("=" * 65)
print(f"  {'전략':>18}  {'수익률':>8}  {'거래수':>5}  {'승률':>6}  {'손익비':>6}  {'MDD':>6}")
print("  " + "-" * 58)

for name, r in results.items():
    print(f"  {r['label']:>18}  {r['ret']:>+7.1f}%  {r['n']:>5}"
          f"  {r['wr']:>5.1f}%  {r['pf']:>6.2f}  {r['mdd']:>5.1f}%")

print("  " + "-" * 58)

# B&H 비교
bah_ret = (df_test_raw['close'].iloc[-1] - df_test_raw['close'].iloc[0]) \
           / df_test_raw['close'].iloc[0] * 100
print(f"  {'B&H':>18}  {bah_ret:>+7.1f}%  {'  -':>5}  {'  -':>6}  {'  -':>6}  {'  -':>6}")
print()


# ── CELL 13: 확률 분포 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, prob, name in zip(axes, [prob_lstm, prob_tf], ['LSTM', 'Transformer']):
    ax.hist(prob[y_test==1], bins=50, alpha=0.6, label='실제 상승', color='green', density=True)
    ax.hist(prob[y_test==0], bins=50, alpha=0.6, label='실제 하락', color='red',   density=True)
    ax.axvline(0.60, color='black', linestyle='--', label='임계값 0.60')
    ax.set_title(f'{name} 예측 확률 분포')
    ax.set_xlabel('예측 확률 (상승)')
    ax.set_ylabel('밀도')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('probability_dist.png', dpi=150)
plt.show()
print("probability_dist.png 저장됨")

print("\n완료! 모델 파일 저장 중...")
lstm_model.save('lstm_model.keras')
tf_model.save('transformer_model.keras')
print("lstm_model.keras, transformer_model.keras 저장됨")
