"""
단타 전략 백테스트 v3: 4H Trend + 15M 눌림목 진입
------------------------------------------------------
핵심 원칙:
  1. 크로스오버 "후" 눌림목 진입 (후행 진입 문제 해결)
  2. 4H 방향 + 일봉 200MA 이중 필터
  3. 죽은 시장 제외 (최소 변동성)
  4. R:R 2:1

전략:
  방향 이중 필터:
    - 일봉 200MA 위 (대세 상승장)
    - 4H EMA21 기울기 상승
  진입 "눌림목" (15M):
    1. EMA9 > EMA21 상태 유지 (크로스오버 후 최대 6캔들 이내)
    2. 현재 가격이 EMA9에 닿거나 아래 (눌림목)
    3. RSI(14): 40~58 (눌림 구간, 과열 아님)
    4. 거래량 > MA20 x 1.2 (최소 확인)
    5. ATR > 가격의 0.2% (최소 변동성)
  청산:
    익절: ATR x 2.0
    손절: ATR x 1.0
    R:R = 2:1
    최대 보유: 10캔들 (150분) 초과시 강제 청산
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timezone

import ccxt

UPBIT_FEE        = 0.0005
YEARS            = [2022, 2023, 2024, 2025]
TP_ATR           = 2.0
SL_ATR           = 1.0
MAX_CANDLES      = 10      # 최대 보유 (150분)
MIN_ATR_PCT      = 0.002   # 최소 변동성: 가격의 0.2%
CROSSOVER_WINDOW = 6       # 크로스오버 후 최대 몇 캔들 이내에서 눌림목 찾을지


# ── 데이터 ────────────────────────────────────────────────────────────────────

def fetch_15m(year: int) -> pd.DataFrame:
    cache = Path(f'data/BTC_KRW_15m_{year}.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f'  {year}: 캐시 로드 ({len(df):,}개)', flush=True)
        return df

    end_year = min(year + 1, 2026)
    start_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime(end_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    if year == 2025:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f'  {year}: 15분봉 다운로드 중...', flush=True)
    exchange = ccxt.upbit({'enableRateLimit': True})
    all_candles = []
    since = start_ts

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv('BTC/KRW', timeframe='15m', since=since, limit=200)
        except Exception as e:
            time.sleep(2); continue
        if not candles:
            break
        candles = [c for c in candles if c[0] < end_ts]
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df.to_csv(cache)
    print(f'  {year}: {len(df):,}개 저장', flush=True)
    return df


# ── 지표 ─────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema9']      = ta.ema(df['close'], length=9)
    df['ema21']     = ta.ema(df['close'], length=21)
    df['rsi']       = ta.rsi(df['close'], length=14)
    df['atr']       = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volume_ma'] = ta.sma(df['volume'], length=20)
    df['vwap']      = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    return df


def build_filters(df_15m: pd.DataFrame):
    """
    이중 방향 필터:
      - 일봉 200MA 위
      - 4H EMA21 기울기 상승
    각 15m 캔들에 bool 매핑 반환
    """
    # 일봉 200MA
    daily_close = df_15m['close'].resample('1D').last().dropna()
    ma200       = daily_close.rolling(200).mean()

    # 4H EMA21 기울기
    close_4h  = df_15m['close'].resample('4h').last().dropna()
    ema21_4h  = ta.ema(close_4h, length=21)
    trend_4h  = ema21_4h > ema21_4h.shift(2)

    result = pd.Series(False, index=df_15m.index)
    for ts in df_15m.index:
        price = df_15m.loc[ts, 'close']

        # 일봉 200MA 확인
        d_idx = ma200.index.searchsorted(ts.normalize() if hasattr(ts,'normalize') else ts, side='right') - 1
        if d_idx < 0:
            continue
        ma200_val = ma200.iloc[d_idx]
        if pd.isna(ma200_val) or price < ma200_val:
            continue

        # 4H 기울기 확인
        h4_idx = trend_4h.index.searchsorted(ts, side='right') - 1
        if h4_idx < 0 or not bool(trend_4h.iloc[h4_idx]):
            continue

        result[ts] = True
    return result


# ── 백테스트 ─────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame) -> dict:
    allow = build_filters(df)

    krw = 1_000_000.0; btc = 0.0; position = None; trades = []
    last_crossover_idx = -999   # 마지막 EMA 크로스오버가 일어난 캔들 인덱스

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']

        # 포지션 청산 체크
        if position:
            held   = i - position['entry_idx']
            reason = None
            if price >= position['tp']:
                reason = 'TP'
            elif price <= position['sl']:
                reason = 'SL'
            elif held >= MAX_CANDLES:
                reason = 'timeout'

            if reason:
                r   = btc * price * (1 - UPBIT_FEE)
                pnl = r - btc * position['price']
                trades.append({'pnl': pnl, 'win': pnl > 0, 'reason': reason, 'held': held})
                krw, btc, position = r, 0.0, None
                continue

        # NaN 방어
        if any(pd.isna(row[c]) for c in ['ema9','ema21','rsi','atr','vwap','volume_ma']):
            continue

        # 크로스오버 감지: 기록만 해둠 (진입 X)
        if prev['ema9'] <= prev['ema21'] and row['ema9'] > row['ema21']:
            last_crossover_idx = i

        # 1단계: 이중 방향 필터 (200MA + 4H)
        if not allow.iloc[i]:
            continue

        # 2단계: 크로스오버 후 CROSSOVER_WINDOW 캔들 이내
        candles_since_cross = i - last_crossover_idx
        if not (1 <= candles_since_cross <= CROSSOVER_WINDOW):
            continue

        # 3단계: 눌림목 — EMA9 위에서 가격이 EMA9에 닿거나 그 아래 (최대 0.1% 이내)
        pullback = price <= row['ema9'] * 1.001

        # 4단계: 나머지 조건
        rsi_ok    = 40 <= row['rsi'] <= 58
        volume_ok = row['volume'] > row['volume_ma'] * 1.2
        atr_ok    = row['atr'] > price * MIN_ATR_PCT

        if pullback and rsi_ok and volume_ok and atr_ok and position is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            position = {
                'price':     price,
                'tp':        price + row['atr'] * TP_ATR,
                'sl':        price - row['atr'] * SL_ATR,
                'entry_idx': i,
            }
            krw = 0.0

    if position:
        lp  = df['close'].iloc[-1]
        r   = btc * lp * (1 - UPBIT_FEE)
        pnl = r - btc * position['price']
        trades.append({'pnl': pnl, 'win': pnl > 0, 'reason': 'end',
                       'held': len(df) - position['entry_idx']})
        krw = r

    if not trades:
        return dict(ret=0.0, n=0, wr=0.0, pf=0.0, mdd=0.0,
                    avg_hold=0.0, tp=0.0, sl=0.0, timeout=0.0)

    ret    = (krw - 1_000_000) / 1_000_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 99.0

    equity = peak = 1_000_000.0; max_dd = 0.0
    for t in trades:
        equity += t['pnl']
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak * 100)

    n = len(trades)
    return dict(
        ret=ret, n=n, wr=wr, pf=pf, mdd=max_dd,
        avg_hold=sum(t['held'] for t in trades) / n * 15,
        tp=sum(1 for t in trades if t['reason']=='TP')      / n * 100,
        sl=sum(1 for t in trades if t['reason']=='SL')      / n * 100,
        timeout=sum(1 for t in trades if t['reason']=='timeout') / n * 100,
    )


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 72)
    print('  단타 백테스트 v3: 200MA+4H 이중필터 + 15M 눌림목 진입')
    print(f'  TP={TP_ATR}xATR  SL={SL_ATR}xATR  R:R={TP_ATR/SL_ATR:.1f}:1  최대보유={MAX_CANDLES*15}분')
    print('=' * 72)

    print('\n[1/2] 데이터 로드 중...')
    yearly = {}
    for year in YEARS:
        try:
            df = fetch_15m(year)
            df = add_indicators(df).dropna()
            yearly[year] = df
        except Exception as e:
            print(f'  {year}: 오류 - {e}')

    print('\n[2/2] 백테스트 실행 중...\n')

    mkt = {2022:'폭락 -63%', 2023:'강세+170%', 2024:'강세+143%', 2025:'하락 -27%'}

    print(f'  {"연도":>4}  {"시장":>10}  {"수익률":>8}  {"거래수":>5}  {"승률":>6}'
          f'  {"손익비":>6}  {"MDD":>6}  {"평균보유":>8}  {"TP%":>5}  {"SL%":>5}  {"시간초과":>7}')
    print('  ' + '-' * 86)

    cum = 1_000_000.0
    for year in YEARS:
        if year not in yearly:
            continue
        r   = backtest(yearly[year])
        cum *= (1 + r['ret'] / 100)
        print(f'  {year}  {mkt.get(year,""):>10}  {r["ret"]:>+7.1f}%  {r["n"]:>5}'
              f'  {r["wr"]:>5.1f}%  {r["pf"]:>6.2f}  {r["mdd"]:>5.1f}%'
              f'  {r["avg_hold"]:>6.0f}분  {r["tp"]:>5.1f}%  {r["sl"]:>5.1f}%  {r["timeout"]:>6.1f}%')

    print('  ' + '-' * 86)
    print(f'\n  {len(YEARS)}년 누적: {cum:,.0f}원  ({(cum-1_000_000)/1_000_000*100:+.1f}%)')
    print()
    print('  * TP/SL/시간초과 = 해당 사유로 청산된 비율')
    print()


if __name__ == '__main__':
    main()
