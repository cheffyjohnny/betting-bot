"""
멀티코인 전략A 백테스트 (2020~2025)
대상: BTC, ETH, XRP, SOL / 업비트 KRW 마켓
전략: 200MA + 4H Donchian + 15M EMA/RSI + ATR×2 트레일링 스탑
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import ccxt, time

UPBIT_FEE = 0.0005
YEARS     = [2020, 2021, 2022, 2023, 2024, 2025]
COINS     = ['BTC', 'ETH', 'XRP', 'SOL']

# 코인별 BTC 시장 레이블 (BTC 기준 참고용)
BTC_MKT = {
    2020:'BTC+306%', 2021:'BTC +76%',
    2022:'BTC -63%', 2023:'BTC+170%',
    2024:'BTC+143%', 2025:'BTC -27%',
}


# ── 데이터 로드 / 다운로드 ────────────────────────────────────────────────────

def load_or_download(coin, year):
    cache = Path(f'data/{coin}_KRW_15m_{year}.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f'  {coin} {year}: 캐시 {len(df):,}개', flush=True)
        return df

    end_year = min(year + 1, 2026)
    start_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime(end_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    if year == 2025:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f'  {coin} {year}: 다운로드 중...', flush=True)
    ex   = ccxt.upbit({'enableRateLimit': True})
    sym  = f'{coin}/KRW'
    rows = []
    since = start_ts

    while since < end_ts:
        try:
            c = ex.fetch_ohlcv(sym, '15m', since=since, limit=200)
        except Exception as e:
            print(f'    재시도 ({e})')
            time.sleep(3)
            continue
        if not c:
            break
        c = [x for x in c if x[0] < end_ts]
        if not c:
            break
        rows.extend(c)
        last = c[-1][0]
        if last <= since:
            break
        since = last + 1
        time.sleep(0.2)

    if not rows:
        print(f'  {coin} {year}: 데이터 없음')
        return None

    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df = df.astype(float)
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache)
    print(f'  {coin} {year}: {len(df):,}개 저장', flush=True)
    return df


# ── 지표 ─────────────────────────────────────────────────────────────────────

def add_ind(df):
    d = df.copy()
    d['ema9']     = ta.ema(d['close'], length=9)
    d['ema21']    = ta.ema(d['close'], length=21)
    d['rsi']      = ta.rsi(d['close'], length=14)
    d['atr']      = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['vol_ma']   = ta.sma(d['volume'], length=20)
    d['don_high'] = d['high'].rolling(20).max().shift(1)
    return d.dropna()


def build_ma200(df):
    daily = df['close'].resample('1D').last().dropna()
    ma    = daily.rolling(200).mean()
    def get(ts):
        i = ma.index.searchsorted(ts, side='right') - 1
        return float(ma.iloc[i]) if i >= 0 else None
    return get


def build_4h(df):
    d4 = df.resample('4h').agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna()
    d4 = add_ind(d4)
    def get(ts):
        i = d4.index.searchsorted(ts, side='right') - 1
        return d4.iloc[i] if i >= 0 else None
    return get


# ── 백테스트 ─────────────────────────────────────────────────────────────────

def run_backtest(df):
    get_ma200 = build_ma200(df)
    get_4h    = build_4h(df)

    krw = 1_000_000; coin_qty = 0.0; pos = None; trades = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = row['close']
        ts    = df.index[i]

        r4h = get_4h(ts)
        if r4h is None or pd.isna(r4h['don_high']):
            continue

        # 트레일링 스탑
        if pos:
            new_stop = price - r4h['atr'] * 2
            if new_stop > pos['stop']:
                pos['stop'] = new_stop
            if price <= pos['stop']:
                r = coin_qty * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - coin_qty * pos['price'], 'win': r > coin_qty * pos['price']})
                krw, coin_qty, pos = r, 0.0, None
                continue

        # 200MA 필터
        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200):
            continue
        if price < ma200:
            if pos:
                r = coin_qty * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - coin_qty * pos['price'], 'win': r > coin_qty * pos['price']})
                krw, coin_qty, pos = r, 0.0, None
            continue

        # 4H 돌파 필터
        if not (price > r4h['don_high']
                and r4h['volume'] > r4h['vol_ma'] * 1.5
                and r4h['rsi'] < 75):
            continue

        # 15M 진입 조건
        if any(pd.isna(row[c]) for c in ['ema9', 'ema21', 'rsi', 'atr']):
            continue
        entry_ok = row['rsi'] < 65 and price > row['ema21']

        if entry_ok and pos is None and krw > 10000:
            coin_qty = (krw - krw * UPBIT_FEE) / price
            pos = {'price': price, 'stop': price - r4h['atr'] * 2}
            krw = 0.0

    # 미청산 종가 처리
    if pos:
        lp = df['close'].iloc[-1]
        r  = coin_qty * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - coin_qty * pos['price'], 'win': r > coin_qty * pos['price']})
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


def bah(df):
    s = df['close'].iloc[0]; e = df['close'].iloc[-1]
    pk = df['close'].cummax()
    return (e - s) / s * 100, ((pk - df['close']) / pk * 100).max()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('  멀티코인 전략A 백테스트  2020~2025')
    print(f'  대상: {" / ".join(COINS)}')
    print('=' * 80)

    # 데이터 로드
    print('\n[1/2] 데이터 로드 중...')
    data = {}  # data[coin][year] = df
    for coin in COINS:
        data[coin] = {}
        for year in YEARS:
            df = load_or_download(coin, year)
            if df is not None and len(df) > 500:
                data[coin][year] = add_ind(df)

    # 코인별 백테스트
    print('\n[2/2] 백테스트 실행...\n')
    summary = {}  # summary[coin] = (cum, cr, avg_mdd)

    for coin in COINS:
        print(f'\n{"─"*80}')
        print(f'  [{coin}/KRW]')
        print(f'  {"연도":>4}  {"BTC시장":>9}  {"수익률":>8}  {"거래수":>5}  {"승률":>6}  {"손익비":>6}  {"MDD":>6}  {"B&H":>8}')
        print(f'  {"─"*68}')

        cum = 1_000_000.0
        mdd_list = []

        for year in YEARS:
            if year not in data[coin]:
                print(f'  {year}  {"데이터 없음":>30}')
                continue

            r = run_backtest(data[coin][year])
            br, _ = bah(data[coin][year])
            cum *= (1 + r['ret'] / 100)
            mdd_list.append(r['mdd'])

            print(f'  {year}  {BTC_MKT.get(year,""):>9}  {r["ret"]:>+7.1f}%'
                  f'  {r["n"]:>5}  {r["wr"]:>5.1f}%  {r["pf"]:>6.2f}'
                  f'  {r["mdd"]:>5.1f}%  {br:>+7.1f}%')

        cr   = (cum - 1_000_000) / 1_000_000 * 100
        avg  = (1 + cr / 100) ** (1 / 6) * 100 - 100
        mmdd = max(mdd_list) if mdd_list else 0
        summary[coin] = (cum, cr, avg, mmdd)

        print(f'  {"─"*68}')
        print(f'  6년 누적: {cum:>12,.0f}원  ({cr:>+.1f}%)')

    # 최종 비교표
    print(f'\n{"="*80}')
    print(f'  최종 비교 (코인별 전략A 성과)')
    print(f'  {"코인":>6}  {"6년 누적":>13}  {"누적수익":>9}  {"연평균":>8}  {"최대MDD":>8}')
    print(f'  {"─"*58}')

    best_coin = max(summary, key=lambda c: summary[c][1])
    for coin, (cum, cr, avg, mmdd) in summary.items():
        tag = '  <-- 최고' if coin == best_coin else ''
        print(f'  {coin:>6}  {cum:>12,.0f}원  {cr:>+8.1f}%  {avg:>+6.1f}%/년  {mmdd:>6.1f}%{tag}')
    print()

    # 포트폴리오 시뮬레이션 (4코인 균등 분배)
    print(f'  {"─"*58}')
    print(f'  [포트폴리오] 4코인 균등 분배 (각 25%)')
    port_cum = 1_000_000.0
    for year in YEARS:
        year_ret = []
        for coin in COINS:
            if year in data[coin]:
                r = run_backtest(data[coin][year])
                year_ret.append(r['ret'])
        if year_ret:
            avg_ret = sum(year_ret) / len(COINS)
            port_cum *= (1 + avg_ret / 100)

    port_cr = (port_cum - 1_000_000) / 1_000_000 * 100
    port_avg = (1 + port_cr / 100) ** (1 / 6) * 100 - 100
    print(f'  6년 누적: {port_cum:>12,.0f}원  ({port_cr:>+.1f}%)  연평균 {port_avg:>+.1f}%')
    print()


if __name__ == '__main__':
    main()
