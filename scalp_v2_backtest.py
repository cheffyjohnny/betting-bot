"""
양방향 단타 전략 연도별 백테스트 (2020~2025)
- 상승장 (200일MA 위): 롱 전략 (골든크로스 후 눌림목)
- 하락장 (200일MA 아래): 숏 전략 (데드크로스 후 되돌림)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import datetime, timezone
import ccxt, time

YEARS        = [2020, 2021, 2022, 2023, 2024, 2025]
FEE          = 0.0004
TP_ATR       = 2.0
SL_ATR       = 1.0
MAX_CANDLES  = 10
CW           = 6        # crossover window
MIN_ATR_PCT  = 0.002


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_year(year):
    cache = Path(f'data/BTC_KRW_15m_{year}.csv')
    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f'  {year}: {len(df):,}개 로드', flush=True)
        return df

    end_year = min(year + 1, 2026)
    start_ts = int(datetime(year,1,1,tzinfo=timezone.utc).timestamp()*1000)
    end_ts   = int(datetime(end_year,1,1,tzinfo=timezone.utc).timestamp()*1000)
    if year == 2025:
        end_ts = int(datetime.now(timezone.utc).timestamp()*1000)

    print(f'  {year}: 다운로드 중...', flush=True)
    ex = ccxt.upbit({'enableRateLimit': True})
    rows = []; since = start_ts
    while since < end_ts:
        try:
            c = ex.fetch_ohlcv('BTC/KRW', '15m', since=since, limit=200)
        except: time.sleep(2); continue
        if not c: break
        c = [x for x in c if x[0] < end_ts]
        if not c: break
        rows.extend(c)
        last = c[-1][0]
        if last <= since: break
        since = last + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df.to_csv(cache)
    print(f'  {year}: {len(df):,}개 저장', flush=True)
    return df


# ── 지표 ─────────────────────────────────────────────────────────────────────

def add_ind(df):
    d = df.copy()
    d['ema9']   = ta.ema(d['close'], length=9)
    d['ema21']  = ta.ema(d['close'], length=21)
    d['rsi']    = ta.rsi(d['close'], length=14)
    d['atr']    = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['vol_ma'] = ta.sma(d['volume'], length=20)
    return d.dropna()


def build_daily_ma(df, period=200):
    daily = df['close'].resample('1D').last().dropna()
    ma    = daily.rolling(period).mean()
    def get(ts):
        day = ts.normalize() if hasattr(ts, 'normalize') else ts
        i   = ma.index.searchsorted(day, side='right') - 1
        return float(ma.iloc[i]) if i >= 0 else None
    return get


def build_4h_trend(df):
    c4h      = df['close'].resample('4h').last().dropna()
    ema21_4h = ta.ema(c4h, length=21)
    up       = ema21_4h > ema21_4h.shift(2)
    dn       = ema21_4h < ema21_4h.shift(2)
    def get(ts, direction):
        i = up.index.searchsorted(ts, side='right') - 1
        if i < 0: return False
        return bool(up.iloc[i]) if direction == 'long' else bool(dn.iloc[i])
    return get


# ── 백테스트 ─────────────────────────────────────────────────────────────────

def backtest(df):
    get_ma200  = build_daily_ma(df)
    get_4h     = build_4h_trend(df)

    usdt = 1_000.0
    pos  = None
    trades = []
    last_long_x  = -999
    last_short_x = -999
    ci = 0

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        ts   = df.index[i]
        price = row['close']
        ci   += 1

        # ── 청산 체크 ────────────────────────────────────────────────────────
        if pos:
            held = i - pos['ei']
            rsn  = None
            if pos['dir'] == 'long':
                if price >= pos['tp']: rsn = 'TP'
                elif price <= pos['sl']: rsn = 'SL'
            else:
                if price <= pos['tp']: rsn = 'TP'
                elif price >= pos['sl']: rsn = 'SL'
            if held >= MAX_CANDLES: rsn = 'timeout'

            if rsn:
                if pos['dir'] == 'long':
                    pnl_pct = (price - pos['price']) / pos['price'] * 100 - FEE * 200
                else:
                    pnl_pct = (pos['price'] - price) / pos['price'] * 100 - FEE * 200
                pnl  = usdt * pnl_pct / 100
                usdt = max(usdt + pnl, 0.01)
                trades.append({'pnl': pnl, 'win': pnl > 0,
                               'reason': rsn, 'held': held * 15,
                               'dir': pos['dir']})
                pos = None
                continue

        # ── 공통 사전 조건 ───────────────────────────────────────────────────
        if any(pd.isna(row[c]) for c in ['ema9','ema21','rsi','atr','vol_ma']):
            continue

        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200):
            continue

        above = price > ma200   # True=상승장(롱) / False=하락장(숏)

        # 크로스오버 감지
        if prev['ema9'] <= prev['ema21'] and row['ema9'] > row['ema21']:
            last_long_x = ci
        if prev['ema9'] >= prev['ema21'] and row['ema9'] < row['ema21']:
            last_short_x = ci

        atr_ok    = row['atr'] > price * MIN_ATR_PCT
        volume_ok = row['volume'] > row['vol_ma'] * 1.2

        # ── 롱 진입 ─────────────────────────────────────────────────────────
        if above and not pos:
            since = ci - last_long_x
            if (1 <= since <= CW
                    and get_4h(ts, 'long')
                    and price <= row['ema9'] * 1.001
                    and 40 <= row['rsi'] <= 58
                    and volume_ok and atr_ok):
                pos = {'dir': 'long', 'price': price, 'ei': i,
                       'tp': price + row['atr'] * TP_ATR,
                       'sl': price - row['atr'] * SL_ATR}

        # ── 숏 진입 ─────────────────────────────────────────────────────────
        elif not above and not pos:
            since = ci - last_short_x
            if (1 <= since <= CW
                    and get_4h(ts, 'short')
                    and price >= row['ema9'] * 0.999
                    and 42 <= row['rsi'] <= 60
                    and volume_ok and atr_ok):
                pos = {'dir': 'short', 'price': price, 'ei': i,
                       'tp': price - row['atr'] * TP_ATR,
                       'sl': price + row['atr'] * SL_ATR}

    # 미청산 종가 처리
    if pos:
        lp = df['close'].iloc[-1]
        pnl_pct = ((lp - pos['price']) / pos['price'] if pos['dir'] == 'long'
                   else (pos['price'] - lp) / pos['price']) * 100 - FEE * 200
        pnl  = usdt * pnl_pct / 100
        usdt = max(usdt + pnl, 0.01)
        trades.append({'pnl': pnl, 'win': pnl > 0, 'reason': 'end',
                       'held': (len(df) - pos['ei']) * 15, 'dir': pos['dir']})

    if not trades:
        return dict(ret=0, n=0, wr=0, pf=0, mdd=0,
                    n_long=0, n_short=0, avg_held=0,
                    tp=0, sl=0, timeout=0)

    ret    = (usdt - 1_000) / 1_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 99.0
    n      = len(trades)

    equity = peak = 1_000.0
    max_dd = 0.0
    for t in trades:
        equity += t['pnl']
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak * 100)

    return dict(
        ret=ret, n=n, wr=wr, pf=pf, mdd=max_dd,
        n_long=sum(1 for t in trades if t['dir']=='long'),
        n_short=sum(1 for t in trades if t['dir']=='short'),
        avg_held=sum(t['held'] for t in trades)/n,
        tp=sum(1 for t in trades if t['reason']=='TP')/n*100,
        sl=sum(1 for t in trades if t['reason']=='SL')/n*100,
        timeout=sum(1 for t in trades if t['reason']=='timeout')/n*100,
    )


def bah(df):
    s = df['close'].iloc[0]; e = df['close'].iloc[-1]
    peak = df['close'].cummax()
    mdd  = ((peak - df['close']) / peak * 100).max()
    return (e - s) / s * 100, mdd


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 76)
    print('  양방향 단타 백테스트  2020~2025  (상승장=롱 / 하락장=숏)')
    print(f'  TP={TP_ATR}xATR  SL={SL_ATR}xATR  R:R=2:1  최대보유={MAX_CANDLES*15}분')
    print('=' * 76)

    print('\n[1/2] 데이터 로드...')
    yearly = {}
    for y in YEARS:
        try:
            df = load_year(y)
            yearly[y] = add_ind(df)
        except Exception as e:
            print(f'  {y}: 오류 - {e}')

    mkt_label = {
        2020: '상승 +306%', 2021: '강세  +76%',
        2022: '폭락  -63%', 2023: '강세 +170%',
        2024: '강세 +143%', 2025: '하락  -27%',
    }

    print('\n[2/2] 백테스트 실행...\n')
    print(f'  {"연도":>4}  {"시장":>11}  {"전략":>5}  {"수익률":>8}  {"거래":>5}'
          f'  {"(롱/숏)":>8}  {"승률":>6}  {"손익비":>6}  {"MDD":>6}  {"B&H":>8}')
    print('  ' + '-' * 80)

    cum_strat = 1_000.0
    cum_bah   = 1_000.0

    for y in YEARS:
        if y not in yearly: continue
        df = yearly[y]
        r  = backtest(df)
        bah_ret, bah_mdd = bah(df)
        regime = '롱만' if bah_ret > 0 else '숏만'

        cum_strat *= (1 + r['ret'] / 100)
        cum_bah   *= (1 + bah_ret / 100)

        print(f'  {y}  {mkt_label.get(y,""):>11}  {regime:>5}'
              f'  {r["ret"]:>+7.1f}%  {r["n"]:>5}'
              f'  ({r["n_long"]:>2}L/{r["n_short"]:>2}S)'
              f'  {r["wr"]:>5.1f}%  {r["pf"]:>6.2f}'
              f'  {r["mdd"]:>5.1f}%  {bah_ret:>+7.1f}%')

    print('  ' + '-' * 80)
    cum_ret  = (cum_strat - 1_000) / 1_000 * 100
    bah_cum  = (cum_bah   - 1_000) / 1_000 * 100
    print(f'\n  6년 누적 | 전략: ${cum_strat:>8,.0f}  ({cum_ret:>+.1f}%)'
          f'   vs   B&H: ${cum_bah:>8,.0f}  ({bah_cum:>+.1f}%)')
    print()
    print('  * L=롱거래수 / S=숏거래수 / B&H=바이앤홀드 연간수익')
    print()


if __name__ == '__main__':
    main()
