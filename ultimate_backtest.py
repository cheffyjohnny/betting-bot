"""
전략 비교 백테스트 (2020~2025)
  전략A        : 200MA + 4H Donchian + 15M EMA/RSI
  전략A+캔들   : + 15M 캔들스틱 패턴 확인 (망치/장악형/샛별)
  전략A+구조   : + 1H 시장 구조 (고점/저점 상승 확인)
  Ultimate     : 위 전부 합산
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

UPBIT_FEE = 0.0005
YEARS     = [2020, 2021, 2022, 2023, 2024, 2025]
MKT_LABEL = {
    2020:'상승+306%', 2021:'강세 +76%',
    2022:'폭락 -63%', 2023:'강세+170%',
    2024:'강세+143%', 2025:'하락 -27%',
}


# ── 지표 ─────────────────────────────────────────────────────────────────────

def add_ind(df):
    d = df.copy()
    d['ema9']    = ta.ema(d['close'], length=9)
    d['ema21']   = ta.ema(d['close'], length=21)
    d['rsi']     = ta.rsi(d['close'], length=14)
    d['atr']     = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['vol_ma']  = ta.sma(d['volume'], length=20)
    d['don_high']= d['high'].rolling(20).max().shift(1)

    # ── 캔들스틱 패턴 (TA-Lib 없이 직접 구현) ────────────────────────────────
    op = d['open']; hi = d['high']; lo = d['low']; cl = d['close']
    body     = (cl - op).abs()
    rng      = hi - lo
    upper_sh = hi - cl.where(cl >= op, op)   # 위꼬리
    lower_sh = op.where(cl >= op, cl) - lo   # 아래꼬리

    # 망치(Hammer): 아래꼬리 >= 2x 몸통, 위꼬리 작음, 범위의 하단 1/3에 몸통
    d['hammer'] = (
        (lower_sh >= 2 * body) &
        (upper_sh <= 0.3 * body) &
        (body > 0)
    ).astype(int)

    # 상승장악형(Bullish Engulfing): 전봉 음봉, 현봉 양봉으로 완전히 감쌈
    d['engulfing'] = (
        (op.shift(1) > cl.shift(1)) &   # 전봉 음봉
        (cl > op) &                       # 현봉 양봉
        (op <= cl.shift(1)) &             # 현봉 시가 <= 전봉 종가
        (cl >= op.shift(1))               # 현봉 종가 >= 전봉 시가
    ).astype(int)

    # 샛별(Morning Star): 3봉 패턴 - 큰음봉 + 작은몸통 + 큰양봉
    big_bear  = (op.shift(2) > cl.shift(2)) & (body.shift(2) > rng.shift(2) * 0.6)
    small_mid = body.shift(1) < rng.shift(1) * 0.3
    big_bull  = (cl > op) & (body > rng * 0.5)
    closes_above_mid = cl > (op.shift(2) + cl.shift(2)) / 2
    d['morningstar'] = (big_bear & small_mid & big_bull & closes_above_mid).astype(int)

    return d.dropna()


def build_ma200(df):
    daily = df['close'].resample('1D').last().dropna()
    ma    = daily.rolling(200).mean()
    def get(ts):
        day = ts.normalize() if hasattr(ts,'normalize') else ts
        i   = ma.index.searchsorted(day, side='right') - 1
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


def build_1h_structure(df):
    """
    1H 시장 구조: 최근 피벗들이 상승 구조인지 확인
    HH (Higher High) + HL (Higher Low) = 상승 구조
    """
    d1h = df['close'].resample('1h').last().dropna()

    def find_pivots(series, window=5):
        highs, lows = [], []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                highs.append((series.index[i], series.iloc[i]))
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                lows.append((series.index[i], series.iloc[i]))
        return highs, lows

    highs, lows = find_pivots(d1h, window=3)

    # 각 타임스탬프에서 가장 최근 피벗 2개씩 가져와 상승 구조 확인
    def is_bull_structure(ts):
        past_highs = [(t, v) for t, v in highs if t < ts][-2:]
        past_lows  = [(t, v) for t, v in lows  if t < ts][-2:]
        if len(past_highs) < 2 or len(past_lows) < 2:
            return False
        hh = past_highs[-1][1] > past_highs[-2][1]  # 고점 상승
        hl = past_lows[-1][1]  > past_lows[-2][1]   # 저점 상승
        return hh and hl

    return is_bull_structure


# ── 백테스트 코어 ─────────────────────────────────────────────────────────────

def run_backtest(df, use_candle=False, use_structure=False):
    get_ma200 = build_ma200(df)
    get_4h    = build_4h(df)
    get_struct = build_1h_structure(df) if use_structure else None

    krw = 1_000_000; btc = 0.0; pos = None; trades = []

    for i in range(1, len(df)):
        row   = df.iloc[i]
        price = row['close']
        ts    = df.index[i]

        r4h = get_4h(ts)
        if r4h is None or pd.isna(r4h['don_high']): continue

        # 트레일링 스탑
        if pos:
            new_stop = price - r4h['atr'] * 2
            if new_stop > pos['stop']: pos['stop'] = new_stop
            if price <= pos['stop']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
                continue

        # 200MA 필터
        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200): continue
        if price < ma200:
            if pos:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc*pos['price'], 'win': r > btc*pos['price']})
                krw, btc, pos = r, 0.0, None
            continue

        # 4H 돌파 필터
        if not (price > r4h['don_high']
                and r4h['volume'] > r4h['vol_ma'] * 1.5
                and r4h['rsi'] < 75): continue

        # 1H 시장 구조 필터 (선택)
        if use_structure and not get_struct(ts): continue

        # 15M 진입 조건
        any(pd.isna(row[c]) for c in ['ema9','ema21','rsi','atr'])
        ema_ok = row['rsi'] < 65 and price > row['ema21']

        # 캔들 패턴 (선택)
        if use_candle:
            bull_candle = (
                row.get('hammer', 0) > 0 or
                row.get('engulfing', 0) > 0 or
                row.get('morningstar', 0) > 0
            )
            entry_ok = ema_ok or bull_candle   # 둘 중 하나면 진입
        else:
            entry_ok = ema_ok

        if entry_ok and pos is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            pos = {'price': price, 'stop': price - r4h['atr'] * 2}
            krw = 0.0

    if pos:
        lp = df['close'].iloc[-1]
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


def bah(df):
    s = df['close'].iloc[0]; e = df['close'].iloc[-1]
    pk = df['close'].cummax()
    return (e-s)/s*100, ((pk-df['close'])/pk*100).max()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('  종합 전략 비교 백테스트  2020~2025')
    print('  전략A / A+캔들패턴 / A+시장구조 / Ultimate(전부)')
    print('=' * 80)

    strategies = [
        ('전략A',      False, False),
        ('A+캔들',     True,  False),
        ('A+구조',     False, True),
        ('Ultimate',   True,  True),
    ]

    print('\n데이터 로드 중...')
    yearly = {}
    for y in YEARS:
        f = Path(f'data/BTC_KRW_15m_{y}.csv')
        if not f.exists(): print(f'  {y}: 없음'); continue
        df = pd.read_csv(f, index_col='timestamp', parse_dates=True)
        yearly[y] = add_ind(df)
        print(f'  {y}: {len(yearly[y]):,}개', flush=True)

    # 연도별 출력
    for sname, use_c, use_s in strategies:
        print(f'\n{"─"*80}')
        print(f'  [{sname}]  캔들패턴={use_c}  시장구조={use_s}')
        print(f'  {"연도":>4}  {"시장":>10}  {"수익률":>8}  {"거래수":>5}  {"승률":>6}  {"손익비":>6}  {"MDD":>6}  {"B&H":>8}')
        print(f'  {"─"*70}')
        cum = 1_000_000.0
        for y in YEARS:
            if y not in yearly: continue
            r = run_backtest(yearly[y], use_candle=use_c, use_structure=use_s)
            br, _ = bah(yearly[y])
            cum *= (1 + r['ret']/100)
            print(f'  {y}  {MKT_LABEL.get(y,""):>10}  {r["ret"]:>+7.1f}%'
                  f'  {r["n"]:>5}  {r["wr"]:>5.1f}%  {r["pf"]:>6.2f}'
                  f'  {r["mdd"]:>5.1f}%  {br:>+7.1f}%')
        cr = (cum - 1_000_000) / 1_000_000 * 100
        print(f'  {"─"*70}')
        print(f'  6년 누적: {cum:>12,.0f}원  ({cr:>+.1f}%)')

    # 최종 비교표
    print(f'\n{"="*80}')
    print(f'  최종 비교')
    print(f'  {"전략":>12}  {"6년 누적":>12}  {"연평균":>8}  {"특징"}')
    print(f'  {"─"*60}')

    results = {}
    for sname, use_c, use_s in strategies:
        cum = 1_000_000.0
        for y in YEARS:
            if y not in yearly: continue
            r = run_backtest(yearly[y], use_candle=use_c, use_structure=use_s)
            cum *= (1 + r['ret']/100)
        cr = (cum - 1_000_000) / 1_000_000 * 100
        avg = (1 + cr/100) ** (1/6) * 100 - 100
        results[sname] = (cum, cr, avg)

    cum_bah = 1_000_000.0
    for y in YEARS:
        if y not in yearly: continue
        br, _ = bah(yearly[y])
        cum_bah *= (1 + br/100)
    cbr = (cum_bah - 1_000_000) / 1_000_000 * 100

    for sname, (cum, cr, avg) in results.items():
        tag = ''
        if cr == max(r[1] for r in results.values()):
            tag = '<-- 최고'
        print(f'  {sname:>12}  {cum:>10,.0f}원  {cr:>+7.1f}%  {avg:>+6.1f}%/년  {tag}')
    print(f'  {"B&H":>12}  {cum_bah:>10,.0f}원  {cbr:>+7.1f}%  (비교 기준선)')
    print()


if __name__ == '__main__':
    main()
