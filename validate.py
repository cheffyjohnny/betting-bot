"""
연도별 아웃오브샘플 백테스트 검증
- 2021~2025 각 연도별로 구전략 vs 전략 A (Macro+Donchian) 비교
- 2025년은 바이앤홀드 벤치마크 포함
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import ccxt

from indicators import add_indicators

UPBIT_FEE = 0.0005
YEARS     = [2021, 2022, 2023, 2024, 2025]


# ── 데이터 페치 ───────────────────────────────────────────────────────────────

def fetch_year(year: int) -> pd.DataFrame:
    end_year = min(year + 1, 2026)
    start_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts   = int(datetime(end_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    if year == 2025:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    cache = Path(f'data/BTC_KRW_15m_{year}.csv')
    cache.parent.mkdir(exist_ok=True)

    if cache.exists():
        df = pd.read_csv(cache, index_col='timestamp', parse_dates=True)
        print(f'  {year}: 캐시 로드 ({len(df)}개)', flush=True)
        return df

    print(f'  {year}: 업비트에서 다운로드 중...', flush=True)
    exchange = ccxt.upbit({'enableRateLimit': True})
    all_candles = []
    since = start_ts

    while since < end_ts:
        try:
            candles = exchange.fetch_ohlcv('BTC/KRW', timeframe='15m', since=since, limit=200)
        except Exception as e:
            print(f'    재시도... ({e})', flush=True)
            time.sleep(2)
            continue
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
        print(f'    {len(all_candles)}개...', end='\r', flush=True)
        time.sleep(0.2)

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    df.to_csv(cache)
    print(f'  {year}: {len(df)}개 캔들 저장', flush=True)
    return df


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def calc_stats(trades: list, final_krw: float) -> dict:
    if not trades:
        return {'return': 0.0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'max_dd': 0.0}
    ret    = (final_krw - 1_000_000) / 1_000_000 * 100
    wins   = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 99.0
    equity, peak, max_dd = 1_000_000, 1_000_000, 0.0
    for t in trades:
        equity += t['pnl']
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak * 100)
    return {'return': ret, 'trades': len(trades), 'win_rate': wr, 'profit_factor': pf, 'max_dd': max_dd}


# ── 구전략 백테스트 ───────────────────────────────────────────────────────────

def backtest_old(df: pd.DataFrame) -> dict:
    """기존 전략: ADX 레짐 + RSI/BB/Supertrend, 고정 SL/TP"""
    rsi_buy = 35; rsi_sell = 65; adx_trend = 25; vol_mult = 1.2; sl_atr = 2.0; tp_atr = 4.0
    krw = 1_000_000; btc = 0.0; position = None; trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]; prev = df.iloc[i - 1]; price = row['close']

        if position:
            if price <= position['sl']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc * position['price'], 'win': False})
                krw, btc, position = r, 0.0, None; continue
            elif price >= position['tp']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc * position['price'], 'win': True})
                krw, btc, position = r, 0.0, None; continue

        adx    = row['adx']
        regime = 'trend' if adx > adx_trend else ('range' if adx < adx_trend - 5 else 'neutral')
        sig    = 'hold'

        if regime == 'trend':
            if (row['supertrend_dir'] == 1 and prev['macd'] < prev['macd_signal']
                    and row['macd'] > row['macd_signal']
                    and row['volume'] > row['volume_ma'] * vol_mult):
                sig = 'buy'
            elif (row['supertrend_dir'] == -1 and prev['macd'] > prev['macd_signal']
                  and row['macd'] < row['macd_signal']):
                sig = 'sell'
        elif regime == 'range':
            if row['close'] <= row['bb_lower'] * 1.01 and row['rsi'] < rsi_buy and row['close'] > row['vwap']:
                sig = 'buy'
            elif row['close'] >= row['bb_upper'] * 0.99 and row['rsi'] > rsi_sell:
                sig = 'sell'
        else:
            if row['ema9'] > row['ema21'] and prev['ema9'] <= prev['ema21']:
                sig = 'buy'
            elif row['ema9'] < row['ema21'] and prev['ema9'] >= prev['ema21']:
                sig = 'sell'

        if sig == 'buy' and position is None and krw > 10000:
            btc = (krw - krw * UPBIT_FEE) / price
            position = {'price': price, 'sl': price - row['atr'] * sl_atr, 'tp': price + row['atr'] * tp_atr}
            krw = 0.0
        elif sig == 'sell' and position is not None:
            r = btc * price * (1 - UPBIT_FEE)
            trades.append({'pnl': r - btc * position['price'], 'win': r > btc * position['price']})
            krw, btc, position = r, 0.0, None

    if position:
        lp = df['close'].iloc[-1]; r = btc * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - btc * position['price'], 'win': r > btc * position['price']}); krw = r

    return calc_stats(trades, krw)


# ── 전략 A 백테스트 ───────────────────────────────────────────────────────────

def backtest_strategy_a(df_15m: pd.DataFrame) -> dict:
    """
    전략 A (멀티 타임프레임):
    - Daily 200MA 매크로 필터
    - 4H Donchian 20캔들 돌파 + 거래량 1.5x + RSI < 75
    - 15M RSI < 65 + EMA21 위에서 정밀 진입
    - 청산: 4H ATR x 2 트레일링 스탑
    """
    # 15m → 4h 리샘플링
    df_4h = df_15m.resample('4h').agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'),   close=('close', 'last'), volume=('volume', 'sum')
    ).dropna()
    df_4h = add_indicators(df_4h)

    # 15m → 일봉으로 200일 MA
    daily   = df_15m['close'].resample('1D').last().dropna()
    ma200_s = daily.rolling(200).mean()

    def get_ma200(ts):
        # tz-naive 비교를 위해 정규화
        day = ts.normalize() if hasattr(ts, 'normalize') else ts
        idx = ma200_s.index.searchsorted(day, side='right') - 1
        return ma200_s.iloc[idx] if idx >= 0 else None

    def get_4h_row(ts):
        idx = df_4h.index.searchsorted(ts, side='right') - 1
        return df_4h.iloc[idx] if idx >= 0 else None

    krw = 1_000_000; btc = 0.0; position = None; trades = []
    SL_ATR = 2.0

    for i in range(1, len(df_15m)):
        row_15m = df_15m.iloc[i]
        price   = row_15m['close']
        ts      = df_15m.index[i]

        row_4h = get_4h_row(ts)
        if row_4h is None or pd.isna(row_4h['don_high']):
            continue

        # 트레일링 스탑: 4H ATR 기준
        if position:
            new_stop = price - row_4h['atr'] * SL_ATR
            if new_stop > position['trailing_stop']:
                position['trailing_stop'] = new_stop
            if price <= position['trailing_stop']:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc * position['price'], 'win': r > btc * position['price']})
                krw, btc, position = r, 0.0, None
                continue

        # 1단계: 매크로 필터
        ma200 = get_ma200(ts)
        if ma200 is None or pd.isna(ma200):
            continue
        if price < ma200:
            if position is not None:
                r = btc * price * (1 - UPBIT_FEE)
                trades.append({'pnl': r - btc * position['price'], 'win': r > btc * position['price']})
                krw, btc, position = r, 0.0, None
            continue

        # 2단계: 4H 돌파 확인
        if not (price > row_4h['don_high']
                and row_4h['volume'] > row_4h['volume_ma'] * 1.5
                and row_4h['rsi'] < 75):
            continue

        # 3단계: 15M 정밀 진입
        if (row_15m['rsi'] < 65
                and price > row_15m['ema21']
                and position is None
                and krw > 10000):
            btc      = (krw - krw * UPBIT_FEE) / price
            position = {'price': price, 'trailing_stop': price - row_4h['atr'] * SL_ATR}
            krw      = 0.0

    if position:
        lp = df_15m['close'].iloc[-1]; r = btc * lp * (1 - UPBIT_FEE)
        trades.append({'pnl': r - btc * position['price'], 'win': r > btc * position['price']}); krw = r

    return calc_stats(trades, krw)


def buy_and_hold(df: pd.DataFrame) -> dict:
    start = df['close'].iloc[0]; end = df['close'].iloc[-1]
    ret   = (end - start) / start * 100
    peak  = df['close'].cummax()
    return {'return': ret, 'max_dd': ((peak - df['close']) / peak * 100).max()}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 72)
    print('  연도별 백테스트 비교 (BTC/KRW 15m)')
    print('  구전략 vs 전략A MTF (Daily+4H+15M) vs B&H (2025)')
    print('=' * 72)

    print('\n[1/2] 데이터 수집 중...')
    yearly_data = {}
    for year in YEARS:
        try:
            df = fetch_year(year)
            df = add_indicators(df).dropna()
            yearly_data[year] = df
        except Exception as e:
            print(f'  {year}: 오류 - {e}')

    print('\n[2/2] 백테스트 실행 중...\n')

    header = (f'  {"연도 [시장]":>19}  {"구분":>6}  {"수익률":>8}  {"거래수":>5}'
              f'  {"승률":>6}  {"손익비":>6}  {"MDD":>7}')
    print(header)
    print('  ' + '-' * 75)

    for year in YEARS:
        if year not in yearly_data:
            continue
        df = yearly_data[year]

        bah     = buy_and_hold(df)
        mkt_ret = bah['return']
        if mkt_ret >= 50:
            mkt_label = '강세장'
        elif mkt_ret >= 10:
            mkt_label = '상승장'
        elif mkt_ret >= -10:
            mkt_label = '횡보장'
        elif mkt_ret >= -40:
            mkt_label = '하락장'
        else:
            mkt_label = '폭락장'

        rows = [
            ('구전략', backtest_old(df)),
            ('전략A',  backtest_strategy_a(df)),
            ('B&H',   {'return': mkt_ret, 'trades': 0, 'win_rate': 0,
                       'profit_factor': 0, 'max_dd': bah['max_dd']}),
        ]

        for i, (label, r) in enumerate(rows):
            if i == 0:
                prefix = f'  {year} [{mkt_label:>4} {mkt_ret:>+6.1f}%]'
            else:
                prefix = f'  {"":>19}'
            if label == 'B&H':
                print(f'{prefix}  {label:>6}  {r["return"]:>+7.1f}%  {"  -":>5}'
                      f'  {"  -":>6}  {"  -":>6}  {r["max_dd"]:>6.1f}%')
            else:
                print(f'{prefix}  {label:>6}  {r["return"]:>+7.1f}%  {r["trades"]:>5}'
                      f'  {r["win_rate"]:>5.1f}%  {r["profit_factor"]:>6.2f}  {r["max_dd"]:>6.1f}%')

        print('  ' + '-' * 75)

    # ── 5년 누적 수익 및 종합 평가 ───────────────────────────────────────────
    print()
    print('=' * 75)
    print('  5년 누적 성과 비교 (100만원 투자 시)')
    print('=' * 75)

    labels   = ['구전략', '전략A', 'B&H']
    equity   = {l: 1_000_000.0 for l in labels}
    all_rets = {l: [] for l in labels}

    for year in YEARS:
        if year not in yearly_data:
            continue
        df  = yearly_data[year]
        bah = buy_and_hold(df)

        year_rets = {
            '구전략': backtest_old(df)['return'],
            '전략A':  backtest_strategy_a(df)['return'],
            'B&H':   bah['return'],
        }
        for l in labels:
            equity[l] *= (1 + year_rets[l] / 100)
            all_rets[l].append(year_rets[l])

    print(f'\n  {"전략":>6}  {"최종 자산":>12}  {"누적 수익률":>10}  {"평균 연수익":>10}  {"손실 연도":>8}')
    print('  ' + '-' * 58)
    for l in labels:
        final      = equity[l]
        total_ret  = (final - 1_000_000) / 1_000_000 * 100
        avg_ret    = sum(all_rets[l]) / len(all_rets[l])
        loss_years = sum(1 for r in all_rets[l] if r < 0)
        print(f'  {l:>6}  {final:>11,.0f}원  {total_ret:>+9.1f}%  {avg_ret:>+9.1f}%  {loss_years:>5}년 / {len(all_rets[l])}년')

    print()
    print('  [시장 포착률 분석]')
    print(f'  {"연도":>4}  {"시장":>8}  {"구전략 포착":>10}  {"전략A 포착":>10}')
    print('  ' + '-' * 40)
    for year in YEARS:
        if year not in yearly_data:
            continue
        df      = yearly_data[year]
        mkt     = buy_and_hold(df)['return']
        old_ret = backtest_old(df)['return']
        a_ret   = backtest_strategy_a(df)['return']
        # 포착률: 상승장은 내 수익/시장 수익, 하락장은 손실 방어율
        if mkt > 0:
            old_cap = old_ret / mkt * 100
            a_cap   = a_ret   / mkt * 100
            note    = f'상승 {mkt:>+.0f}%'
        else:
            old_cap = (1 - old_ret / mkt) * 100  # 손실 방어: 내가 덜 잃을수록 높음
            a_cap   = (1 - a_ret   / mkt) * 100
            note    = f'하락 {mkt:>+.0f}%'
        print(f'  {year}  {note:>8}  {old_cap:>+8.1f}%  {a_cap:>+8.1f}%')

    print()
    print('  * 상승장 포착률: 내 수익 / 시장 수익  (100% = 시장과 동일, 높을수록 좋음)')
    print('  * 하락장 방어율: 내 손실이 시장 손실보다 적을수록 높음  (100% = 완벽 방어)')
    print()
    print('* B&H = 바이앤홀드 | MDD = 최대 낙폭')
    print('* 시장 구분: 강세(+50%~) / 상승(+10~50%) / 횡보(-10~+10%) / 하락(-40~-10%) / 폭락(-40%~)')
    print()


if __name__ == '__main__':
    main()
