"""
모멘텀 로테이션 결과 검증
- 매 리밸런싱 시점의 보유 코인 / 진입가 / 청산가 / 수익 추적
- 연도별 실제 보유 내역 출력
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

COINS = ['BTC', 'ETH', 'XRP', 'SOL']
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FEE   = 0.0005


def load_daily(coin):
    frames = []
    for year in YEARS:
        f = Path(f'data/{coin}_KRW_15m_{year}.csv')
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col='timestamp', parse_dates=True)
        frames.append(df['close'])
    if not frames:
        return None
    s = pd.concat(frames).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    return s.resample('1D').last().dropna()


def run_with_log(prices, lookback=14, hold=7, top_n=1):
    all_dates  = sorted(set.union(*[set(s.index) for s in prices.values()]))
    all_dates  = pd.DatetimeIndex(all_dates)

    krw        = 1_000_000.0
    holdings   = {}
    last_rebal = None
    log        = []   # 리밸런싱 로그
    equity     = []   # 일별 자산 추적

    for i, date in enumerate(all_dates):
        # 일별 포트폴리오 가치 계산
        port = krw
        for c, qty in holdings.items():
            s = prices[c]
            if date in s.index and not pd.isna(s[date]):
                port += qty * s[date]
        equity.append({'date': date, 'value': port})

        if i < lookback:
            continue
        if last_rebal is not None and (date - last_rebal).days < hold:
            continue

        available = {c: s for c, s in prices.items()
                     if date in s.index and not pd.isna(s[date])}
        if len(available) < 2:
            continue

        past_date_idx = i - lookback
        if past_date_idx < 0:
            continue
        past_date = all_dates[past_date_idx]

        moms = {}
        for c, s in available.items():
            if past_date in s.index and not pd.isna(s[past_date]):
                moms[c] = (s[date] - s[past_date]) / s[past_date]
        if not moms:
            continue

        ranked  = sorted(moms, key=moms.get, reverse=True)
        targets = ranked[:min(top_n, len(ranked))]

        current = set(holdings.keys())
        if current == set(targets):
            last_rebal = date
            continue

        # 매도
        sell_val = 0.0
        for c in list(holdings.keys()):
            if c in available:
                sell_p = available[c][date]
                val    = holdings[c] * sell_p * (1 - FEE)
                krw   += val
                sell_val += val
            holdings.pop(c)

        # 매수
        alloc = krw / len(targets)
        buy_info = []
        for c in targets:
            buy_p = available[c][date]
            qty   = alloc * (1 - FEE) / buy_p
            holdings[c] = qty
            krw -= alloc
            buy_info.append(f'{c}@{buy_p:,.0f}')

        # 모멘텀 순위 기록
        mom_str = ' | '.join([f'{c}:{moms[c]*100:+.1f}%' for c in ranked[:4] if c in moms])

        log.append({
            'date'    : date.date(),
            'buy'     : ', '.join(buy_info),
            'moms'    : mom_str,
            'krw'     : krw + sum(holdings[c] * available[c][date] * (1-FEE)
                                  for c in holdings if c in available),
        })
        last_rebal = date

    # 마지막 청산
    last_date = all_dates[-1]
    for c, qty in list(holdings.items()):
        s = prices[c]
        p = s[last_date] if last_date in s.index else s[s.index <= last_date].iloc[-1]
        krw += qty * p * (1 - FEE)

    eq_df  = pd.DataFrame(equity).set_index('date')
    eq_val = eq_df['value']
    peak   = eq_val.cummax()
    mdd    = ((peak - eq_val) / peak * 100).max()

    return krw, log, eq_val, mdd


def main():
    print('=' * 76)
    print('  모멘텀 로테이션 검증  (룩백=14일 / 보유=7일 / 상위1개)')
    print('=' * 76)

    print('\n데이터 로드 중...')
    prices = {}
    for c in COINS:
        s = load_daily(c)
        if s is not None:
            prices[c] = s

    # ── 전체 실행 + 로그 ──────────────────────────────────────────────────────
    final_krw, log, eq_val, mdd = run_with_log(prices, lookback=14, hold=7, top_n=1)
    ret = (final_krw - 1_000_000) / 1_000_000 * 100

    print(f'\n최종 자산: {final_krw:>15,.0f}원')
    print(f'누적 수익: {ret:>+.1f}%')
    print(f'최대 MDD : {mdd:.1f}%')
    print(f'총 리밸런싱: {len(log)}회')

    # ── 연도별 요약 ───────────────────────────────────────────────────────────
    print(f'\n{"─"*76}')
    print('  연도별 요약')
    print(f'  {"연도":>4}  {"시작 자산":>13}  {"종료 자산":>13}  {"수익률":>9}  {"리밸횟수":>6}')
    print(f'  {"─"*60}')

    eq_val.index = pd.DatetimeIndex(eq_val.index)
    for year in YEARS:
        yr = eq_val[eq_val.index.year == year]
        if len(yr) == 0:
            continue
        start_v = yr.iloc[0]
        end_v   = yr.iloc[-1]
        yr_ret  = (end_v - start_v) / start_v * 100
        n_reb   = sum(1 for l in log if l['date'].year == year)
        print(f'  {year}  {start_v:>13,.0f}원  {end_v:>13,.0f}원  {yr_ret:>+8.1f}%  {n_reb:>6}회')

    # ── 리밸런싱 상세 로그 (연도별) ───────────────────────────────────────────
    print(f'\n{"─"*76}')
    print('  리밸런싱 상세 (각 연도 첫 5회)')
    for year in YEARS:
        yr_log = [l for l in log if l['date'].year == year]
        if not yr_log:
            continue
        print(f'\n  [{year}년]  총 {len(yr_log)}회 리밸런싱')
        print(f'  {"날짜":>12}  {"매수":>20}  {"14일 모멘텀 순위"}')
        print(f'  {"─"*72}')
        for l in yr_log[:5]:
            print(f'  {str(l["date"]):>12}  {l["buy"]:>20}  {l["moms"]}')
        if len(yr_log) > 5:
            print(f'  ... 이하 {len(yr_log)-5}회 생략')

    # ── 특이 구간 점검 (MDD 발생 시점) ───────────────────────────────────────
    print(f'\n{"─"*76}')
    print('  MDD 발생 구간')
    eq_idx = pd.DatetimeIndex(eq_val.index)
    peak   = eq_val.cummax()
    dd     = (peak - eq_val) / peak * 100
    top5   = dd.nlargest(5)
    for dt, v in top5.items():
        print(f'  {str(dt.date()):>12}  낙폭 {v:.1f}%  (자산: {eq_val[dt]:>12,.0f}원)')

    # ── 코인별 보유 비중 ─────────────────────────────────────────────────────
    print(f'\n{"─"*76}')
    print('  코인별 보유 횟수 (전체 리밸런싱 기준)')
    from collections import Counter
    coin_count = Counter()
    for l in log:
        for c in l['buy'].split(', '):
            coin_count[c.split('@')[0]] += 1
    for c, cnt in coin_count.most_common():
        pct = cnt / len(log) * 100
        print(f'  {c:>5}: {cnt:>4}회  ({pct:.1f}%)')
    print()


if __name__ == '__main__':
    main()
