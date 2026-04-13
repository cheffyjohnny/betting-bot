"""
모멘텀 로테이션 백테스트 (2020~2025)
전략: 매 리밸런싱 주기마다 가장 강한 코인으로 갈아타기
200MA 필터: 200MA 위 코인만 후보 (전부 아래면 현금 보유)
대상: BTC, ETH, XRP, SOL
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

COINS = ['BTC', 'ETH', 'XRP', 'SOL']
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FEE   = 0.0005


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

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


def build_ma200(prices):
    """코인별 200일 MA 딕셔너리 반환"""
    ma200 = {}
    for c, s in prices.items():
        ma200[c] = s.rolling(200).mean()
    return ma200


# ── 모멘텀 로테이션 백테스트 ─────────────────────────────────────────────────

def run_momentum(prices, ma200, lookback=14, hold=7, top_n=1, use_ma_filter=True):
    all_dates  = sorted(set.union(*[set(s.index) for s in prices.values()]))
    all_dates  = pd.DatetimeIndex(all_dates)

    krw        = 1_000_000.0
    holdings   = {}
    last_rebal = None
    equity     = []

    for i, date in enumerate(all_dates):
        # 일별 포트폴리오 가치
        port = krw
        for c, qty in holdings.items():
            s = prices[c]
            if date in s.index and not pd.isna(s[date]):
                port += qty * s[date]
        equity.append(port)

        if i < lookback:
            continue
        if last_rebal is not None and (date - last_rebal).days < hold:
            continue

        # 현재 가격 있는 코인
        available = {c: s for c, s in prices.items()
                     if date in s.index and not pd.isna(s[date])}

        # 200MA 필터: 200MA 위 코인만 후보
        if use_ma_filter:
            above_ma = {}
            for c, s in available.items():
                if date in ma200[c].index and not pd.isna(ma200[c][date]):
                    if s[date] > ma200[c][date]:
                        above_ma[c] = s
            candidates = above_ma
        else:
            candidates = available

        # 후보가 없으면 → 전부 현금
        if len(candidates) < 1:
            # 보유 중이면 청산
            if holdings:
                for c in list(holdings.keys()):
                    if c in available:
                        krw += holdings[c] * available[c][date] * (1 - FEE)
                    holdings.pop(c)
            last_rebal = date
            continue

        if len(candidates) < 2 and len(available) >= 2:
            pass  # 후보 1개면 그냥 진행

        # 모멘텀 계산
        past_date_idx = i - lookback
        if past_date_idx < 0:
            continue
        past_date = all_dates[past_date_idx]

        moms = {}
        for c, s in candidates.items():
            if past_date in s.index and not pd.isna(s[past_date]):
                moms[c] = (s[date] - s[past_date]) / s[past_date]
        if not moms:
            continue

        ranked  = sorted(moms, key=moms.get, reverse=True)
        targets = ranked[:min(top_n, len(ranked))]

        # 현재 보유와 동일하면 패스
        if set(holdings.keys()) == set(targets):
            last_rebal = date
            continue

        # 매도
        for c in list(holdings.keys()):
            if c in available:
                krw += holdings[c] * available[c][date] * (1 - FEE)
            holdings.pop(c)

        # 매수
        alloc = krw / len(targets)
        for c in targets:
            qty = alloc * (1 - FEE) / available[c][date]
            holdings[c] = qty
            krw -= alloc

        last_rebal = date

    # 마지막 청산
    last_date = all_dates[-1]
    for c, qty in list(holdings.items()):
        s = prices[c]
        p = s[last_date] if last_date in s.index else s[s.index <= last_date].iloc[-1]
        krw += qty * p * (1 - FEE)

    eq  = pd.Series(equity, index=all_dates)
    pk  = eq.cummax()
    mdd = ((pk - eq) / pk * 100).max()
    ret = (krw - 1_000_000) / 1_000_000 * 100
    return ret, krw, mdd, eq


# ── 연도별 수익률 ─────────────────────────────────────────────────────────────

def yearly_rets(eq):
    rets = {}
    for year in YEARS:
        yr = eq[eq.index.year == year]
        if len(yr) < 2:
            continue
        rets[year] = (yr.iloc[-1] - yr.iloc[0]) / yr.iloc[0] * 100
    return rets


def bah_ret(prices, coin='BTC'):
    s = prices[coin].dropna()
    return (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 76)
    print('  모멘텀 로테이션 + 200MA 필터  백테스트  2020~2025')
    print(f'  대상: {" / ".join(COINS)}')
    print('=' * 76)

    print('\n데이터 로드 중...')
    prices = {}
    for c in COINS:
        s = load_daily(c)
        if s is not None:
            prices[c] = s
            print(f'  {c}: {len(s):,}일')

    ma200 = build_ma200(prices)

    # ── 1. 필터 없음 vs 필터 있음 (최고 조합 기준) ───────────────────────────
    LB, HD, TN = 14, 7, 1   # 이전 최고 조합

    ret_no,  krw_no,  mdd_no,  eq_no  = run_momentum(prices, ma200, LB, HD, TN, use_ma_filter=False)
    ret_yes, krw_yes, mdd_yes, eq_yes = run_momentum(prices, ma200, LB, HD, TN, use_ma_filter=True)

    print(f'\n{"─"*76}')
    print(f'  [200MA 필터 효과 비교]  룩백={LB}일 / 보유={HD}일 / 상위{TN}개')
    print(f'  {"":>22}  {"6년 누적":>12}  {"연평균":>9}  {"MDD":>7}')
    print(f'  {"─"*58}')

    avg_no  = (1 + ret_no  / 100) ** (1/6) * 100 - 100
    avg_yes = (1 + ret_yes / 100) ** (1/6) * 100 - 100
    print(f'  {"필터 없음":>22}  {krw_no:>10,.0f}원  {avg_no:>+7.1f}%/년  {mdd_no:>6.1f}%')
    print(f'  {"200MA 필터":>22}  {krw_yes:>10,.0f}원  {avg_yes:>+7.1f}%/년  {mdd_yes:>6.1f}%')

    # ── 2. 연도별 비교 ────────────────────────────────────────────────────────
    print(f'\n{"─"*76}')
    print(f'  [연도별 수익률 비교]')
    print(f'  {"연도":>4}  {"필터없음":>10}  {"200MA필터":>10}  {"BTC B&H":>10}  {"현금 구간?"}')
    print(f'  {"─"*60}')

    yr_no  = yearly_rets(eq_no)
    yr_yes = yearly_rets(eq_yes)

    BTC_BAH = {
        2020: '+306%', 2021: '+77%', 2022: '-63%',
        2023: '+170%', 2024: '+143%', 2025: '-27%'
    }

    for year in YEARS:
        r_no  = yr_no.get(year, 0)
        r_yes = yr_yes.get(year, 0)
        btc   = BTC_BAH.get(year, '')
        # 현금 구간 표시: 수익률이 크게 줄었으면 필터가 작동한 것
        cash_flag = '← 현금 구간 있음' if (r_no - r_yes) > 10 else ''
        print(f'  {year}  {r_no:>+9.1f}%  {r_yes:>+9.1f}%  {btc:>10}  {cash_flag}')

    # ── 3. 파라미터 스캔 (200MA 필터 적용) ───────────────────────────────────
    print(f'\n{"─"*76}')
    print(f'  [파라미터 스캔 — 200MA 필터 적용]')
    print(f'  {"룩백":>5}  {"보유":>5}  {"상위N":>5}  {"누적수익":>10}  {"연평균":>9}  {"MDD":>7}')
    print(f'  {"─"*55}')

    scan_results = []
    for lb, hd, tn in product([7, 14, 30], [7, 14, 30], [1, 2]):
        r, krw, mdd, _ = run_momentum(prices, ma200, lb, hd, tn, use_ma_filter=True)
        avg = (1 + r / 100) ** (1/6) * 100 - 100
        scan_results.append((lb, hd, tn, r, krw, avg, mdd))
        print(f'  {lb:>5}일  {hd:>5}일  {tn:>5}개  {r:>+9.1f}%  {avg:>+7.1f}%/년  {mdd:>6.1f}%')

    best = max(scan_results, key=lambda x: x[3])
    print(f'\n  ★ 최고 조합: 룩백={best[0]}일 / 보유={best[1]}일 / 상위{best[2]}개'
          f'  →  누적 {best[3]:>+.1f}%  연평균 {best[5]:>+.1f}%  MDD {best[6]:.1f}%')

    # ── 4. 최종 비교표 ────────────────────────────────────────────────────────
    r_best, krw_best, mdd_best, eq_best = run_momentum(
        prices, ma200, best[0], best[1], best[2], use_ma_filter=True)
    avg_best = (1 + r_best / 100) ** (1/6) * 100 - 100

    btc_r   = bah_ret(prices, 'BTC')
    btc_krw = 1_000_000 * (1 + btc_r / 100)
    btc_avg = (1 + btc_r / 100) ** (1/6) * 100 - 100

    print(f'\n{"="*76}')
    print(f'  최종 비교')
    print(f'  {"전략":>28}  {"6년 누적":>13}  {"연평균":>9}  {"MDD":>7}')
    print(f'  {"─"*62}')
    print(f'  {"BTC B&H":>28}  {btc_krw:>11,.0f}원  {btc_avg:>+7.1f}%/년    N/A')
    print(f'  {"모멘텀(필터없음)":>28}  {krw_no:>11,.0f}원  {avg_no:>+7.1f}%/년  {mdd_no:>6.1f}%')
    print(f'  {"모멘텀+200MA":>28}  {krw_yes:>11,.0f}원  {avg_yes:>+7.1f}%/년  {mdd_yes:>6.1f}%')
    print(f'  {f"최고조합+200MA({best[0]}d/{best[1]}d/top{best[2]})":>28}'
          f'  {krw_best:>11,.0f}원  {avg_best:>+7.1f}%/년  {mdd_best:>6.1f}%')
    print()


if __name__ == '__main__':
    main()
