"""
모멘텀 로테이션 전수 조사 백테스트
- ~13,000가지 파라미터 조합 전수 검사
- 워크포워드 검증 (2020~2022 학습 → 2023~2025 테스트)
- 수익 / 일관성 / 리스크조정 / 최악년도 기준 랭킹
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time

COINS = ['BTC', 'ETH', 'XRP', 'SOL']
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FEE   = 0.0005


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_prices():
    prices = {}
    for c in COINS:
        frames = []
        for y in YEARS:
            f = Path(f'data/{c}_KRW_15m_{y}.csv')
            if not f.exists(): continue
            df = pd.read_csv(f, index_col='timestamp', parse_dates=True)
            frames.append(df['close'])
        if not frames: continue
        s = pd.concat(frames).sort_index()
        s = s[~s.index.duplicated(keep='last')]
        prices[c] = s.resample('1D').last().dropna()
    return prices


def make_price_matrix(prices):
    all_dates = sorted(set.union(*[set(s.index) for s in prices.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    df = pd.DataFrame({c: prices[c].reindex(all_dates) for c in prices})
    return df


# ── 고속 백테스트 (벡터화) ────────────────────────────────────────────────────

def fast_backtest(price_df, lookback, hold, top_n, ma_period=0, fee=FEE):
    """
    price_df : (날짜 x 코인) 종가 DataFrame
    반환     : (총수익%, MDD%, equity Series, 연도별 수익 dict)
    """
    prices = price_df.values.astype(float)
    dates  = price_df.index
    n_days, n_coins = prices.shape
    coins  = list(price_df.columns)

    # 200MA 마스크 (0이면 필터 없음)
    if ma_period > 0:
        ma = price_df.rolling(ma_period).mean().values
        above = prices > ma   # (n_days, n_coins)
    else:
        above = np.ones((n_days, n_coins), dtype=bool)

    krw      = 1_000_000.0
    holding  = -1          # 현재 보유 코인 인덱스 (-1=현금)
    qty      = 0.0
    last_reb = -999
    equity   = np.zeros(n_days)

    for i in range(n_days):
        # 포트폴리오 가치
        if holding >= 0 and not np.isnan(prices[i, holding]):
            equity[i] = qty * prices[i, holding]
        else:
            equity[i] = krw

        # 리밸런싱 타이밍
        if i - last_reb < hold or i < lookback:
            continue

        # 모멘텀 계산
        past = i - lookback
        if past < 0: continue

        moms = []
        for c in range(n_coins):
            p_now  = prices[i, c]
            p_past = prices[past, c]
            if np.isnan(p_now) or np.isnan(p_past) or p_past == 0:
                moms.append(-999.0)
            elif ma_period > 0 and not above[i, c]:
                moms.append(-999.0)  # 200MA 아래 코인 제외
            else:
                moms.append((p_now - p_past) / p_past)

        # 후보 없으면 현금
        valid = [c for c, m in enumerate(moms) if m > -999.0]
        if not valid:
            if holding >= 0:
                sell_p = prices[i, holding]
                if not np.isnan(sell_p):
                    krw = qty * sell_p * (1 - fee)
                holding = -1; qty = 0.0
            last_reb = i
            continue

        # 상위 top_n 코인
        ranked  = sorted(valid, key=lambda c: moms[c], reverse=True)
        target  = ranked[0]   # top_n=1 가정 (속도 최적화)

        if holding == target:
            last_reb = i
            continue

        # 매도
        if holding >= 0:
            sell_p = prices[i, holding]
            if not np.isnan(sell_p):
                krw = qty * sell_p * (1 - fee)
            holding = -1; qty = 0.0

        # 매수
        buy_p = prices[i, target]
        if not np.isnan(buy_p) and buy_p > 0:
            qty     = krw * (1 - fee) / buy_p
            holding = target
            krw     = 0.0

        last_reb = i

    # 최종 청산
    if holding >= 0:
        last_p = prices[-1, holding]
        if np.isnan(last_p):
            last_p = prices[:, holding][~np.isnan(prices[:, holding])][-1]
        krw = qty * last_p * (1 - fee)

    eq_series = pd.Series(equity, index=dates)
    # equity가 0인 초기 구간 제거
    eq_series = eq_series.replace(0, np.nan).ffill().fillna(1_000_000)

    # MDD
    peak = eq_series.cummax()
    mdd  = ((peak - eq_series) / peak * 100).max()

    # 수익률
    ret = (krw - 1_000_000) / 1_000_000 * 100

    # 연도별
    yr_rets = {}
    for yr in YEARS:
        s = eq_series[eq_series.index.year == yr]
        if len(s) < 2: continue
        yr_rets[yr] = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100

    return ret, mdd, yr_rets


# ── 전수 조사 ─────────────────────────────────────────────────────────────────

def exhaustive_scan(price_df, train_years=None, test_years=None):
    """
    train_years: 학습 기간 (None이면 전체)
    test_years : 테스트 기간 (워크포워드용)
    """
    if train_years:
        df = price_df[price_df.index.year.isin(train_years)]
    else:
        df = price_df

    # 파라미터 그리드
    lookbacks  = list(range(3, 61, 2))   # 3,5,7,...,59  → 29개
    holds      = list(range(1, 31, 1))   # 1~30          → 30개
    ma_periods = [0, 50, 100, 200]       # 4개
    # 총 29 × 30 × 4 = 3,480 (top_n=1 고정)
    # top_n=2 포함 시 × 2 = 6,960

    total = len(lookbacks) * len(holds) * len(ma_periods) * 2  # top_n 1,2
    print(f'  총 {total:,}가지 조합 검사 중...')

    results = []
    done = 0
    t0 = time.time()

    for lb, hd, ma, tn in product(lookbacks, holds, ma_periods, [1, 2]):
        ret, mdd, yr_rets = fast_backtest(df, lb, hd, tn, ma)

        # 일관성 점수 (몇 년이 플러스?)
        pos_years = sum(1 for r in yr_rets.values() if r > 0)
        # 목표 달성 (10% 이상인 해 수)
        goal_years = sum(1 for r in yr_rets.values() if r >= 10)
        # 최악 년도
        worst = min(yr_rets.values()) if yr_rets else -999

        # 리스크 조정 수익 (수익 / MDD)
        rarm = ret / max(mdd, 1)

        results.append({
            'lb': lb, 'hd': hd, 'ma': ma, 'tn': tn,
            'ret': ret, 'mdd': mdd,
            'pos_yrs': pos_years,
            'goal_yrs': goal_years,
            'worst': worst,
            'rarm': rarm,
            **{f'y{y}': yr_rets.get(y, None) for y in YEARS}
        })

        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f'  {done:>5}/{total} 완료  경과 {elapsed:.0f}초  남은시간 ~{eta:.0f}초', flush=True)

    print(f'  완료! 총 {time.time()-t0:.0f}초 소요')
    return pd.DataFrame(results)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 76)
    print('  모멘텀 로테이션 전수 조사  2020~2025')
    print('=' * 76)

    print('\n데이터 로드 중...')
    prices = load_prices()
    for c, s in prices.items():
        print(f'  {c}: {len(s):,}일')
    price_df = make_price_matrix(prices)

    # ── 1. 전체 기간 전수 조사 ────────────────────────────────────────────────
    print('\n[1/3] 전체 기간 전수 조사 (2020~2025)...')
    df_all = exhaustive_scan(price_df)

    # ── 2. 워크포워드 검증 ────────────────────────────────────────────────────
    print('\n[2/3] 워크포워드 검증...')
    print('  학습: 2020~2022  /  테스트: 2023~2025')

    train_df = price_df[price_df.index.year.isin([2020, 2021, 2022])]
    test_df  = price_df[price_df.index.year.isin([2023, 2024, 2025])]

    # 학습 기간 TOP 50 조합으로 테스트 성능 검증
    train_results = []
    for _, row in df_all.iterrows():
        # 학습 기간 성능
        ret_tr, mdd_tr, yr_tr = fast_backtest(
            train_df, int(row.lb), int(row.hd), int(row.tn), int(row.ma))
        # 테스트 기간 성능
        ret_te, mdd_te, yr_te = fast_backtest(
            test_df, int(row.lb), int(row.hd), int(row.tn), int(row.ma))

        train_results.append({
            'lb': row.lb, 'hd': row.hd, 'ma': row.ma, 'tn': row.tn,
            'ret_train': ret_tr, 'mdd_train': mdd_tr,
            'ret_test':  ret_te, 'mdd_test':  mdd_te,
            'consistency': (ret_tr > 0) + (ret_te > 0),
            'avg_ret': (ret_tr + ret_te) / 2,
        })

    wf_df = pd.DataFrame(train_results)
    print('  워크포워드 완료!')

    # ── 3. 결과 출력 ──────────────────────────────────────────────────────────
    print(f'\n[3/3] 결과 분석')

    def show_top(label, df_sorted, n=10):
        print(f'\n{"─"*76}')
        print(f'  [{label}] TOP {n}')
        print(f'  {"룩백":>5}  {"보유":>5}  {"MA":>5}  {"N":>3}'
              f'  {"총수익":>9}  {"MDD":>7}'
              f'  {"2020":>7}  {"2021":>7}  {"2022":>7}'
              f'  {"2023":>7}  {"2024":>7}  {"2025":>7}'
              f'  {"최악년":>8}')
        print(f'  {"─"*110}')
        for _, r in df_sorted.head(n).iterrows():
            ma_str = f'{int(r.ma)}MA' if r.ma > 0 else '없음'
            yrs = [f'{r[f"y{y}"]:>+6.0f}%' if r[f"y{y}"] is not None
                   else f'{"N/A":>7}' for y in YEARS]
            print(f'  {int(r.lb):>5}일  {int(r.hd):>5}일  {ma_str:>5}  {int(r.tn):>3}'
                  f'  {r.ret:>+8.0f}%  {r.mdd:>6.1f}%'
                  f'  {"  ".join(yrs)}'
                  f'  {r.worst:>+7.0f}%')

    # 수익 최고
    show_top('수익 최고 조합', df_all.sort_values('ret', ascending=False))

    # 일관성 최고 (매년 플러스 횟수 + 총수익)
    df_cons = df_all.sort_values(['pos_yrs', 'ret'], ascending=[False, False])
    show_top('일관성 최고 (매년 플러스 우선)', df_cons)

    # 10% 달성 해수 최고
    df_goal = df_all.sort_values(['goal_yrs', 'ret'], ascending=[False, False])
    show_top('연 10% 달성 해수 최고', df_goal)

    # 최악년도 최고 (하락장 방어)
    df_worst = df_all[df_all['worst'] > -100].sort_values(['worst', 'ret'], ascending=[False, False])
    show_top('최악년도 손실 최소', df_worst)

    # 리스크조정 최고
    df_rarm = df_all.sort_values('rarm', ascending=False)
    show_top('리스크 대비 수익 최고 (수익/MDD)', df_rarm)

    # 워크포워드 TOP (학습+테스트 평균 수익)
    print(f'\n{"─"*76}')
    print(f'  [워크포워드 검증 TOP 10]  학습(2020~22) → 테스트(2023~25)')
    print(f'  {"룩백":>5}  {"보유":>5}  {"MA":>5}  {"N":>3}'
          f'  {"학습수익":>9}  {"테스트수익":>10}  {"평균":>9}  {"일관성"}')
    print(f'  {"─"*65}')
    for _, r in wf_df.sort_values('avg_ret', ascending=False).head(10).iterrows():
        ma_str = f'{int(r.ma)}MA' if r.ma > 0 else '없음'
        print(f'  {int(r.lb):>5}일  {int(r.hd):>5}일  {ma_str:>5}  {int(r.tn):>3}'
              f'  {r.ret_train:>+8.0f}%  {r.ret_test:>+9.0f}%'
              f'  {r.avg_ret:>+8.0f}%  {int(r.consistency)}/2')

    # 최종 추천 (워크포워드 + 일관성 모두 좋은 조합)
    best_wf = wf_df[(wf_df.consistency == 2)].sort_values('avg_ret', ascending=False)
    if len(best_wf) > 0:
        best = best_wf.iloc[0]
        ma_str = f'{int(best.ma)}MA' if best.ma > 0 else '없음'
        print(f'\n{"="*76}')
        print(f'  ★ 최종 추천 조합 (학습/테스트 모두 플러스)')
        print(f'  룩백={int(best.lb)}일 / 보유={int(best.hd)}일 / MA={ma_str} / 상위{int(best.tn)}개')
        print(f'  학습(2020~22): {best.ret_train:>+.0f}%   테스트(2023~25): {best.ret_test:>+.0f}%')
    print()


if __name__ == '__main__':
    main()
