"""
Apex Mode Backtest — 1개월 시뮬레이션
기간: 2026-03-19 ~ 2026-04-19

apex_bot.py 전략을 그대로 재현:
- 매일 장 시작: 레짐 판단 + 진입/로테이션/피라미딩
- 장중 stop 체크: 일봉 low로 스탑 트리거 여부 판단
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time

# ── 상수 (apex_bot.py와 동일) ────────────────────────────────────────────────
COINS           = ['BTC', 'ETH', 'XRP', 'SOL']
FEE             = 0.0005
INIT_KRW        = 1_000_000

GEAR_ALLOC      = {'BEAST': 0.90, 'CRUISE': 0.70, 'CAUTION': 0.40, 'BUNKER': 0.00}
GEAR_MAX        = {'BEAST': 1,    'CRUISE': 2,     'CAUTION': 1,    'BUNKER': 0}

LOT_SPLITS      = [0.40, 0.30, 0.30]
PYRAMID_TRIGGER = 0.03
HARD_STOP_PCT   = 0.08
TRAIL_ATR_MULT  = 2.0
PARTIAL_PROFIT  = 0.15
ATR_PERIOD      = 14

BEAST_COND_NEED  = 3
BUNKER_DROP      = -12.0
BUNKER_RSI       = 80
BEAST_RSI_MAX    = 70
BEAST_MOM_MIN    = 5.0
BEAST_VOL_MIN    = 1.2

# ── 전략 버전 파라미터 (비교 시 변경) ────────────────────────────────────────
# 1 = 원본:  3개 조건 중 1개만 걸려도 BUNKER
# 2 = 개선1: 3개 조건 중 2개 이상이어야 BUNKER
# 3 = 개선2: RSI과열 or 7일급락 → BUNKER / BTC<MA50 단독 → CAUTION(40%)
# 4 = 동적:  BTC>MA200이면 V3 / BTC<MA200이면 V1 (매크로 레짐 자동 전환)
BUNKER_COND_NEED = 1

SIM_START = date(2024, 1, 1)   # 기본값 (비교 분석 시 덮어씀)
SIM_END   = date(2024, 12, 31)


# ── 데이터 수집 ───────────────────────────────────────────────────────────────

def fetch_all_history():
    """시뮬레이션 기간 + MA200 계산용 과거 데이터를 페이지네이션으로 수집"""
    ex = ccxt.upbit({'enableRateLimit': True})
    # 수집 시작 시점: SIM_START - 220일 (MA200 여유분)
    from datetime import datetime
    fetch_from = SIM_START - timedelta(days=220)
    since_ms   = int(datetime(fetch_from.year, fetch_from.month, fetch_from.day).timestamp() * 1000)
    all_data   = {}

    for coin in COINS:
        print(f'  {coin} 수집 중 ({fetch_from} ~ {SIM_END})...')
        try:
            rows   = []
            cursor = since_ms
            while True:
                batch = ex.fetch_ohlcv(f'{coin}/KRW', '1d', since=cursor, limit=200)
                if not batch:
                    break
                rows  += batch
                last_ts = batch[-1][0]
                # SIM_END 이후 데이터까지 받았으면 종료
                last_date = pd.to_datetime(last_ts, unit='ms', utc=True).date()
                if last_date >= SIM_END:
                    break
                cursor = last_ts + 24 * 3600 * 1000  # 다음 배치 시작
                time.sleep(0.3)

            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.date
            df = df.set_index('date').sort_index()
            df = df[~df.index.duplicated()].astype({'open': float, 'high': float,
                                                    'low': float, 'close': float,
                                                    'volume': float})
            print(f'    → {len(df)}일 수집 완료 ({df.index[0]} ~ {df.index[-1]})')
            all_data[coin] = df
        except Exception as e:
            print(f'  {coin} 오류: {e}')
            all_data[coin] = None
        time.sleep(0.5)
    return all_data


def slice_df(df, as_of_date):
    """as_of_date 포함 이전 데이터만 반환 (look-ahead 방지)"""
    return df[df.index <= as_of_date]


# ── 지표 (apex_bot.py와 동일) ─────────────────────────────────────────────────

def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - 100 / (1 + rs)).iloc[-1]


def calc_ma(series, period):
    return series.rolling(period).mean().iloc[-1]


def detect_gear(btc_df):
    close     = btc_df['close']
    price     = close.iloc[-1]
    ma50      = calc_ma(close, 50)
    ma200     = calc_ma(close, 200)
    rsi       = calc_rsi(close)
    mom7      = (price - close.iloc[-8]) / close.iloc[-8] * 100
    vol       = btc_df['volume'].iloc[-1]
    vol_ma    = btc_df['volume'].rolling(20).mean().iloc[-1]
    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0

    below_ma50  = price < ma50
    rsi_hot     = rsi > BUNKER_RSI
    crash7      = mom7 < BUNKER_DROP

    if BUNKER_COND_NEED == 3:
        # V3: RSI과열 or 7일급락 → BUNKER / BTC<MA50 단독 → CRUISE 상한
        go_bunker = rsi_hot or crash7
    else:
        # V1(1개) or V2(2개)
        bunker_score = int(below_ma50) + int(rsi_hot) + int(crash7)
        go_bunker    = bunker_score >= BUNKER_COND_NEED

    active = ([k for k, v in {'BTC<MA50': below_ma50, 'RSI과열': rsi_hot,
                               '7일급락': crash7}.items() if v])
    base_details = {'price': price, 'ma50': ma50, 'ma200': ma200,
                    'rsi': rsi, 'mom7': mom7, 'vol_ratio': vol_ratio,
                    'trigger': active}

    if go_bunker:
        return 'BUNKER', base_details

    beast_conds = {
        'BTC>MA50'  : price > ma50,
        'BTC>MA200' : price > ma200,
        'RSI적정'   : rsi < BEAST_RSI_MAX,
        '거래량확장': vol_ratio >= BEAST_VOL_MIN,
        '7일모멘텀' : mom7 >= BEAST_MOM_MIN,
    }
    beast_score = sum(beast_conds.values())
    details = {**base_details, 'beast_score': beast_score, 'conds': beast_conds}

    # V4: 매크로 레짐(MA200) 기반 동적 전환
    # V3: BTC<MA50 단독이면 CAUTION (40% 투자, 1코인)
    in_bull_macro = price > ma200
    use_v3 = (BUNKER_COND_NEED == 3) or (BUNKER_COND_NEED == 4 and in_bull_macro)

    if use_v3 and below_ma50:
        return 'CAUTION', details

    if beast_score >= BEAST_COND_NEED:
        return 'BEAST', details
    return 'CRUISE', details


def calc_momentum_scores(all_data, as_of_date):
    scores = {}
    for coin, df in all_data.items():
        if df is None:
            continue
        sub = slice_df(df, as_of_date)
        if len(sub) < 35:
            continue
        close = sub['close']
        price = close.iloc[-1]
        ma50  = calc_ma(close, 50)
        if price < ma50:
            continue
        ret7  = (price - close.iloc[-8])  / close.iloc[-8]  * 100
        ret30 = (price - close.iloc[-31]) / close.iloc[-31] * 100
        vol      = sub['volume'].iloc[-1]
        vol_ma   = sub['volume'].rolling(20).mean().iloc[-1]
        vol_score = (vol / vol_ma - 1) * 100 if vol_ma > 0 else 0
        score = ret7 * 0.4 + ret30 * 0.3 + vol_score * 0.3
        atr   = calc_atr(sub)
        scores[coin] = {
            'price': price, 'ma50': ma50,
            'ret7': round(ret7, 2), 'ret30': round(ret30, 2),
            'vol_ratio': round(vol / vol_ma if vol_ma > 0 else 1, 2),
            'score': round(score, 2), 'atr': atr,
        }
    return dict(sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True))


# ── 거래 실행 (apex_bot.py와 동일) ───────────────────────────────────────────

def total_value(state, scores):
    val = state['krw']
    for coin, pos in state['positions'].items():
        price = scores[coin]['price'] if coin in scores else pos['avg_entry']
        val  += pos['total_qty'] * price
    return val


def paper_sell_all(state, coin, price):
    if coin not in state['positions']:
        return 0
    pos  = state['positions'].pop(coin)
    recv = pos['total_qty'] * price * (1 - FEE)
    state['krw'] += recv
    return recv


def paper_sell_partial(state, coin, price, ratio=0.5):
    if coin not in state['positions']:
        return 0
    pos      = state['positions'][coin]
    sell_qty = pos['total_qty'] * ratio
    recv     = sell_qty * price * (1 - FEE)
    pos['total_qty']     -= sell_qty
    pos['partial_exited'] = True
    state['krw'] += recv
    for lot in pos['lots']:
        lot['qty'] *= (1 - ratio)
    return recv


def paper_buy(state, coin, price, krw_amount, lot_num, atr):
    qty = krw_amount * (1 - FEE) / price
    state['krw'] -= krw_amount
    if coin not in state['positions']:
        state['positions'][coin] = {
            'lots': [], 'total_qty': 0.0, 'avg_entry': price,
            'hwm': price,
            'hard_stop'     : round(price * (1 - HARD_STOP_PCT)),
            'trailing_stop' : round(price - atr * TRAIL_ATR_MULT),
            'partial_exited': False,
        }
    pos = state['positions'][coin]
    pos['lots'].append({'lot': lot_num, 'qty': qty, 'entry': price})
    pos['total_qty'] += qty
    total_cost = sum(l['qty'] * l['entry'] for l in pos['lots'])
    pos['avg_entry']  = round(total_cost / pos['total_qty'])
    pos['hard_stop']  = round(pos['avg_entry'] * (1 - HARD_STOP_PCT))
    return qty


def update_stops(state, scores):
    for coin, pos in state['positions'].items():
        if coin not in scores:
            continue
        price = scores[coin]['price']
        atr   = scores[coin]['atr']
        if price > pos['hwm']:
            pos['hwm'] = price
        new_trail = round(pos['hwm'] - atr * TRAIL_ATR_MULT)
        if new_trail > pos['trailing_stop']:
            pos['trailing_stop'] = new_trail


# ── 장중 스탑 체크 (일봉 low 사용) ───────────────────────────────────────────

def intraday_stop_check(state, all_data, sim_date, actions):
    """일봉 low로 스탑 트리거 여부 시뮬레이션"""
    for coin in list(state['positions'].keys()):
        if coin not in all_data or all_data[coin] is None:
            continue
        df = all_data[coin]
        if sim_date not in df.index:
            continue
        day_low  = df.loc[sim_date, 'low']
        day_high = df.loc[sim_date, 'high']
        pos      = state['positions'].get(coin)
        if pos is None:
            continue

        # HWM 갱신
        if day_high > pos['hwm']:
            pos['hwm'] = day_high
            atr = calc_atr(slice_df(df, sim_date))
            new_trail = round(pos['hwm'] - atr * TRAIL_ATR_MULT)
            if new_trail > pos['trailing_stop']:
                pos['trailing_stop'] = new_trail

        # 부분 익절 체크 (day_high 기준)
        if not pos['partial_exited'] and day_high >= pos['avg_entry'] * (1 + PARTIAL_PROFIT):
            exit_price = pos['avg_entry'] * (1 + PARTIAL_PROFIT)
            recv = paper_sell_partial(state, coin, exit_price)
            actions.append(f'  [스탑체커 부분익절] {coin} @ {exit_price:,.0f}원 ({recv:,.0f}원)')

        # 스탑 체크 (day_low 기준)
        pos = state['positions'].get(coin)
        if pos is None:
            continue
        stop_price = max(pos['trailing_stop'], pos['hard_stop'])
        if day_low <= stop_price:
            exit_price = stop_price
            recv = paper_sell_all(state, coin, exit_price)
            reason = ('하드스탑' if day_low <= pos['hard_stop'] else '트레일링스탑')
            actions.append(f'  [스탑체커 청산] {coin} @ {exit_price:,.0f}원  {reason} ({recv:,.0f}원)')


# ── 일별 전략 실행 ────────────────────────────────────────────────────────────

def run_day(state, gear, gear_details, scores, actions):
    alloc    = GEAR_ALLOC[gear]
    max_pos  = GEAR_MAX[gear]
    portfolio = total_value(state, scores)

    update_stops(state, scores)

    to_sell = []
    for coin, pos in list(state['positions'].items()):
        price  = scores[coin]['price'] if coin in scores else pos['avg_entry']
        reason = None
        if gear == 'BUNKER':
            reason = 'BUNKER 전환'
        elif price <= pos['hard_stop']:
            reason = f'하드스탑 ({price:,.0f}<={pos["hard_stop"]:,.0f})'
        elif price <= pos['trailing_stop']:
            reason = f'트레일링 ({price:,.0f}<={pos["trailing_stop"]:,.0f})'
        elif coin not in scores:
            reason = 'MA50 하향'

        if reason:
            to_sell.append((coin, price, reason, 'full'))
        elif not pos['partial_exited'] and price >= pos['avg_entry'] * (1 + PARTIAL_PROFIT):
            to_sell.append((coin, price, f'+{PARTIAL_PROFIT*100:.0f}% 부분익절', 'half'))

    for coin, price, reason, sell_type in to_sell:
        if sell_type == 'full':
            recv = paper_sell_all(state, coin, price)
            actions.append(f'  [매도] {coin} @ {price:,.0f}  {recv:,.0f}원  ({reason})')
        else:
            recv = paper_sell_partial(state, coin, price)
            actions.append(f'  [부분매도] {coin} @ {price:,.0f}  {recv:,.0f}원  ({reason})')

    if gear == 'BUNKER':
        return

    # 피라미딩
    if gear == 'BEAST':
        for coin, pos in state['positions'].items():
            if coin not in scores or len(pos['lots']) >= 3:
                continue
            price      = scores[coin]['price']
            atr        = scores[coin]['atr']
            lot_done   = len(pos['lots'])
            prev_entry = pos['lots'][-1]['entry']
            if price < prev_entry * (1 + PYRAMID_TRIGGER):
                continue
            lot_frac = LOT_SPLITS[lot_done]
            budget   = portfolio * alloc * lot_frac
            if state['krw'] < budget * 0.5:
                continue
            budget = min(budget, state['krw'] * 0.95)
            qty = paper_buy(state, coin, price, budget, lot_done + 1, atr)
            actions.append(f'  [피라미딩 {lot_done+1}차] {coin} @ {price:,.0f}  {qty:.6f}개')

    # 신규 진입 / 로테이션
    top_coins  = list(scores.keys())[:max_pos]
    held_coins = set(state['positions'].keys())

    for coin in list(held_coins):
        if coin not in top_coins:
            price = scores[coin]['price'] if coin in scores else state['positions'][coin]['avg_entry']
            recv  = paper_sell_all(state, coin, price)
            actions.append(f'  [로테이션 매도] {coin} @ {price:,.0f}  {recv:,.0f}원')

    for coin in top_coins:
        if coin in state['positions']:
            continue
        price = scores[coin]['price']
        atr   = scores[coin]['atr']
        if gear == 'BEAST':
            budget = portfolio * alloc * LOT_SPLITS[0]
        else:
            budget = portfolio * alloc / len(top_coins)
        if state['krw'] < budget * 0.5:
            actions.append(f'  {coin} 진입 실패 — KRW 부족')
            continue
        budget = min(budget, state['krw'] * 0.95)
        qty    = paper_buy(state, coin, price, budget, 1, atr)
        pos    = state['positions'][coin]
        actions.append(f'  [매수 Lot1] {coin} @ {price:,.0f}  {qty:.6f}개  '
                       f'트레일:{pos["trailing_stop"]:,.0f} 하드:{pos["hard_stop"]:,.0f}')


# ── 시뮬레이션 코어 ──────────────────────────────────────────────────────────

def simulate(all_data, start, end):
    """단일 기간 시뮬레이션. 상세 로그 리스트 반환."""
    state = {
        'gear': 'CRUISE', 'positions': {},
        'krw': float(INIT_KRW), 'init_krw': float(INIT_KRW),
    }
    daily_logs = []
    sim_date   = start

    while sim_date <= end:
        btc_df = all_data.get('BTC')
        if btc_df is None or sim_date not in btc_df.index:
            sim_date += timedelta(days=1)
            continue

        btc_sub = slice_df(btc_df, sim_date)
        if len(btc_sub) < 55:
            sim_date += timedelta(days=1)
            continue

        gear, gear_details = detect_gear(btc_sub)
        scores = calc_momentum_scores(all_data, sim_date)

        actions = []
        intraday_stop_check(state, all_data, sim_date, actions)
        run_day(state, gear, gear_details, scores, actions)
        state['gear'] = gear

        portfolio = total_value(state, scores)
        pnl       = (portfolio - INIT_KRW) / INIT_KRW * 100

        daily_logs.append({
            'date'        : sim_date,
            'gear'        : gear,
            'gear_details': gear_details,
            'portfolio'   : round(portfolio),
            'pnl_pct'     : round(pnl, 2),
            'krw'         : round(state['krw']),
            'held'        : list(state['positions'].keys()),
            'actions'     : actions,
            'btc_price'   : round(gear_details['price']),
            'btc_ma50'    : round(gear_details['ma50']),
            'btc_rsi'     : round(gear_details['rsi'], 1),
            'btc_mom7'    : round(gear_details['mom7'], 2),
        })
        sim_date += timedelta(days=1)

    return daily_logs


def calc_stats(daily_logs, all_data, start, end):
    """통계 계산. stats dict 반환."""
    if not daily_logs:
        return {}

    # 기본 수익률
    final    = daily_logs[-1]
    peak     = max(l['portfolio'] for l in daily_logs)
    trough   = min(l['portfolio'] for l in daily_logs)
    mdd      = (trough - peak) / peak * 100
    gear_cnt = {'BEAST': 0, 'CRUISE': 0, 'CAUTION': 0, 'BUNKER': 0}
    for l in daily_logs:
        gear_cnt[l['gear']] += 1

    # BUNKER 트리거 원인 집계
    bunker_triggers = {'BTC<MA50': 0, 'RSI과열': 0, '7일급락': 0}
    for l in daily_logs:
        if l['gear'] == 'BUNKER':
            for t in l['gear_details'].get('trigger', []):
                bunker_triggers[t] = bunker_triggers.get(t, 0) + 1

    # BUNKER 연속 구간 분석
    bunker_runs = []
    run_len = 0
    for l in daily_logs:
        if l['gear'] == 'BUNKER':
            run_len += 1
        else:
            if run_len > 0:
                bunker_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        bunker_runs.append(run_len)

    # 매매 횟수
    buy_cnt  = sum(1 for l in daily_logs for a in l['actions'] if '매수 Lot1' in a)
    sell_cnt = sum(1 for l in daily_logs for a in l['actions'] if '매도' in a or '청산' in a)

    # BTC 홀딩 비교
    btc_df   = all_data.get('BTC')
    btc_data = {d: btc_df.loc[d, 'close'] for d in btc_df.index
                if start <= d <= end} if btc_df is not None else {}
    btc_dates       = sorted(btc_data.keys())
    btc_start_price = btc_data[btc_dates[0]]  if btc_dates else 1
    btc_end_price   = btc_data[btc_dates[-1]] if btc_dates else 1
    btc_hold_qty    = INIT_KRW * (1 - FEE) / btc_start_price
    btc_vals        = [btc_hold_qty * btc_data[d] for d in btc_dates]
    btc_peak        = max(btc_vals)
    btc_trough      = min(btc_vals)
    btc_pnl         = (btc_end_price - btc_start_price) / btc_start_price * 100
    btc_mdd         = (btc_trough - btc_peak) / btc_peak * 100

    return {
        'days'            : len(daily_logs),
        'apex_pnl'        : final['pnl_pct'],
        'apex_final'      : final['portfolio'],
        'apex_mdd'        : round(mdd, 2),
        'btc_pnl'         : round(btc_pnl, 2),
        'btc_mdd'         : round(btc_mdd, 2),
        'btc_start'       : round(btc_start_price),
        'btc_end'         : round(btc_end_price),
        'gear_cnt'        : gear_cnt,
        'bunker_triggers' : bunker_triggers,
        'bunker_runs'     : bunker_runs,
        'buy_cnt'         : buy_cnt,
        'sell_cnt'        : sell_cnt,
    }


# ── 비교 분석 출력 ────────────────────────────────────────────────────────────

def print_comparison(label_a, stats_a, logs_a, label_b, stats_b, logs_b):
    W = 70

    def row(label, va, vb):
        print(f'  {label:<26}  {va:>18}  {vb:>18}')

    print()
    print('=' * W)
    print(f'  {"Apex Mode 전략 비교 분석":^{W-4}}')
    print('=' * W)
    print(f'  {"":26}  {label_a:>18}  {label_b:>18}')
    print(f'  {"─"*(W-2)}')

    # ── 1. 수익률 ─────────────────────────────────────────────────────────────
    print(f'\n  [1] 수익률 비교')
    row('BTC 시작가', f'{stats_a["btc_start"]:,}원', f'{stats_b["btc_start"]:,}원')
    row('BTC 종료가', f'{stats_a["btc_end"]:,}원', f'{stats_b["btc_end"]:,}원')
    row('BTC 홀딩 수익률', f'{stats_a["btc_pnl"]:+.2f}%', f'{stats_b["btc_pnl"]:+.2f}%')
    row('Apex 수익률', f'{stats_a["apex_pnl"]:+.2f}%', f'{stats_b["apex_pnl"]:+.2f}%')
    row('수익률 차이 (Apex-BTC)',
        f'{stats_a["apex_pnl"]-stats_a["btc_pnl"]:+.2f}%p',
        f'{stats_b["apex_pnl"]-stats_b["btc_pnl"]:+.2f}%p')

    # ── 2. 리스크 ─────────────────────────────────────────────────────────────
    print(f'\n  [2] 리스크 비교')
    row('BTC 홀딩 MDD', f'{stats_a["btc_mdd"]:.2f}%', f'{stats_b["btc_mdd"]:.2f}%')
    row('Apex MDD', f'{stats_a["apex_mdd"]:.2f}%', f'{stats_b["apex_mdd"]:.2f}%')
    row('MDD 방어율',
        f'{(1 - stats_a["apex_mdd"]/stats_a["btc_mdd"])*100:.1f}%',
        f'{(1 - stats_b["apex_mdd"]/stats_b["btc_mdd"])*100:.1f}%')

    # ── 3. 기어 분포 ──────────────────────────────────────────────────────────
    print(f'\n  [3] 기어 분포')
    for g in ['BEAST', 'CRUISE', 'BUNKER']:
        ca = stats_a['gear_cnt'][g]
        cb = stats_b['gear_cnt'][g]
        ra = ca / stats_a['days'] * 100
        rb = cb / stats_b['days'] * 100
        row(f'{g} 비중', f'{ca}일 ({ra:.0f}%)', f'{cb}일 ({rb:.0f}%)')

    # ── 4. BUNKER 트리거 원인 ─────────────────────────────────────────────────
    print(f'\n  [4] BUNKER 트리거 원인 (연일 중복 집계)')
    for t in ['BTC<MA50', 'RSI과열', '7일급락']:
        row(t,
            f'{stats_a["bunker_triggers"].get(t, 0)}일',
            f'{stats_b["bunker_triggers"].get(t, 0)}일')

    # BUNKER 구간 분포
    def run_dist(runs):
        if not runs:
            return '없음'
        avg = sum(runs) / len(runs)
        mx  = max(runs)
        return f'평균 {avg:.1f}일 / 최장 {mx}일 ({len(runs)}회)'

    row('BUNKER 연속 구간', run_dist(stats_a['bunker_runs']), run_dist(stats_b['bunker_runs']))

    # ── 5. 매매 활동 ──────────────────────────────────────────────────────────
    print(f'\n  [5] 매매 활동')
    row('신규 진입 횟수', f'{stats_a["buy_cnt"]}회', f'{stats_b["buy_cnt"]}회')
    row('청산 횟수', f'{stats_a["sell_cnt"]}회', f'{stats_b["sell_cnt"]}회')

    # ── 6. 핵심 원인 분석 ─────────────────────────────────────────────────────
    print(f'\n  [6] 핵심 분석')
    print()

    # 상승장에서 BUNKER 기간 중 BTC 상승분 계산 (놓친 수익)
    btc_df = None
    missed_a = _calc_missed_upside(logs_a)
    missed_b = _calc_missed_upside(logs_b)
    row('BUNKER 기간 BTC 평균 일수익',
        f'{missed_a:+.3f}%/일',
        f'{missed_b:+.3f}%/일')

    print()
    print(f'  {"─"*(W-2)}')
    print(f'  결론:')
    if stats_a['btc_pnl'] > 30:
        print(f'  {label_a}: BTC가 {stats_a["btc_pnl"]:+.1f}% 오르는 동안 BUNKER {stats_a["gear_cnt"]["BUNKER"]}일')
        print(f'           → "BTC<MA50" 트리거가 상승장 단기조정마다 과민 반응')
    if stats_b['btc_pnl'] < 0:
        print(f'  {label_b}: BTC가 {stats_b["btc_pnl"]:+.1f}% 하락하는 동안 MDD {stats_b["apex_mdd"]:.1f}%로 방어')
        print(f'           → 하락장 방어가 전략의 핵심 강점')
    print('=' * W)


def _calc_missed_upside(logs):
    """BUNKER 기간 동안의 BTC 일평균 등락률"""
    changes = []
    for i in range(1, len(logs)):
        if logs[i]['gear'] == 'BUNKER':
            prev = logs[i-1]['btc_price']
            curr = logs[i]['btc_price']
            if prev > 0:
                changes.append((curr - prev) / prev * 100)
    return sum(changes) / len(changes) if changes else 0.0


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run_backtest():
    """단일 기간 백테스트 (SIM_START ~ SIM_END)"""
    global SIM_START, SIM_END
    print('=' * 65)
    print(f'  Apex Mode Backtest  {SIM_START} ~ {SIM_END}')
    print('=' * 65)
    print('\n[데이터 수집]')
    all_data = fetch_all_history()
    logs  = simulate(all_data, SIM_START, SIM_END)
    stats = calc_stats(logs, all_data, SIM_START, SIM_END)

    btc_df   = all_data.get('BTC')
    btc_data = {d: btc_df.loc[d, 'close'] for d in btc_df.index
                if SIM_START <= d <= SIM_END} if btc_df is not None else {}
    btc_dates       = sorted(btc_data.keys())
    btc_start_price = btc_data[btc_dates[0]]  if btc_dates else 1
    btc_hold_qty    = INIT_KRW * (1 - FEE) / btc_start_price
    btc_hold_vals   = {d: round(btc_hold_qty * btc_data[d]) for d in btc_dates}
    btc_hold_pnls   = {d: (btc_hold_vals[d] - INIT_KRW) / INIT_KRW * 100 for d in btc_dates}

    print(f'\n{"날짜":>12}  {"기어":>8}  {"Apex평가액":>13}  {"Apex수익":>8}  {"BTC홀딩":>12}  {"BTC수익":>8}  보유')
    print('─' * 90)
    for log in logs:
        gear_str = {'BEAST': '★ BEAST', 'CRUISE': '◎ CRUISE', 'BUNKER': '■ BUNKER'}[log['gear']]
        held_str = ','.join(log['held']) if log['held'] else '현금'
        btc_val  = btc_hold_vals.get(log['date'], 0)
        btc_pnl  = btc_hold_pnls.get(log['date'], 0)
        print(f'  {log["date"]}  {gear_str:>8}  {log["portfolio"]:>12,}원  '
              f'{log["pnl_pct"]:>+7.2f}%  {btc_val:>11,}원  {btc_pnl:>+7.2f}%  {held_str}')
        for a in log['actions']:
            print(f'           {a}')

    print(f'\n  수익률: Apex {stats["apex_pnl"]:+.2f}% / BTC홀딩 {stats["btc_pnl"]:+.2f}%')
    print(f'  MDD: Apex {stats["apex_mdd"]:.2f}% / BTC홀딩 {stats["btc_mdd"]:.2f}%')
    print(f'  기어: BEAST {stats["gear_cnt"]["BEAST"]}일 / CRUISE {stats["gear_cnt"]["CRUISE"]}일 / BUNKER {stats["gear_cnt"]["BUNKER"]}일')


def run_comparison():
    """원본(BUNKER 1/3) vs 개선(BUNKER 2/3) — 상승장·하락장 4개 시나리오 비교"""
    global SIM_START, SIM_END, BUNKER_COND_NEED

    BULL  = (date(2024, 1, 1),  date(2024, 12, 31))
    BEAR  = (date(2025, 4, 19), date(2026, 4, 19))

    SIM_START = min(BULL[0], BEAR[0])
    SIM_END   = max(BULL[1], BEAR[1])

    print('=' * 75)
    print('  Apex Mode 개선 분석: V1 원본 vs V3 개선(CAUTION 기어 추가)')
    print('  V3: RSI과열/7일급락→BUNKER / BTC<MA50 단독→CAUTION(40%/1코인)')
    print('  기간: 2024 상승장 / 2025~2026 하락장 각각 비교')
    print('=' * 75)
    print('\n[데이터 수집]')
    all_data = fetch_all_history()

    results = {}
    for version, cond in [('V1_원본', 1), ('V3_개선', 3), ('V4_동적', 4)]:
        BUNKER_COND_NEED = cond
        for label, period in [('상승장', BULL), ('하락장', BEAR)]:
            key = f'{version}_{label}'
            print(f'  [{version} / {label}] 시뮬 중...')
            logs  = simulate(all_data, period[0], period[1])
            stats = calc_stats(logs, all_data, period[0], period[1])
            results[key] = stats

    # ── 출력 ──────────────────────────────────────────────────────────────────
    W = 80
    cols = ['V1_원본_상승장', 'V3_개선_상승장', 'V4_동적_상승장',
            'V1_원본_하락장', 'V3_개선_하락장', 'V4_동적_하락장']
    hdrs = ['V1 원본 2024', 'V3 개선 2024', 'V4 동적 2024',
            'V1 원본 25~26', 'V3 개선 25~26', 'V4 동적 25~26']

    def row(label, vals):
        print(f'  {label:<22}' + ''.join(f'  {v:>14}' for v in vals))

    print()
    print('=' * W)
    print(f'  {"항목":<22}' + ''.join(f'  {h:>14}' for h in hdrs))
    print(f'  {"─"*(W-2)}')

    print('\n  [수익률]')
    row('BTC 홀딩 수익률', [f'{results[c]["btc_pnl"]:+.2f}%' for c in cols])
    row('Apex 수익률',     [f'{results[c]["apex_pnl"]:+.2f}%' for c in cols])
    row('Apex - BTC 차이', [f'{results[c]["apex_pnl"]-results[c]["btc_pnl"]:+.2f}%p' for c in cols])

    print('\n  [리스크 MDD]')
    row('BTC 홀딩 MDD',    [f'{results[c]["btc_mdd"]:.2f}%' for c in cols])
    row('Apex MDD',        [f'{results[c]["apex_mdd"]:.2f}%' for c in cols])
    row('MDD 방어율',      [f'{(1-results[c]["apex_mdd"]/results[c]["btc_mdd"])*100:.1f}%' for c in cols])

    print('\n  [기어 분포]')
    for g in ['BEAST', 'CRUISE', 'CAUTION', 'BUNKER']:
        row(f'{g} 비중',
            [f'{results[c]["gear_cnt"][g]}일 ({results[c]["gear_cnt"][g]/results[c]["days"]*100:.0f}%)'
             for c in cols])

    print('\n  [매매 활동]')
    row('신규 진입 횟수', [f'{results[c]["buy_cnt"]}회' for c in cols])
    row('청산 횟수',      [f'{results[c]["sell_cnt"]}회' for c in cols])

    print('\n  [BUNKER 트리거]')
    for t in ['BTC<MA50', 'RSI과열', '7일급락']:
        row(t, [f'{results[c]["bunker_triggers"].get(t,0)}일' for c in cols])

    def run_dist(runs):
        if not runs:
            return '없음'
        return f'avg{sum(runs)/len(runs):.1f}d/{len(runs)}회'
    row('연속구간', [run_dist(results[c]['bunker_runs']) for c in cols])

    print()
    print('=' * W)
    print('  [핵심 비교]')
    v1b = results['V1_원본_상승장']
    v3b = results['V3_개선_상승장']
    v4b = results['V4_동적_상승장']
    v1r = results['V1_원본_하락장']
    v3r = results['V3_개선_하락장']
    v4r = results['V4_동적_하락장']

    print(f'  {"":18}  {"V1 원본":>10}  {"V3 개선":>10}  {"V4 동적":>10}')
    print(f'  {"─"*52}')
    print(f'  {"[상승장 2024]":18}')
    print(f'  {"  수익률":18}  {v1b["apex_pnl"]:>+9.2f}%  {v3b["apex_pnl"]:>+9.2f}%  {v4b["apex_pnl"]:>+9.2f}%')
    print(f'  {"  MDD":18}  {v1b["apex_mdd"]:>9.2f}%  {v3b["apex_mdd"]:>9.2f}%  {v4b["apex_mdd"]:>9.2f}%')
    print(f'  {"  BUNKER일수":18}  {v1b["gear_cnt"]["BUNKER"]:>9}일  {v3b["gear_cnt"]["BUNKER"]:>9}일  {v4b["gear_cnt"]["BUNKER"]:>9}일')
    print(f'  {"  CAUTION일수":18}  {v1b["gear_cnt"]["CAUTION"]:>9}일  {v3b["gear_cnt"]["CAUTION"]:>9}일  {v4b["gear_cnt"]["CAUTION"]:>9}일')
    print(f'  {"[하락장 25~26]":18}')
    print(f'  {"  수익률":18}  {v1r["apex_pnl"]:>+9.2f}%  {v3r["apex_pnl"]:>+9.2f}%  {v4r["apex_pnl"]:>+9.2f}%')
    print(f'  {"  MDD":18}  {v1r["apex_mdd"]:>9.2f}%  {v3r["apex_mdd"]:>9.2f}%  {v4r["apex_mdd"]:>9.2f}%')
    print(f'  {"  BUNKER일수":18}  {v1r["gear_cnt"]["BUNKER"]:>9}일  {v3r["gear_cnt"]["BUNKER"]:>9}일  {v4r["gear_cnt"]["BUNKER"]:>9}일')
    print(f'  {"  CAUTION일수":18}  {v1r["gear_cnt"]["CAUTION"]:>9}일  {v3r["gear_cnt"]["CAUTION"]:>9}일  {v4r["gear_cnt"]["CAUTION"]:>9}일')
    print('=' * W)


if __name__ == '__main__':
    run_comparison()
