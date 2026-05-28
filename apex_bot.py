"""
Apex Mode Bot — 전설의 집합체 전략
PTJ × Soros × Druckenmiller × Livermore × Richard Dennis

GEAR 1 BEAST:  공격 (최대 90% 투입, 피라미딩 3단계)
GEAR 2 CRUISE: 표준 (70% 투입, 상위 2코인)
GEAR 3 BUNKER: 방어 (전량 현금, 시장 떠남)

실행: 매일 1회 (GitHub Actions UTC 00:00 = KST 09:00)
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import ccxt
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime, date

# ── 설정 ─────────────────────────────────────────────────────────────────────
COINS       = ['BTC', 'ETH', 'XRP', 'SOL']
FEE         = 0.0005
INIT_KRW    = 1_000_000
PAPER_MODE  = True
API_KEY     = ''
API_SECRET  = ''

# 기어별 투자 비율 / 최대 보유 코인 수
# CAUTION: BTC>MA200(장기 상승) 구간에서 BTC<MA50(단기 조정) 시 — 40% 보수적 투자
GEAR_ALLOC  = {'BEAST': 0.90, 'CRUISE': 0.70, 'CAUTION': 0.40, 'BUNKER': 0.00}
GEAR_MAX    = {'BEAST': 1,    'CRUISE': 2,     'CAUTION': 1,    'BUNKER': 0}

# 피라미딩 (BEAST 전용) — 3단계 40/30/30
LOT_SPLITS       = [0.40, 0.30, 0.30]
PYRAMID_TRIGGER  = 0.03   # 전 단계 진입가 +3% 돌파시 추가 매수

# 리스크 관리
HARD_STOP_PCT       = 0.08   # 진입 평균가 기준 -8% 하드스탑
TRAIL_ATR_MULT      = 2.0    # 고점 - ATR×2 트레일링 스탑 (하락장/기본)
TRAIL_ATR_MULT_BULL = 3.0    # 고점 - ATR×3 트레일링 스탑 (장기 상승장 BTC>MA200)
PARTIAL_PROFIT      = 0.15   # +15% 도달시 50% 부분 익절
ATR_PERIOD          = 14
ROTATION_THRESHOLD  = 0.20   # 상승장 로테이션 조건: 신규 스코어가 현재보다 20% 이상 높을 때만
MIN_CASH_RATIO      = 0.05   # 총 자산 대비 최소 현금 비율 (5%)
BEAST_STREAK_MIN    = 2      # 피라미딩 허용 최소 연속 BEAST 일수

# 볼린저 밴드
BB_PERIOD       = 20
BB_STD          = 2.0
BB_WIDTH_MULT   = 1.1   # 밴드 폭 20일 평균 대비 확장 배율 (BEAST 조건)
BB_ENTRY_MAX    = 0.8   # %B 이상이면 신규 진입 보류

# 레짐 판단 임계값
BEAST_COND_NEED  = 3      # BEAST 조건 5개 중 3개 이상
BUNKER_DROP      = -12.0  # BTC 7일 낙폭 -12% 이하 → BUNKER
BUNKER_RSI       = 80     # BTC RSI > 80 → 과열 BUNKER
BEAST_RSI_MAX    = 70     # BTC RSI < 70 이어야 BEAST 가능
BEAST_MOM_MIN    = 5.0    # BTC 7일 모멘텀 +5% 이상
BEAST_VOL_MIN    = 1.2    # 거래량 20일 평균 대비 1.2배 이상

STATE_FILE  = Path('data/apex_bot_state.json')
LOG_FILE    = Path('data/apex_bot_log.jsonl')


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

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


def calc_bollinger(series, period=BB_PERIOD, std_mult=BB_STD):
    ma    = series.rolling(period).mean()
    sd    = series.rolling(period).std()
    upper = ma + sd * std_mult
    lower = ma - sd * std_mult
    bw    = (upper - lower) / ma
    pct_b = (series - lower) / (upper - lower)
    return {
        'upper'    : upper.iloc[-1],
        'lower'    : lower.iloc[-1],
        'mid'      : ma.iloc[-1],
        'bandwidth': bw.iloc[-1],
        'bw_ma'    : bw.rolling(period).mean().iloc[-1],
        'pct_b'    : pct_b.iloc[-1],
    }


# ── 데이터 수집 ───────────────────────────────────────────────────────────────

def fetch_ohlcv(coin, limit=70):
    try:
        ex = ccxt.upbit({'enableRateLimit': True})
        rows = ex.fetch_ohlcv(f'{coin}/KRW', '1d', limit=limit)
        df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.date
        df = df.set_index('date').sort_index().astype(float)
        return df
    except Exception as e:
        print(f'  {coin} 데이터 오류: {e}')
        return None


# ── 레짐 감지 ─────────────────────────────────────────────────────────────────

def detect_gear(btc_df):
    """
    BTC 일봉 데이터로 시장 레짐 판단
    반환: ('BEAST' | 'CRUISE' | 'BUNKER', 조건 상세)
    """
    close  = btc_df['close']
    price  = close.iloc[-1]

    ma50   = calc_ma(close, 50)
    ma200  = calc_ma(close, 200)
    rsi    = calc_rsi(close)
    mom7   = (price - close.iloc[-8]) / close.iloc[-8] * 100
    vol    = btc_df['volume'].iloc[-1]
    vol_ma = btc_df['volume'].rolling(20).mean().iloc[-1]
    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0

    bb          = calc_bollinger(close)
    band_expand = bool(bb['bandwidth'] > bb['bw_ma'] * BB_WIDTH_MULT)

    below_ma50   = price < ma50
    above_ma200  = price > ma200
    rsi_hot      = rsi > BUNKER_RSI
    crash7       = mom7 < BUNKER_DROP

    active = [k for k, v in {'BTC<MA50': below_ma50, 'RSI과열': rsi_hot,
                              '7일급락': crash7}.items() if v]
    base_details = {
        'price': price, 'ma50': ma50, 'ma200': ma200,
        'rsi': rsi, 'mom7': mom7, 'vol_ratio': vol_ratio,
        'trigger': active,
        'bb_pct_b'    : round(bb['pct_b'], 3),
        'bb_expanding': band_expand,
    }

    # V4 동적 전환: MA200 기준으로 레짐 구분
    # - 장기 하락장(BTC<MA200): 원본 V1 — BTC<MA50 하나만으로도 BUNKER
    # - 장기 상승장(BTC>MA200): 개선 V3 — RSI과열/7일급락만 BUNKER, BTC<MA50 단독 → CAUTION
    if above_ma200:
        go_bunker = rsi_hot or crash7
    else:
        go_bunker = below_ma50 or rsi_hot or crash7

    if go_bunker:
        return 'BUNKER', base_details

    # BEAST 조건 채점
    beast_conds = {
        'BTC>MA50'    : price > ma50,
        'BTC>MA200'   : above_ma200,
        'RSI적정'     : rsi < BEAST_RSI_MAX,
        '거래량확장'  : vol_ratio >= BEAST_VOL_MIN,
        '7일모멘텀'   : mom7 >= BEAST_MOM_MIN,
        '밴드확장'    : band_expand,
    }
    beast_score = sum(beast_conds.values())
    details = {**base_details, 'beast_score': beast_score, 'conds': beast_conds}

    # 장기 상승장(BTC>MA200)에서 BTC<MA50 단독 조정 → CAUTION (40% 보수적 투자)
    if above_ma200 and below_ma50:
        return 'CAUTION', details

    if beast_score >= BEAST_COND_NEED:
        return 'BEAST', details
    return 'CRUISE', details


# ── 모멘텀 스코어 ─────────────────────────────────────────────────────────────

def calc_momentum_scores(all_data):
    """
    스코어 = (7일수익률 × 0.4) + (30일수익률 × 0.3) + (거래량증가율 × 0.3)
    MA50 아래 코인은 후보에서 제외
    """
    scores = {}
    for coin, df in all_data.items():
        if df is None or len(df) < 35:
            continue
        close = df['close']
        price = close.iloc[-1]
        ma50  = calc_ma(close, 50)

        if price < ma50:
            continue  # MA50 필터

        ret7  = (price - close.iloc[-8])  / close.iloc[-8]  * 100
        ret30 = (price - close.iloc[-31]) / close.iloc[-31] * 100

        vol       = df['volume'].iloc[-1]
        vol_ma    = df['volume'].rolling(20).mean().iloc[-1]
        vol_score = (vol / vol_ma - 1) * 100 if vol_ma > 0 else 0

        score = ret7 * 0.4 + ret30 * 0.3 + vol_score * 0.3

        atr = calc_atr(df)
        above_ma200 = None
        if len(close) >= 200:
            above_ma200 = bool(price > calc_ma(close, 200))

        bb    = calc_bollinger(close)
        pct_b = round(bb['pct_b'], 3)

        scores[coin] = {
            'price'      : price,
            'ma50'       : ma50,
            'ret7'       : round(ret7, 2),
            'ret30'      : round(ret30, 2),
            'vol_ratio'  : round(vol / vol_ma if vol_ma > 0 else 1, 2),
            'score'      : round(score, 2),
            'atr'        : atr,
            'above_ma200': above_ma200,
            'pct_b'      : pct_b,
        }

    return dict(sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True))


# ── 상태 관리 ─────────────────────────────────────────────────────────────────

def load_state():
    STATE_FILE.parent.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {
        'gear'         : 'CRUISE',
        'positions'    : {},
        'krw'          : float(INIT_KRW),
        'init_krw'     : float(INIT_KRW),
        'last_date'    : None,
        'beast_streak' : 0,
    }


def save_state(state):
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(STATE_FILE)


def append_log(record):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def total_value(state, scores):
    val = state['krw']
    for coin, pos in state['positions'].items():
        price = scores[coin]['price'] if coin in scores else pos['avg_entry']
        val  += pos['total_qty'] * price
    return val


# ── 거래 실행 ─────────────────────────────────────────────────────────────────

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
    pos        = state['positions'][coin]
    sell_qty   = pos['total_qty'] * ratio
    recv       = sell_qty * price * (1 - FEE)
    pos['total_qty']  -= sell_qty
    pos['partial_exited'] = True
    state['krw'] += recv
    # lots 비율 축소
    for lot in pos['lots']:
        lot['qty'] *= (1 - ratio)
    return recv


def paper_buy(state, coin, price, krw_amount, lot_num, atr, bull_macro=False):
    qty        = krw_amount * (1 - FEE) / price
    trail_mult = TRAIL_ATR_MULT_BULL if bull_macro else TRAIL_ATR_MULT
    state['krw'] -= krw_amount

    if coin not in state['positions']:
        state['positions'][coin] = {
            'lots'           : [],
            'total_qty'      : 0.0,
            'avg_entry'      : price,
            'hwm'            : price,
            'hard_stop'      : round(price * (1 - HARD_STOP_PCT)),
            'trailing_stop'  : round(price - atr * trail_mult),
            'partial_exited' : False,
            'bull_macro'     : bull_macro,
        }

    pos = state['positions'][coin]
    pos['lots'].append({'lot': lot_num, 'qty': qty, 'entry': price})
    pos['total_qty'] += qty

    # 평균 진입가 재계산
    total_cost = sum(l['qty'] * l['entry'] for l in pos['lots'])
    pos['avg_entry'] = round(total_cost / pos['total_qty'])

    # 하드스탑은 평균 진입가 기준으로 갱신
    pos['hard_stop'] = round(pos['avg_entry'] * (1 - HARD_STOP_PCT))

    return qty


def update_stops(state, scores):
    """보유 포지션의 고점/트레일링 스탑 갱신"""
    for coin, pos in state['positions'].items():
        if coin not in scores:
            continue
        price      = scores[coin]['price']
        atr        = scores[coin]['atr']
        trail_mult = TRAIL_ATR_MULT_BULL if pos.get('bull_macro') else TRAIL_ATR_MULT

        if price > pos['hwm']:
            pos['hwm'] = price

        new_trail = round(pos['hwm'] - atr * trail_mult)
        if new_trail > pos['trailing_stop']:
            pos['trailing_stop'] = new_trail


# ── 메인 로직 ─────────────────────────────────────────────────────────────────

def run(state, gear, gear_details, scores):
    today      = str(date.today())
    actions    = []
    alloc      = GEAR_ALLOC[gear]
    max_pos    = GEAR_MAX[gear]
    bull_macro = bool(gear_details.get('price', 0) > gear_details.get('ma200', 0))

    portfolio = total_value(state, scores)

    # ── 0. 스탑 업데이트 ─────────────────────────────────────────────────────
    update_stops(state, scores)

    # ── 1. 청산 체크 (우선순위 순) ───────────────────────────────────────────
    to_sell = []

    for coin, pos in list(state['positions'].items()):
        price = scores[coin]['price'] if coin in scores else pos['avg_entry']
        reason = None

        if gear == 'BUNKER':
            reason = 'BUNKER 모드 전환 — 전량 청산'
        elif price <= pos['hard_stop']:
            reason = f'하드스탑 ({price:,.0f} <= {pos["hard_stop"]:,.0f})'
        elif price <= pos['trailing_stop']:
            reason = f'트레일링 스탑 ({price:,.0f} <= {pos["trailing_stop"]:,.0f})'
        elif coin not in scores and gear != 'BUNKER':
            reason = 'MA50 하향돌파 — 후보 제외'

        if reason:
            to_sell.append((coin, price, reason, 'full'))
        elif not pos['partial_exited'] and price >= pos['avg_entry'] * (1 + PARTIAL_PROFIT):
            to_sell.append((coin, price, f'+{PARTIAL_PROFIT*100:.0f}% 부분 익절', 'half'))

    for coin, price, reason, sell_type in to_sell:
        if sell_type == 'full':
            recv = paper_sell_all(state, coin, price)
            actions.append(f'[매도 전량] {coin} @ {price:,.0f}원  수익 {recv:,.0f}원  사유: {reason}')
        else:
            recv = paper_sell_partial(state, coin, price)
            actions.append(f'[매도 50%] {coin} @ {price:,.0f}원  수익 {recv:,.0f}원  사유: {reason}')

    # ── 2. BUNKER — 여기서 종료 ──────────────────────────────────────────────
    if gear == 'BUNKER':
        if not actions:
            actions.append('BUNKER 모드 — 현금 보유 (매매 없음)')
        return actions

    # ── 3. 현금 버퍼 + 피라미딩 안정성 계산 ────────────────────────────────
    min_cash      = portfolio * MIN_CASH_RATIO
    available_krw = max(0.0, state['krw'] - min_cash)
    beast_streak  = state.get('beast_streak', 0)

    # ── 4. 피라미딩 체크 (BEAST 연속 N일+ + 기존 보유) ──────────────────────
    if gear == 'BEAST' and beast_streak < BEAST_STREAK_MIN:
        actions.append(f'BEAST {beast_streak}일차 — 피라미딩 대기 (최소 {BEAST_STREAK_MIN}일 연속 필요)')

    if gear == 'BEAST' and beast_streak >= BEAST_STREAK_MIN:
        for coin, pos in state['positions'].items():
            if coin not in scores:
                continue
            price    = scores[coin]['price']
            atr      = scores[coin]['atr']
            lot_done = len(pos['lots'])

            if lot_done >= 3:
                continue

            prev_entry = pos['lots'][-1]['entry']
            if price < prev_entry * (1 + PYRAMID_TRIGGER):
                continue

            lot_num  = lot_done + 1
            lot_frac = LOT_SPLITS[lot_done]
            budget   = portfolio * alloc * lot_frac

            if available_krw < budget * 0.5:
                continue  # 버퍼 고려 KRW 부족

            budget = min(budget, available_krw)
            qty = paper_buy(state, coin, price, budget, lot_num, atr, bull_macro)
            actions.append(
                f'[피라미딩 {lot_num}차] {coin} @ {price:,.0f}원  {qty:.6f}개  '
                f'({budget:,.0f}원)'
            )

    # ── 5. 신규 진입 ─────────────────────────────────────────────────────────
    top_coins  = list(scores.keys())[:max_pos]
    held_coins = set(state['positions'].keys())

    # 보유 중이지만 top에 없으면 청산 (포트폴리오 로테이션)
    for coin in list(held_coins):
        if coin not in top_coins:
            # 상승장에서는 스코어 차이가 ROTATION_THRESHOLD 이상일 때만 로테이션
            if bull_macro and coin in scores:
                held_score = scores[coin]['score']
                top_score  = scores[list(scores.keys())[0]]['score'] if scores else 0
                if top_score <= 0 or (top_score - held_score) / abs(top_score) < ROTATION_THRESHOLD:
                    actions.append(f'{coin} 유지 (로테이션 임계값 미달)')
                    continue
            price = scores[coin]['price'] if coin in scores else state['positions'][coin]['avg_entry']
            recv  = paper_sell_all(state, coin, price)
            actions.append(f'[로테이션 매도] {coin} @ {price:,.0f}원  {recv:,.0f}원')

    for coin in top_coins:
        if coin in state['positions']:
            actions.append(f'{coin} 유지 (Lot {len(state["positions"][coin]["lots"])}차)')
            continue

        price      = scores[coin]['price']
        atr        = scores[coin]['atr']
        num_to_buy = len(top_coins)

        if gear == 'BEAST':
            budget = portfolio * alloc * LOT_SPLITS[0]
        else:
            budget = portfolio * alloc / num_to_buy

        # Bear macro 코인 비중 50% 축소 (CRUISE / CAUTION)
        coin_bull = scores[coin].get('above_ma200')
        if gear in ('CRUISE', 'CAUTION') and coin_bull is False:
            budget *= 0.5
            actions.append(f'  ※ {coin} MA200 하향 (bear macro) — 비중 50% 축소')

        pct_b = scores[coin].get('pct_b')
        if pct_b is not None and pct_b >= BB_ENTRY_MAX:
            actions.append(f'{coin} 진입 보류 — %B {pct_b:.2f} (상단밴드 근처, {BB_ENTRY_MAX} 초과)')
            continue

        if available_krw < budget * 0.5:
            actions.append(f'{coin} 진입 실패 — KRW 부족')
            continue

        budget = min(budget, available_krw)
        qty    = paper_buy(state, coin, price, budget, 1, atr, bull_macro)
        pos    = state['positions'][coin]
        actions.append(
            f'[매수 Lot1] {coin} @ {price:,.0f}원  {qty:.6f}개  '
            f'스탑: {pos["trailing_stop"]:,.0f} / 하드: {pos["hard_stop"]:,.0f}'
        )

    return actions


# ── 대시보드 ──────────────────────────────────────────────────────────────────

GEAR_ICON = {'BEAST': '★ BEAST', 'CRUISE': '◎ CRUISE', 'CAUTION': '▲ CAUTION', 'BUNKER': '■ BUNKER'}
GEAR_DESC = {
    'BEAST'  : '공격 모드 (90% 투입, 피라미딩)',
    'CRUISE' : '표준 모드 (70% 투입, 분산)',
    'CAUTION': '주의 모드 (40% 투입, 1코인) — MA200 상승장 + MA50 단기조정',
    'BUNKER' : '방어 모드 (전량 현금)',
}

def print_dashboard(state, gear, gear_details, scores, actions):
    today = str(date.today())
    total = total_value(state, scores)
    init  = state['init_krw']
    pnl   = (total - init) / init * 100

    print()
    print('=' * 65)
    print(f'  Apex Mode Bot  {today}')
    print(f'  {GEAR_ICON[gear]}  —  {GEAR_DESC[gear]}')
    print('=' * 65)

    # 레짐 상세
    d = gear_details
    print(f'\n  [레짐 판단]')
    print(f'  BTC {d["price"]:,.0f}원  MA50:{d["ma50"]:,.0f}  MA200:{d["ma200"]:,.0f}')
    print(f'  RSI:{d["rsi"]:.1f}  7일모멘텀:{d["mom7"]:+.1f}%  거래량:{d["vol_ratio"]:.2f}x')
    print(f'  BB %B:{d.get("bb_pct_b", "N/A")}  밴드확장:{"O" if d.get("bb_expanding") else "X"}')
    if gear == 'BUNKER':
        print(f'  트리거: {d["trigger"]}')
    else:
        score = d.get('beast_score', '-')
        conds_str = ', '.join(f'{k}:{"O" if v else "X"}' for k, v in d.get('conds', {}).items())
        print(f'  BEAST 점수: {score}/6  [{conds_str}]')

    # 코인 스코어
    print(f'\n  [코인 모멘텀 스코어]')
    print(f'  {"코인":>5}  {"현재가":>13}  {"7일":>7}  {"30일":>7}  {"거래량":>7}  {"스코어":>8}  {"%B":>6}  {"선택"}')
    print(f'  {"─"*70}')
    selected = list(scores.keys())[:GEAR_MAX[gear]]
    for i, (coin, s) in enumerate(scores.items()):
        sel  = '★' if coin in selected else ''
        pb   = s.get('pct_b')
        pb_s = f'{pb:.2f}' if pb is not None else '  N/A'
        print(f'  {coin:>5}  {s["price"]:>12,.0f}원  {s["ret7"]:>+6.1f}%  '
              f'{s["ret30"]:>+6.1f}%  {s["vol_ratio"]:>5.2f}x  {s["score"]:>8.1f}  {pb_s:>6}  {sel}')

    # 포지션
    print(f'\n  [포지션]')
    if state['positions']:
        for coin, pos in state['positions'].items():
            price   = scores[coin]['price'] if coin in scores else pos['avg_entry']
            unreal  = (price - pos['avg_entry']) / pos['avg_entry'] * 100
            hwm_pct = (pos['hwm'] - pos['avg_entry']) / pos['avg_entry'] * 100
            print(f'  {coin}: {pos["total_qty"]:.6f}개  평균진입:{pos["avg_entry"]:,}  현재:{price:,}  '
                  f'미실현:{unreal:+.2f}%')
            print(f'       고점:{pos["hwm"]:,}({hwm_pct:+.1f}%)  '
                  f'트레일:{pos["trailing_stop"]:,}  하드:{pos["hard_stop"]:,}  '
                  f'Lot{len(pos["lots"])}차  부분익절:{pos["partial_exited"]}')
    else:
        print(f'  현금 보유 중')

    # 손익
    print(f'\n  {"─"*63}')
    print(f'  총 평가액  : {total:>14,.0f}원')
    print(f'  초기 자본  : {init:>14,.0f}원')
    print(f'  손익       : {pnl:>+.2f}%  ({total-init:>+,.0f}원)')
    print(f'  현금       : {state["krw"]:>14,.0f}원')

    # 오늘 실행
    if actions:
        print(f'\n  [오늘 실행]')
        for a in actions:
            print(f'  → {a}')
    print()


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    args        = sys.argv[1:]
    status_only = 'status' in args

    print('Apex Mode Bot 시작...')

    # 1. 데이터 수집
    print('\n[데이터 수집]')
    all_data = {}
    for coin in COINS:
        print(f'  {coin} 수집 중...')
        all_data[coin] = fetch_ohlcv(coin, limit=220)  # MA200 계산 위해 전 코인 220봉
        time.sleep(0.3)

    btc_df = all_data.get('BTC')
    if btc_df is None or len(btc_df) < 50:
        print('BTC 데이터 부족. 종료.')
        return

    # 2. 레짐 판단
    print('\n[레짐 판단]')
    gear, gear_details = detect_gear(btc_df)
    print(f'  → {GEAR_ICON[gear]}')

    # 3. 모멘텀 스코어
    print('\n[모멘텀 스코어 계산]')
    scores = calc_momentum_scores(all_data)
    for coin, s in scores.items():
        print(f'  {coin}: 스코어={s["score"]:.1f}  7일={s["ret7"]:+.1f}%  30일={s["ret30"]:+.1f}%')

    # 상태 로드
    state = load_state()

    if status_only:
        print_dashboard(state, gear, gear_details, scores, [])
        return

    # 오늘 이미 실행했으면 스킵
    today = str(date.today())
    if state.get('last_date') == today:
        print(f'오늘({today}) 이미 실행 완료. status 모드로 전환.')
        print_dashboard(state, gear, gear_details, scores, [])
        return

    # 4. 전략 실행
    if gear == 'BEAST':
        state['beast_streak'] = state.get('beast_streak', 0) + 1
    else:
        state['beast_streak'] = 0

    print('\n[전략 실행]')
    actions = run(state, gear, gear_details, scores)
    state['gear']      = gear
    state['last_date'] = today

    # 5. 저장 & 로그
    save_state(state)
    append_log({
        'date'         : today,
        'gear'         : gear,
        'beast_streak' : state.get('beast_streak', 0),
        'btc_ma50'        : round(gear_details['ma50']),
        'btc_rsi'         : round(gear_details['rsi'], 1),
        'btc_mom7'        : round(gear_details['mom7'], 2),
        'btc_bb_pct_b'    : gear_details.get('bb_pct_b'),
        'btc_bb_expanding': gear_details.get('bb_expanding'),
        'scores'          : {c: {'score': s['score'], 'price': round(s['price']),
                                 'above_ma200': s.get('above_ma200'),
                                 'pct_b': s.get('pct_b')}
                             for c, s in scores.items()},
        'positions'    : {c: {'avg_entry': p['avg_entry'], 'total_qty': round(p['total_qty'], 6)}
                          for c, p in state['positions'].items()},
        'krw'          : round(state['krw']),
        'actions'      : actions,
    })

    # 6. 대시보드
    print_dashboard(state, gear, gear_details, scores, actions)


if __name__ == '__main__':
    main()
