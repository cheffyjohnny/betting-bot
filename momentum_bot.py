"""
모멘텀 로테이션 봇
전략: 룩백=15일 / 보유=1일 / 50MA 필터
대상: BTC / ETH / XRP / SOL (업비트 KRW)
실행: 매일 1회 (아무 시간이나 동일 시간대에)

사용법:
  python momentum_bot.py          → 오늘 신호 확인 + 리밸런싱
  python momentum_bot.py status   → 현재 상태만 확인 (거래 안 함)
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import ccxt
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timezone, date

# ── 설정 ─────────────────────────────────────────────────────────────────────
COINS       = ['BTC', 'ETH', 'XRP', 'SOL']
LOOKBACK    = 15       # 모멘텀 측정 기간 (일)
MA_PERIOD   = 50       # MA 필터 기간
FEE         = 0.0005   # 업비트 수수료 0.05%
INIT_KRW    = 1_000_000  # 페이퍼 초기 자본

PAPER_MODE  = True     # True=페이퍼 트레이딩 / False=실거래 (업비트 API 키 필요)
API_KEY     = ''       # 실거래 시 입력
API_SECRET  = ''       # 실거래 시 입력

STATE_FILE  = Path('data/momentum_bot_state.json')
TRADE_FILE  = Path('data/momentum_bot_trades.json')
LOG_FILE    = Path('data/bot_log.jsonl')

# ── 데이터 ────────────────────────────────────────────────────────────────────

def fetch_daily(coin, limit=70):
    """업비트에서 일봉 데이터 수집"""
    try:
        ex = ccxt.upbit({'enableRateLimit': True})
        rows = ex.fetch_ohlcv(f'{coin}/KRW', '1d', limit=limit)
        df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
        df['date'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.date
        df = df.set_index('date').sort_index()
        return df['close'].astype(float)
    except Exception as e:
        print(f'  {coin} 데이터 오류: {e}')
        return None


def get_current_price(coin):
    """현재가 조회"""
    try:
        ex = ccxt.upbit({'enableRateLimit': True})
        ticker = ex.fetch_ticker(f'{coin}/KRW')
        return float(ticker['last'])
    except Exception as e:
        print(f'  {coin} 현재가 오류: {e}')
        return None


# ── 신호 계산 ─────────────────────────────────────────────────────────────────

def calc_signal():
    """
    모멘텀 + 50MA 기반 오늘의 신호 계산
    반환: (target_coin, signal_info_dict)
    """
    print('  데이터 수집 중...')
    prices = {}
    for c in COINS:
        s = fetch_daily(c, limit=70)
        if s is not None and len(s) >= MA_PERIOD + LOOKBACK:
            prices[c] = s
        time.sleep(0.2)

    if not prices:
        return None, {}

    signals = {}
    for c, s in prices.items():
        price_now  = s.iloc[-1]
        price_past = s.iloc[-(LOOKBACK + 1)]  # 15일 전
        ma50       = s.iloc[-MA_PERIOD:].mean()

        momentum = (price_now - price_past) / price_past * 100
        above_ma = price_now > ma50

        signals[c] = {
            'price'   : price_now,
            'ma50'    : ma50,
            'momentum': momentum,
            'above_ma': above_ma,
        }

    # 50MA 위 코인만 후보
    candidates = {c: v for c, v in signals.items() if v['above_ma']}

    if not candidates:
        target = None   # 전부 현금
    else:
        target = max(candidates, key=lambda c: candidates[c]['momentum'])

    return target, signals


# ── 상태 관리 ─────────────────────────────────────────────────────────────────

def load_state():
    STATE_FILE.parent.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'holding' : None,
        'qty'     : 0.0,
        'krw'     : float(INIT_KRW),
        'init_krw': float(INIT_KRW),
        'last_date': None,
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_trades():
    if TRADE_FILE.exists():
        with open(TRADE_FILE) as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADE_FILE, 'w') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)


def append_log(today, signals, target, actions):
    """매일 실행 결과를 bot_log.jsonl에 한 줄씩 append"""
    LOG_FILE.parent.mkdir(exist_ok=True)
    record = {
        'date'   : today,
        'target' : target,
        'actions': actions or [],
        'signals': {
            c: {
                'price'   : round(v['price']),
                'ma50'    : round(v['ma50']),
                'momentum': round(v['momentum'], 2),
                'above_ma': bool(v['above_ma']),
            }
            for c, v in signals.items()
        },
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ── 거래 실행 ─────────────────────────────────────────────────────────────────

def paper_sell(state, coin, price):
    qty = state['qty']
    krw = qty * price * (1 - FEE)
    state['krw'] = krw
    state['qty'] = 0.0
    state['holding'] = None
    return krw


def paper_buy(state, coin, price):
    krw   = state['krw']
    qty   = krw * (1 - FEE) / price
    state['qty']     = qty
    state['holding'] = coin
    state['krw']     = 0.0
    return qty


def real_sell(coin, qty):
    """실거래 매도 (업비트 API)"""
    ex = ccxt.upbit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    })
    order = ex.create_market_sell_order(f'{coin}/KRW', qty)
    return order


def real_buy(coin, krw_amount):
    """실거래 매수 (업비트 API)"""
    ex = ccxt.upbit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    })
    price  = get_current_price(coin)
    qty    = krw_amount * (1 - FEE) / price
    order  = ex.create_market_buy_order(f'{coin}/KRW', qty)
    return order


# ── 메인 로직 ─────────────────────────────────────────────────────────────────

def rebalance(state, target, signals, trades):
    today     = str(date.today())
    holding   = state.get('holding')
    action_log = []

    # ── 1. 매도 ──────────────────────────────────────────────────────────────
    if holding and holding != target:
        price = signals[holding]['price'] if holding in signals else get_current_price(holding)
        if price is None:
            print(f'  {holding} 현재가 조회 실패, 매도 취소')
            return

        if PAPER_MODE:
            krw_recv = paper_sell(state, holding, price)
        else:
            real_sell(holding, state['qty'])
            krw_recv = state['qty'] * price * (1 - FEE)
            state['krw']     = krw_recv
            state['qty']     = 0.0
            state['holding'] = None

        trade = {
            'date'  : today,
            'action': 'SELL',
            'coin'  : holding,
            'price' : price,
            'qty'   : state.get('qty', 0),
            'krw'   : krw_recv,
        }
        trades.append(trade)
        action_log.append(f'매도 {holding} @ {price:,.0f}원 → {krw_recv:,.0f}원')

    # ── 2. 현금 유지 (후보 없음) ─────────────────────────────────────────────
    if target is None:
        if holding:
            action_log.append('→ 50MA 위 코인 없음: 현금 보유')
        else:
            action_log.append('현금 유지 (50MA 위 코인 없음)')
        return action_log

    # ── 3. 매수 ──────────────────────────────────────────────────────────────
    if holding != target:
        price = signals[target]['price']

        if PAPER_MODE:
            qty = paper_buy(state, target, price)
        else:
            order = real_buy(target, state['krw'])
            qty   = float(order.get('amount', state['krw'] * (1-FEE) / price))
            state['qty']     = qty
            state['holding'] = target
            state['krw']     = 0.0

        trade = {
            'date'  : today,
            'action': 'BUY',
            'coin'  : target,
            'price' : price,
            'qty'   : qty,
            'krw'   : qty * price,
        }
        trades.append(trade)
        action_log.append(f'매수 {target} @ {price:,.0f}원  수량 {qty:.6f}')

    else:
        action_log.append(f'{target} 유지 (변경 없음)')

    return action_log


# ── 대시보드 출력 ──────────────────────────────────────────────────────────────

def print_dashboard(state, signals, actions=None):
    today = str(date.today())
    print()
    print('=' * 60)
    print(f'  모멘텀 로테이션 봇  {today}')
    print(f'  {"페이퍼 트레이딩" if PAPER_MODE else "실거래"}  |  '
          f'룩백={LOOKBACK}일 / MA={MA_PERIOD}일')
    print('=' * 60)

    # 코인별 신호
    print(f'\n  {"코인":>5}  {"현재가":>14}  {"50MA":>14}  {"MA위":>5}  {"모멘텀(15일)":>12}  {"후보"}')
    print(f'  {"─"*65}')
    for c in COINS:
        if c not in signals:
            continue
        s = signals[c]
        cand = '★ 1위' if c == max(
            {k: v for k, v in signals.items() if v['above_ma']},
            key=lambda x: signals[x]['momentum'],
            default=None) else ''
        ma_flag = '✓' if s['above_ma'] else '✗'
        print(f'  {c:>5}  {s["price"]:>13,.0f}원  {s["ma50"]:>13,.0f}원  '
              f'{ma_flag:>5}  {s["momentum"]:>+10.1f}%  {cand}')

    # 포지션
    print(f'\n  {"─"*60}')
    holding = state.get('holding')
    krw     = state.get('krw', 0)
    qty     = state.get('qty', 0)
    init    = state.get('init_krw', INIT_KRW)

    if holding and holding in signals:
        cur_val = qty * signals[holding]['price']
        total   = cur_val
    else:
        cur_val = 0
        total   = krw

    pnl = (total - init) / init * 100

    print(f'  보유: {holding if holding else "현금"}  '
          f'{"수량: " + f"{qty:.6f}" if holding else ""}')
    print(f'  평가액: {total:>14,.0f}원')
    print(f'  초기자본: {init:>12,.0f}원')
    print(f'  손익: {pnl:>+.2f}%  ({total-init:>+,.0f}원)')

    # 오늘 액션
    if actions:
        print(f'\n  [오늘 실행]')
        for a in actions:
            print(f'  → {a}')

    print()


def print_status_only(state, signals):
    """거래 없이 현재 상태만 출력"""
    print_dashboard(state, signals)

    # 최근 거래 내역
    trades = load_trades()
    if trades:
        print(f'  [최근 거래 내역]')
        for t in trades[-5:]:
            print(f'  {t["date"]}  {t["action"]:>4}  {t["coin"]:>4}  '
                  f'@ {t["price"]:>14,.0f}원')
    print()


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    status_only = 'status' in args

    print('모멘텀 로테이션 봇 시작...')
    print(f'모드: {"상태 확인" if status_only else "리밸런싱"}')

    # 신호 계산
    print('\n[신호 계산]')
    target, signals = calc_signal()
    if not signals:
        print('데이터 수집 실패. 종료.')
        return

    # 상태 로드
    state  = load_state()
    trades = load_trades()

    if status_only:
        print_status_only(state, signals)
        return

    # 오늘 이미 리밸런싱했으면 스킵
    today = str(date.today())
    if state.get('last_date') == today:
        print(f'오늘({today}) 이미 리밸런싱 완료. status 모드로 전환.')
        print_status_only(state, signals)
        return

    # 리밸런싱 실행
    print('\n[리밸런싱]')
    actions = rebalance(state, target, signals, trades)
    state['last_date'] = today

    # 저장
    save_state(state)
    save_trades(trades)
    append_log(today, signals, target, actions)

    # 대시보드
    print_dashboard(state, signals, actions)


if __name__ == '__main__':
    main()
