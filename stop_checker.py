"""
Apex Mode — Stop Checker
역할: 진입/로테이션 없이 스탑 조건만 체크 (4시간마다 실행)

체크 항목:
  1. 하드스탑   — 평균진입가 기준 -8% 이하
  2. 트레일링   — 고점 - ATR×2 이하
  3. 부분 익절  — 평균진입가 기준 +15% 이상 (50% 청산)
  4. HWM 갱신  — 현재가 > 고점이면 고점 + 트레일링 스탑 업데이트
"""

import sys
import ccxt
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from apex_bot import (
    fetch_ohlcv, calc_atr,
    paper_sell_all, paper_sell_partial,
    total_value, append_log,
    HARD_STOP_PCT, TRAIL_ATR_MULT, TRAIL_ATR_MULT_BULL, PARTIAL_PROFIT, ATR_PERIOD,
    STATE_FILE,
)

LOG_FILE     = Path('data/stop_checker_log.jsonl')
COINS        = ['BTC', 'ETH', 'XRP', 'SOL']


def now_kst():
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')


def load_state():
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print('상태 파일이 비어있음.')
                return None
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f'[경고] 상태 파일 JSON 오류: {e}')
        print('상태 파일이 손상되었습니다. apex_bot.py를 실행해 재초기화 필요.')
        return None


def save_state(state):
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(STATE_FILE)


def append_stop_log(record):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_current_price(coin):
    try:
        ex = ccxt.upbit({'enableRateLimit': True})
        ticker = ex.fetch_ticker(f'{coin}/KRW')
        return float(ticker['last'])
    except Exception as e:
        print(f'  {coin} 현재가 오류: {e}')
        return None


def main():
    ts = now_kst()
    print(f'\n[Stop Checker] {ts}')

    state = load_state()
    if state is None:
        print('상태 파일 없음. 종료.')
        return

    positions = state.get('positions', {})
    if not positions:
        print('보유 포지션 없음. 종료.')
        return

    print(f'보유 코인: {list(positions.keys())}')

    actions     = []
    changed     = False
    coin_prices = {}

    for coin in list(positions.keys()):
        pos   = positions[coin]
        price = get_current_price(coin)
        time.sleep(0.3)

        if price is None:
            print(f'  {coin} 가격 조회 실패 — 스킵')
            continue

        coin_prices[coin] = price

        avg_entry = pos['avg_entry']
        hwm       = pos['hwm']
        trail_stop = pos['trailing_stop']
        hard_stop  = pos['hard_stop']
        pct        = (price - avg_entry) / avg_entry * 100

        print(f'\n  {coin}: 현재가={price:,}  평균진입={avg_entry:,}  '
              f'미실현={pct:+.2f}%  트레일={trail_stop:,}  하드={hard_stop:,}')

        # ATR 갱신 (트레일링 스탑 업데이트용)
        df = fetch_ohlcv(coin, limit=30)
        time.sleep(0.3)
        atr = calc_atr(df) if df is not None else None

        # ── 1. 하드스탑 ────────────────────────────────────────────────
        if price <= hard_stop:
            recv = paper_sell_all(state, coin, price)
            msg  = f'[하드스탑] {coin} @ {price:,}원  ({pct:+.2f}%)  회수={recv:,.0f}원'
            print(f'  ★ {msg}')
            actions.append(msg)
            changed = True
            continue

        # ── 2. 트레일링 스탑 ───────────────────────────────────────────
        if price <= trail_stop:
            recv = paper_sell_all(state, coin, price)
            msg  = f'[트레일링 스탑] {coin} @ {price:,}원  ({pct:+.2f}%)  회수={recv:,.0f}원'
            print(f'  ★ {msg}')
            actions.append(msg)
            changed = True
            continue

        # ── 3. 부분 익절 (+15%) ────────────────────────────────────────
        if not pos['partial_exited'] and price >= avg_entry * (1 + PARTIAL_PROFIT):
            recv = paper_sell_partial(state, coin, price)
            msg  = f'[부분 익절 50%] {coin} @ {price:,}원  ({pct:+.2f}%)  회수={recv:,.0f}원'
            print(f'  ★ {msg}')
            actions.append(msg)
            changed = True

        # ── 4. HWM + 트레일링 스탑 갱신 ───────────────────────────────
        if price > hwm:
            pos['hwm'] = price
            print(f'  → HWM 갱신: {hwm:,} → {price:,}')
            changed = True

        if atr is not None:
            atr_mult  = TRAIL_ATR_MULT_BULL if pos.get('bull_macro') else TRAIL_ATR_MULT
            new_trail = round(pos['hwm'] - atr * atr_mult)
            if new_trail > pos['trailing_stop']:
                print(f'  → 트레일링 스탑 갱신: {pos["trailing_stop"]:,} → {new_trail:,}')
                pos['trailing_stop'] = new_trail
                changed = True

        if not actions:
            print(f'  → 이상 없음 (스탑 유지)')

    # 로그 기록 (변경 여부와 무관하게 항상)
    log_positions = {}
    for c, p in state.get('positions', {}).items():
        price_c = coin_prices.get(c)
        log_positions[c] = {
            'avg_entry'    : p['avg_entry'],
            'hwm'          : p['hwm'],
            'trailing_stop': p['trailing_stop'],
            'price'        : price_c,
            'pct'          : round((price_c - p['avg_entry']) / p['avg_entry'] * 100, 2)
                             if price_c else None,
        }

    append_stop_log({
        'ts'       : ts,
        'actions'  : actions,
        'positions': log_positions,
        'krw'      : round(state['krw']),
    })

    # 변경사항 저장
    if changed:
        save_state(state)
        print(f'\n상태 저장 완료.')
    else:
        print(f'\n변경사항 없음.')


if __name__ == '__main__':
    main()
