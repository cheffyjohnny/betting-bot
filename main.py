"""
업비트 BTC/KRW 페이퍼 트레이딩 봇
전략 A (멀티 타임프레임): Daily 200MA + 4H Donchian 돌파 + 15M 정밀 진입
"""

import time
import threading
import traceback
from datetime import datetime

import ccxt
import pandas as pd

from strategy import get_signal
from paper_trader import (
    load_state, buy, sell, update_trailing_stop,
    check_trailing_stop, get_summary
)

INITIAL_KRW     = 1_000_000
PRICE_INTERVAL  = 10       # 가격 체크 (초)
SIGNAL_INTERVAL = 60 * 15  # 전략 계산 (초) — 15분마다 3단계 필터 체크

exchange   = ccxt.upbit()
state      = load_state(INITIAL_KRW)
state_lock = threading.Lock()


def fetch_ohlcv(timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv('BTC/KRW', timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')


def log(msg: str):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}')


# ── 가격 모니터 (10초마다) ──────────────────────────────────────────────────────
def price_monitor():
    log('가격 모니터 시작 (10초 간격)')
    while True:
        try:
            price = exchange.fetch_ticker('BTC/KRW')['last']
            with state_lock:
                if check_trailing_stop(state, price):
                    trade = sell(state, price, reason='트레일링 스탑')
                    if trade:
                        log(f'[매도] 트레일링 스탑 | {price:,.0f} KRW | '
                            f'수익률: {trade["pnl_pct"]:.2f}% | PnL: {trade["pnl"]:,.0f} KRW')
                else:
                    if state['position']:
                        entry = state['position']['price']
                        stop  = state['position']['trailing_stop']
                        unrl  = (price - entry) / entry * 100
                        print(f'\r  {price:,.0f} KRW | 미실현: {unrl:+.2f}% | 트레일링스탑: {stop:,.0f}  ',
                              end='', flush=True)
                    else:
                        print(f'\r  {price:,.0f} KRW | 대기 중...  ', end='', flush=True)

        except Exception as e:
            log(f'[가격모니터 오류] {e}')

        time.sleep(PRICE_INTERVAL)


# ── 전략 신호 모니터 (15분마다) ─────────────────────────────────────────────────
def signal_monitor():
    log('전략 모니터 시작 (15M 기준, 3단계 MTF 필터)')
    while True:
        try:
            print()
            log('--- 전략 분석 중 ---')
            df_15m   = fetch_ohlcv('15m', 200)   # 15분봉 200개
            df_4h    = fetch_ohlcv('4h',  100)   # 4시간봉 100개 (Donchian + ATR)
            df_daily = fetch_ohlcv('1d',  210)   # 일봉 210개 (200일 MA)
            result   = get_signal(df_15m, df_4h, df_daily)
            price    = result['price']

            log(f'레짐: {result["regime"]} | 200MA: {result["ma200"]:,.0f} '
                f'| 4H 신고가: {result["don_high"]:,.0f} | RSI: {result["rsi"]:.1f}'
                f'| 신호: {result["signal"]}')
            log(f'  {result["reason"]}')

            with state_lock:
                # 트레일링 스탑 업데이트 (4H ATR 기준)
                if state['position']:
                    update_trailing_stop(state, price, result['atr'])

                if result['signal'] == 'buy':
                    trade = buy(state, price, result['atr'], result['reason'])
                    if trade:
                        log(f'[매수] {price:,.0f} KRW | 트레일링스탑: {trade["trailing_stop"]:,.0f}')
                    else:
                        log('매수 신호 — 이미 포지션 보유 중')

                elif result['regime'] == 'bear' and state['position']:
                    trade = sell(state, price, '200일MA 하향 돌파 — 보유 포지션 청산')
                    if trade:
                        log(f'[매도] 매크로 필터 청산 | {price:,.0f} KRW | '
                            f'수익률: {trade["pnl_pct"]:.2f}% | PnL: {trade["pnl"]:,.0f} KRW')

                summary = get_summary(state, price)

            log(f'총 자산: {summary["total_value"]:,.0f} KRW | '
                f'수익률: {summary["pnl_pct"]:.2f}% | '
                f'거래: {summary["total_trades"]}회 | '
                f'승률: {summary["win_rate"]:.1f}%')

        except Exception as e:
            log(f'[전략모니터 오류] {e}')
            traceback.print_exc()

        time.sleep(SIGNAL_INTERVAL)


# ── 메인 ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log('=== 페이퍼 트레이딩 봇 시작 ===')
    log('전략: Daily 200MA + 4H Donchian 돌파 + 15M 정밀 진입 + 4H ATR 트레일링 스탑')
    log(f'초기 잔고: {INITIAL_KRW:,.0f} KRW')

    t_signal = threading.Thread(target=signal_monitor, daemon=True)
    t_signal.start()

    t_price = threading.Thread(target=price_monitor, daemon=True)
    t_price.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        log('봇 종료')
