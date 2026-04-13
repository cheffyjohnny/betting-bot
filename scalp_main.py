"""
BTC/KRW 단타 페이퍼 트레이딩 (실시간)
전략: 200MA + 4H EMA21 이중필터 + 15M 눌림목 진입

실행: python scalp_main.py
종료: Ctrl+C
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd
import pandas_ta as ta

# ── 설정 ─────────────────────────────────────────────────────────────────────
INITIAL_KRW      = 1_000_000
UPBIT_FEE        = 0.0005
TP_ATR           = 2.0
SL_ATR           = 1.0
MAX_CANDLES      = 10        # 최대 보유 캔들 수 (150분)
CROSSOVER_WINDOW = 6         # 크로스오버 후 눌림목 유효 캔들 수
MIN_ATR_PCT      = 0.002     # 최소 변동성 (가격의 0.2%)

STATE_FILE  = Path('data/scalp_state.json')
TRADES_FILE = Path('data/scalp_trades.json')

exchange = ccxt.upbit()

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
state_lock  = threading.Lock()
display_lock = threading.Lock()

live = {
    'price':       0,
    'price_change': 0.0,
    'last_signal': 'HOLD',
    'signal_reason': '초기화 중...',
    'filter_200ma':  (False, 0),    # (pass, ma값)
    'filter_4h':     (False, ''),
    'crossover':     (False, -1),   # (최근크로스오버여부, 몇캔들전)
    'last_check':  '없음',
    'status':      '초기화 중...',
    'entry_count': 0,
}

# ── 잔고/거래 관리 ─────────────────────────────────────────────────────────────

def load_state():
    STATE_FILE.parent.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    return {'krw': INITIAL_KRW, 'btc': 0.0, 'initial_krw': INITIAL_KRW,
            'position': None, 'candle_entry_time': None}

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')

def load_trades():
    if TRADES_FILE.exists():
        return json.loads(TRADES_FILE.read_text(encoding='utf-8'))
    return []

def save_trade(t):
    trades = load_trades()
    trades.append(t)
    TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False, indent=2), encoding='utf-8')

state = load_state()

# ── 데이터 / 지표 ──────────────────────────────────────────────────────────────

def fetch(timeframe, limit):
    ohlcv = exchange.fetch_ohlcv('BTC/KRW', timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def add_indicators(df):
    df = df.copy()
    df['ema9']      = ta.ema(df['close'], length=9)
    df['ema21']     = ta.ema(df['close'], length=21)
    df['rsi']       = ta.rsi(df['close'], length=14)
    df['atr']       = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volume_ma'] = ta.sma(df['volume'], length=20)
    df['vwap']      = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    return df.dropna()

# ── 매수 / 매도 ────────────────────────────────────────────────────────────────

def do_buy(price, atr, reason, candle_idx):
    global state
    if state['position'] or state['krw'] < 10000:
        return
    btc = (state['krw'] - state['krw'] * UPBIT_FEE) / price
    state['krw'] = 0.0
    state['btc'] = btc
    state['position'] = {
        'price': price, 'btc': btc,
        'tp': price + atr * TP_ATR,
        'sl': price - atr * SL_ATR,
        'atr': atr,
        'candle_entry': candle_idx,
        'time': datetime.now().isoformat(),
    }
    save_state(state)
    save_trade({'type':'BUY','time':datetime.now().isoformat(),
                'price':price,'btc':btc,'reason':reason,
                'tp':price+atr*TP_ATR,'sl':price-atr*SL_ATR})

def do_sell(price, reason):
    global state
    if not state['position']:
        return
    btc         = state['btc']
    received    = btc * price * (1 - UPBIT_FEE)
    entry_price = state['position']['price']
    pnl         = received - btc * entry_price
    pnl_pct     = (price - entry_price) / entry_price * 100

    state['krw']      = received
    state['btc']      = 0.0
    state['position'] = None
    state['candle_entry_time'] = None
    save_state(state)
    save_trade({'type':'SELL','time':datetime.now().isoformat(),
                'price':price,'pnl':pnl,'pnl_pct':pnl_pct,'reason':reason})
    return pnl_pct

# ── 디스플레이 ─────────────────────────────────────────────────────────────────

def clr():
    os.system('cls' if os.name == 'nt' else 'clear')

def pct_bar(pct, width=20):
    filled = int(abs(pct) / 5 * width)  # 5%당 1칸
    filled = min(filled, width)
    bar    = ('█' * filled).ljust(width)
    return f'[{bar}]' if pct >= 0 else f'[{bar}]'

def render():
    with display_lock:
        clr()
        price  = live['price']
        trades = load_trades()
        sells  = [t for t in trades if t['type'] == 'SELL']
        total  = state['krw'] + state['btc'] * price
        ret    = (total - INITIAL_KRW) / INITIAL_KRW * 100
        wins   = [t for t in sells if t['pnl'] > 0]
        wr     = len(wins) / len(sells) * 100 if sells else 0

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('=' * 60)
        print(f'  BTC/KRW 단타 페이퍼 트레이딩              {now}')
        print('=' * 60)

        # 현재가
        arrow = '+' if live['price_change'] >= 0 else ''
        print(f'\n  현재가   {price:>15,.0f} KRW  ({arrow}{live["price_change"]:.2f}% 15분)')

        # 잔고
        print(f'\n  {"─"*56}')
        print(f'  [잔고]')
        print(f'    초기 자본:  {INITIAL_KRW:>12,.0f} 원')
        print(f'    현재 자산:  {total:>12,.0f} 원  ({ret:>+.2f}%)')
        print(f'    KRW:        {state["krw"]:>12,.0f} 원')
        if state['btc'] > 0:
            print(f'    BTC:        {state["btc"]:>16.8f}')

        # 포지션
        print(f'\n  {"─"*56}')
        print(f'  [포지션]')
        if state['position']:
            pos     = state['position']
            unrl    = (price - pos['price']) / pos['price'] * 100
            dist_tp = (pos['tp'] - price) / pos['atr']
            dist_sl = (price - pos['sl']) / pos['atr']
            arrow2  = '+' if unrl >= 0 else ''
            print(f'    진입가:  {pos["price"]:>13,.0f} 원   ({pos["time"][11:16]} 진입)')
            print(f'    현재:    {price:>13,.0f} 원   미실현 {arrow2}{unrl:.2f}%')
            print(f'    익절(TP):{pos["tp"]:>13,.0f} 원   남은거리 {dist_tp:.1f}xATR')
            print(f'    손절(SL):{pos["sl"]:>13,.0f} 원   남은거리 {dist_sl:.1f}xATR')
        else:
            print(f'    없음 — 진입 대기 중')

        # 필터 상태
        print(f'\n  {"─"*56}')
        print(f'  [필터 상태]  (마지막 확인: {live["last_check"]})')
        ok200, ma200 = live['filter_200ma']
        print(f'    200일 MA:  {ma200:>13,.0f}  {"✓ OK" if ok200 else "✗ 현재가 아래"}')
        ok4h, slope = live['filter_4h']
        print(f'    4H EMA21:  {slope:<20}  {"✓ OK" if ok4h else "✗ 하락 중"}')
        cross, n = live['crossover']
        if cross:
            print(f'    크로스오버: {n}캔들 전 발생  ✓ 눌림목 대기 가능')
        else:
            print(f'    크로스오버: 없음 (마지막 {n}캔들 전)')

        # 신호
        print(f'\n  {"─"*56}')
        sig = live['last_signal']
        sig_display = f'  ▶ {sig}' if sig == 'BUY' else f'    {sig}'
        print(f'  [신호]  {sig_display}')
        print(f'    {live["signal_reason"]}')

        # 거래 통계
        print(f'\n  {"─"*56}')
        print(f'  [통계]  총 {len(sells)}건  승률 {wr:.1f}%')
        if sells:
            recent = sells[-5:]
            for t in reversed(recent):
                icon = '▲' if t['pnl'] > 0 else '▼'
                print(f'    {icon} {t["time"][5:16]}  {t["pnl_pct"]:>+6.2f}%  {t["reason"]}')

        print(f'\n  {live["status"]}')
        print('=' * 60)
        print('  Ctrl+C 로 종료')

# ── 전략 체크 스레드 (15분마다) ────────────────────────────────────────────────

candle_counter = {'last_cross': -999, 'idx': 0}

def strategy_loop():
    global state
    while True:
        try:
            with state_lock:
                live['status'] = '데이터 수집 중...'
            render()

            df_15m   = add_indicators(fetch('15m', 200))
            df_4h    = add_indicators(fetch('4h',  50))
            df_1d    = fetch('1d', 210)

            price    = df_15m['close'].iloc[-1]
            cur      = df_15m.iloc[-1]
            prev     = df_15m.iloc[-2]
            cur_4h   = df_4h.iloc[-1]

            # 200일 MA
            ma200    = df_1d['close'].rolling(200).mean().iloc[-1]
            ok_200   = price > ma200 and not pd.isna(ma200)

            # 4H EMA21 기울기
            ema21_4h = ta.ema(df_4h['close'], length=21)
            slope_ok = bool(ema21_4h.iloc[-1] > ema21_4h.iloc[-3])
            slope_str = f'상승 ({ema21_4h.iloc[-1]:,.0f}원)' if slope_ok else f'하락 ({ema21_4h.iloc[-1]:,.0f}원)'

            # 크로스오버 감지
            candle_counter['idx'] += 1
            if prev['ema9'] <= prev['ema21'] and cur['ema9'] > cur['ema21']:
                candle_counter['last_cross'] = candle_counter['idx']

            candles_since = candle_counter['idx'] - candle_counter['last_cross']
            cross_valid   = 1 <= candles_since <= CROSSOVER_WINDOW

            with state_lock:
                live['price']       = price
                live['filter_200ma'] = (ok_200, int(ma200) if not pd.isna(ma200) else 0)
                live['filter_4h']    = (slope_ok, slope_str)
                live['crossover']    = (cross_valid, candles_since)
                live['last_check']   = datetime.now().strftime('%H:%M:%S')

            # ── 포지션 보유 중: 캔들 타임아웃 체크 ──────────────────────────────
            if state['position']:
                pos_time    = datetime.fromisoformat(state['position']['time'])
                held_mins   = (datetime.now() - pos_time).seconds // 60
                held_candles = held_mins // 15
                if held_candles >= MAX_CANDLES:
                    pnl_pct = do_sell(price, f'시간초과 ({held_mins}분 보유)')
                    with state_lock:
                        live['last_signal']   = 'SELL (시간초과)'
                        live['signal_reason'] = f'{held_mins}분 보유 후 강제 청산 ({pnl_pct:+.2f}%)'
                    render()
                    time.sleep(1)

            # ── 신규 진입 조건 ────────────────────────────────────────────────
            if not state['position']:
                atr_ok      = cur['atr'] > price * MIN_ATR_PCT
                pullback    = price <= cur['ema9'] * 1.001
                rsi_ok      = 40 <= cur['rsi'] <= 58
                volume_ok   = cur['volume'] > cur['volume_ma'] * 1.2

                all_pass    = ok_200 and slope_ok and cross_valid and pullback and rsi_ok and volume_ok and atr_ok

                if all_pass:
                    reason = (f'200MA({int(ma200):,}) 위 + 4H 상승 + 크로스{candles_since}캔들후 눌림목'
                              f' | RSI {cur["rsi"]:.1f} | 거래량 {cur["volume"]/cur["volume_ma"]:.1f}x')
                    with state_lock:
                        do_buy(price, cur['atr'], reason, candle_counter['idx'])
                        live['last_signal']   = 'BUY'
                        live['signal_reason'] = reason
                else:
                    reasons = []
                    if not ok_200:    reasons.append(f'200MA({int(ma200):,}) 아래')
                    if not slope_ok:  reasons.append('4H 하락')
                    if not cross_valid: reasons.append(f'크로스오버 없음 ({candles_since}캔들 전)')
                    if not pullback:  reasons.append(f'눌림목 아님 (EMA9 대비 +{(price/cur["ema9"]-1)*100:.2f}%)')
                    if not rsi_ok:    reasons.append(f'RSI {cur["rsi"]:.1f} 범위 외')
                    if not volume_ok: reasons.append(f'거래량 {cur["volume"]/cur["volume_ma"]:.1f}x 부족')
                    if not atr_ok:    reasons.append('변동성 부족')
                    with state_lock:
                        live['last_signal']   = 'HOLD'
                        live['signal_reason'] = ' / '.join(reasons[:3])

            with state_lock:
                live['status'] = '다음 체크: 15분 후'

            render()

        except Exception as e:
            with state_lock:
                live['status'] = f'[오류] {e}'
            render()

        time.sleep(60 * 15)


# ── 가격 모니터 스레드 (10초마다) ─────────────────────────────────────────────

def price_loop():
    global state
    prev_price = None
    while True:
        try:
            price = exchange.fetch_ticker('BTC/KRW')['last']

            with state_lock:
                live['price'] = price
                if prev_price:
                    live['price_change'] = (price - prev_price) / prev_price * 100

            # TP / SL 체크
            if state['position']:
                pos    = state['position']
                reason = None
                if price >= pos['tp']:
                    reason = f'TP 달성 (+{TP_ATR}x ATR)'
                elif price <= pos['sl']:
                    reason = f'SL 손절 (-{SL_ATR}x ATR)'

                if reason:
                    with state_lock:
                        pnl_pct = do_sell(price, reason)
                        live['last_signal']   = f'SELL ({reason})'
                        live['signal_reason'] = f'{price:,.0f}원에 청산 ({pnl_pct:+.2f}%)'

            render()
            prev_price = price

        except Exception as e:
            with state_lock:
                live['status'] = f'[가격조회 오류] {e}'

        time.sleep(10)


# ── 메인 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)

    print('단타 페이퍼 트레이딩 시작 중...')

    # 전략 스레드 먼저
    t1 = threading.Thread(target=strategy_loop, daemon=True)
    t1.start()
    time.sleep(3)

    # 가격 모니터
    t2 = threading.Thread(target=price_loop, daemon=True)
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        clr()
        trades = load_trades()
        sells  = [t for t in trades if t['type'] == 'SELL']
        price  = live['price']
        total  = state['krw'] + state['btc'] * price
        ret    = (total - INITIAL_KRW) / INITIAL_KRW * 100
        print('\n=== 종료 ===')
        print(f'최종 자산: {total:,.0f}원  ({ret:+.2f}%)')
        print(f'총 거래:   {len(sells)}건')
        print('상태 저장됨:', STATE_FILE)
