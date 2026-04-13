"""
BTC/USDT 양방향 단타 페이퍼 트레이딩 (바이낸스 선물 시뮬레이션)
--------------------------------------------------------------
전략: 200MA 기준으로 상승장/하락장 판단 후 방향 결정

  상승장 (200MA 위) → 롱만
    진입: 4H 상승 + 15M EMA 골든크로스 후 눌림목
    청산: ATR x 2.0 (TP) / ATR x 1.0 (SL)

  하락장 (200MA 아래) → 숏만
    진입: 4H 하락 + 15M EMA 데드크로스 후 되돌림
    청산: ATR x 2.0 (TP) / ATR x 1.0 (SL)

공통: 최대 보유 10캔들(150분) 강제 청산
실행: python -X utf8 scalp_v2_main.py
종료: Ctrl+C
"""

import os, sys, json, time, threading
from datetime import datetime
from pathlib import Path

import ccxt
import pandas as pd
import pandas_ta as ta

# ── 설정 ──────────────────────────────────────────────────────────────────────
INITIAL_USDT     = 1_000.0      # 초기 가상 잔고 (USDT)
UPBIT_FEE        = 0.0004       # 바이낸스 선물 수수료 (Maker 0.02% x2 = 0.04%)
TP_ATR           = 2.0
SL_ATR           = 1.0
MAX_CANDLES      = 10
CROSSOVER_WINDOW = 6
MIN_ATR_PCT      = 0.002        # 최소 변동성 0.2%

STATE_FILE  = Path('data/scalp_v2_state.json')
TRADES_FILE = Path('data/scalp_v2_trades.json')

exchange = ccxt.binance()

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
lock = threading.Lock()
live = {
    'price': 0.0, 'price_change': 0.0,
    'market': '확인 중',           # '상승장(롱)' or '하락장(숏)'
    'signal': 'HOLD', 'reason': '초기화 중...',
    'filter_200ma': (False, 0.0),
    'filter_4h': (False, ''),
    'crossover': (False, 999),
    'last_check': '없음',
    'status': '초기화 중...',
}

# ── 잔고 관리 ─────────────────────────────────────────────────────────────────

def load_state():
    STATE_FILE.parent.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    return {'usdt': INITIAL_USDT, 'initial_usdt': INITIAL_USDT, 'position': None}

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

# ── 진입 / 청산 ───────────────────────────────────────────────────────────────

def do_enter(price, atr, direction, reason):
    """direction: 'LONG' or 'SHORT'"""
    if state['position'] or state['usdt'] < 10:
        return
    size = (state['usdt'] * (1 - UPBIT_FEE)) / price   # BTC 수량

    if direction == 'LONG':
        tp = price + atr * TP_ATR
        sl = price - atr * SL_ATR
    else:  # SHORT
        tp = price - atr * TP_ATR
        sl = price + atr * SL_ATR

    state['position'] = {
        'direction': direction,
        'price': price, 'size': size,
        'tp': tp, 'sl': sl, 'atr': atr,
        'time': datetime.now().isoformat(),
        'invest_usdt': state['usdt'],
    }
    state['usdt'] = 0.0
    save_state(state)
    save_trade({'type': f'ENTER_{direction}', 'time': datetime.now().isoformat(),
                'price': price, 'size': size, 'tp': tp, 'sl': sl, 'reason': reason})

def do_exit(price, reason):
    if not state['position']:
        return None
    pos    = state['position']
    size   = pos['size']
    invest = pos['invest_usdt']

    if pos['direction'] == 'LONG':
        received = size * price * (1 - UPBIT_FEE)
        pnl      = received - invest
    else:  # SHORT: 가격이 내려갈수록 수익
        received = invest + (pos['price'] - price) * size * (1 - UPBIT_FEE)
        pnl      = received - invest

    pnl_pct = pnl / invest * 100

    state['usdt']     = received
    state['position'] = None
    save_state(state)
    save_trade({'type': 'EXIT', 'time': datetime.now().isoformat(),
                'price': price, 'pnl': pnl, 'pnl_pct': pnl_pct,
                'direction': pos['direction'], 'reason': reason})
    return pnl_pct

# ── 데이터 / 지표 ─────────────────────────────────────────────────────────────

def fetch(tf, limit):
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('ts')

def add_ind(df):
    df = df.copy()
    df['ema9']   = ta.ema(df['close'], length=9)
    df['ema21']  = ta.ema(df['close'], length=21)
    df['rsi']    = ta.rsi(df['close'], length=14)
    df['atr']    = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['vol_ma'] = ta.sma(df['volume'], length=20)
    return df.dropna()

# ── 디스플레이 ────────────────────────────────────────────────────────────────

def render():
    os.system('cls' if os.name == 'nt' else 'clear')
    price  = live['price']
    trades = load_trades()
    exits  = [t for t in trades if t['type'] == 'EXIT']
    total  = state['usdt'] if not state['position'] else state['position']['invest_usdt']
    if state['position']:
        pos = state['position']
        if pos['direction'] == 'LONG':
            total = pos['size'] * price * (1 - UPBIT_FEE)
        else:
            total = pos['invest_usdt'] + (pos['price'] - price) * pos['size']
    ret    = (total - INITIAL_USDT) / INITIAL_USDT * 100
    wins   = [t for t in exits if t['pnl'] > 0]
    wr     = len(wins) / len(exits) * 100 if exits else 0.0

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mkt = live['market']
    mkt_color = '↑ 상승장 (롱 전략)' if '상승' in mkt else '↓ 하락장 (숏 전략)'

    print('=' * 62)
    print(f'  BTC/USDT 양방향 단타 페이퍼 트레이딩       {now}')
    print('=' * 62)

    ch = live['price_change']
    print(f'\n  현재가  ${price:>12,.2f}   ({ch:>+.2f}% 15분)')
    print(f'  시장    {mkt_color}')

    # 잔고
    print(f'\n  {"─"*58}')
    print(f'  [잔고]')
    print(f'    초기:    ${INITIAL_USDT:>10,.2f}')
    print(f'    현재:    ${total:>10,.2f}   ({ret:>+.2f}%)')
    if state['usdt'] > 0:
        print(f'    USDT:    ${state["usdt"]:>10,.2f}')

    # 포지션
    print(f'\n  {"─"*58}')
    print(f'  [포지션]')
    if state['position']:
        pos  = state['position']
        d    = pos['direction']
        if d == 'LONG':
            unrl = (price - pos['price']) / pos['price'] * 100
        else:
            unrl = (pos['price'] - price) / pos['price'] * 100
        icon = '▲ LONG ' if d == 'LONG' else '▼ SHORT'
        print(f'    {icon}  진입: ${pos["price"]:,.2f}   ({pos["time"][11:16]})')
        print(f'           현재: ${price:,.2f}   미실현 {unrl:>+.2f}%')
        print(f'    TP: ${pos["tp"]:>10,.2f}   SL: ${pos["sl"]:>10,.2f}')
    else:
        print(f'    없음 — 신호 대기 중')

    # 필터 상태
    print(f'\n  {"─"*58}')
    print(f'  [필터]  마지막 확인: {live["last_check"]}')
    ok200, ma200 = live['filter_200ma']
    ok4h,  slope = live['filter_4h']
    cross, n     = live['crossover']
    status200 = '✓' if ok200 else '✗'
    status4h  = '✓' if ok4h  else '✗'
    cross_str = f'{n}캔들 전  ✓' if cross else f'{n}캔들 전  ✗'
    print(f'    200일MA  ${ma200:>10,.2f}  {status200}  {"현재가 위 (롱)" if ok200 else "현재가 아래 (숏)"}')
    print(f'    4H EMA21  {slope:<28}  {status4h}')
    print(f'    크로스오버  {cross_str}')

    # 신호
    print(f'\n  {"─"*58}')
    sig = live['signal']
    tag = f'▶ {sig}' if sig in ('LONG', 'SHORT') else f'  {sig}'
    print(f'  [신호]  {tag}')
    print(f'    {live["reason"]}')

    # 거래 기록
    print(f'\n  {"─"*58}')
    print(f'  [거래]  총 {len(exits)}건   승률 {wr:.1f}%')
    if exits:
        for t in reversed(exits[-5:]):
            icon = '▲' if t['pnl'] > 0 else '▼'
            dir_str = t.get('direction', '')
            print(f'    {icon} {t["time"][5:16]}  {dir_str:<5}  {t["pnl_pct"]:>+6.2f}%  {t["reason"]}')

    print(f'\n  {live["status"]}')
    print('=' * 62)
    print('  Ctrl+C 로 종료')

# ── 전략 스레드 ───────────────────────────────────────────────────────────────

cross_tracker = {'last_long': -999, 'last_short': -999, 'idx': 0}

def strategy_loop():
    while True:
        try:
            with lock:
                live['status'] = '데이터 수집 중...'

            df_15m = add_ind(fetch('15m', 200))
            df_4h  = add_ind(fetch('4h',  50))
            df_1d  = fetch('1d', 210)

            price  = df_15m['close'].iloc[-1]
            cur    = df_15m.iloc[-1]
            prev   = df_15m.iloc[-2]

            # 200일 MA
            ma200   = df_1d['close'].rolling(200).mean().iloc[-1]
            above   = price > ma200 and not pd.isna(ma200)  # True = 상승장(롱), False = 하락장(숏)

            # 4H EMA21 기울기
            ema21_4h = ta.ema(df_4h['close'], length=21)
            if above:
                trend_ok = bool(ema21_4h.iloc[-1] > ema21_4h.iloc[-3])   # 롱: 4H 상승
                slope_str = f'상승 ({ema21_4h.iloc[-1]:,.2f})'
            else:
                trend_ok = bool(ema21_4h.iloc[-1] < ema21_4h.iloc[-3])   # 숏: 4H 하락
                slope_str = f'하락 ({ema21_4h.iloc[-1]:,.2f})'

            # 크로스오버 감지
            cross_tracker['idx'] += 1
            idx = cross_tracker['idx']
            if prev['ema9'] <= prev['ema21'] and cur['ema9'] > cur['ema21']:
                cross_tracker['last_long'] = idx
            if prev['ema9'] >= prev['ema21'] and cur['ema9'] < cur['ema21']:
                cross_tracker['last_short'] = idx

            if above:
                since = idx - cross_tracker['last_long']
            else:
                since = idx - cross_tracker['last_short']
            cross_valid = 1 <= since <= CROSSOVER_WINDOW

            with lock:
                live['price']        = price
                live['market']       = '상승장' if above else '하락장'
                live['filter_200ma'] = (above, float(ma200) if not pd.isna(ma200) else 0.0)
                live['filter_4h']    = (trend_ok, slope_str)
                live['crossover']    = (cross_valid, since)
                live['last_check']   = datetime.now().strftime('%H:%M:%S')

            # ── 포지션 타임아웃 체크 ──────────────────────────────────────────
            if state['position']:
                held_min = (datetime.now() - datetime.fromisoformat(state['position']['time'])).seconds // 60
                if held_min // 15 >= MAX_CANDLES:
                    pct = do_exit(price, f'시간초과 ({held_min}분)')
                    with lock:
                        live['signal'] = f'EXIT (시간초과)'
                        live['reason'] = f'{held_min}분 보유 후 강제 청산  {pct:+.2f}%'
                    render(); time.sleep(1)

            # ── 공통 조건 ─────────────────────────────────────────────────────
            atr_ok     = cur['atr'] > price * MIN_ATR_PCT
            volume_ok  = cur['volume'] > cur['vol_ma'] * 1.2
            all_base   = trend_ok and cross_valid and atr_ok and volume_ok

            # ── 롱 진입 ───────────────────────────────────────────────────────
            if above and not state['position']:
                pullback = price <= cur['ema9'] * 1.001
                rsi_ok   = 40 <= cur['rsi'] <= 58
                if all_base and pullback and rsi_ok:
                    reason = (f'상승장 롱 | 4H 상승 + {since}캔들후 눌림목'
                              f' RSI {cur["rsi"]:.1f} | 거래량 {cur["volume"]/cur["vol_ma"]:.1f}x')
                    with lock:
                        do_enter(price, cur['atr'], 'LONG', reason)
                        live['signal'] = 'LONG'
                        live['reason'] = reason
                else:
                    reasons = _fail_reasons(above, trend_ok, cross_valid, since,
                                            price, cur, atr_ok, volume_ok)
                    with lock:
                        live['signal'] = 'HOLD'
                        live['reason'] = reasons

            # ── 숏 진입 ───────────────────────────────────────────────────────
            elif not above and not state['position']:
                rebound  = price >= cur['ema9'] * 0.999   # 되돌림: EMA9 근처까지 반등
                rsi_ok   = 42 <= cur['rsi'] <= 60         # 중립~약과매수 구간에서 숏
                if all_base and rebound and rsi_ok:
                    reason = (f'하락장 숏 | 4H 하락 + {since}캔들후 되돌림'
                              f' RSI {cur["rsi"]:.1f} | 거래량 {cur["volume"]/cur["vol_ma"]:.1f}x')
                    with lock:
                        do_enter(price, cur['atr'], 'SHORT', reason)
                        live['signal'] = 'SHORT'
                        live['reason'] = reason
                else:
                    reasons = _fail_reasons(above, trend_ok, cross_valid, since,
                                            price, cur, atr_ok, volume_ok)
                    with lock:
                        live['signal'] = 'HOLD'
                        live['reason'] = reasons

            with lock:
                live['status'] = '다음 체크: 15분 후'
            render()

        except Exception as e:
            with lock:
                live['status'] = f'[오류] {e}'
            render()

        time.sleep(60 * 15)


def _fail_reasons(above, trend_ok, cross_valid, since, price, cur, atr_ok, volume_ok):
    r = []
    if not trend_ok:
        r.append(f'4H {"하락" if above else "상승"} (방향 불일치)')
    if not cross_valid:
        r.append(f'크로스오버 없음 ({since}캔들 전)')
    if above:
        if not (price <= cur['ema9'] * 1.001):
            r.append(f'눌림목 없음 (EMA9 대비 +{(price/cur["ema9"]-1)*100:.2f}%)')
        if not (40 <= cur['rsi'] <= 58):
            r.append(f'RSI {cur["rsi"]:.1f} (40~58 범위 외)')
    else:
        if not (price >= cur['ema9'] * 0.999):
            r.append(f'되돌림 없음 (EMA9 대비 {(price/cur["ema9"]-1)*100:.2f}%)')
        if not (42 <= cur['rsi'] <= 60):
            r.append(f'RSI {cur["rsi"]:.1f} (42~60 범위 외)')
    if not volume_ok:
        r.append(f'거래량 {cur["volume"]/cur["vol_ma"]:.1f}x 부족')
    if not atr_ok:
        r.append('변동성 부족')
    return ' / '.join(r[:3]) if r else 'HOLD'


# ── 가격 모니터 스레드 ────────────────────────────────────────────────────────

def price_loop():
    prev = None
    while True:
        try:
            price = exchange.fetch_ticker('BTC/USDT')['last']
            with lock:
                live['price'] = price
                if prev:
                    live['price_change'] = (price - prev) / prev * 100

            if state['position']:
                pos    = state['position']
                reason = None
                if pos['direction'] == 'LONG':
                    if price >= pos['tp']: reason = f'TP +{TP_ATR}xATR'
                    elif price <= pos['sl']: reason = f'SL -{SL_ATR}xATR'
                else:  # SHORT
                    if price <= pos['tp']: reason = f'TP +{TP_ATR}xATR'
                    elif price >= pos['sl']: reason = f'SL -{SL_ATR}xATR'

                if reason:
                    with lock:
                        pct = do_exit(price, reason)
                        live['signal'] = f'EXIT ({reason})'
                        live['reason'] = f'${price:,.2f}에 청산  {pct:+.2f}%'

            render()
            prev = price

        except Exception as e:
            with lock:
                live['status'] = f'[가격오류] {e}'

        time.sleep(10)


# ── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    print('양방향 단타 페이퍼 트레이딩 시작 중...')

    t1 = threading.Thread(target=strategy_loop, daemon=True)
    t1.start()
    time.sleep(4)

    t2 = threading.Thread(target=price_loop, daemon=True)
    t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trades = load_trades()
        exits  = [t for t in trades if t['type'] == 'EXIT']
        price  = live['price']
        total  = state['usdt']
        ret    = (total - INITIAL_USDT) / INITIAL_USDT * 100
        print(f'\n=== 종료 ===')
        print(f'최종 자산: ${total:,.2f}  ({ret:+.2f}%)')
        print(f'총 거래:   {len(exits)}건')
        print(f'상태 저장: {STATE_FILE}')
