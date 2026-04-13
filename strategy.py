"""
전략 A (멀티 타임프레임):
  Daily  → 200MA 매크로 필터 (대세 방향)
  4H     → Donchian 20캔들 돌파 감지 (큰 흐름 진입 타이밍)
  15M    → RSI + EMA21 정밀 진입 (노이즈 제거)

청산: 4H ATR × 2 트레일링 스탑 (고정 익절 없음)
"""

import pandas as pd
from indicators import add_indicators


def get_signal(df_15m: pd.DataFrame, df_4h: pd.DataFrame, df_daily: pd.DataFrame) -> dict:
    df_15m = add_indicators(df_15m)
    df_4h  = add_indicators(df_4h)

    cur_15m = df_15m.iloc[-1]
    cur_4h  = df_4h.iloc[-1]
    price   = cur_15m['close']

    # ── 1단계: 매크로 필터 (Daily 200MA) ────────────────────────────────────────
    ma200    = df_daily['close'].rolling(200).mean().iloc[-1]
    macro_ok = price > ma200

    base = {
        'price':    price,
        'rsi':      cur_15m['rsi'],
        'adx':      cur_4h['adx'],
        'atr':      cur_4h['atr'],       # 트레일링 스탑은 4H ATR 기준
        'ma200':    ma200,
        'don_high': cur_4h['don_high'],
    }

    if not macro_ok:
        return {**base,
                'signal': 'hold',
                'regime': 'bear',
                'reason': f'200일MA({ma200:,.0f}) 아래 — 매매 금지'}

    # ── 2단계: 4H Donchian 돌파 확인 ────────────────────────────────────────────
    breakout_4h  = price > cur_4h['don_high']
    volume_ok_4h = cur_4h['volume'] > cur_4h['volume_ma'] * 1.5
    rsi_ok_4h    = cur_4h['rsi'] < 75

    if not breakout_4h:
        return {**base, 'signal': 'hold', 'regime': 'bull',
                'reason': f'대기 — 4H 신고가({cur_4h["don_high"]:,.0f}) 미돌파'}
    if not volume_ok_4h:
        return {**base, 'signal': 'hold', 'regime': 'bull',
                'reason': f'대기 — 4H 거래량 부족({cur_4h["volume"] / cur_4h["volume_ma"]:.2f}x)'}
    if not rsi_ok_4h:
        return {**base, 'signal': 'hold', 'regime': 'bull',
                'reason': f'대기 — 4H RSI 과열({cur_4h["rsi"]:.1f})'}

    # ── 3단계: 15M 정밀 진입 타이밍 ─────────────────────────────────────────────
    # 4H 돌파 확인됨 → 15M에서 RSI 적정 + EMA21 위에서 진입 (추세 내 눌림 후 회복)
    rsi_ok_15m  = cur_15m['rsi'] < 65
    ema_ok_15m  = price > cur_15m['ema21']

    if rsi_ok_15m and ema_ok_15m:
        reason = (f'진입 | 4H 신고가 {cur_4h["don_high"]:,.0f} 돌파'
                  f' + 4H 거래량 {cur_4h["volume"] / cur_4h["volume_ma"]:.1f}x'
                  f' | 15M RSI {cur_15m["rsi"]:.1f} / EMA21 위')
        return {**base, 'signal': 'buy', 'regime': 'bull', 'reason': reason}

    reason = (f'대기 — 4H 돌파 확인, 15M 진입 대기'
              f' (RSI {cur_15m["rsi"]:.1f}{"<65 OK" if rsi_ok_15m else " 과열"}'
              f', EMA21 {"위 OK" if ema_ok_15m else "아래"})')
    return {**base, 'signal': 'hold', 'regime': 'bull', 'reason': reason}
