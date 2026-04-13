import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 데이터프레임에 모든 지표 추가"""
    df = df.copy()

    # RSI (14)
    df['rsi'] = ta.rsi(df['close'], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # 볼린저밴드 (20, 2)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0_2.0']
    df['bb_mid'] = bb['BBM_20_2.0_2.0']
    df['bb_lower'] = bb['BBL_20_2.0_2.0']

    # ADX (14) — 추세 강도
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14']

    # ATR (14) — 변동성
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Supertrend (7, 3)
    st = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
    df['supertrend'] = st['SUPERT_7_3']
    df['supertrend_dir'] = st['SUPERTd_7_3']  # 1=상승, -1=하락

    # EMA (9, 21)
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)

    # VWAP
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

    # 거래량 이동평균
    df['volume_ma'] = ta.sma(df['volume'], length=20)

    # Donchian Channel (20) — 이전 20캔들 고/저점 (현재 캔들 제외)
    df['don_high'] = df['high'].rolling(20).max().shift(1)
    df['don_low']  = df['low'].rolling(20).min().shift(1)

    return df


def get_market_regime(df: pd.DataFrame) -> str:
    """ADX 기반 시장 레짐 감지"""
    adx = df['adx'].iloc[-1]
    if adx > 25:
        return 'trend'
    elif adx < 20:
        return 'range'
    else:
        return 'neutral'
