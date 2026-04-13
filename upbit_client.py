import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

def get_client():
    return ccxt.upbit({
        'apiKey': os.getenv('UPBIT_ACCESS_KEY'),
        'secret': os.getenv('UPBIT_SECRET_KEY'),
    })

def get_balance(client):
    balance = client.fetch_balance()
    krw = balance['KRW']['free']
    btc = balance['BTC']['free'] if 'BTC' in balance else 0
    return {'KRW': krw, 'BTC': btc}

def get_btc_price(client):
    ticker = client.fetch_ticker('BTC/KRW')
    return ticker['last']

def get_ohlcv(client, timeframe='4h', limit=200):
    """BTC/KRW OHLCV 데이터 가져오기
    timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
    """
    ohlcv = client.fetch_ohlcv('BTC/KRW', timeframe=timeframe, limit=limit)
    import pandas as pd
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == '__main__':
    client = get_client()

    print('=== 업비트 연결 테스트 ===')
    price = get_btc_price(client)
    print(f'BTC 현재가: {price:,.0f} KRW')

    balance = get_balance(client)
    print(f'보유 KRW: {balance["KRW"]:,.0f}')
    print(f'보유 BTC: {balance["BTC"]:.8f}')

    df = get_ohlcv(client, timeframe='4h', limit=10)
    print(f'\n최근 4시간봉 10개:')
    print(df[['open', 'high', 'low', 'close', 'volume']].to_string())
