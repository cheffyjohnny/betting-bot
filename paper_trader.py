import json
from datetime import datetime
from pathlib import Path

TRADES_FILE = Path('data/trades.json')
STATE_FILE  = Path('data/state.json')
UPBIT_FEE   = 0.0005
SL_ATR_MULT = 2.0  # 트레일링 스탑: ATR × 2


def load_state(initial_krw: float = 1_000_000) -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'krw':         initial_krw,
        'btc':         0.0,
        'initial_krw': initial_krw,
        'position':    None,
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def load_trades() -> list:
    if TRADES_FILE.exists():
        return json.loads(TRADES_FILE.read_text())
    return []


def save_trade(trade: dict):
    trades = load_trades()
    trades.append(trade)
    TRADES_FILE.parent.mkdir(exist_ok=True)
    TRADES_FILE.write_text(json.dumps(trades, ensure_ascii=False, indent=2))


def buy(state: dict, price: float, atr: float, reason: str) -> dict | None:
    if state['position'] is not None:
        return None
    if state['krw'] < 10000:
        return None

    invest_krw = state['krw']
    btc_bought = (invest_krw - invest_krw * UPBIT_FEE) / price
    trailing_stop = price - atr * SL_ATR_MULT

    state['krw'] = 0.0
    state['btc'] = btc_bought
    state['position'] = {
        'price':         price,
        'btc':           btc_bought,
        'trailing_stop': trailing_stop,
        'atr':           atr,
    }

    trade = {
        'type':          'BUY',
        'time':          datetime.now().isoformat(),
        'price':         price,
        'btc':           btc_bought,
        'invest_krw':    invest_krw,
        'trailing_stop': trailing_stop,
        'reason':        reason,
    }
    save_trade(trade)
    save_state(state)
    return trade


def update_trailing_stop(state: dict, price: float, atr: float):
    """가격이 올라갈수록 트레일링 스탑을 함께 올린다 (절대 내려가지 않음)."""
    if state['position'] is None:
        return
    new_stop = price - atr * SL_ATR_MULT
    if new_stop > state['position']['trailing_stop']:
        state['position']['trailing_stop'] = new_stop
        state['position']['atr'] = atr
        save_state(state)


def check_trailing_stop(state: dict, price: float) -> bool:
    """트레일링 스탑 이하로 내려오면 True."""
    if state['position'] is None:
        return False
    return price <= state['position']['trailing_stop']


def sell(state: dict, price: float, reason: str) -> dict | None:
    if state['position'] is None:
        return None

    btc           = state['btc']
    received_krw  = btc * price * (1 - UPBIT_FEE)
    entry_price   = state['position']['price']
    pnl           = received_krw - (btc * entry_price)
    pnl_pct       = (price - entry_price) / entry_price * 100

    state['krw']      = received_krw
    state['btc']      = 0.0
    state['position'] = None

    trade = {
        'type':         'SELL',
        'time':         datetime.now().isoformat(),
        'price':        price,
        'btc':          btc,
        'received_krw': received_krw,
        'pnl':          pnl,
        'pnl_pct':      pnl_pct,
        'reason':       reason,
    }
    save_trade(trade)
    save_state(state)
    return trade


def get_summary(state: dict, current_price: float) -> dict:
    total_value = state['krw'] + state['btc'] * current_price
    initial     = state['initial_krw']
    pnl         = total_value - initial
    pnl_pct     = pnl / initial * 100

    trades      = load_trades()
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    wins        = [t for t in sell_trades if t['pnl'] > 0]
    win_rate    = len(wins) / len(sell_trades) * 100 if sell_trades else 0

    return {
        'total_value':   total_value,
        'initial_krw':   initial,
        'pnl':           pnl,
        'pnl_pct':       pnl_pct,
        'krw':           state['krw'],
        'btc':           state['btc'],
        'total_trades':  len(sell_trades),
        'win_rate':      win_rate,
        'position':      state['position'],
    }
