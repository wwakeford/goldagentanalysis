"""Pluggable broker abstraction so the strategy logic never talks to MT5 directly.

The live runner (live_mt5.py) drives the shared GridEngine and turns its intents
(open / add-leg / close-basket) into calls on a ``Broker``. Two implementations:

  * ``SimBroker``  -- a paper broker that fills at a supplied quote with a modeled
    spread + commission and tracks equity / margin / margin level. Runs on ANY OS,
    so the whole live stack is testable on macOS (and is what backs the build-week
    dry run + the integration parity test).
  * ``MT5Broker``  -- thin wrapper over the ``MetaTrader5`` package. That package
    is **Windows-only** (it attaches to a local MT5 terminal), so it is imported
    lazily; on macOS it raises a clear, actionable error. The competition account
    or a Windows VM/cloud box drops in here with no strategy changes.

Margin model matches sizing.py: XAUUSD 1 lot = 100 oz, margin/lot = price*100/lev,
margin level = equity / used_margin (the quantity the comp checks against 30%).
"""
from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sizing import MARGIN_CALL_LEVEL, margin_level


@dataclass
class Quote:
    time_ns: int
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class Position:
    ticket: int
    side: int          # +1 long, -1 short
    lots: float
    open_price: float


@dataclass
class Account:
    equity: float
    balance: float
    used_margin: float

    @property
    def margin_level(self) -> float:
        return margin_level(self.equity, self.used_margin)


class Broker(ABC):
    """Minimal surface the live runner needs. Symbol is fixed per instance."""
    symbol: str
    leverage: float
    usd_per_point_per_lot: float = 100.0   # XAUUSD

    @abstractmethod
    def quote(self) -> Quote: ...
    @abstractmethod
    def market_order(self, side: int, lots: float) -> Position: ...
    @abstractmethod
    def positions(self) -> list[Position]: ...
    @abstractmethod
    def close_position(self, ticket: int) -> float: ...   # returns realized pnl
    @abstractmethod
    def account(self) -> Account: ...

    def close_all(self) -> float:
        return sum(self.close_position(pos.ticket) for pos in list(self.positions()))

    def margin_per_lot(self, price: float) -> float:
        return price * self.usd_per_point_per_lot / self.leverage


# --------------------------------------------------------------------------- #
# Paper broker -- fills at a fed quote; full equity/margin bookkeeping.
# --------------------------------------------------------------------------- #
class SimBroker(Broker):
    """Paper-trading broker. The runner feeds it the current quote via
    ``set_quote`` (from MT5 in live, or recorded ticks in replay) and it fills
    market orders at bid/ask, tracking floating P&L, equity and margin level.

    Equity = balance + floating P&L of open positions, marked at the current
    quote (long marked at bid, short at ask -- the price you'd close at)."""

    def __init__(self, symbol: str = "XAUUSD", balance: float = 1_000_000.0,
                 leverage: float = 30.0, commission_per_lot_side: float = 0.0,
                 usd_per_point_per_lot: float = 100.0):
        self.symbol = symbol
        self.leverage = leverage
        self.usd_per_point_per_lot = usd_per_point_per_lot
        self.commission_per_lot_side = commission_per_lot_side
        self.balance = balance
        self._q: Quote | None = None
        self._pos: dict[int, Position] = {}
        self._next_ticket = 1
        self.realized = 0.0

    def set_quote(self, q: Quote) -> None:
        self._q = q

    def quote(self) -> Quote:
        if self._q is None:
            raise RuntimeError("SimBroker has no quote yet; call set_quote first")
        return self._q

    def _floating(self) -> float:
        if self._q is None or not self._pos:
            return 0.0
        tot = 0.0
        for p in self._pos.values():
            mark = self._q.bid if p.side > 0 else self._q.ask
            tot += (mark - p.open_price) * p.side * p.lots * self.usd_per_point_per_lot
        return tot

    def market_order(self, side: int, lots: float) -> Position:
        q = self.quote()
        px = q.ask if side > 0 else q.bid
        self.balance -= self.commission_per_lot_side * lots   # entry-side commission
        pos = Position(self._next_ticket, side, lots, px)
        self._pos[pos.ticket] = pos
        self._next_ticket += 1
        return pos

    def close_position(self, ticket: int) -> float:
        p = self._pos.pop(ticket)
        q = self.quote()
        exit_px = q.bid if p.side > 0 else q.ask
        pnl = (exit_px - p.open_price) * p.side * p.lots * self.usd_per_point_per_lot
        pnl -= self.commission_per_lot_side * p.lots   # exit-side commission
        self.balance += pnl
        self.realized += pnl
        return pnl

    def positions(self) -> list[Position]:
        return list(self._pos.values())

    def account(self) -> Account:
        q = self._q
        used = 0.0
        if q is not None:
            used = sum(p.lots for p in self._pos.values()) * self.margin_per_lot(q.mid)
        return Account(equity=self.balance + self._floating(),
                       balance=self.balance, used_margin=used)


# --------------------------------------------------------------------------- #
# Live MT5 broker -- lazy import (Windows-only package).
# --------------------------------------------------------------------------- #
class MT5Broker(Broker):
    """Wraps the MetaTrader5 package. Connect with login/server/password, or
    attach to an already-logged-in terminal. Windows-only (see module docstring)."""

    def __init__(self, symbol: str = "XAUUSD", leverage: float = 30.0,
                 login: int | None = None, password: str | None = None,
                 server: str | None = None, path: str | None = None,
                 deviation: int = 20, magic: int = 23765):
        try:
            import MetaTrader5 as mt5  # noqa: N813
        except Exception as e:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "MetaTrader5 package unavailable. It is Windows-only and attaches "
                "to a local MT5 terminal. On macOS run the bot on a Windows VM / "
                "cloud box, or via the comp's API; use SimBroker for local dry-run."
            ) from e
        self.mt5 = mt5
        self.symbol = symbol
        self.leverage = leverage
        self.deviation = deviation
        self.magic = magic
        kw = {k: v for k, v in
              dict(login=login, password=password, server=server, path=path).items()
              if v is not None}
        if not mt5.initialize(**kw):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 cannot select symbol {symbol}")
        info = mt5.symbol_info(symbol)
        # contract size * 1 point -> $/point/lot; for XAUUSD this is 100.
        self.usd_per_point_per_lot = float(info.trade_contract_size)
        # --- server-tz -> naive UTC offset ---------------------------------------- #
        # The feature pipeline expects bar timestamps in TRUE UTC (it adds +3h internally
        # for the UTC+3 broker-hour session windows). MT5 tick/rates times are in the
        # SERVER's timezone, so we strip the offset on every quote()/history_m1. The comp
        # server (FTWorldwide) is NOT the usual EET — auto-detect it (round to whole hours,
        # only trust a live/recent tick; sanity-clamp), overridable via SERVER_OFFSET_H.
        import os as _os, time as _time
        env_off = _os.environ.get("SERVER_OFFSET_H")
        if env_off not in (None, ""):
            self.server_offset_h = int(env_off)
        else:
            tk = mt5.symbol_info_tick(symbol)
            det = int(round((float(tk.time) - _time.time()) / 3600.0)) if (tk and tk.time) else 3
            self.server_offset_h = det if -14 <= det <= 14 else 3   # stale-tick fallback
        print(f"[MT5Broker] server->UTC offset = +{self.server_offset_h}h "
              f"(auto-detected; set SERVER_OFFSET_H to override). Verify vs the session table.")

    def quote(self) -> Quote:
        t = self.mt5.symbol_info_tick(self.symbol)
        # strip server tz -> naive UTC (the feature pipeline's broker-hour math expects UTC)
        utc_ns = int(t.time_msc) * 1_000_000 - self.server_offset_h * 3_600_000_000_000
        return Quote(time_ns=utc_ns, bid=t.bid, ask=t.ask)

    def market_order(self, side: int, lots: float) -> Position:
        mt5 = self.mt5
        t = mt5.symbol_info_tick(self.symbol)
        px = t.ask if side > 0 else t.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
            "price": px, "deviation": self.deviation, "magic": self.magic,
            "type_filling": mt5.ORDER_FILLING_IOC, "comment": "grid",
        }
        r = mt5.order_send(req)
        if r.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"order_send failed: {r.retcode} {r.comment}")
        return Position(int(r.order), side, lots, float(r.price))

    def positions(self) -> list[Position]:
        out = []
        for p in (self.mt5.positions_get(symbol=self.symbol) or ()):
            side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            out.append(Position(int(p.ticket), side, float(p.volume),
                                float(p.price_open)))
        return out

    def close_position(self, ticket: int) -> float:
        mt5 = self.mt5
        ps = mt5.positions_get(ticket=ticket)
        if not ps:
            return 0.0
        p = ps[0]
        side = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        t = mt5.symbol_info_tick(self.symbol)
        px = t.bid if side > 0 else t.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": float(p.volume), "position": int(ticket),
            "type": mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY,
            "price": px, "deviation": self.deviation, "magic": self.magic,
            "type_filling": mt5.ORDER_FILLING_IOC, "comment": "grid-close",
        }
        r = mt5.order_send(req)
        if r.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"close failed: {r.retcode} {r.comment}")
        return float(p.profit)

    def account(self) -> Account:
        a = self.mt5.account_info()
        return Account(equity=float(a.equity), balance=float(a.balance),
                       used_margin=float(a.margin))

    def history_m1(self, n_bars: int):
        """Last ``n_bars`` M1 bars as a DataFrame (timestamp UTC, bid/ask/mid close).

        Used by the Quantum live stack to warm its rolling features at startup so the
        htf trend-flip is armed from the first live bar. MT5 ``copy_rates`` returns
        mid-ish close; bid/ask are reconstructed from the symbol's current spread
        (features use mid only, so this is exact for warm-up)."""
        import numpy as np
        import pandas as pd
        mt5 = self.mt5
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, int(n_bars))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 copy_rates returned nothing for {self.symbol}")
        df = pd.DataFrame(rates)
        ts = pd.to_datetime(df["time"], unit="s")           # MT5 server time
        # strip the detected server tz -> naive UTC (matches the feature pipeline + quote())
        ts = ts - pd.Timedelta(hours=self.server_offset_h)
        spread = float(getattr(mt5.symbol_info(self.symbol), "spread", 0)) * \
            float(getattr(mt5.symbol_info(self.symbol), "point", 0.01))
        close = df["close"].astype(float).values
        half = spread / 2.0
        return pd.DataFrame({
            "timestamp": ts.values,
            "bid_close": close - half,
            "ask_close": close + half,
            "mid_close": close,
        })

    def shutdown(self) -> None:
        self.mt5.shutdown()


def make_broker(kind: str = "sim", **kw) -> Broker:
    """Factory: 'sim' everywhere, 'mt5' on Windows / a terminal host."""
    if kind == "sim":
        return SimBroker(**kw)
    if kind == "mt5":
        return MT5Broker(**kw)
    raise ValueError(f"unknown broker kind {kind!r}")


if __name__ == "__main__":
    # SimBroker smoke test + margin-level sanity (runs on macOS).
    b = SimBroker(balance=1_000_000.0, leverage=30.0)
    b.set_quote(Quote(0, bid=4300.0, ask=4300.2))
    p1 = b.market_order(1, 5.0)               # buy 5 lots
    acc = b.account()
    print(f"after buy 5 lots @4300.2: equity={acc.equity:,.0f} "
          f"used_margin={acc.used_margin:,.0f} margin_level={acc.margin_level:.2f}")
    b.set_quote(Quote(1, bid=4310.0, ask=4310.2))   # +$10 in our favour
    acc = b.account()
    print(f"price +10: equity={acc.equity:,.0f} (floating +5000 expected), "
          f"margin_level={acc.margin_level:.2f}")
    pnl = b.close_position(p1.ticket)
    print(f"closed: realized pnl={pnl:,.0f} (expect ~ (4310.0-4300.2)*5*100=4900)")
    print(f"platform={platform.system()}  MT5Broker available only on Windows")
