import asyncio
import logging
import time
from collections.abc import Callable

from alpaca.data.live import StockDataStream

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 10   # seconds between heartbeat checks
HEARTBEAT_TIMEOUT  = 30   # warn if no quote arrives within this window


class WebSocketManager:
    """
    Wraps Alpaca's StockDataStream with:
      - heartbeat monitor (warns when no data for HEARTBEAT_TIMEOUT seconds)
      - exponential-backoff auto-reconnect
      - simple subscribe() API for attaching quote callbacks
    """

    def __init__(self, api_key: str, api_secret: str, symbols: list[str]) -> None:
        self._api_key    = api_key
        self._api_secret = api_secret
        self.symbols     = symbols
        self._callbacks: list[Callable] = []
        self._last_msg_time: float = time.monotonic()

    def subscribe(self, callback: Callable) -> None:
        """Register a callback. May be sync or async: fn(quote) -> None."""
        self._callbacks.append(callback)

    async def _dispatch(self, quote) -> None:
        self._last_msg_time = time.monotonic()
        for cb in self._callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(quote)
            else:
                cb(quote)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            lag = time.monotonic() - self._last_msg_time
            if lag > HEARTBEAT_TIMEOUT:
                log.warning(
                    "Heartbeat: no quote received for %.0f s "
                    "(market closed or connection stalled)",
                    lag,
                )
            else:
                log.debug("Heartbeat OK — last quote %.1f s ago", lag)

    async def run(self) -> None:
        """
        Start streaming. Blocking coroutine; run with asyncio.run() or await it.
        Reconnects automatically with exponential backoff (max 60 s).
        """
        asyncio.create_task(self._heartbeat_loop())
        attempt = 0
        while True:
            attempt += 1
            if attempt > 1:
                delay = min(2 ** (attempt - 1), 60)
                log.info("Reconnecting in %d s (attempt %d) ...", delay, attempt)
                await asyncio.sleep(delay)

            log.info("WebSocket connect #%d — subscribing to %s", attempt, self.symbols)
            try:
                stream = StockDataStream(
                    self._api_key,
                    self._api_secret,
                    feed="iex",
                )
                stream.subscribe_quotes(self._dispatch, *self.symbols)
                # _run_forever is the internal async runner that stream.run() delegates to;
                # we call it directly so we stay inside our own async event loop.
                await stream._run_forever()
            except asyncio.CancelledError:
                log.info("WebSocket cancelled — shutting down cleanly.")
                return
            except Exception as exc:
                log.error("WebSocket error (will retry): %s", exc)
