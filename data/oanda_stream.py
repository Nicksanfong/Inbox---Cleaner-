import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable

import oandapyV20
import oandapyV20.endpoints.pricing as pricing

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 10   # seconds between heartbeat checks
HEARTBEAT_TIMEOUT  = 30   # warn if no PRICE tick arrives within this window


class OandaStreamManager:
    """
    Streams real-time bid/ask prices from OANDA for forex instruments.

    Uses oandapyV20's synchronous streaming generator in a background thread,
    feeding ticks into an asyncio.Queue so the main event loop (and the
    heartbeat coroutine) can run unblocked.

    Instruments use OANDA format: EUR_USD, GBP_USD, USD_JPY, etc.
    """

    def __init__(self, api_token: str, account_id: str, instruments: list[str]) -> None:
        self._token      = api_token
        self._account_id = account_id
        self.instruments = instruments
        self._env        = os.getenv("OANDA_ENVIRONMENT", "practice")
        self._callbacks: list[Callable] = []
        self._last_price_time: float = time.monotonic()

    def subscribe(self, callback: Callable) -> None:
        """Register a callback. May be sync or async: fn(tick: dict) -> None."""
        self._callbacks.append(callback)

    async def _dispatch(self, tick: dict) -> None:
        self._last_price_time = time.monotonic()
        for cb in self._callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(tick)
            else:
                cb(tick)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            lag = time.monotonic() - self._last_price_time
            if lag > HEARTBEAT_TIMEOUT:
                log.warning(
                    "OANDA heartbeat: no PRICE tick for %.0f s "
                    "(OANDA sends a heartbeat every 5 s — check your connection)",
                    lag,
                )
            else:
                log.debug("OANDA heartbeat OK — last tick %.1f s ago", lag)

    def _stream_worker(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        stop_event: threading.Event,
    ) -> None:
        """
        Blocking generator loop that runs in a background thread.
        Pushes each tick dict into the asyncio Queue.
        Pushes {"_error": msg} on exception so the async side can detect it.
        """
        api = oandapyV20.API(access_token=self._token, environment=self._env)
        params = {"instruments": ",".join(self.instruments)}
        r = pricing.PricingStream(accountID=self._account_id, params=params)
        try:
            for tick in api.request(r):
                if stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(tick), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put({"_error": str(exc)}), loop)

    async def run(self) -> None:
        """
        Start streaming. Blocking coroutine; reconnects automatically with
        exponential backoff (max 60 s) on errors or disconnects.
        """
        asyncio.create_task(self._heartbeat_loop())
        loop = asyncio.get_running_loop()
        attempt = 0

        while True:
            attempt += 1
            if attempt > 1:
                delay = min(2 ** (attempt - 1), 60)
                log.info("OANDA reconnect in %d s (attempt %d) ...", delay, attempt)
                await asyncio.sleep(delay)

            log.info(
                "OANDA stream connect #%d — instruments: %s",
                attempt, self.instruments,
            )

            queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
            stop_event = threading.Event()

            thread = threading.Thread(
                target=self._stream_worker,
                args=(queue, loop, stop_event),
                daemon=True,
            )
            thread.start()

            disconnected = False
            while not disconnected:
                try:
                    tick = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    if not thread.is_alive():
                        log.warning("OANDA stream thread died unexpectedly.")
                        disconnected = True
                    continue
                except asyncio.CancelledError:
                    stop_event.set()
                    thread.join(timeout=3.0)
                    log.info("OANDA stream cancelled — shut down cleanly.")
                    return

                if "_error" in tick:
                    log.error("OANDA stream error: %s", tick["_error"])
                    disconnected = True
                    continue

                tick_type = tick.get("type", "")
                if tick_type == "HEARTBEAT":
                    # OANDA sends its own heartbeat every 5 s when no price moves
                    self._last_price_time = time.monotonic()
                    log.debug("OANDA server heartbeat received.")
                elif tick_type == "PRICE":
                    await self._dispatch(tick)

            stop_event.set()
            thread.join(timeout=3.0)
