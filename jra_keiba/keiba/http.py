"""Polite HTTP client: one request stream, a minimum interval plus jitter, retries.

netkeiba answers HTTP 400 once it considers a client too busy (seen after ~3,800
requests at ~1.2 s spacing), so the default spacing is deliberately slow.
"""
from __future__ import annotations

import random
import time

import requests

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


class Fetcher:
    def __init__(self, min_interval: float = 3.0, timeout: float = 30.0, retries: int = 4):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"}
        )
        self._last = 0.0
        self.n_requests = 0

    def _wait(self) -> None:
        gap = self.min_interval * random.uniform(1.0, 1.5) - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def get_bytes(self, url: str, referer: str | None = None) -> tuple[int, bytes]:
        """Return (status, body). Retries on 5xx / network errors with backoff.

        4xx responses are returned as-is (a 400/403 from the site usually means
        "slow down" or "blocked", which the caller must handle by stopping).
        """
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            self._wait()
            try:
                headers = {"Referer": referer} if referer else {}
                r = self.session.get(url, timeout=self.timeout, headers=headers)
                self.n_requests += 1
                if r.status_code >= 500:
                    time.sleep(5 * (attempt + 1))
                    continue
                return r.status_code, r.content
            except requests.RequestException as exc:  # network hiccup
                last_exc = exc
                time.sleep(5 * (attempt + 1))
        if last_exc:
            raise last_exc
        return 599, b""


def decode(body: bytes) -> str:
    """netkeiba serves EUC-JP on db.* and UTF-8 on newer pages; detect by trying."""
    head = body[:2000].lower()
    if b"euc-jp" in head:
        return body.decode("euc_jp", errors="replace")
    if b"utf-8" in head:
        return body.decode("utf-8", errors="replace")
    for enc in ("utf-8", "euc_jp", "cp932"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")
