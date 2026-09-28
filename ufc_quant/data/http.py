"""Polite HTTP downloader: local cache, per-host rate limiting, bounded retries, provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class PoliteDownloader:
    def __init__(self, min_interval: float = 2.0, max_retries: int = 3, backoff: float = 4.0,
                 timeout: float = 60.0, user_agent: str = "ufc-quant-academic/0.1"):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_request: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: dict) -> "PoliteDownloader":
        h = cfg["http"]
        return cls(h["min_interval_seconds"], h["max_retries"], h["backoff_seconds"],
                   h["timeout_seconds"], h["user_agent"])

    def _wait_for_host(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_request.get(host, -1e9)
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request[host] = time.monotonic()

    def fetch_to_file(self, url: str, dest: Path, refresh: bool = False) -> dict:
        """Download url to dest unless cached. Uses ETag for incremental refresh.

        A sidecar `<dest>.provenance.json` records url, time, sha256, size and ETag.
        Returns the provenance dict.
        """
        dest = Path(dest)
        meta_path = dest.with_suffix(dest.suffix + ".provenance.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if dest.exists() and meta and not refresh:
            log.info("cache hit: %s", dest.name)
            return meta

        headers = {}
        if dest.exists() and meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_host(url)
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 304:
                    log.info("not modified: %s", dest.name)
                    meta["checked_at"] = _now()
                    meta_path.write_text(json.dumps(meta, indent=2))
                    return meta
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                meta = {
                    "url": url,
                    "downloaded_at": _now(),
                    "checked_at": _now(),
                    "sha256": hashlib.sha256(resp.content).hexdigest(),
                    "bytes": len(resp.content),
                    "etag": resp.headers.get("ETag"),
                    "last_modified": resp.headers.get("Last-Modified"),
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                return meta
            except requests.RequestException as exc:  # network or HTTP error
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    break  # client errors (403, 404) are not retried
                log.warning("attempt %d/%d failed for %s: %s", attempt, self.max_retries, url, exc)
                time.sleep(self.backoff * attempt)
        raise RuntimeError(f"download failed for {url}: {last_exc}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
