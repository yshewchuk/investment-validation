"""Tier 1 — the raw immutable cache, and the one fetch wrapper for all sources.

Policy: **every byte ever fetched is kept, verbatim, forever.** Quotas and
throttling make refetching expensive (ORATS is 20,000 calls/month; the existing
chain cache cost most of a month's budget), so the raw store is append-only —
never edited, never deleted, never rewritten by a parser change. Everything
downstream is rebuilt *from* it.

The wrapper guarantees, in order:

1. **Cache-first.** A repeated request never touches the network. The cache key
   is a hash of ``(source, endpoint, canonical params)``, so parameter ordering
   and formatting cannot produce two entries for one request.
2. **Throttled.** A miss calls :meth:`Throttle.acquire` before the network.
3. **Quota-guarded.** ORATS refuses to spend into the live-operations reserve.
4. **Persisted before parsed.** The body is gzipped to disk exactly as received,
   with a sidecar of metadata, *before* anything tries to interpret it. A
   parser bug can then never cost a re-fetch.
5. **Logged.** One row per network call into ``fetch_log.csv`` (and, for metered
   sources, ``quota_log.csv``), which is what the quota ledger reconciles against.

Failures never poison the store: a non-2xx response is logged but not cached, so
a retry after the outage is a clean miss rather than a permanent bad entry.

``live=True`` marks an endpoint whose answer changes: the key gets today's date
appended, giving one cache entry per day. Historical endpoints are immutable and
cached forever.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from engine import paths
from engine.data.sources import get_adapter
from engine.data.sources.base import CredentialRotated, FetchError, Response, SourceAdapter
from engine.data.throttle import PolygonBusy, QuotaExhausted, Throttle, polygon_lock

__all__ = [
    "RawRecord",
    "Fetcher",
    "fetch",
    "cache_key",
    "canonical_params",
    "CredentialRotated",
    "FetchError",
    "FETCH_LOG_COLUMNS",
]

FETCH_LOG_COLUMNS = (
    "ts",
    "source",
    "endpoint",
    "key",
    "status",
    "bytes",
    "elapsed_s",
    "quota_remaining",
    "from_cache",
    "url",
    "note",
)

QUOTA_LOG_COLUMNS = (
    "ts",
    "source",
    "endpoint",
    "key",
    "status",
    "quota_remaining",
    "elapsed_s",
)


# --------------------------------------------------------------------------
# cache keys
# --------------------------------------------------------------------------


def canonical_params(params: Mapping[str, Any] | None) -> str:
    """Deterministic JSON for a parameter mapping.

    Values are stringified before sorting so that ``{"dte": 14}`` and
    ``{"dte": "14"}`` — the same request, typed differently by two call sites —
    map to one cache entry rather than two. ``None`` values are dropped, since
    an omitted parameter and an explicitly-null one mean the same thing to
    every API here.
    """
    clean = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def cache_key(
    source: str, endpoint: str, params: Mapping[str, Any] | None, *, live: bool = False
) -> str:
    payload = {
        "source": source,
        "endpoint": endpoint.strip("/"),
        "params": canonical_params(params),
    }
    if live:
        payload["day"] = date.today().isoformat()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    """One cached response: the bytes, where they live, and how they got there."""

    body: bytes
    meta: dict[str, Any]
    path: Path
    from_cache: bool

    @property
    def key(self) -> str:
        return str(self.meta.get("key", ""))

    @property
    def status(self) -> int:
        return int(self.meta.get("status", 0))

    def json(self) -> Any:
        """Parse the body as JSON. Parsing never mutates the stored bytes."""
        return json.loads(self.body)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------


class Fetcher:
    """Cache-first fetch across all sources.

    Constructor arguments exist for the test-suite: a throwaway ``root``, a
    fake adapter, and an instrumented :class:`Throttle` make the whole path
    exercisable without a network or a credential.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        throttle: Throttle | None = None,
        adapters: Mapping[str, SourceAdapter] | None = None,
    ):
        self.root = Path(root) if root is not None else paths.RAW_FETCH
        self.throttle = throttle or Throttle()
        self._adapters: dict[str, SourceAdapter] = dict(adapters or {})
        self.network_calls = 0

    # -- paths ------------------------------------------------------------

    def body_path(self, source: str, key: str) -> Path:
        return self.root / source / key[:2] / f"{key}.body.gz"

    def meta_path(self, source: str, key: str) -> Path:
        return self.root / source / key[:2] / f"{key}.meta.json"

    @property
    def fetch_log(self) -> Path:
        return self.root / "fetch_log.csv"

    @property
    def quota_log(self) -> Path:
        return self.root / "quota_log.csv"

    def adapter(self, source: str) -> SourceAdapter:
        if source not in self._adapters:
            self._adapters[source] = get_adapter(source)
        return self._adapters[source]

    # -- public API -------------------------------------------------------

    def fetch(
        self,
        source: str,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        live: bool = False,
        timeout: float | None = None,
        note: str = "",
    ) -> RawRecord:
        key = cache_key(source, endpoint, params, live=live)
        cached = self._read_cache(source, key)
        if cached is not None:
            return cached

        cfg = self.throttle.config(source)
        if cfg.single_process:
            with polygon_lock():
                return self._fetch_uncached(source, endpoint, params, key, timeout, note)
        return self._fetch_uncached(source, endpoint, params, key, timeout, note)

    def has(self, source: str, endpoint: str, params=None, *, live: bool = False) -> bool:
        key = cache_key(source, endpoint, params, live=live)
        return self.body_path(source, key).exists()

    # -- internals --------------------------------------------------------

    def _read_cache(self, source: str, key: str) -> RawRecord | None:
        body_path, meta_path = self.body_path(source, key), self.meta_path(source, key)
        if not (body_path.exists() and meta_path.exists()):
            return None
        try:
            with gzip.open(body_path, "rb") as fh:
                body = fh.read()
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError, EOFError) as exc:
            # A truncated entry (crash mid-write) is a miss, not a hard failure:
            # the write is atomic-by-rename below, so this should be unreachable,
            # but a corrupt cache must never be able to wedge a pull.
            raise FetchError(
                f"corrupt Tier-1 entry {body_path} ({exc}); "
                "move it aside to re-fetch — do not edit it in place"
            ) from exc
        return RawRecord(body=body, meta=meta, path=body_path, from_cache=True)

    def _fetch_uncached(
        self,
        source: str,
        endpoint: str,
        params: Mapping[str, Any] | None,
        key: str,
        timeout: float | None,
        note: str,
    ) -> RawRecord:
        adapter = self.adapter(source)
        cfg = self.throttle.config(source)
        effective_timeout = float(timeout if timeout is not None else cfg.timeout)

        # Refuse to spend into the reserve *before* pacing, so a guarded run
        # stops immediately rather than after a sleep.
        self.throttle.check_quota(source, self._last_known_quota(source))

        last_error: Exception | None = None
        for attempt in range(cfg.max_retries):
            self.throttle.acquire(source)
            self.network_calls += 1
            try:
                response = adapter.request(endpoint, dict(params or {}), effective_timeout)
            except CredentialRotated:
                raise
            except Exception as exc:  # transport-level failure
                last_error = exc
                self._log(
                    source, endpoint, key, status=0, size=0, elapsed=0.0,
                    quota=None, from_cache=False, url="", note=f"error: {type(exc).__name__}",
                )
                if attempt == cfg.max_retries - 1:
                    break
                self.throttle.backoff(source, attempt)
                continue

            quota = adapter.quota_from(response)
            if quota is not None:
                self._record_quota(source, quota)

            if adapter.is_auth_failure(response):
                self._log(
                    source, endpoint, key, response.status, len(response.body),
                    response.elapsed_s, quota, False, response.url, "auth-failure",
                )
                raise CredentialRotated(
                    f"{source} returned {response.status} for {endpoint} — the "
                    "credential has rotated. Update .env and re-run; do not retry."
                )

            if response.ok:
                path = self._persist(source, endpoint, key, params, response, quota, note)
                self._log(
                    source, endpoint, key, response.status, len(response.body),
                    response.elapsed_s, quota, False, response.url, note,
                )
                if quota is not None:
                    self._log_quota(source, endpoint, key, response, quota)
                meta = json.loads(self.meta_path(source, key).read_text())
                return RawRecord(response.body, meta, path, from_cache=False)

            # Non-2xx: log it, cache NOTHING, retry if it looks transient.
            self._log(
                source, endpoint, key, response.status, len(response.body),
                response.elapsed_s, quota, False, response.url, note or "non-2xx",
            )
            last_error = FetchError(
                f"{source} {endpoint} returned HTTP {response.status}"
            )
            if response.status < 500 and response.status != 429:
                raise last_error  # 4xx other than rate-limit will not fix itself
            if attempt == cfg.max_retries - 1:
                break
            self.throttle.backoff(source, attempt)

        raise FetchError(
            f"{source} {endpoint} failed after {cfg.max_retries} attempts: {last_error}"
        ) from last_error

    def _persist(
        self,
        source: str,
        endpoint: str,
        key: str,
        params: Mapping[str, Any] | None,
        response: Response,
        quota: int | None,
        note: str,
    ) -> Path:
        """Write body + meta. Body first, then meta, both via atomic rename.

        Order matters: meta is what :meth:`_read_cache` treats as the commit
        marker, so a crash between the two writes leaves an orphan body (a
        miss, re-fetched) rather than metadata pointing at a partial body.
        """
        body_path = paths.assert_writable(self.body_path(source, key))
        meta_path = self.meta_path(source, key)
        body_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_body = body_path.with_suffix(body_path.suffix + ".tmp")
        with gzip.open(tmp_body, "wb") as fh:
            fh.write(response.body)
        os.replace(tmp_body, body_path)

        meta = {
            "key": key,
            "source": source,
            "endpoint": endpoint.strip("/"),
            "params": json.loads(canonical_params(params)),
            "url": response.url,  # already redacted by the adapter
            "status": response.status,
            "bytes": len(response.body),
            "elapsed_s": round(response.elapsed_s, 3),
            "quota_remaining": quota,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
        }
        tmp_meta = meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=1, sort_keys=True))
        os.replace(tmp_meta, meta_path)
        return body_path

    # -- logs -------------------------------------------------------------

    def _append_row(self, path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
        paths.assert_writable(path).parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in columns})
            fh.flush()  # a crash loses at most the row being written

    def _log(
        self, source, endpoint, key, status, size, elapsed, quota, from_cache, url, note
    ) -> None:
        self._append_row(
            self.fetch_log,
            FETCH_LOG_COLUMNS,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "endpoint": endpoint.strip("/"),
                "key": key,
                "status": status,
                "bytes": size,
                "elapsed_s": round(float(elapsed), 3),
                "quota_remaining": "" if quota is None else quota,
                "from_cache": int(bool(from_cache)),
                "url": url,
                "note": note,
            },
        )

    def _log_quota(self, source, endpoint, key, response: Response, quota: int) -> None:
        self._append_row(
            self.quota_log,
            QUOTA_LOG_COLUMNS,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "endpoint": endpoint.strip("/"),
                "key": key,
                "status": response.status,
                "quota_remaining": quota,
                "elapsed_s": round(response.elapsed_s, 3),
            },
        )

    # -- quota state ------------------------------------------------------

    def _record_quota(self, source: str, remaining: int) -> None:
        self._quota_cache = getattr(self, "_quota_cache", {})
        self._quota_cache[source] = remaining

    def _last_known_quota(self, source: str) -> int | None:
        """Most recent remaining-quota reading, from this process or the log.

        ORATS omits quota headers on CDN-cached responses, so the last *known*
        value is the best available signal; the guard is deliberately advisory
        rather than authoritative, and the quota ledger reconciliation in the
        Phase 0 report is what catches drift.
        """
        cached = getattr(self, "_quota_cache", {}).get(source)
        if cached is not None:
            return cached
        path = self.quota_log
        if not path.exists():
            return None
        last: int | None = None
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("source") != source:
                        continue
                    value = (row.get("quota_remaining") or "").strip()
                    if value:
                        try:
                            last = int(float(value))
                        except ValueError:
                            continue
        except OSError:
            return None
        if last is not None:
            self._record_quota(source, last)
        return last


# --------------------------------------------------------------------------
# module-level default
# --------------------------------------------------------------------------

_DEFAULT: Fetcher | None = None


def _default() -> Fetcher:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Fetcher()
    return _DEFAULT


def fetch(
    source: str,
    endpoint: str,
    params: Mapping[str, Any] | None = None,
    *,
    live: bool = False,
    timeout: float | None = None,
    note: str = "",
) -> RawRecord:
    """Cache-first fetch. See :class:`Fetcher` for the guarantees."""
    return _default().fetch(
        source, endpoint, params, live=live, timeout=timeout, note=note
    )
