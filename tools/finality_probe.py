import argparse
import errno
import hashlib
import json
import math
import os
import re
import resource
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlparse

DEFAULT_ORATS_ENDPOINTS = "hist/strikes,hist/summaries"
ORATS_BASE = "https://api.orats.io/datav2/"
MASSIVE_BASE = "https://api.massive.com/v3/snapshot/options/"
MASSIVE_ORIGIN = "https://api.massive.com/"
MASSIVE_ENDPOINT = "v3/snapshot/options"
DROP_HEADERS = frozenset({"set-cookie", "authorization", "cookie"})
MAX_MASSIVE_PAGES = 20
REQUEST_TIMEOUT_S = 30
MAX_META_BYTES = 1024 * 1024
MAX_BODY_BYTES = 64 * 1024 * 1024
HASH_POLICY = "rows-sorted-canonical-json-v1"
OPEN_RO_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TICKER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENDPOINT_RE = re.compile(r"^[a-z0-9_/]+$")


def _fail(msg, code=2):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _stem(provider, endpoint, ticker, page):
    stem = f"{provider}__{endpoint.replace('/', '.')}__{ticker}"
    if page is not None:
        stem = f"{stem}__p{page}"
    return stem


def _valid_session(s):
    if not isinstance(s, str) or not SESSION_RE.match(s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _errno_name(exc):
    eno = getattr(exc, "errno", None)
    if isinstance(eno, int) and eno in errno.errorcode:
        return errno.errorcode[eno]
    return getattr(exc, "strerror", None) or type(exc).__name__


def _fail_path(exc):
    _fail(f"cannot use output path: {_errno_name(exc)}", 2)


def _refuse_repo(out_dir):
    try:
        path = Path(out_dir).resolve()
        for base in (path, *path.parents):
            if (base / ".git").exists() or (base / ".git").is_symlink():
                _fail("refusing to write provider data inside a repository", 2)
    except OSError as exc:
        _fail_path(exc)


def _mkdirs(path):
    try:
        missing, cur = [], Path(path)
        while not cur.exists():
            missing.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
        for d in reversed(missing):
            d.mkdir(mode=0o700)
            os.chmod(d, 0o700)
    except OSError as exc:
        _fail_path(exc)


def _redacted_url(url, params):
    p = dict(params or {})
    if "token" in p:
        p["token"] = "REDACTED"
    return f"{url}?{urlencode(p)}" if p else url


def _leaks_key(key, meta, body):
    if not key:
        return False
    meta_b = _canon(meta).encode("utf-8")
    encoded = [quote(key, safe=""), quote_plus(key)]
    variants = [key] + encoded + [re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(), v) for v in encoded]
    return any(v.encode("utf-8") in meta_b or v.encode("utf-8") in body for v in variants)


_TMP_COUNTER = 0


def _atomic_write(path, data):
    global _TMP_COUNTER
    while True:
        _TMP_COUNTER += 1
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{_TMP_COUNTER}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _content_sha256(rows):
    order = sorted(range(len(rows)), key=lambda i: _canon(rows[i]))
    return hashlib.sha256(_canon([rows[i] for i in order]).encode("utf-8")).hexdigest()


def _envelope_rows(provider, parsed):
    if not isinstance(parsed, dict):
        return None
    rows = parsed.get("data" if provider == "orats" else "results")
    return rows if isinstance(rows, list) else None


def _content_digest(provider, body):
    try:
        parsed = json.loads(body)
    except Exception:
        return None, None
    rows = _envelope_rows(provider, parsed)
    if rows is None:
        return None, None
    return _content_sha256(rows), len(rows)


def _max_updated_orats(rows):
    best_val, best_ts = None, None
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = r.get("updatedAt")
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            continue
        t = _ts(v)
        if t is None:
            continue
        if best_ts is None or t > best_ts:
            best_val, best_ts = v, t
    return best_val


def _massive_next_ok(url, ticker):
    if not (isinstance(url, str) and url.startswith("https://") and url.startswith(MASSIVE_ORIGIN)):
        return False
    u = urlparse(url)
    if u.path != f"/v3/snapshot/options/{ticker}":
        return False
    return not any(k.lower() == "apikey" for k, _ in parse_qsl(u.query, keep_blank_values=True))


def _max_updated_massive(rows):
    vals = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for ck in ("day", "last_quote"):
            c = r.get(ck)
            if not isinstance(c, dict) or c.get("last_updated") is None:
                continue
            v = c["last_updated"]
            if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
                continue
            try:
                vals.append(int(v))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None


def _rows_and_max(provider, parsed):
    rows = _envelope_rows(provider, parsed)
    if rows is None:
        rows = []
    if provider == "orats":
        return rows, _max_updated_orats(rows)
    return rows, _max_updated_massive(rows)


def _default_http_get(req):
    import requests
    headers = dict(req.get("headers") or {})
    headers.setdefault("Accept-Encoding", "identity")
    resp = requests.get(req["url"], params=req.get("params"), headers=headers,
                        timeout=req.get("timeout", REQUEST_TIMEOUT_S),
                        allow_redirects=False, stream=True)
    try:
        chunks, total = [], 0
        for chunk in resp.iter_content(65536):
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                raise ValueError("BodyTooLarge")
            if chunk:
                chunks.append(chunk)
    finally:
        resp.close()
    return int(resp.status_code), {str(k).lower(): v for k, v in resp.headers.items()}, b"".join(chunks)


def capture(args_ns, http_get=None, now=None):
    session = getattr(args_ns, "session", None)
    label = getattr(args_ns, "label", None)
    out_dir = getattr(args_ns, "out", None)
    tickers_arg = getattr(args_ns, "tickers", None)
    endpoints_arg = getattr(args_ns, "orats_endpoints", DEFAULT_ORATS_ENDPOINTS)
    massive_arg = getattr(args_ns, "massive", False)
    replace_arg = getattr(args_ns, "replace", False)
    if not ((now is None or callable(now))
            and (http_get is None or callable(http_get))
            and isinstance(session, str) and "\x00" not in session
            and isinstance(label, str) and "\x00" not in label
            and isinstance(out_dir, str) and "\x00" not in out_dir
            and isinstance(tickers_arg, str) and "\x00" not in tickers_arg
            and (endpoints_arg is None or isinstance(endpoints_arg, str))
            and isinstance(massive_arg, bool)
            and isinstance(replace_arg, bool)):
        _fail("invalid argument types", 2)
    if http_get is None:
        http_get = _default_http_get
    if now is None:
        def now():
            return datetime.now(timezone.utc)
    if not _valid_session(session):
        _fail(f"invalid session {session!r}: expected YYYY-MM-DD", 2)
    if not LABEL_RE.match(label or "") or label in (".", ".."):
        _fail(f"invalid label {label!r}", 2)
    tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]
    if not tickers:
        _fail("empty ticker list", 2)
    for t in tickers:
        if not TICKER_RE.match(t):
            _fail(f"invalid ticker {t!r}", 2)
    if len(set(tickers)) != len(tickers):
        _fail(f"duplicate tickers {tickers!r}", 2)
    endpoints = [e.strip() for e in (endpoints_arg or "").split(",") if e.strip()]
    for e in endpoints:
        if not ENDPOINT_RE.match(e) or ".." in e:
            _fail(f"invalid ORATS endpoint {e!r}", 2)
    if len(set(endpoints)) != len(endpoints):
        _fail(f"duplicate ORATS endpoints {endpoints!r}", 2)
    want_massive = bool(massive_arg)
    if not endpoints and not want_massive:
        _fail("no provider selected", 2)
    orats_key = os.environ.get("ORATS_API_KEY") or ""
    massive_key = os.environ.get("MASSIVE_API_KEY") or ""
    if endpoints and not orats_key:
        _fail("missing ORATS_API_KEY", 2)
    if want_massive and not massive_key:
        _fail("missing MASSIVE_API_KEY", 2)
    out_base = Path(out_dir).resolve()
    unit_dir = (Path(out_dir) / session / label).resolve()
    _refuse_repo(unit_dir)
    try:
        unit_dir.relative_to(out_base)
    except ValueError:
        _fail("refusing to write outside the output directory", 2)
    replace = bool(replace_arg)
    if (not replace
            and unit_dir.is_dir()
            and any(unit_dir.glob("*.meta.json"))):
        _fail("label already captured; pass --replace to invalidate the prior capture", 2)
    try:
        _mkdirs(unit_dir)
    except OSError as exc:
        _fail_path(exc)
    lock_path = unit_dir / ".capture.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        _fail("label is locked by another capture (remove .capture.lock if stale)", 2)
    if not replace and any(unit_dir.glob("*.meta.json")):
        try:
            os.close(lock_fd)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except OSError:
            pass
        _fail("label already captured; pass --replace to invalidate the prior capture", 2)
    try:
        os.write(lock_fd, str(os.getpid()).encode("utf-8"))
        os.close(lock_fd)
        lock_fd = None
        if replace and unit_dir.is_dir():
            for entry in unit_dir.iterdir():
                if entry.name == lock_path.name or not entry.name.endswith((".meta.json", ".body")):
                    continue
                try:
                    st = entry.lstat()
                except OSError as exc:
                    _fail_path(exc)
                if stat.S_ISREG(st.st_mode):
                    try:
                        entry.unlink()
                    except OSError as exc:
                        _fail_path(exc)
        run_id = uuid.uuid4().hex
        orats_planned = len(endpoints) * len(tickers)
        start_units, massive_note = orats_planned, ""
        if orats_planned == 0 and want_massive:
            start_units = len(tickers)
            massive_note = f" massive_tickers={start_units}"
        t_start = time.perf_counter()
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[capture] start units={start_units}{massive_note} label={label} elapsed=0 rss={start_rss:.1f}MB", file=sys.stderr)
        total = not_ok = 0
        saw_massive = False
        last_prog_t = time.perf_counter()
        mpages = {"t0": None, "total": 0, "k": 0, "n": 0}

        def _progress(n, m=None):
            el = time.perf_counter() - t_start
            rate = n / el if el > 0 else 0.0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if m is None:
                print(f"[capture] {n} units elapsed={int(el)}s rate={rate:.1f}/s rss={rss:.1f}MB", file=sys.stderr)
            else:
                eta = int(el / n * (m - n)) if n else 0
                print(f"[capture] {n}/{m} units elapsed={int(el)}s rate={rate:.1f}/s eta={eta}s rss={rss:.1f}MB", file=sys.stderr)

        def store(provider, endpoint, ticker, page, req, url_redacted, key):
            nonlocal total, not_ok, saw_massive, last_prog_t
            t0 = time.perf_counter()
            status, rhdrs, body, error = 0, {}, b"", None
            try:
                status, rhdrs, body = http_get(req)
                body, status = bytes(body), int(status)
                if len(body) > MAX_BODY_BYTES:
                    raise ValueError("BodyTooLarge")
            except Exception as exc:
                error, status, rhdrs, body = type(exc).__name__, 0, {}, b""
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            fetched_at = now().isoformat()
            headers = {str(k).lower(): v for k, v in (rhdrs or {}).items() if str(k).lower() not in DROP_HEADERS}
            row_count = max_updated_at = content_sha = parsed = None
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = None
            env_rows = _envelope_rows(provider, parsed)
            if env_rows is not None:
                rows, max_updated_at = _rows_and_max(provider, parsed)
                row_count = len(rows)
                content_sha = _content_sha256(rows)
            truncated = False
            has_next = False
            if provider == "massive":
                nu = parsed.get("next_url") if isinstance(parsed, dict) else None
                has_next = _massive_next_ok(nu, ticker)
                if page == MAX_MASSIVE_PAGES - 1 and has_next:
                    truncated = True
                    print(f"[capture] warning: truncated massive pagination for {ticker} after {MAX_MASSIVE_PAGES} pages", file=sys.stderr)
                elif nu is not None and not has_next:
                    truncated = True
                    print(f"[capture] warning: non-massive next_url for {ticker} page {page}; marking truncated", file=sys.stderr)
            envelope_ok = env_rows is not None
            ok = envelope_ok and 200 <= status < 300 and not truncated
            meta = {
                "provider": provider, "endpoint": endpoint, "ticker": ticker, "session": session,
                "label": label, "page": page, "url_redacted": url_redacted, "status": status,
                "ok": ok, "fetched_at_utc": fetched_at, "elapsed_ms": elapsed_ms,
                "headers": headers, "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_bytes": len(body), "row_count": row_count, "max_updated_at": max_updated_at,
                "content_sha256": content_sha, "hash_policy": HASH_POLICY, "run_id": run_id,
            }
            if provider == "massive":
                meta["has_next"] = has_next
            if truncated:
                meta["truncated"] = True
            if error is not None:
                meta["error"] = error
            if _leaks_key(key, meta, body):
                _fail("refusing to write provider record containing API key", 2)
            stem = _stem(provider, endpoint, ticker, page)
            _atomic_write(unit_dir / f"{stem}.body", body)
            _atomic_write(unit_dir / f"{stem}.meta.json", (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            total += 1
            if not ok:
                not_ok += 1
            if provider == "massive":
                saw_massive = True
                mpages["total"] += 1
            c12 = content_sha[:12] if content_sha else None
            print(f"{label} {provider} {ticker} status={status} rows={row_count} max_updated_at={max_updated_at} content={c12}")
            now_t = time.perf_counter()
            if now_t - last_prog_t >= 30.0:
                last_prog_t = now_t
                if provider == "massive":
                    el = now_t - t_start
                    rate = total / el if el > 0 else 0.0
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    avg_pg = ((now_t - mpages["t0"]) / mpages["total"]
                              if mpages["t0"] is not None and mpages["total"] else 0.0)
                    eta_max = int(avg_pg * (MAX_MASSIVE_PAGES - (page + 1)))
                    tickers_left = max(0, mpages["n"] - mpages["k"] + 1)
                    print(f"[capture] {total} pages elapsed={int(el)}s rate={rate:.1f}/s eta_max={eta_max}s tickers_left={tickers_left} rss={rss:.1f}MB", file=sys.stderr)
                else:
                    _progress(total, None if orats_planned == 0 else orats_planned)
            return parsed, ok

        for endpoint in endpoints:
            for ticker in tickers:
                params = {"ticker": ticker, "tradeDate": session, "token": orats_key}
                url = f"{ORATS_BASE}{endpoint}"
                req = {"url": url, "params": params, "headers": {"Accept-Encoding": "identity"}, "timeout": REQUEST_TIMEOUT_S}
                store("orats", endpoint, ticker, None, req, _redacted_url(url, params), orats_key)
        if want_massive:
            n_tickers = len(tickers)
            m_start = time.perf_counter()
            mpages["n"] = n_tickers
            mpages["t0"] = m_start
            for k, ticker in enumerate(tickers, 1):
                mpages["k"] = k
                url, params = f"{MASSIVE_BASE}{ticker}", {"limit": "250"}
                pages_stored = 0
                ticker_ok = True
                for page in range(MAX_MASSIVE_PAGES):
                    req = {"url": url, "params": params, "headers": {"Accept-Encoding": "identity", "Authorization": f"Bearer {massive_key}"}, "timeout": REQUEST_TIMEOUT_S}
                    parsed, page_ok = store("massive", MASSIVE_ENDPOINT, ticker, page, req, _redacted_url(url, params), massive_key)
                    pages_stored = page + 1
                    if not page_ok:
                        ticker_ok = False
                    next_url = parsed.get("next_url") if isinstance(parsed, dict) else None
                    if not _massive_next_ok(next_url, ticker):
                        break
                    url, params = next_url, None
                if ticker_ok and pages_stored:
                    last = pages_stored - 1
                    for pattern in (f"massive__*__{ticker}__p*.body", f"massive__*__{ticker}__p*.meta.json"):
                        for stale in unit_dir.glob(pattern):
                            m = re.search(r"__p(\d+)\.(?:body|meta\.json)$", stale.name)
                            if m and int(m.group(1)) > last:
                                stale.unlink()
                el = time.perf_counter() - m_start
                eta = int(el / k * (n_tickers - k))
                print(f"[capture] massive ticker {k}/{n_tickers} pages={pages_stored} elapsed={int(el)}s eta={eta}s", file=sys.stderr)
        _progress(total, None if saw_massive or orats_planned == 0 else orats_planned)
        el = time.perf_counter() - t_start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[capture] end units={total} elapsed={el:.1f}s rss={rss:.1f}MB", file=sys.stderr)
        return 1 if total == 0 or not_ok else 0
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        try:
            lock_path.unlink()
        except OSError:
            pass


def _ts(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        try:
            x = float(s)
        except ValueError:
            try:
                x = datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    if not math.isfinite(x):
        return None
    a = abs(x)
    return x / (1e9 if a > 1e15 else 1e3 if a > 1e11 else 1)


def _meta_ok(m):
    if not isinstance(m, dict):
        return False
    if "ok" in m:
        return bool(m["ok"])
    return m.get("status") == 200 and m.get("content_sha256") is not None


def _header_tracks(ok_rows, hn, composite=False):
    if composite or len(ok_rows) < 2:
        return "n/a"
    pairs = []
    for i in range(len(ok_rows) - 1):
        h0 = (ok_rows[i].get("headers") or {}).get(hn)
        h1 = (ok_rows[i + 1].get("headers") or {}).get(hn)
        c0, c1 = ok_rows[i].get("content_sha256"), ok_rows[i + 1].get("content_sha256")
        pairs.append((h0, h1, c0 != c1))
    if not any(content_diff for _, _, content_diff in pairs):
        return "n/a"
    for h0, h1, content_diff in pairs:
        if h0 is None or h1 is None:
            return "no"
        if (h0 != h1) != content_diff:
            return "no"
    return "yes"


def _skip(name, reason):
    print(f"skipped {name}: {reason}", file=sys.stderr)
    return None


def _open_ro(path):
    return os.open(path, OPEN_RO_FLAGS)


def _verified_body(meta_path, meta):
    name = meta_path.name
    if not name.endswith(".meta.json"):
        return None
    body_path = meta_path.with_name(name[: -len(".meta.json")] + ".body")
    try:
        fd = _open_ro(body_path)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_size > MAX_BODY_BYTES:
            return None
        if meta.get("body_bytes") != st.st_size:
            return None
        data = b""
        while len(data) <= MAX_BODY_BYTES:
            chunk = os.read(fd, min(65536, MAX_BODY_BYTES + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) != st.st_size:
        return None
    expected = meta.get("body_sha256")
    if not (isinstance(expected, str) and hashlib.sha256(data).hexdigest() == expected):
        return None
    return data


def _read_meta(path, session_dir, label_dir):
    name = f"{session_dir}/{label_dir}/{Path(path).name}"
    try:
        fd = _open_ro(path)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            try:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    return _skip(name, "symlink")
            except OSError:
                pass
        return _skip(name, _errno_name(exc))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            reason = "directory" if stat.S_ISDIR(st.st_mode) else "fifo" if stat.S_ISFIFO(st.st_mode) else "not a regular file"
            return _skip(name, reason)
        if st.st_size > MAX_META_BYTES:
            return _skip(name, "larger than 1 MiB")
        data = b""
        while len(data) <= MAX_META_BYTES:
            chunk = os.read(fd, min(65536, MAX_META_BYTES + 1 - len(data)))
            if not chunk:
                break
            data += chunk
    except OSError as exc:
        return _skip(name, _errno_name(exc))
    finally:
        os.close(fd)
    if len(data) > MAX_META_BYTES:
        return _skip(name, "larger than 1 MiB")
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _skip(name, "invalid JSON")
    if not isinstance(obj, dict):
        return _skip(name, "not a JSON object")
    if obj.get("hash_policy") != HASH_POLICY:
        return _skip(name, "hash policy mismatch")
    if obj.get("provider") not in ("orats", "massive"):
        return _skip(name, "invalid provider")
    endpoint = obj.get("endpoint")
    if not isinstance(endpoint, str) or not ENDPOINT_RE.match(endpoint):
        return _skip(name, "invalid endpoint")
    if not isinstance(obj.get("ticker"), str) or not TICKER_RE.match(obj.get("ticker")):
        return _skip(name, "invalid ticker")
    run_id = obj.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        return _skip(name, "invalid run_id")
    page = obj.get("page")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
        return _skip(name, "invalid page")
    file_stem = Path(path).name[: -len(".meta.json")]
    if _stem(obj.get("provider"), endpoint, obj.get("ticker"), page) != file_stem:
        return _skip(name, "identity does not match filename")
    if obj.get("provider") == "orats" and page is not None:
        return _skip(name, "invalid page")
    if obj.get("provider") == "massive" and (page is None or page < 0 or page >= MAX_MASSIVE_PAGES):
        return _skip(name, "invalid page")
    if obj.get("label") != label_dir:
        return _skip(name, "label mismatch")
    session = obj.get("session")
    if not isinstance(session, str) or not _valid_session(session):
        return _skip(name, "invalid session")
    if session != session_dir:
        return _skip(name, "session mismatch")
    if not isinstance(obj.get("status"), int) or isinstance(obj.get("status"), bool):
        return _skip(name, "invalid status")
    row_count = obj.get("row_count")
    if row_count is not None and (not isinstance(row_count, int) or isinstance(row_count, bool)):
        return _skip(name, "invalid row_count")
    fetched = obj.get("fetched_at_utc")
    if not isinstance(fetched, str) or _ts(fetched) is None:
        return _skip(name, "invalid fetched_at_utc")
    if "ok" in obj and not isinstance(obj["ok"], bool):
        return _skip(name, "invalid ok")
    if not isinstance(obj.get("body_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", obj.get("body_sha256")):
        return _skip(name, "invalid body_sha256")
    if not isinstance(obj.get("body_bytes"), int) or isinstance(obj.get("body_bytes"), bool):
        return _skip(name, "invalid body_bytes")
    if not isinstance(obj.get("headers"), dict):
        obj["headers"] = {}
    body = _verified_body(Path(path), obj)
    if body is None:
        return _skip(name, "body mismatch")
    try:
        parsed_body = json.loads(body)
    except Exception:
        parsed_body = None
    provider = obj.get("provider")
    expected_trunc = False
    if provider == "massive":
        nu = parsed_body.get("next_url") if isinstance(parsed_body, dict) else None
        derived_next = _massive_next_ok(nu, obj.get("ticker"))
        if nu is not None and not derived_next:
            expected_trunc = True
        elif derived_next and page is not None and page >= MAX_MASSIVE_PAGES - 1:
            expected_trunc = True
        stored_next = obj.get("has_next", False)
        stored_trunc = obj.get("truncated", False)
        if (not isinstance(stored_next, bool) or stored_next != derived_next
                or not isinstance(stored_trunc, bool) or stored_trunc != expected_trunc):
            return _skip(name, "continuation mismatch")
    envelope_ok = _envelope_rows(provider, parsed_body) is not None
    if obj.get("ok") != (envelope_ok and 200 <= obj["status"] < 300 and not expected_trunc):
        return _skip(name, "ok receipt mismatch")
    _, calc_max = _rows_and_max(provider, parsed_body)
    if calc_max != obj.get("max_updated_at"):
        return _skip(name, "max_updated_at mismatch")
    sha, n_rows = _content_digest(obj.get("provider"), body)
    if sha != obj.get("content_sha256") or n_rows != obj.get("row_count"):
        return _skip(name, "content digest mismatch")
    return obj


def _composite(key, pages):
    ordered = sorted(pages, key=lambda r: int(r.get("page") or 0))
    n = len(ordered)
    nums = [int(r.get("page") or 0) for r in ordered]
    runs = {r.get("run_id") for r in ordered}
    last = ordered[-1]
    complete = (nums == list(range(n))
                and not last.get("has_next")
                and not last.get("truncated")
                and n <= MAX_MASSIVE_PAGES
                and all(r.get("has_next") for r in ordered[:-1])
                and all(_meta_ok(r) for r in ordered)
                and len(runs) == 1
                and ordered[0].get("run_id") is not None)
    sha = hashlib.sha256("|".join(f"{int(r.get('page') or 0)}:{r.get('content_sha256')}" for r in ordered).encode("utf-8")).hexdigest()
    rcs = [r.get("row_count") for r in ordered if r.get("row_count") is not None]
    mus = [r.get("max_updated_at") for r in ordered if r.get("max_updated_at") is not None]
    sts = [r.get("status") for r in ordered if isinstance(r.get("status"), int)]
    base = dict(ordered[0])
    base.update({
        "session": key[0], "provider": key[1], "endpoint": key[2], "ticker": key[3], "label": key[4],
        "page": None, "pages": len(ordered), "content_sha256": sha,
        "row_count": sum(rcs) if rcs else None,
        "max_updated_at": max(mus) if mus else None,
        "fetched_at_utc": max(ordered, key=lambda r: _ts(r.get("fetched_at_utc")) or 0.0).get("fetched_at_utc"),
        "headers": next((r.get("headers") for r in ordered if int(r.get("page") or 0) == 0), ordered[0].get("headers")),
        "status": max(sts) if sts else 0,
        "ok": complete,
    })
    return base


def compare(args_ns):
    rule = "a refetch at T+X whose content hash equals the fetch at T, with max(updatedAt) at T earlier than T, counts as final at T"
    hdrs = ("etag", "last-modified", "age", "date", "content-length")
    out = getattr(args_ns, "out", None)
    session = getattr(args_ns, "session", None)
    if not (isinstance(out, str) and "\x00" not in out
            and (session is None or (isinstance(session, str) and "\x00" not in session))):
        _fail("invalid argument types", 2)
    if session is not None and not _valid_session(session):
        _fail(f"invalid session {session!r}: expected YYYY-MM-DD", 2)
    try:
        root = Path(out).resolve()
        root_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        _fail_path(exc)
    sessions = [session] if session else root_dirs
    metas = []
    for s in sessions:
        sdir = root / s
        if not sdir.resolve().is_relative_to(root):
            _skip(s, "outside output root")
            continue
        if not sdir.is_dir():
            continue
        for f in sorted(sdir.glob("*/*.meta.json")):
            if not f.parent.resolve().is_relative_to(root):
                _skip(f"{sdir.name}/{f.parent.name}", "outside output root")
                continue
            if not f.resolve().is_relative_to(root):
                _skip(f"{sdir.name}/{f.parent.name}/{f.name}", "outside output root")
                continue
            m = _read_meta(f, sdir.name, f.parent.name)
            if m is not None:
                metas.append(m)
    plain, by_page_label = [], {}
    for m in metas:
        if m.get("page") is None:
            plain.append(m)
        else:
            k = (m.get("session"), m.get("provider"), m.get("endpoint"), m.get("ticker"), m.get("label"))
            by_page_label.setdefault(k, []).append(m)
    metas = plain + [_composite(k, v) for k, v in by_page_label.items()]
    groups = {}
    for m in metas:
        groups.setdefault((m.get("session"), m.get("provider"), m.get("endpoint"), m.get("ticker"), m.get("page")), []).append(m)
    ff = tnf = 0
    group_trk = {}
    print("# finality compare")
    print("- evaluated headers: etag, last-modified — yes only when some pair has a content change and header-differs equals content-differs with headers present in every pair; n/a when content never changed or on composite multi-page groups (only page 0 headers kept); age, date, content-length are displayed only")
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = sorted(groups[key], key=lambda r: _ts(r.get("fetched_at_utc")) or 0.0)
        ok_rows = [r for r in rows if _meta_ok(r)]
        bad = [(r.get("label"), r.get("status")) for r in rows if not _meta_ok(r)]
        n, shas = len(ok_rows), [r.get("content_sha256") for r in ok_rows]
        session, provider, endpoint, ticker, page = key
        present = [h for h in hdrs if any((r.get("headers") or {}).get(h) is not None for r in rows)]
        has_pages = any("pages" in r for r in rows)
        cols = ["label", "ok", "fetched_at_utc", "status", "row_count", "max_updated_at", "content_sha256[:12]"]
        if has_pages:
            cols.append("pages")
        print(f"\n## {session} {provider} {endpoint} {ticker} page={page}\n| " + " | ".join(cols + present) + " |\n| " + " | ".join(["---"] * (len(cols) + len(present))) + " |")
        for r in rows:
            h, sha = r.get("headers") or {}, r.get("content_sha256")
            cells = [str(r.get("label")), "yes" if _meta_ok(r) else "no", str(r.get("fetched_at_utc")), str(r.get("status")), str(r.get("row_count")), str(r.get("max_updated_at")), sha[:12] if sha else ""]
            if has_pages:
                cells.append(f"pages={r.get('pages', '')}")
            cells.extend(str(h.get(c, "")) for c in present)
            print("| " + " | ".join(cells) + " |")
        if n <= 1:
            stable = "n/a"
        elif shas[-1] != shas[-2]:
            stable = "never"
        else:
            stable = next((str(ok_rows[i].get("label")) for i in range(n) if all(shas[i] == shas[j] for j in range(i + 1, n))), "never")
        changed = [(ok_rows[i].get("label"), ok_rows[i + 1].get("label")) for i in range(n - 1) if shas[i] != shas[i + 1]]
        uvf = []
        for r in ok_rows:
            t0, t1 = _ts(r.get("max_updated_at")), _ts(r.get("fetched_at_utc"))
            uvf.append(f"{r.get('label')}: n/a" if t0 is None or t1 is None else f"{r.get('label')}: {'yes' if t0 < t1 else 'no'} gap={t1 - t0:.3f}s")
        print(f"- stable_from: {stable}\n- changed_between: {changed}\n- updatedAt_vs_fetch: {'; '.join(uvf)}\n- Rule R: \"{rule}\"")
        print(f"- excluded_not_ok: [{', '.join(f'{lab}: {st}' for lab, st in bad)}]")
        pages_hdr = any((r.get("pages") or 0) > 1 for r in rows)
        ht = {hn: _header_tracks(ok_rows, hn, composite=pages_hdr) for hn in ("etag", "last-modified")}
        group_trk[key] = ht
        print(f"- header_tracking: etag={ht['etag']} last-modified={ht['last-modified']}")
        for i in range(n - 1):
            ti, tf = _ts(ok_rows[i].get("max_updated_at")), _ts(ok_rows[i].get("fetched_at_utc"))
            decl = shas[i] == shas[i + 1] and ti is not None and tf is not None and ti < tf
            ff_i, tnf_i = decl and any(shas[j] != shas[i] for j in range(i + 2, n)), shas[i] != shas[i + 1]
            ff += ff_i
            tnf += tnf_i
            print(f"- pair {ok_rows[i].get('label')}->{ok_rows[i + 1].get('label')}: R_declares_final_at_i={decl} false_final={ff_i} true_not_final={tnf_i}")
    print(f"\n## Overall\n- groups: {len(groups)}\n- false_final: {ff}\n- true_not_final: {tnf}")
    if not groups:
        print("- header: inconclusive (no captures)")
    else:
        for hn in ("etag", "last-modified"):
            print(f"- header {hn}: {'header tracks content' if all(group_trk[k][hn] == 'yes' for k in group_trk) else 'header does not track content'}")


def build_parser():
    parser = argparse.ArgumentParser(prog="finality_probe")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--session", required=True)
    cap.add_argument("--tickers", required=True)
    cap.add_argument("--label", required=True)
    cap.add_argument("--out", required=True)
    cap.add_argument("--orats-endpoints", default=DEFAULT_ORATS_ENDPOINTS)
    cap.add_argument("--massive", action="store_true")
    cap.add_argument("--replace", action="store_true")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--session", default=None)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    if ns.command == "capture":
        sys.exit(capture(ns))
    compare(ns)


if __name__ == "__main__":
    main()
