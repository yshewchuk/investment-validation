import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, quote_plus, urlencode

DEFAULT_ORATS_ENDPOINTS = "hist/strikes,hist/summaries"
ORATS_BASE = "https://api.orats.io/datav2/"
MASSIVE_BASE = "https://api.massive.com/v3/snapshot/options/"
MASSIVE_ORIGIN = "https://api.massive.com/"
MASSIVE_ENDPOINT = "v3/snapshot/options"
DROP_HEADERS = frozenset({"set-cookie", "authorization", "cookie"})
MAX_MASSIVE_PAGES = 20
REQUEST_TIMEOUT_S = 30
MAX_META_BYTES = 1024 * 1024
SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TICKER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENDPOINT_RE = re.compile(r"^[a-z0-9_/]+$")


def _fail(msg, code=2):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _refuse_repo(out_dir):
    path = Path(out_dir).resolve()
    for base in (path, *path.parents):
        if (base / ".git").exists() or (base / ".git").is_symlink():
            _fail("refusing to write provider data inside a repository", 2)


def _mkdirs(path):
    missing, cur = [], Path(path)
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for d in reversed(missing):
        d.mkdir(mode=0o700)
        os.chmod(d, 0o700)


def _redacted_url(url, params):
    p = dict(params or {})
    if "token" in p:
        p["token"] = "REDACTED"
    return f"{url}?{urlencode(p)}" if p else url


def _leaks_key(key, meta, body):
    if not key:
        return False
    meta_b = _canon(meta).encode("utf-8")
    return any(v.encode("utf-8") in meta_b or v.encode("utf-8") in body
               for v in (key, quote(key, safe=""), quote_plus(key)))


def _atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _drop_keys(obj, keys):
    if not keys: return obj
    if isinstance(obj, dict): return {k: _drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list): return [_drop_keys(v, keys) for v in obj]
    return obj


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _content_sha256(rows, ignore_keys):
    cleaned = [_drop_keys(r, ignore_keys) for r in rows]
    order = sorted(range(len(cleaned)), key=lambda i: _canon(cleaned[i]))
    return hashlib.sha256(_canon([cleaned[i] for i in order]).encode("utf-8")).hexdigest()


def _max_updated_orats(rows):
    vals = [r.get("updatedAt") for r in rows if isinstance(r, dict) and r.get("updatedAt") is not None]
    return max(vals) if vals else None


def _max_updated_massive(rows):
    vals = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for ck in ("day", "last_quote"):
            c = r.get(ck)
            if not isinstance(c, dict) or c.get("last_updated") is None:
                continue
            try:
                vals.append(int(c["last_updated"]))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None


def _default_http_get(req):
    import requests
    headers = dict(req.get("headers") or {})
    headers.setdefault("Accept-Encoding", "identity")
    resp = requests.get(req["url"], params=req.get("params"), headers=headers,
                        timeout=req.get("timeout", REQUEST_TIMEOUT_S), allow_redirects=False)
    return int(resp.status_code), {str(k).lower(): v for k, v in resp.headers.items()}, bytes(resp.content)


def capture(args_ns, http_get=None, now=None):
    if http_get is None: http_get = _default_http_get
    if now is None: now = lambda: datetime.now(timezone.utc)
    session, label, out_dir = args_ns.session, args_ns.label, args_ns.out
    if not SESSION_RE.match(session or ""): _fail(f"invalid session {session!r}: expected YYYY-MM-DD", 2)
    if not LABEL_RE.match(label or "") or label in (".", ".."): _fail(f"invalid label {label!r}", 2)
    tickers = [t.strip() for t in args_ns.tickers.split(",") if t.strip()]
    for t in tickers:
        if not TICKER_RE.match(t): _fail(f"invalid ticker {t!r}", 2)
    endpoints = [e.strip() for e in (getattr(args_ns, "orats_endpoints", DEFAULT_ORATS_ENDPOINTS) or "").split(",") if e.strip()]
    for e in endpoints:
        if not ENDPOINT_RE.match(e) or ".." in e: _fail(f"invalid ORATS endpoint {e!r}", 2)
    want_massive = bool(getattr(args_ns, "massive", False))
    ignore_keys = [k.strip() for k in (getattr(args_ns, "ignore_keys", "") or "").split(",") if k.strip()]
    orats_key = os.environ.get("ORATS_API_KEY") or ""
    massive_key = os.environ.get("MASSIVE_API_KEY") or ""
    if endpoints and not orats_key: _fail("missing ORATS_API_KEY", 2)
    if want_massive and not massive_key: _fail("missing MASSIVE_API_KEY", 2)
    out_base = Path(out_dir).resolve()
    unit_dir = (Path(out_dir) / session / label).resolve()
    _refuse_repo(unit_dir)
    try:
        unit_dir.relative_to(out_base)
    except ValueError:
        _fail("refusing to write outside the output directory", 2)
    _mkdirs(unit_dir)
    orats_planned = len(endpoints) * len(tickers)
    t_start = time.perf_counter()
    print(f"[capture] start units={orats_planned} label={label}", file=sys.stderr)
    total = not_ok = 0
    saw_massive = False

    def _progress(n, m=None):
        el = time.perf_counter() - t_start
        if m is None:
            print(f"[capture] {n} units elapsed={int(el)}s", file=sys.stderr)
        else:
            eta = int(el / n * (m - n)) if n else 0
            print(f"[capture] {n}/{m} units elapsed={int(el)}s eta={eta}s", file=sys.stderr)

    def store(provider, endpoint, ticker, page, req, url_redacted, key):
        nonlocal total, not_ok, saw_massive
        t0 = time.perf_counter()
        status, rhdrs, body, error = 0, {}, b"", None
        try:
            status, rhdrs, body = http_get(req)
            body, status = bytes(body), int(status)
        except Exception as exc:
            error, status, rhdrs, body = type(exc).__name__, 0, {}, b""
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        headers = {str(k).lower(): v for k, v in (rhdrs or {}).items() if str(k).lower() not in DROP_HEADERS}
        row_count = max_updated_at = content_sha = parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            rows = parsed.get("data" if provider == "orats" else "results")
            if not isinstance(rows, list): rows = []
            row_count = len(rows)
            max_updated_at = _max_updated_orats(rows) if provider == "orats" else _max_updated_massive(rows)
            content_sha = _content_sha256(rows, ignore_keys)
        truncated = False
        if provider == "massive" and page == MAX_MASSIVE_PAGES - 1:
            nu = parsed.get("next_url") if isinstance(parsed, dict) else None
            if isinstance(nu, str) and nu.startswith(MASSIVE_ORIGIN):
                truncated = True
                print(f"[capture] warning: truncated massive pagination for {ticker} after {MAX_MASSIVE_PAGES} pages", file=sys.stderr)
        ok = isinstance(parsed, dict) and 200 <= status < 300 and not truncated
        meta = {
            "provider": provider, "endpoint": endpoint, "ticker": ticker, "session": session,
            "label": label, "page": page, "url_redacted": url_redacted, "status": status,
            "ok": ok, "fetched_at_utc": now().isoformat(), "elapsed_ms": elapsed_ms,
            "headers": headers, "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_bytes": len(body), "row_count": row_count, "max_updated_at": max_updated_at,
            "content_sha256": content_sha,
        }
        if truncated: meta["truncated"] = True
        if error is not None: meta["error"] = error
        if _leaks_key(key, meta, body):
            _fail("refusing to write provider record containing API key", 2)
        stem = f"{provider}__{endpoint.replace('/', '.')}__{ticker}"
        if page is not None: stem = f"{stem}__p{page}"
        _atomic_write(unit_dir / f"{stem}.body", body)
        _atomic_write(unit_dir / f"{stem}.meta.json", (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        total += 1
        if not ok: not_ok += 1
        if provider == "massive": saw_massive = True
        c12 = content_sha[:12] if content_sha else None
        print(f"{label} {provider} {ticker} status={status} rows={row_count} max_updated_at={max_updated_at} content={c12}")
        if total % 10 == 0:
            _progress(total, None if provider == "massive" or orats_planned == 0 else orats_planned)
        return parsed

    for endpoint in endpoints:
        for ticker in tickers:
            params = {"ticker": ticker, "tradeDate": session, "token": orats_key}
            url = f"{ORATS_BASE}{endpoint}"
            req = {"url": url, "params": params, "headers": {"Accept-Encoding": "identity"}, "timeout": REQUEST_TIMEOUT_S}
            store("orats", endpoint, ticker, None, req, _redacted_url(url, params), orats_key)
    if want_massive:
        for ticker in tickers:
            url, params = f"{MASSIVE_BASE}{ticker}", {"limit": "250"}
            for page in range(MAX_MASSIVE_PAGES):
                req = {"url": url, "params": params, "headers": {"Accept-Encoding": "identity", "Authorization": f"Bearer {massive_key}"}, "timeout": REQUEST_TIMEOUT_S}
                parsed = store("massive", MASSIVE_ENDPOINT, ticker, page, req, _redacted_url(url, params), massive_key)
                next_url = parsed.get("next_url") if isinstance(parsed, dict) else None
                if not isinstance(next_url, str) or not next_url.startswith(MASSIVE_ORIGIN):
                    break
                url, params = next_url, None
    _progress(total, None if saw_massive or orats_planned == 0 else orats_planned)
    return 1 if total == 0 or not_ok else 0


def _ts(v):
    if v is None: return None
    if isinstance(v, (int, float)): x = float(v)
    else:
        s = str(v).strip()
        try: x = float(s)
        except ValueError:
            try: return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except ValueError: return None
    a = abs(x)
    return x / (1e9 if a > 1e15 else 1e3 if a > 1e11 else 1)


def _meta_ok(m):
    if not isinstance(m, dict): return False
    if "ok" in m: return bool(m["ok"])
    return m.get("status") == 200 and m.get("content_sha256") is not None


def _header_tracks(ok_rows, hn):
    if len(ok_rows) < 2: return "n/a"
    for i in range(len(ok_rows) - 1):
        h0 = (ok_rows[i].get("headers") or {}).get(hn)
        h1 = (ok_rows[i + 1].get("headers") or {}).get(hn)
        if h0 is None or h1 is None: return "no"
        c0, c1 = ok_rows[i].get("content_sha256"), ok_rows[i + 1].get("content_sha256")
        if (h0 != h1) != (c0 != c1): return "no"
    return "yes"


def _skip(name, reason):
    print(f"skipped {name}: {reason}", file=sys.stderr)
    return None


def _read_meta(path):
    name = str(path)
    try:
        st = os.lstat(path)
    except OSError as exc:
        return _skip(name, exc.strerror or exc)
    if not stat.S_ISREG(st.st_mode):
        reason = "symlink" if stat.S_ISLNK(st.st_mode) else "directory" if stat.S_ISDIR(st.st_mode) else "fifo" if stat.S_ISFIFO(st.st_mode) else "not a regular file"
        return _skip(name, reason)
    if st.st_size > MAX_META_BYTES:
        return _skip(name, "larger than 1 MiB")
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_META_BYTES + 1)
    except OSError as exc:
        return _skip(name, exc.strerror or exc)
    if len(data) > MAX_META_BYTES:
        return _skip(name, "larger than 1 MiB")
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _skip(name, f"invalid JSON: {exc}")
    if not isinstance(obj, dict):
        return _skip(name, "not a JSON object")
    return obj


def compare(args_ns):
    rule = "a refetch at T+X whose content hash equals the fetch at T, with max(updatedAt) at T earlier than T, counts as final at T"
    hdrs = ("etag", "last-modified", "age", "date", "content-length")
    root = Path(args_ns.out)
    sessions = [args_ns.session] if args_ns.session else (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else [])
    metas = []
    for s in sessions:
        sdir = root / s
        if not sdir.is_dir(): continue
        for f in sorted(sdir.glob("*/*.meta.json")):
            m = _read_meta(f)
            if m is not None: metas.append(m)
    groups = {}
    for m in metas:
        groups.setdefault((m.get("session"), m.get("provider"), m.get("endpoint"), m.get("ticker"), m.get("page")), []).append(m)
    cols = ["label", "ok", "fetched_at_utc", "status", "row_count", "max_updated_at", "content_sha256[:12]"]
    ff = tnf = 0
    group_trk = {}
    print("# finality compare")
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = sorted(groups[key], key=lambda r: _ts(r.get("fetched_at_utc")) or 0.0)
        ok_rows = [r for r in rows if _meta_ok(r)]
        bad = [(r.get("label"), r.get("status")) for r in rows if not _meta_ok(r)]
        n, shas = len(ok_rows), [r.get("content_sha256") for r in ok_rows]
        present = [h for h in hdrs if any((r.get("headers") or {}).get(h) is not None for r in rows)]
        session, provider, endpoint, ticker, page = key
        print(f"\n## {session} {provider} {endpoint} {ticker} page={page}\n| " + " | ".join(cols + present) + " |\n| " + " | ".join(["---"] * (len(cols) + len(present))) + " |")
        for r in rows:
            h, sha = r.get("headers") or {}, r.get("content_sha256")
            cells = [str(r.get("label")), "yes" if _meta_ok(r) else "no", str(r.get("fetched_at_utc")), str(r.get("status")), str(r.get("row_count")), str(r.get("max_updated_at")), sha[:12] if sha else ""] + [str(h.get(c, "")) for c in present]
            print("| " + " | ".join(cells) + " |")
        if n <= 1: stable = "n/a"
        elif shas[-1] != shas[-2]: stable = "never"
        else: stable = next((str(ok_rows[i].get("label")) for i in range(n) if all(shas[i] == shas[j] for j in range(i + 1, n))), "never")
        changed = [(ok_rows[i].get("label"), ok_rows[i + 1].get("label")) for i in range(n - 1) if shas[i] != shas[i + 1]]
        uvf = []
        for r in ok_rows:
            t0, t1 = _ts(r.get("max_updated_at")), _ts(r.get("fetched_at_utc"))
            uvf.append(f"{r.get('label')}: n/a" if t0 is None or t1 is None else f"{r.get('label')}: {'yes' if t0 < t1 else 'no'} gap={t1 - t0:.3f}s")
        print(f"- stable_from: {stable}\n- changed_between: {changed}\n- updatedAt_vs_fetch: {'; '.join(uvf)}\n- Rule R: \"{rule}\"")
        print(f"- excluded_not_ok: [{', '.join(f'{lab}: {st}' for lab, st in bad)}]")
        ht = {hn: _header_tracks(ok_rows, hn) for hn in ("etag", "last-modified")}
        group_trk[key] = ht
        print(f"- header_tracking: etag={ht['etag']} last-modified={ht['last-modified']}")
        for i in range(n - 1):
            ti, tf = _ts(ok_rows[i].get("max_updated_at")), _ts(ok_rows[i].get("fetched_at_utc"))
            decl = shas[i] == shas[i + 1] and ti is not None and tf is not None and ti < tf
            ff_i, tnf_i = decl and any(shas[j] != shas[i] for j in range(i + 2, n)), shas[i] != shas[i + 1]
            ff += ff_i; tnf += tnf_i
            print(f"- pair {ok_rows[i].get('label')}->{ok_rows[i + 1].get('label')}: R_declares_final_at_i={decl} false_final={ff_i} true_not_final={tnf_i}")
    print(f"\n## Overall\n- groups: {len(groups)}\n- false_final: {ff}\n- true_not_final: {tnf}")
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
    cap.add_argument("--ignore-keys", default="")
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
