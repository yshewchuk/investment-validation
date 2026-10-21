import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

DEFAULT_ORATS_ENDPOINTS = "hist/strikes,hist/summaries"
ORATS_BASE = "https://api.orats.io/datav2/"
MASSIVE_BASE = "https://api.massive.com/v3/snapshot/options/"
MASSIVE_ORIGIN = "https://api.massive.com/"
MASSIVE_ENDPOINT = "v3/snapshot/options"
DROP_HEADERS = frozenset({"set-cookie", "authorization", "cookie"})
MAX_MASSIVE_PAGES = 20
REQUEST_TIMEOUT_S = 30


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


def _redact(url, key):
    return url.replace(key, "REDACTED") if key and key in url else url


def _drop_keys(obj, keys):
    if not keys:
        return obj
    if isinstance(obj, dict):
        return {k: _drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop_keys(v, keys) for v in obj]
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
            container = r.get(ck)
            if not isinstance(container, dict) or container.get("last_updated") is None:
                continue
            try:
                vals.append(int(container["last_updated"]))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None


def _default_http_get(req):
    import requests

    headers = dict(req.get("headers") or {})
    headers.setdefault("Accept-Encoding", "identity")
    resp = requests.get(
        req["url"],
        params=req.get("params"),
        headers=headers,
        timeout=req.get("timeout", REQUEST_TIMEOUT_S),
        allow_redirects=False,
    )
    hdrs = {str(k).lower(): v for k, v in resp.headers.items()}
    return int(resp.status_code), hdrs, bytes(resp.content)


def capture(args_ns, http_get=None, now=None):
    if http_get is None:
        http_get = _default_http_get
    if now is None:
        now = lambda: datetime.now(timezone.utc)
    session, label, out_dir = args_ns.session, args_ns.label, args_ns.out
    tickers = [t.strip() for t in args_ns.tickers.split(",") if t.strip()]
    endpoints = [
        e.strip()
        for e in (getattr(args_ns, "orats_endpoints", DEFAULT_ORATS_ENDPOINTS) or "").split(",")
        if e.strip()
    ]
    want_massive = bool(getattr(args_ns, "massive", False))
    ignore_keys = [k.strip() for k in (getattr(args_ns, "ignore_keys", "") or "").split(",") if k.strip()]
    orats_key = os.environ.get("ORATS_API_KEY") or ""
    massive_key = os.environ.get("MASSIVE_API_KEY") or ""
    if endpoints and not orats_key:
        _fail("missing ORATS_API_KEY", 2)
    if want_massive and not massive_key:
        _fail("missing MASSIVE_API_KEY", 2)
    _refuse_repo(out_dir)
    unit_dir = Path(out_dir) / session / label
    _mkdirs(unit_dir)
    total = failed = 0

    def store(provider, endpoint, ticker, page, req, url_redacted, key):
        nonlocal total, failed
        t0 = time.perf_counter()
        status, rhdrs, body, error = 0, {}, b"", None
        try:
            status, rhdrs, body = http_get(req)
            body, status = bytes(body), int(status)
        except Exception as exc:
            error, status, rhdrs, body = type(exc).__name__, 0, {}, b""
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        headers = {
            str(k).lower(): v
            for k, v in (rhdrs or {}).items()
            if str(k).lower() not in DROP_HEADERS
        }
        row_count = max_updated_at = content_sha = parsed = None
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            rows = parsed.get("data" if provider == "orats" else "results")
            if not isinstance(rows, list):
                rows = []
            row_count = len(rows)
            max_updated_at = _max_updated_orats(rows) if provider == "orats" else _max_updated_massive(rows)
            content_sha = _content_sha256(rows, ignore_keys)
        meta = {
            "provider": provider,
            "endpoint": endpoint,
            "ticker": ticker,
            "session": session,
            "label": label,
            "page": page,
            "url_redacted": url_redacted,
            "status": status,
            "fetched_at_utc": now().isoformat(),
            "elapsed_ms": elapsed_ms,
            "headers": headers,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_bytes": len(body),
            "row_count": row_count,
            "max_updated_at": max_updated_at,
            "content_sha256": content_sha,
        }
        if error is not None:
            meta["error"] = error
        key_b = key.encode("utf-8") if key else b""
        if key_b and (key_b in _canon(meta).encode("utf-8") or key_b in body):
            _fail("refusing to write provider record containing API key", 2)
        stem = f"{provider}__{endpoint.replace('/', '.')}__{ticker}"
        if page is not None:
            stem = f"{stem}__p{page}"
        (unit_dir / f"{stem}.body").write_bytes(body)
        (unit_dir / f"{stem}.meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        total += 1
        if status == 0 or status >= 400:
            failed += 1
        c12 = content_sha[:12] if content_sha else None
        print(f"{label} {provider} {ticker} status={status} rows={row_count} max_updated_at={max_updated_at} content={c12}")
        return parsed

    for endpoint in endpoints:
        for ticker in tickers:
            params = {"ticker": ticker, "tradeDate": session, "token": orats_key}
            url = f"{ORATS_BASE}{endpoint}"
            req = {"url": url, "params": params, "headers": {"Accept-Encoding": "identity"}, "timeout": REQUEST_TIMEOUT_S}
            store("orats", endpoint, ticker, None, req, _redact(f"{url}?{urlencode(params)}", orats_key), orats_key)
    if want_massive:
        for ticker in tickers:
            url, params = f"{MASSIVE_BASE}{ticker}", {"limit": "250"}
            for page in range(MAX_MASSIVE_PAGES):
                req = {
                    "url": url,
                    "params": params,
                    "headers": {"Accept-Encoding": "identity", "Authorization": f"Bearer {massive_key}"},
                    "timeout": REQUEST_TIMEOUT_S,
                }
                query = f"?{urlencode(params)}" if params else ""
                parsed = store("massive", MASSIVE_ENDPOINT, ticker, page, req, _redact(f"{url}{query}", massive_key), massive_key)
                next_url = parsed.get("next_url") if isinstance(parsed, dict) else None
                if not isinstance(next_url, str) or not next_url.startswith(MASSIVE_ORIGIN):
                    break
                url, params = next_url, None
    if total == 0:
        return 1
    return 1 if failed == total else 0


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


def compare(args_ns):
    rule = "a refetch at T+X whose content hash equals the fetch at T, with max(updatedAt) at T earlier than T, counts as final at T"
    hdrs = ("etag", "last-modified", "age", "date", "content-length")
    root = Path(args_ns.out)
    sessions = [args_ns.session] if args_ns.session else (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else [])
    metas = [json.loads(f.read_text()) for s in sessions if (root / s).is_dir() for f in sorted((root / s).glob("*/*.meta.json"))]
    groups = {}
    for m in metas: groups.setdefault((m.get("provider"), m.get("endpoint"), m.get("ticker"), m.get("page")), []).append(m)
    cols = ["label", "fetched_at_utc", "status", "row_count", "max_updated_at", "content_sha256[:12]"]
    ff = tnf = 0; trk = {"etag": [True, False], "last-modified": [True, False]}
    print("# finality compare")
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = sorted(groups[key], key=lambda r: _ts(r.get("fetched_at_utc")) or 0.0)
        n, shas = len(rows), [r.get("content_sha256") for r in rows]
        present = [h for h in hdrs if any((r.get("headers") or {}).get(h) is not None for r in rows)]
        print(f"\n## {key[0]} {key[1]} {key[2]} page={key[3]}\n| " + " | ".join(cols + present) + " |\n| " + " | ".join(["---"] * (len(cols) + len(present))) + " |")
        for r in rows:
            h, sha = r.get("headers") or {}, r.get("content_sha256")
            print("| " + " | ".join([str(r.get("label")), str(r.get("fetched_at_utc")), str(r.get("status")), str(r.get("row_count")), str(r.get("max_updated_at")), sha[:12] if sha else ""] + [str(h.get(c, "")) for c in present]) + " |")
        stable = "n/a" if n == 1 else ("never" if shas[-1] != shas[-2] else next((str(rows[i].get("label")) for i in range(n) if all(shas[i] == shas[j] for j in range(i + 1, n))), "never"))
        changed = [(rows[i].get("label"), rows[i + 1].get("label")) for i in range(n - 1) if shas[i] != shas[i + 1]]
        uvf = []
        for r in rows:
            t0, t1 = _ts(r.get("max_updated_at")), _ts(r.get("fetched_at_utc"))
            uvf.append(f"{r.get('label')}: n/a" if t0 is None or t1 is None else f"{r.get('label')}: {'yes' if t0 < t1 else 'no'} gap={t1 - t0:.3f}s")
        print(f"- stable_from: {stable}\n- changed_between: {changed}\n- updatedAt_vs_fetch: {'; '.join(uvf)}\n- Rule R: \"{rule}\"")
        for i in range(n - 1):
            ti, tf = _ts(rows[i].get("max_updated_at")), _ts(rows[i].get("fetched_at_utc"))
            decl = shas[i] == shas[i + 1] and ti is not None and tf is not None and ti < tf
            ff_i, tnf_i = decl and any(shas[j] != shas[i] for j in range(i + 2, n)), shas[i] != shas[i + 1]
            ff += ff_i; tnf += tnf_i
            print(f"- pair {rows[i].get('label')}->{rows[i + 1].get('label')}: R_declares_final_at_i={decl} false_final={ff_i} true_not_final={tnf_i}")
            if shas[i] is not None and shas[i + 1] is not None:
                for hn in ("etag", "last-modified"):
                    h0, h1 = (rows[i].get("headers") or {}).get(hn), (rows[i + 1].get("headers") or {}).get(hn)
                    if h0 is None and h1 is None: continue
                    trk[hn][1] = True
                    if (h0 != h1) != (shas[i] != shas[i + 1]): trk[hn][0] = False
    print(f"\n## Overall\n- groups: {len(groups)}\n- false_final: {ff}\n- true_not_final: {tnf}")
    print("- header: " + ("header tracks content" if any(t and s for t, s in trk.values()) else "header does not track content"))


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
