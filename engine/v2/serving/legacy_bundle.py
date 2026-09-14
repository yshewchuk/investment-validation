"""The real legacy render-bundle adapter — rearchitecture phase-3 guide §5.4
(P3-4): read the bundle `engine/dashboard/render.py::render_bundle` actually
writes, not the flat `{ticker}.json` simplification `tools/v2_dashboard_
project.py` used until now.

**Layout read** (`render.py` module docstring, `_write_pair`, `render_bundle`):

    {root}/data/board.json          {root}/data/board.js
    {root}/data/tickers/{T}.json    {root}/data/tickers/{T}.js

``board.json`` is ``{"as_of", "n_rows", "rows"}`` — every scored row
(main-board **and** ladder rows concatenated, exactly as `render_bundle`
receives them; `bridge.py` tells the two apart by `strike_offset`). Each
``data/tickers/{T}.json`` is ``{"ticker", "as_of", "events", "history",
"analogs"}``; ``events`` is a list of ``{"event_date", "session", "rows"}``,
and a ticker's rows for :func:`engine.v2.serving.bridge.build_bridges` are
every event's ``rows`` flattened together — main-board and ladder rows both,
since a ladder row shares its parent's ``event_date`` and lands in the same
event group, distinguished only by a non-null ``strike_offset``.

``.js`` siblings (``window.BOARD = {...};``, `` window.TICKER_DATA["T"] =
{...};``) exist only so the bundle opens from ``file://`` where ``fetch()``
is blocked (render.py module docstring) — and it is what the live legacy
dashboard and the compatibility preview actually load in a browser. When
only one form exists this loader reads that one; when BOTH exist, it reads
and hashes both and requires their parsed payloads to agree — a bundle
whose ``.js`` disagrees with its own ``.json`` would let this loader project
something a user never sees, refused as :class:`LegacyBundleError`
``BUNDLE_FORM_MISMATCH`` rather than silently trusting whichever form
happened to be read first.

**Strict, byte-bound, no silent skips.** Every file this module actually
reads is hashed into the returned manifest (`bundle_manifest`), so a
projection binds to exact bytes rather than to "a bundle that looked right".
A symlink anywhere in a read path, a ticker string that would escape the
bundle root, an unrecognized top-level shape, two files claiming the same
ticker, a ``.js`` wrapper that does not match the one form `render.py`
emits, or a ``.json``/``.js`` pair that disagrees after parsing — each
refuses with a typed :class:`LegacyBundleError` rather than being coerced,
ignored or silently dropped. A ticker the board references with no matching
``data/tickers/`` file is refused the same way: a bundle that quietly leaves
a ticker's evidence out is not a smaller bundle, it is a wrong one.

No legacy `engine.*` import, no `engine.v2.ops` import — this module only
reads bytes and parses JSON.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

__all__ = ["LegacyBundleError", "load_legacy_bundle", "load_score_document"]


class LegacyBundleError(Exception):
    """A refused or malformed legacy render-bundle/score-document read.

    ``code`` is stable and machine-checkable (mirrors ``ArtifactError`` /
    ``engine.v2.ops.errors.fail``'s style without importing either — a peer
    package this layer may not import and a foundation type this module has
    no need of).
    """

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details or {})


# --------------------------------------------------------------------------
# the real bundle's top-level shapes (render.py, transcribed — this module
# may not import engine.dashboard.render)
# --------------------------------------------------------------------------

_BOARD_KEYS = frozenset({"as_of", "n_rows", "rows"})
_TICKER_KEYS = frozenset({"ticker", "as_of", "events", "history", "analogs"})

_JS_BOARD_PREFIX = re.compile(r"^window\.BOARD\s*=\s*")
_JS_TICKER_PREFIX = re.compile(r'window\.TICKER_DATA\["[^"]*"\]\s*=\s*')

_SCORE_DOC_REQUIRED = frozenset({"rows", "ladder", "expected_population"})
_SCORE_DOC_OPTIONAL = frozenset({
    "observed_population", "tickers", "context_tickers",
    "analog_entry_coverage", "session", "requested_session",
})
_SCORE_DOC_ALLOWED = _SCORE_DOC_REQUIRED | _SCORE_DOC_OPTIONAL


# --------------------------------------------------------------------------
# strict, hashed, symlink-refusing file reads
# --------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _assert_no_symlink(bundle_root: Path, path: Path) -> None:
    """Refuse a symlink at ``path`` or at any directory between it and the root."""
    relative = path.relative_to(bundle_root)
    probe = bundle_root
    for part in relative.parts:
        probe = probe / part
        if probe.is_symlink():
            raise LegacyBundleError(
                "SYMLINK_REFUSED", f"{probe} is a symlink; refusing to follow it",
                details={"path": str(probe.relative_to(bundle_root).as_posix())})


def _safe_ticker(ticker: Any) -> str:
    """A ticker string safe to use as exactly one path segment."""
    text = str(ticker)
    if not text or text in (".", "..") or "/" in text or "\\" in text or "\x00" in text:
        raise LegacyBundleError(
            "PATH_TRAVERSAL",
            f"ticker {text!r} is not a safe single path segment",
            details={"ticker": text})
    return text


def _exists_or_symlink(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _read_and_hash(bundle_root: Path, path: Path, manifest: dict[str, str]) -> bytes:
    """Refuse a symlink, then read+hash the exact bytes into ``manifest``."""
    _assert_no_symlink(bundle_root, path)
    if not path.is_file():
        raise LegacyBundleError("MISSING_FILE", f"{path} is missing",
                                details={"path": str(path.relative_to(bundle_root).as_posix())})
    data = path.read_bytes()
    manifest[str(path.relative_to(bundle_root).as_posix())] = _sha256_hex(data)
    return data


def _parse_json_text(text: str, path: Path) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{path}: not valid JSON") from exc


def _strip_js_wrapper(text: str, prefix: re.Pattern[str], path: Path) -> str:
    """``window.X = <json>;`` (or ``window.TICKER_DATA["T"] = <json>;``) → ``<json>``.

    The one wrapper shape `render.py` `_write_pair` / `render_bundle` emits.
    Anything else — no matching assignment, no trailing ``;`` — is a
    malformed wrapper, refused rather than guessed at.
    """
    match = prefix.search(text)
    if match is None:
        raise LegacyBundleError(
            "MALFORMED_WRAPPER",
            f"{path}: does not contain the expected window.* assignment")
    body = text[match.end():].rstrip()
    if not body.endswith(";"):
        raise LegacyBundleError(
            "MALFORMED_WRAPPER", f"{path}: JS wrapper is not terminated with ';'")
    return body[:-1]


def _require_exact_keys(payload: Any, keys: frozenset[str], path: Path) -> dict:
    if not isinstance(payload, dict):
        raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{path}: expected a JSON object")
    found = set(payload)
    if found != keys:
        raise LegacyBundleError(
            "UNKNOWN_STRUCTURE",
            f"{path}: expected exactly {sorted(keys)}, found {sorted(found)}",
            details={"path": str(path), "expected": sorted(keys), "found": sorted(found)})
    return payload


def _read_json_form(bundle_root: Path, json_path: Path, manifest: dict[str, str]) -> Any:
    data = _read_and_hash(bundle_root, json_path, manifest)
    return _parse_json_text(data.decode("utf-8"), json_path)


def _read_js_form(bundle_root: Path, js_path: Path, js_prefix: re.Pattern[str],
                  manifest: dict[str, str]) -> Any:
    data = _read_and_hash(bundle_root, js_path, manifest)
    body = _strip_js_wrapper(data.decode("utf-8"), js_prefix, js_path)
    return _parse_json_text(body, js_path)


def _matching_forms(json_path: Path, js_path: Path, json_payload: Any, js_payload: Any,
                    bundle_root: Path) -> None:
    """The browser (legacy dashboard, compatibility preview) loads the
    ``.js`` wrapper, not the ``.json`` file -- so when both exist, what this
    loader projects must equal what a user actually sees. A parsed
    disagreement is refused rather than silently trusting whichever form
    happened to be read first."""
    if json_payload == js_payload:
        return
    raise LegacyBundleError(
        "BUNDLE_FORM_MISMATCH",
        f"{json_path.name} and {js_path.name} disagree after parsing",
        details={"json_path": str(json_path.relative_to(bundle_root).as_posix()),
                 "js_path": str(js_path.relative_to(bundle_root).as_posix())})


def _load_json_or_js(bundle_root: Path, json_path: Path, js_path: Path, *,
                     allowed_keys: frozenset[str], js_prefix: re.Pattern[str],
                     manifest: dict[str, str]) -> dict:
    """Read every form that exists. A JSON-only or JS-only file keeps the
    prior behavior; when BOTH exist, both are hashed into ``manifest`` and
    their parsed payloads must match (see :func:`_matching_forms`)."""
    json_exists = _exists_or_symlink(json_path)
    js_exists = _exists_or_symlink(js_path)
    if not json_exists and not js_exists:
        raise LegacyBundleError(
            "MISSING_FILE",
            f"neither {json_path.name} nor {js_path.name} exists in "
            f"{json_path.parent.relative_to(bundle_root).as_posix()}")
    json_payload = _read_json_form(bundle_root, json_path, manifest) if json_exists else None
    js_payload = _read_js_form(bundle_root, js_path, js_prefix, manifest) if js_exists else None
    if json_exists and js_exists:
        _matching_forms(json_path, js_path, json_payload, js_payload, bundle_root)
    payload = json_payload if json_exists else js_payload
    return _require_exact_keys(payload, allowed_keys, json_path)


# --------------------------------------------------------------------------
# load_legacy_bundle — split by stage: board, directory scan, one ticker
# --------------------------------------------------------------------------


def _board_tickers(root: Path, manifest: dict[str, str]) -> set[str]:
    """Read+validate ``data/board.json`` (or its ``.js`` wrapper); the set of
    tickers it references, each checked traversal-safe."""
    board_json = root / "data" / "board.json"
    board_js = root / "data" / "board.js"
    board = _load_json_or_js(root, board_json, board_js, allowed_keys=_BOARD_KEYS,
                             js_prefix=_JS_BOARD_PREFIX, manifest=manifest)
    if not isinstance(board["rows"], list):
        raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{board_json}: 'rows' must be a list")
    tickers: set[str] = set()
    for row in board["rows"]:
        if not isinstance(row, dict) or "ticker" not in row:
            raise LegacyBundleError(
                "UNKNOWN_STRUCTURE", f"{board_json}: a board row has no 'ticker'")
        tickers.add(_safe_ticker(row["ticker"]))
    return tickers


def _ticker_file_stems(tickers_dir: Path) -> list[str]:
    """Every ``.json``/``.js`` stem under ``data/tickers/`` — a dangling
    symlink is included too, so it is refused rather than silently skipped."""
    if tickers_dir.is_symlink():
        raise LegacyBundleError("SYMLINK_REFUSED", f"{tickers_dir} is a symlink")
    if not tickers_dir.is_dir():
        return []
    stems: set[str] = set()
    for entry in tickers_dir.iterdir():
        if entry.suffix in (".json", ".js") and (entry.is_symlink() or entry.is_file()):
            stems.add(entry.stem)
    return sorted(stems)


def _flatten_ticker_rows(payload: dict, json_path: Path) -> list[dict]:
    """A ticker payload's ``events[].rows`` flattened — main-board and ladder
    rows together, exactly as :func:`engine.v2.serving.bridge.build_bridges`
    expects (told apart only by ``strike_offset``)."""
    events = payload["events"]
    if not isinstance(events, list):
        raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{json_path}: 'events' must be a list")
    rows: list[dict] = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("rows"), list):
            raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{json_path}: a malformed event entry")
        rows.extend(event["rows"])
    return rows


def _load_one_ticker(root: Path, tickers_dir: Path, stem: str,
                     manifest: dict[str, str]) -> tuple[str, Path, list[dict]]:
    """One ``data/tickers/{stem}.json``(``.js``) file → ``(ticker, source, rows)``."""
    json_path = tickers_dir / f"{stem}.json"
    js_path = tickers_dir / f"{stem}.js"
    payload = _load_json_or_js(root, json_path, js_path, allowed_keys=_TICKER_KEYS,
                               js_prefix=_JS_TICKER_PREFIX, manifest=manifest)
    ticker = _safe_ticker(payload["ticker"])
    source = json_path if _exists_or_symlink(json_path) else js_path
    return ticker, source, _flatten_ticker_rows(payload, json_path)


def _load_all_tickers(root: Path, manifest: dict[str, str]) -> dict[str, list[dict]]:
    """Every ``data/tickers/`` file, deduped by its own internal ``ticker``
    field (never the filename) — two files claiming the same ticker refuse."""
    tickers_dir = root / "data" / "tickers"
    seen: dict[str, Path] = {}
    bundle_rows_by_ticker: dict[str, list[dict]] = {}
    for stem in _ticker_file_stems(tickers_dir):
        ticker, source, rows = _load_one_ticker(root, tickers_dir, stem, manifest)
        if ticker in seen:
            raise LegacyBundleError(
                "DUPLICATE_TICKER",
                f"ticker {ticker!r} is claimed by both "
                f"{seen[ticker].relative_to(root).as_posix()} and "
                f"{source.relative_to(root).as_posix()}",
                details={"ticker": ticker})
        seen[ticker] = source
        bundle_rows_by_ticker[ticker] = rows
    return bundle_rows_by_ticker


def load_legacy_bundle(bundle_root: Path | str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Read a real `render_bundle` output tree into the shape
    :func:`engine.v2.serving.bridge.build_bridges` expects.

    Returns ``(bundle_rows_by_ticker, bundle_manifest)``:

    * ``bundle_rows_by_ticker``: ``{ticker: [row, ...]}``, every row flattened
      out of that ticker's ``data/tickers/{T}.json`` ``events[].rows`` —
      main-board and ladder rows together, exactly as they sit in the real
      bundle; the caller (``build_bridges``) tells them apart on
      ``strike_offset``.
    * ``bundle_manifest``: ``{relative_path: "sha256:<hex>"}`` for every file
      this call actually read (``data/board.json`` plus each ticker file),
      so the projection binds to exact bytes.

    Raises :class:`LegacyBundleError` on a symlink, a traversal-unsafe ticker
    string, an unrecognized top-level shape, two files claiming the same
    ticker, a malformed ``.js`` wrapper, or a ticker the board references
    with no rendered ticker file.
    """
    root = Path(bundle_root)
    if root.is_symlink():
        raise LegacyBundleError("SYMLINK_REFUSED", f"{root} is a symlink")
    manifest: dict[str, str] = {}

    board_tickers = _board_tickers(root, manifest)
    bundle_rows_by_ticker = _load_all_tickers(root, manifest)

    missing = sorted(board_tickers - set(bundle_rows_by_ticker))
    if missing:
        raise LegacyBundleError(
            "MISSING_TICKER_FILE",
            f"board references tickers with no rendered ticker file: {missing}",
            details={"missing_tickers": missing})

    return bundle_rows_by_ticker, manifest


# --------------------------------------------------------------------------
# load_score_document
# --------------------------------------------------------------------------


def load_score_document(path: Path | str) -> dict:
    """Strictly-shaped ``score.json`` read: ``rows``/``ladder``/
    ``expected_population`` present as lists, no unrecognized top-level key,
    no symlink.
    """
    path = Path(path)
    if path.is_symlink():
        raise LegacyBundleError("SYMLINK_REFUSED", f"{path} is a symlink")
    if not path.is_file():
        raise LegacyBundleError("MISSING_FILE", f"{path} is missing")
    doc = _parse_json_text(path.read_bytes().decode("utf-8"), path)
    if not isinstance(doc, dict):
        raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{path}: expected a JSON object")
    unknown = set(doc) - _SCORE_DOC_ALLOWED
    if unknown:
        raise LegacyBundleError(
            "UNKNOWN_STRUCTURE", f"{path}: unknown score-document keys {sorted(unknown)}",
            details={"unknown_keys": sorted(unknown)})
    missing = _SCORE_DOC_REQUIRED - set(doc)
    if missing:
        raise LegacyBundleError(
            "UNKNOWN_STRUCTURE", f"{path}: missing required keys {sorted(missing)}",
            details={"missing_keys": sorted(missing)})
    for key in ("rows", "ladder", "expected_population"):
        if not isinstance(doc[key], list):
            raise LegacyBundleError("UNKNOWN_STRUCTURE", f"{path}: {key!r} must be a list")
    return doc
