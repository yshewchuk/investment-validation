"""Strict typed documents: frozen dataclass <-> JSON-shaped mapping.

contracts §2.3: commands and registrations reject unknown fields and unknown
enum values, and every reader rejects an unsupported major schema version.
``engine/v2/contracts`` declares the shapes and may hold no logic, so the one
decoder that enforces those rules lives here, beside canonical JSON.

Decoding is driven by the dataclass's own annotations, so a contract cannot
drift from its validator: there is no second list of fields to forget.
Supported annotations are the ones the contracts use — ``str``, ``int``,
``float``, ``bool``, ``Literal[...]``, ``X | None``, ``tuple[X, ...]``,
``dict[str, X]``, ``Any`` (strict JSON only) and nested dataclasses. Anything
else is a programming error, raised as ``TypeError`` rather than accepted.

Failures carry a stable ``code`` and the JSON path of the offending value, so a
refused submission can say *which* field was wrong without echoing its value.
"""
from __future__ import annotations

import dataclasses
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

__all__ = ["DocumentError", "from_document", "parse_schema_version", "to_document"]

_VERSION = re.compile(r"^(?P<family>[a-z][a-z0-9_]*)\.v(?P<major>\d+)\.(?P<minor>\d+)$")
_SCALARS: dict[type, tuple[type, ...]] = {
    bool: (bool,), int: (int,), float: (int, float), str: (str,),
}


class DocumentError(ValueError):
    """A document that does not match its declared contract."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(f"{code} at {path}: {message}")
        self.code = code
        self.path = path


def parse_schema_version(text: Any) -> tuple[str, int, int]:
    """``"job_spec.v1.0"`` -> ``("job_spec", 1, 0)``."""
    match = _VERSION.match(text) if isinstance(text, str) else None
    if match is None:
        raise DocumentError("BAD_SCHEMA_VERSION", "$.schema_version",
                            "not of the form family.vMAJOR.MINOR")
    return match["family"], int(match["major"]), int(match["minor"])


def to_document(value: Any) -> Any:
    """A dataclass tree as plain JSON-shaped values; tuples become lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_document(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_document(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_document(v) for k, v in value.items()}
    return value


def from_document(cls: type, doc: Any, *, path: str = "$") -> Any:
    """Build ``cls`` from ``doc``, refusing anything the contract does not allow."""
    if not isinstance(doc, dict):
        raise DocumentError("BAD_TYPE", path, f"expected an object for {cls.__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(doc) - set(fields))
    if unknown:
        raise DocumentError("UNKNOWN_FIELD", f"{path}.{unknown[0]}",
                            f"{cls.__name__} declares no such field")
    _check_version(fields, doc, path)
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, spec in fields.items():
        if name in doc:
            kwargs[name] = _decode(hints[name], doc[name], f"{path}.{name}")
        elif (spec.default is dataclasses.MISSING
              and spec.default_factory is dataclasses.MISSING):
            raise DocumentError("MISSING_FIELD", f"{path}.{name}", "required")
    return cls(**kwargs)


def _check_version(fields: dict, doc: dict, path: str) -> None:
    spec = fields.get("schema_version")
    if spec is None:
        return
    if "schema_version" not in doc:
        raise DocumentError("MISSING_FIELD", f"{path}.schema_version", "required")
    family, major, minor = parse_schema_version(spec.default)
    found = parse_schema_version(doc["schema_version"])
    if found[:2] != (family, major):
        raise DocumentError("UNSUPPORTED_VERSION", f"{path}.schema_version",
                            f"reader supports {family}.v{major}.x only")
    if found[2] > minor:
        # A newer minor may carry optional fields this reader cannot interpret;
        # silently dropping them is exactly the "guess a new safety status"
        # behaviour contracts §2.3 forbids.
        raise DocumentError("UNSUPPORTED_VERSION", f"{path}.schema_version",
                            f"newer than this reader's {family}.v{major}.{minor}")


def _decode(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    if tp is Any:
        return _json_value(value, path)
    if origin is Literal:
        return _literal(tp, value, path)
    if origin in (Union, types.UnionType):
        return _union(tp, value, path)
    if dataclasses.is_dataclass(tp):
        return from_document(tp, value, path=path)
    if origin is tuple:
        return _tuple(tp, value, path)
    if origin is dict:
        return _mapping(tp, value, path)
    return _scalar(tp, value, path)


def _literal(tp: Any, value: Any, path: str) -> Any:
    for allowed in get_args(tp):
        if value == allowed and type(value) is type(allowed):
            return value
    raise DocumentError("BAD_ENUM", path, "not one of the declared values")


def _union(tp: Any, value: Any, path: str) -> Any:
    options = [a for a in get_args(tp) if a is not type(None)]
    if value is None:
        if len(options) < len(get_args(tp)):
            return None
        raise DocumentError("BAD_TYPE", path, "null is not allowed here")
    last: DocumentError | None = None
    for option in options:
        try:
            return _decode(option, value, path)
        except DocumentError as exc:
            last = exc
    assert last is not None
    raise last


def _tuple(tp: Any, value: Any, path: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise DocumentError("BAD_TYPE", path, "expected an array")
    args = get_args(tp)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_decode(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if len(args) != len(value):
        raise DocumentError("BAD_TYPE", path, f"expected {len(args)} items")
    return tuple(_decode(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))


def _mapping(tp: Any, value: Any, path: str) -> dict:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise DocumentError("BAD_TYPE", path, "expected an object with string keys")
    _, value_type = get_args(tp)
    return {k: _decode(value_type, v, f"{path}.{k}") for k, v in value.items()}


def _scalar(tp: Any, value: Any, path: str) -> Any:
    allowed = _SCALARS.get(tp)
    if allowed is None:
        raise TypeError(f"unsupported contract annotation {tp!r} at {path}")
    if (isinstance(value, bool) and tp is not bool) or not isinstance(value, allowed):
        raise DocumentError("BAD_TYPE", path, f"expected {tp.__name__}")
    if tp is float:
        if not math.isfinite(value):
            raise DocumentError("BAD_TYPE", path, "non-finite numbers are not JSON")
        return float(value)
    return value


def _json_value(value: Any, path: str) -> Any:
    """Strict JSON: no NaN, no non-string keys, no Python-only types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _scalar(float, value, path)
    if isinstance(value, (list, tuple)):
        return [_json_value(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v, f"{path}.{k}") for k, v in value.items()}
    raise DocumentError("BAD_TYPE", path, f"{type(value).__name__} is not JSON")
