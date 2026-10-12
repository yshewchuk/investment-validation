"""Synthetic HTTP-edge controls: no credentials, external sockets, or raw store."""
from __future__ import annotations

import json
import signal
import socket
import time
import traceback
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import requests

from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.providers import massive

SYNTHETIC_KEY = "synthetic-massive-key-never-a-real-credential"


def _request(adjusted=False):
    return {
        "endpoint": "/v2/aggs/ticker/AbC/range/1/day/2026-01-02/2026-01-09",
        "params": {"adjusted": adjusted, "sort": "asc", "limit": 50000},
    }


class FakeResponse:
    def __init__(self, *, status=200, headers=None, chunks=(b"synthetic body",)):
        self.status_code = status
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.chunks = chunks
        self.reads = 0
        self.entered = self.closed = False
        self.chunk_size = None

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.closed = True

    @property
    def content(self):
        pytest.fail("the edge must stream rather than eagerly read a response")

    @property
    def text(self):
        pytest.fail("response text must not be accessed")

    def json(self):
        pytest.fail("HTTP transport must not parse the response")

    def iter_content(self, chunk_size):
        self.chunk_size = chunk_size
        chunks = self.chunks() if callable(self.chunks) else self.chunks
        for chunk in chunks:
            self.reads += 1
            yield chunk


@pytest.fixture(autouse=True)
def offline_and_synthetic(monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("synthetic HTTP tests must never reach a network socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    # Replace the module's environment source outright: even fixture setup must
    # not read or save an ambient credential through monkeypatch.setenv.
    monkeypatch.setattr(massive, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": SYNTHETIC_KEY}))


@pytest.fixture(autouse=True)
def fake_deadline(monkeypatch):
    # Ordinary fake HTTP tests retain the pytest harness's SIGALRM hang guard.
    # Only the explicit real_deadline fixture below takes ownership of it.
    original = massive._deadline
    monkeypatch.setattr(massive, "_deadline", nullcontext)
    return original


@pytest.fixture
def real_deadline(monkeypatch, fake_deadline):
    monkeypatch.setattr(massive, "_deadline", fake_deadline)
    previous_handler = signal.getsignal(signal.SIGALRM)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    yield
    timer = signal.getitimer(signal.ITIMER_REAL)
    handler = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler)
    assert timer == (0.0, 0.0), "the edge left an armed timer behind"
    assert handler is previous_handler, "the edge did not restore the caller's handler"


@pytest.fixture
def http(monkeypatch):
    state = SimpleNamespace(response=FakeResponse(), events=[], sessions=[], calls=[],
                            get_error=None, get_delay=0)

    # Retain the real Requests adapter defaults so an accidentally installed
    # retry adapter is observable, while replacing every actual GET with a fake.
    class FakeSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.closed = False
            state.sessions.append(self)

        def __enter__(self):
            state.events.append("session entered")
            return self

        def get(self, url, **kwargs):
            state.events.append("GET")
            assert not self.trust_env, "proxy/netrc environment must be disabled"
            assert self.get_adapter(url).max_retries.total == 0
            state.calls.append((url, kwargs))
            if state.get_delay:
                time.sleep(state.get_delay)
            if state.get_error:
                raise state.get_error
            return state.response

        def close(self):
            self.closed = True
            super().close()

    def charge():
        state.events.append("charged")

    state.charge = charge
    monkeypatch.setattr(massive.requests, "Session", FakeSession)
    return state


@pytest.mark.parametrize(("adjusted", "wire_value"), [(False, "false"), (True, "true")])
def test_fixed_https_header_only_key_explicit_query_and_charge_order(http, adjusted, wire_value):
    request = _request(adjusted)
    serialized = json.dumps(request, sort_keys=True)

    assert massive.fetch(request, http.charge) == (200, b"synthetic body")

    assert http.events == ["session entered", "charged", "GET"]
    assert http.calls == [(
        "https://api.massive.com/v2/aggs/ticker/AbC/range/1/day/2026-01-02/2026-01-09",
        {"params": {"adjusted": wire_value, "sort": "asc", "limit": 50000},
         "headers": {"Authorization": "Bearer " + SYNTHETIC_KEY,
                     "Accept-Encoding": "identity"},
         "stream": True, "allow_redirects": False, "timeout": (10, 15)},
    )]
    assert json.dumps(request, sort_keys=True) == serialized
    assert SYNTHETIC_KEY not in serialized
    assert SYNTHETIC_KEY not in http.calls[0][0]
    assert SYNTHETIC_KEY not in json.dumps(http.calls[0][1]["params"])
    assert http.response.chunk_size == 65536
    assert http.response.closed and http.sessions[0].closed


@pytest.mark.parametrize("key", [None, "", " \t ", "synthetic\rkey", "synthetic\nkey",
                                "synthetic key", "synthetic\0key", "synthetic\u00e9key", "synthetic\ud800key"])
def test_missing_or_header_unsafe_key_refuses_before_charge_or_session(http, key):
    if key is None:
        massive.os.environ.clear()
    else:
        massive.os.environ["MASSIVE_API_KEY"] = key

    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), http.charge)

    assert caught.value.code == "CREDENTIAL_INVALID"
    assert caught.value.problem.message == "Massive API credential is unavailable"
    assert http.events == [] and http.sessions == [] and http.calls == []


def test_key_is_looked_up_at_call_time(http):
    massive.os.environ["MASSIVE_API_KEY"] = "replacement-synthetic-key"
    massive.fetch(_request(), http.charge)
    assert http.calls[0][1]["headers"]["Authorization"] == "Bearer replacement-synthetic-key"


def test_charge_refusal_prevents_get_and_closes_session(http):
    refusal = fail("LEASE_LOST", "synthetic admission refused")

    def refused_charge():
        http.events.append("charge refused")
        raise refusal

    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), refused_charge)

    assert caught.value is refusal
    assert http.events == ["session entered", "charge refused"]
    assert http.calls == []
    assert http.sessions[0].closed


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 404, 429, 500, 503])
def test_http_failures_and_redirects_return_no_body_and_make_one_attempt(http, status):
    http.response = FakeResponse(status=status, headers={"Location": "https://elsewhere.invalid/"},
                                 chunks=(SYNTHETIC_KEY.encode(),))

    assert massive.fetch(_request(), http.charge) == (status, b"")
    assert http.response.reads == 0
    assert http.events.count("charged") == len(http.calls) == 1
    assert http.response.closed and http.sessions[0].closed


@pytest.mark.parametrize("encoding", [None, "identity", "IDENTITY"])
def test_identity_stream_accepts_empty_chunks_without_truncation(http, encoding):
    headers = {} if encoding is None else {"Content-Encoding": encoding}
    http.response = FakeResponse(headers=headers, chunks=(b"", b"one", b"", b"two", b""))
    assert massive.fetch(_request(), http.charge) == (200, b"onetwo")
    assert http.response.reads == 5


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "gzip, identity", ""])
def test_compressed_or_unknown_encoding_refused_before_body_read(http, encoding):
    http.response = FakeResponse(headers={"Content-Encoding": encoding})
    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), http.charge)
    assert caught.value.code == "SOURCE_INVALID"
    assert caught.value.problem.message == "Massive response compression is unsupported"
    assert http.response.reads == 0
    assert http.response.closed and http.sessions[0].closed


@pytest.mark.parametrize("overflow", [False, True])
def test_literal_sixteen_mib_boundary_and_one_extra_byte(http, overflow):
    def chunks():
        yield b""
        for _ in range(256):
            yield b"x" * 65536
        if overflow:
            yield b"x"
            pytest.fail("oversized response continued to stream after refusal")

    http.response = FakeResponse(chunks=chunks)
    if overflow:
        with pytest.raises(OpsError) as caught:
            massive.fetch(_request(), http.charge)
        assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert http.response.reads == 258
    else:
        status, body = massive.fetch(_request(), http.charge)
        assert status == 200
        assert len(body) == 16777216
        assert body.count(b"x") == 16777216
        assert http.response.reads == 257
    assert http.response.closed and http.sessions[0].closed


@pytest.mark.parametrize("phase", ["get", "stream"])
@pytest.mark.parametrize("exception", [requests.ConnectionError, requests.Timeout,
                                       requests.exceptions.ChunkedEncodingError])
def test_requests_exceptions_are_redacted_and_never_retried(http, phase, exception):
    error = exception("https://private.invalid/?apiKey=" + SYNTHETIC_KEY)
    if phase == "get":
        http.get_error = error
    else:
        def failed_chunks():
            yield b"partial bytes"
            raise error

        http.response = FakeResponse(chunks=failed_chunks)

    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), http.charge)

    assert caught.value.code == "TRANSIENT_SOURCE"
    assert caught.value.problem.message == "Massive HTTP request failed"
    rendered = "".join(traceback.format_exception(caught.value))
    assert SYNTHETIC_KEY not in rendered and "private.invalid" not in rendered
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert http.events.count("charged") == len(http.calls) == 1
    assert http.sessions[0].closed
    if phase == "stream":
        assert http.response.closed


def test_credential_echo_across_chunk_boundary_is_refused(http):
    key = SYNTHETIC_KEY.encode()
    http.response = FakeResponse(chunks=(b"prefix", key[:13], b"", key[13:], b"suffix"))
    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), http.charge)
    assert caught.value.code == "SOURCE_INVALID"
    assert caught.value.problem.message == "Massive response contains credential material"
    assert SYNTHETIC_KEY not in str(caught.value)
    assert http.response.closed and http.sessions[0].closed


# These focused timer tests own SIGALRM, so only they disable the competing
# pytest alarm. Their explicit fixture verifies timer and handler restoration.
@pytest.mark.test_timeout(0)
@pytest.mark.parametrize("fail_inside", [False, True])
def test_real_deadline_arms_timer_and_restores_on_exit(real_deadline, fail_inside):
    expected = pytest.raises(RuntimeError, match="synthetic failure") if fail_inside else nullcontext()
    with expected, massive._deadline():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining <= massive.HTTP_DEADLINE_SECONDS
        assert interval == 0
        if fail_inside:
            raise RuntimeError("synthetic failure")


@pytest.mark.test_timeout(0)
@pytest.mark.parametrize("phase", ["get", "stream"])
def test_real_total_deadline_interrupts_delayed_fake_and_restores_timer(
    http, monkeypatch, real_deadline, phase,
):
    assert massive.HTTP_DEADLINE_SECONDS == 45
    monkeypatch.setattr(massive, "HTTP_DEADLINE_SECONDS", 0.06)
    if phase == "get":
        http.get_delay = 0.5
    else:
        def slow_chunks():
            for _ in range(20):
                time.sleep(0.015)
                yield b"x"

        http.response = FakeResponse(chunks=slow_chunks)

    with pytest.raises(OpsError) as caught:
        massive.fetch(_request(), http.charge)

    assert caught.value.code == "TRANSIENT_SOURCE"
    assert caught.value.problem.message == "Massive HTTP deadline exceeded"
    assert http.events.count("charged") == len(http.calls) == 1
    assert http.sessions[0].closed
    if phase == "stream":
        assert http.response.reads < 20
        assert http.response.closed


@pytest.mark.test_timeout(0)
def test_preexisting_timer_refuses_without_overwriting_it_or_charging(http, real_deadline):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        with pytest.raises(OpsError) as caught:
            massive.fetch(_request(), http.charge)
        assert caught.value.code == "RESOURCE_UNAVAILABLE"
        assert signal.getsignal(signal.SIGALRM) is previous_handler
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 30
        assert http.events == [] and http.sessions == [] and http.calls == []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
