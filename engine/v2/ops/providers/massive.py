"""One bounded HTTPS request; only the supervised Massive worker calls this edge."""
import os
import signal
from contextlib import contextmanager

import requests

from engine.v2.ops.errors import fail

MAX_BYTES = 16 * 1024 * 1024
HTTP_DEADLINE_SECONDS = 45


@contextmanager
def _deadline():
    def expired(*_):
        raise fail("TRANSIENT_SOURCE", "Massive HTTP deadline exceeded")

    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise fail("RESOURCE_UNAVAILABLE", "Massive worker timer is unavailable")
    prior = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, HTTP_DEADLINE_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


def fetch(request, before_request):
    """No redirects/client retries; credential lookup occurs only on an admitted miss."""
    key = os.environ.get("MASSIVE_API_KEY", "")
    if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise fail("CREDENTIAL_INVALID", "Massive API credential is unavailable")
    params = {k: str(v).lower() if isinstance(v, bool) else v for k, v in request["params"].items()}
    try:
        with _deadline(), requests.Session() as session:
            session.trust_env = False
            before_request()
            with session.get("https://api.massive.com" + request["endpoint"], params=params,
                             headers={"Authorization": "Bearer " + key, "Accept-Encoding": "identity"}, stream=True,
                             allow_redirects=False, timeout=(10, 15)) as response:
                if response.status_code != 200:
                    return response.status_code, b""
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise fail("SOURCE_INVALID", "Massive response compression is unsupported")
                body = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    if len(body) + len(chunk) > MAX_BYTES:
                        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive response exceeds byte policy")
                    body.extend(chunk)
                if key.encode() in body:
                    raise fail("SOURCE_INVALID", "Massive response contains credential material")
                return 200, bytes(body)
    except requests.RequestException:
        raise fail("TRANSIENT_SOURCE", "Massive HTTP request failed") from None
