"""
HMAC request signing shared by the client (client.py) and the server
(agent_server.py). Stdlib only — no new dependency for the crypto.

The shared secret never travels on the wire: each request carries a signature
bound to its method, path, body and a short time window, so a captured request
cannot be replayed or repurposed into a different command.

Wire format: three headers alongside the normal request.
    X-Agent-Timestamp: <unix seconds, int>
    X-Agent-Nonce:     <16 random bytes, hex>
    X-Agent-Signature: <hex HMAC-SHA256>

Signed string:  f"{method}\n{path}\n{timestamp}\n{nonce}\n{sha256(body).hexdigest()}"
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

HEADER_TIMESTAMP = "X-Agent-Timestamp"
HEADER_NONCE = "X-Agent-Nonce"
HEADER_SIGNATURE = "X-Agent-Signature"

# Key = AGENT_SECRET.encode() (UTF-8 of the literal string), never
# bytes.fromhex(); otherwise every signature mismatches.

# 60s tolerates normal clock skew without a large replay window.
MAX_SKEW_SECONDS = 60


def _to_sign(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return f"{method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}".encode()


def _secret_bytes(secret: str | bytes) -> bytes:
    """Single place the secret is converted to bytes."""
    return secret.encode() if isinstance(secret, str) else secret


def sign_request(secret: str | bytes, method: str, path: str, body: bytes) -> dict[str, str]:
    """Return the three headers to attach to an outgoing request."""
    key = _secret_bytes(secret)
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    msg = _to_sign(method, path, timestamp, nonce, body)
    signature = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return {
        HEADER_TIMESTAMP: timestamp,
        HEADER_NONCE: nonce,
        HEADER_SIGNATURE: signature,
    }


class ReplayCache:
    """In-memory nonce cache for replay rejection.

    No persistence — a server restart just reopens a MAX_SKEW_SECONDS replay
    window until the cache repopulates, an acceptable trade-off for a
    single-operator CI tool. Prune on every call so the dict
    never grows unbounded.
    """

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        expired = [n for n, exp in self._seen.items() if exp <= now]
        for n in expired:
            del self._seen[n]

    def check_and_record(self, nonce: str, now: float | None = None) -> bool:
        """Return True if the nonce is fresh (not seen before) and record it.
        Return False if it has already been used within the window."""
        now = now if now is not None else time.time()
        self._prune(now)
        if nonce in self._seen:
            return False
        self._seen[nonce] = now + MAX_SKEW_SECONDS
        return True


def verify_request(
    secret: str | bytes,
    method: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
    replay_cache: ReplayCache,
    now: float | None = None,
) -> tuple[bool, str]:
    """Verify an incoming request's signature, timestamp, and nonce freshness.

    Returns (ok, reason). reason is empty on success, else a short string
    for logging (never echoed back to the client in detail, to avoid
    leaking which check failed to a potential attacker).
    """
    key = _secret_bytes(secret)
    now = now if now is not None else time.time()

    timestamp = headers.get(HEADER_TIMESTAMP)
    nonce = headers.get(HEADER_NONCE)
    signature = headers.get(HEADER_SIGNATURE)
    if not timestamp or not nonce or not signature:
        return False, "missing auth headers"

    try:
        ts_val = int(timestamp)
    except ValueError:
        return False, "malformed timestamp"

    if abs(now - ts_val) > MAX_SKEW_SECONDS:
        return False, "timestamp outside allowed skew"

    expected_msg = _to_sign(method, path, timestamp, nonce, body)
    expected_sig = hmac.new(key, expected_msg, hashlib.sha256).hexdigest()
    # Constant-time compare — a timing side channel here would let an
    # attacker recover the signature byte-by-byte.
    if not hmac.compare_digest(expected_sig, signature):
        return False, "signature mismatch"

    if not replay_cache.check_and_record(nonce, now):
        return False, "nonce replay"

    return True, ""
