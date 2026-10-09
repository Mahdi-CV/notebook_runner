#!/usr/bin/env python3
"""
Minimal HTTP agent server — runs ON the GPU node, replaces the "ssh host
'cmd'" / "scp file host:path" operations that AMD's egress won't carry
(SSH silently dropped after handshake, TLS refused outright for this IP
range — see docs/PLAN-http-agent-shim.md). Stdlib only.

Endpoints (all require valid HMAC signing — see hmac_auth.py):
  POST /agent/exec    {"cmd": str, "timeout": int} -> {"exit_code","stdout","stderr"}
      Direct equivalent of `ssh host 'cmd'`. Runs via subprocess with a
      shell, exactly as a remote SSH command would.
  POST /agent/upload   body: JSON header line {"path","len"} + raw bytes.
      Equivalent of `scp local host:path`.
  GET  /agent/download?path=<abs path>  -> raw file bytes.
      Equivalent of `scp host:path local` — used by the fixer's Step 4c
      to retrieve <nb>_fixed.ipynb before cleanup deletes the remote copy
      (CLAUDE_FIX.md).

This process should run under systemd (agent-server.service) and be
reverse-proxied by the existing Caddy on :80 at the /agent/ path prefix —
see tools/http_ssh_shim/Caddyfile and tools/http_ssh_shim/agent-server.service.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmac_auth import ReplayCache, verify_request  # noqa: E402

AGENT_SECRET = os.environ.get("AGENT_SECRET", "")  # kept as str — hmac_auth normalizes
LISTEN_PORT = int(os.environ.get("AGENT_LISTEN_PORT", "8899"))
DEFAULT_TIMEOUT = 600  # seconds — generous ceiling; callers pass their own
MAX_BODY_BYTES = 200 * 1024 * 1024  # 200MB — generous for notebook uploads

_replay_cache = ReplayCache()


def _log(msg: str) -> None:
    print(f"[agent-server] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "notebook-runner-agent/1.0"

    def log_message(self, fmt, *args):  # noqa: A003 — quiet default access log
        _log(fmt % args)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_BODY_BYTES:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _verify(self, body: bytes) -> bool:
        if not AGENT_SECRET:
            # Fail closed — a server with no configured secret must refuse
            # everything, not silently accept unauthenticated requests.
            self._send_json(500, {"error": "server misconfigured: AGENT_SECRET unset"})
            return False
        ok, reason = verify_request(
            AGENT_SECRET, self.command, self.path, body, dict(self.headers), _replay_cache
        )
        if not ok:
            _log(f"rejected {self.command} {self.path}: {reason}")
            # Deliberately vague to the client — don't help an attacker
            # narrow down which check failed.
            self._send_json(401, {"error": "unauthorized"})
        return ok

    def do_GET(self):
        # GET requests carry no body; sign/verify against an empty body so
        # the same HMAC scheme covers this endpoint too (method+path are
        # still bound into the signature, so the query string can't be
        # tampered with in transit either).
        parsed_path = self.path
        if not self._verify(b""):
            return
        if parsed_path.startswith("/agent/download"):
            self._handle_download(parsed_path)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_download(self, path_with_query: str) -> None:
        from urllib.parse import urlparse, parse_qs

        query = parse_qs(urlparse(path_with_query).query)
        candidates = query.get("path")
        if not candidates:
            self._send_json(400, {"error": "missing path query parameter"})
            return
        src = Path(candidates[0])
        if not src.is_file():
            self._send_json(404, {"error": f"no such file: {src}"})
            return
        try:
            data = src.read_bytes()
        except OSError as e:
            self._send_json(500, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            body = self._read_body()
        except ValueError as e:
            self._send_json(413, {"error": str(e)})
            return

        if not self._verify(body):
            return

        if self.path == "/agent/exec":
            self._handle_exec(body)
        elif self.path == "/agent/upload":
            self._handle_upload(body)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_exec(self, body: bytes) -> None:
        try:
            payload = json.loads(body)
            cmd = payload["cmd"]
            timeout = int(payload.get("timeout", DEFAULT_TIMEOUT))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        try:
            # Explicit executable=/bin/bash — a real SSH session runs remote
            # commands through the login shell (bash on this deployment), but
            # subprocess.run(shell=True) always uses /bin/sh regardless of
            # $SHELL. That gap is invisible for ordinary commands but breaks
            # CLAUDE.md's own `nohup ... & disown` backgrounding idiom (POSIX
            # sh has no `disown` builtin) — caught by the background/poll
            # verification test during implementation.
            proc = subprocess.run(
                cmd, shell=True, executable="/bin/bash",
                capture_output=True, timeout=timeout,
            )
            self._send_json(200, {
                "exit_code": proc.returncode,
                "stdout": proc.stdout.decode(errors="replace"),
                "stderr": proc.stderr.decode(errors="replace"),
            })
        except subprocess.TimeoutExpired as e:
            self._send_json(200, {
                "exit_code": 124,  # matches the conventional shell timeout(1) exit code
                "stdout": (e.stdout or b"").decode(errors="replace"),
                "stderr": (e.stderr or b"").decode(errors="replace") + "\n[agent-server] command timed out",
            })

    def _handle_upload(self, body: bytes) -> None:
        # Simple framing (not real multipart) — the shim controls both
        # sides, so a minimal length-prefixed-JSON-header + raw-bytes
        # framing avoids pulling in a multipart parser for one use case.
        # Header line: JSON {"path": "...", "len": N} + b"\n", then N raw bytes.
        try:
            newline = body.index(b"\n")
            header = json.loads(body[:newline])
            dest_path = header["path"]
            content = body[newline + 1:]
            if len(content) != header["len"]:
                raise ValueError("length mismatch")
        except Exception as e:
            self._send_json(400, {"error": f"bad upload framing: {e}"})
            return

        try:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            self._send_json(200, {"written": len(content), "path": str(dest)})
        except OSError as e:
            self._send_json(500, {"error": str(e)})


def main() -> int:
    if not AGENT_SECRET:
        _log("FATAL: AGENT_SECRET is not set — refusing to start")
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    _log(f"listening on 127.0.0.1:{LISTEN_PORT} (expect Caddy to proxy /agent/ here)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
