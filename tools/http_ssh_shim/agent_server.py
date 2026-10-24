#!/usr/bin/env python3
"""
HTTP agent server — runs ON the GPU node and executes the commands the `ssh`
and `scp` shims send, for nodes the runner can reach only over HTTP(S).
Stdlib only.

  POST /agent/exec  {"cmd": str, "timeout": int, "stdin": base64}
                    -> {"exit_code", "stdout", "stderr"}

Every request must carry a valid HMAC signature (hmac_auth.py). Runs under
systemd (agent-server.service) behind Caddy's /agent/* route (Caddyfile).
"""

import base64
import binascii
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmac_auth import ReplayCache, verify_request

AGENT_SECRET = os.environ.get("AGENT_SECRET", "")
LISTEN_PORT = int(os.environ.get("AGENT_LISTEN_PORT", "8899"))
DEFAULT_TIMEOUT = 600
MAX_BODY_BYTES = 200 * 1024 * 1024  # notebook uploads arrive base64-encoded in the body

_replay_cache = ReplayCache()


def _log(msg: str) -> None:
    print(f"[agent-server] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "notebook-runner-agent/1.0"

    def log_message(self, fmt, *args):
        _log(fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "body too large"})
            return
        body = self.rfile.read(length) if length else b""

        ok, reason = verify_request(AGENT_SECRET, self.command, self.path, body,
                                    dict(self.headers), _replay_cache)
        if not ok:
            # Vague on purpose: don't tell a caller which check failed.
            _log(f"rejected {self.command} {self.path}: {reason}")
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path != "/agent/exec":
            self._send_json(404, {"error": "not found"})
            return

        try:
            payload = json.loads(body)
            cmd = payload["cmd"]
            timeout = int(payload.get("timeout", DEFAULT_TIMEOUT))
            stdin = base64.b64decode(payload.get("stdin") or "")
        except (ValueError, KeyError, TypeError, binascii.Error) as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        try:
            # bash, not sh: CLAUDE.md's `nohup … & disown` idiom needs bash's disown.
            proc = subprocess.run(cmd, shell=True, executable="/bin/bash", input=stdin,
                                  capture_output=True, timeout=timeout, check=False)
            out, err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            out, code = e.stdout or b"", 124  # timeout(1)'s exit code
            err = (e.stderr or b"") + b"\n[agent-server] command timed out"
        self._send_json(200, {"exit_code": code,
                              "stdout": out.decode(errors="replace"),
                              "stderr": err.decode(errors="replace")})


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
