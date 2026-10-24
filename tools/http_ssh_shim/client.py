"""
Client side of the HTTP shim, shared by the `ssh` and `scp` drop-ins: one
HMAC-signed call to the agent server's /agent/exec endpoint.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

from hmac_auth import sign_request

AGENT_SECRET = os.environ.get("AGENT_SECRET", "")
AGENT_URL = os.environ.get("AGENT_URL", "").rstrip("/")
# Per-call timeout; long jobs use nohup+poll, so each exec is short.
EXEC_TIMEOUT = int(os.environ.get("AGENT_EXEC_TIMEOUT", "600"))


def fail(prog: str, msg: str, code: int = 255) -> None:
    print(f"{prog} (http-shim): {msg}", file=sys.stderr)
    sys.exit(code)


def parse_argv(prog: str, argv: list[str], flags_with_value: tuple[str, ...]) -> list[str]:
    """Drop ssh/scp flags (and the values of those that take one); return the
    remaining bare arguments. Flags configure a real SSH connection and have
    no meaning over HTTP."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in flags_with_value:
            i += 2
        elif argv[i].startswith("-"):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    if not AGENT_SECRET or not AGENT_URL:
        fail(prog, "AGENT_SECRET / AGENT_URL not set in environment")
    return out


def exec_remote(prog: str, cmd: str, stdin: bytes = b"") -> dict:
    """Run `cmd` on the GPU node; return {"exit_code", "stdout", "stderr"}."""
    body = json.dumps({"cmd": cmd, "timeout": EXEC_TIMEOUT,
                       "stdin": base64.b64encode(stdin).decode()}).encode()
    headers = {**sign_request(AGENT_SECRET, "POST", "/agent/exec", body),
               "Content-Type": "application/json"}
    req = urllib.request.Request(f"{AGENT_URL}/exec", data=body, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=EXEC_TIMEOUT + 10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        fail(prog, f"agent server rejected request (HTTP {e.code}): "
                   f"{e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        fail(prog, f"could not reach agent server at {AGENT_URL}: {e}")
    raise AssertionError("unreachable")
