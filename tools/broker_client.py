"""Run a pod-tool command through the credential broker (authoring/broker.py).

An authoring job's container holds no credentials: k8s_pod.py and
radeon_pod.py hand their command line to the broker, which checks it against
the job (its own pod, target and GPU count) and runs it with the real
kubeconfig or Radeon token. The interface, output and exit codes are the
tools' own, so run.py and the agent cannot tell the difference.

NBR_BROKER_URL and NBR_JOB_KEY (the job's key, valid while it runs) come from
docker/run-job.sh.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

EXIT_MARK = "\n__NBR_EXIT__ "


def _post(path: str, body: bytes, content_type: str, timeout: int):
    req = urllib.request.Request(
        os.environ["NBR_BROKER_URL"].rstrip("/") + path, data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('NBR_JOB_KEY', '')}",
                 "Content-Type": content_type})
    return urllib.request.urlopen(req, timeout=timeout)


def _multipart(fields: dict[str, str], files: list[Path]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                   f"{value}\r\n".encode())
    for path in files:
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"push\"; "
                   f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                   .encode() + path.read_bytes() + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def forward(tool: str, argv: list[str]) -> int:
    """Run `tool argv` through the broker; returns the tool's exit code."""
    args = list(argv)
    files: list[Path] = []
    # --push names local files: send their contents, keep only the names.
    for i, a in enumerate(args[:-1]):
        if a == "--push":
            files.append(Path(args[i + 1]))
            args[i + 1] = Path(args[i + 1]).name
    pull_local = None
    if "pull" in args:
        pull_local = Path(args[-1])
        args[-1] = "-"
    body, ctype = _multipart({"argv": json.dumps(args)}, files)
    try:
        resp = _post(f"/tool/{tool}", body, ctype, timeout=4 * 3600)
    except urllib.error.HTTPError as e:
        print(f"FAILED: the credential broker refused this command "
              f"({e.code}: {e.read().decode(errors='replace')[:300]})", flush=True)
        return 2
    except OSError as e:
        print(f"FAILED: cannot reach the credential broker ({e})", flush=True)
        return 2

    if pull_local is not None:
        result = json.loads(resp.read())
        sys.stdout.write(result.get("output", ""))
        if result.get("rc") == 0 and result.get("data") is not None:
            pull_local.parent.mkdir(parents=True, exist_ok=True)
            pull_local.write_bytes(base64.b64decode(result["data"]))
        return int(result.get("rc", 1))

    # Stream the tool's output as it comes; its exit code follows a marker line.
    tail = ""
    while True:
        chunk = resp.read1(65536).decode(errors="replace") if hasattr(resp, "read1") \
            else resp.read(65536).decode(errors="replace")
        if not chunk:
            break
        tail += chunk
        keep = len(EXIT_MARK) + 8
        if len(tail) > keep:
            sys.stdout.write(tail[:-keep])
            sys.stdout.flush()
            tail = tail[-keep:]
    if EXIT_MARK in tail:
        text, code = tail.rsplit(EXIT_MARK, 1)
        sys.stdout.write(text)
        return int(code.strip() or 1)
    sys.stdout.write(tail)
    print("\nFAILED: the credential broker closed the connection", flush=True)
    return 2
