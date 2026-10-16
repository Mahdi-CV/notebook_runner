"""
Provision and drive a Radeon Global instance for the Radeon runner.

Radeon Global exposes no shell and no cluster access: only its Service API
(launch / status / browser handoff / delete) and the JupyterLab inside each
instance. This tool drives that JupyterLab the way a browser would — the
handoff ticket is exchanged for a cookie, and commands run in a kernel through
`subprocess` — and gives run.py and the agent the same interface as
tools/k8s_pod.py. The protocol details (handoff, Front Door header rules,
kernel WebSocket) follow jupyterServer's tests/radeon_notebook_runner.py.

Usage:
  radeon_pod.py --target NAME up   --pod P --gpus N [--memory-gb G] [--env K=V ...]
                                   [--push FILE ...] [--image REPO[:TAG]]
  radeon_pod.py --target NAME exec --pod P '<shell command>'
  radeon_pod.py --target NAME pull --pod P /workspace/x.ipynb local/x.ipynb
  radeon_pod.py --target NAME down --pod P

The API allows one active instance per user name, so `up` deletes whatever
instance the target's user has before launching. `up` is idempotent and
returns within UP_WAIT_SECONDS; its last line is READY (exit 0),
"Waiting for pod: <reason>" (exit 3: run it again), FAILED: <reason> (exit 2)
or UNSCHEDULABLE: <reason> (exit 4).

/workspace in commands and paths is the instance's work directory. Files move
through `nbr-xfer/` under the Jupyter root (the only place the Contents API can
reach): pushes are uploaded there and copied in, pulls copied out and downloaded. Only catalog images can run there, so --image
names a Docker repository that the target maps to one (`images:`), falling back
to its default `image`. Needs RADEON_CLOUD_API_TOKEN in the environment.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "logs" / ".radeon"
API_URL = os.environ.get("RADEON_CLOUD_API_URL", "https://radeon-global.anruicloud.com").rstrip("/")
WORKSPACE = "/workspace"
UP_WAIT_SECONDS = 480
GONE_WAIT_SECONDS = 180
SETTLE_SECONDS = 30  # after delete, before the user name can launch again
UNSCHEDULABLE_GRACE = 300
LAUNCH_GRACE_SECONDS = 180  # an accepted launch can read not_found this long
EXEC_MARKER = "__NBR_EXIT__="
EXIT_READY, EXIT_FAILED, EXIT_WAITING, EXIT_UNSCHEDULABLE = 0, 2, 3, 4

# The API's certificate chains to an issuer the system store trusts but
# requests' bundled list may not; prefer the system store when there is one.
_SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"
if os.path.exists(_SYSTEM_CA):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _SYSTEM_CA)


class Failed(RuntimeError):
    pass


# ── Service API ───────────────────────────────────────────────────────────────

class Service:
    def __init__(self, target: dict):
        self.user = target["user_name"]
        self.target = target
        token = os.environ.get("RADEON_CLOUD_API_TOKEN")
        if not token:
            raise Failed("RADEON_CLOUD_API_TOKEN is not set")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.params = {"user_name": self.user, "uid": self.user}

    def call(self, method: str, path: str, **kw) -> requests.Response:
        return requests.request(method, f"{API_URL}/api/service/{path}",
                                headers={**self.headers, **kw.pop("headers", {})},
                                timeout=kw.pop("timeout", 60), **kw)

    def status(self) -> dict:
        r = self.call("GET", "notebooks/current", params=self.params)
        return {"status": "not_found"} if r.status_code == 404 else r.json()

    def delete_and_wait(self) -> None:
        """Delete the user's instance (whoever launched it) and wait until gone."""
        if self.status().get("status") in ("not_found", None):
            return
        self.call("DELETE", "notebooks/current", params=self.params)
        deadline = time.time() + GONE_WAIT_SECONDS
        while time.time() < deadline:
            time.sleep(5)
            if self.status().get("status") in ("not_found", None):
                time.sleep(SETTLE_SECONDS)
                return
        raise Failed(f"instance for {self.user} still present {GONE_WAIT_SECONDS}s after delete")

    def template(self, gpus: int, memory_gb: int) -> str:
        """Smallest catalog template with exactly `gpus` GPUs and enough memory."""
        catalog = self.call("GET", "resource-catalog").json()
        fits = []
        for t in catalog.get("resource_templates", []):
            limits = t.get("resources", {}).get("limits", {})
            mem = int(re.sub(r"\D", "", str(limits.get("memory", "0"))) or 0)
            if int(limits.get("amd.com/gpu", 0)) == gpus and mem >= memory_gb:
                fits.append((mem, t["id"]))
        if not fits:
            raise Failed(f"no resource template with {gpus} GPU(s) and >= {memory_gb}Gi")
        return min(fits)[1]

    def launch(self, template: str, key: str, image: str | None) -> None:
        t = self.target
        body = {"user_name": self.user, "pod_type": t.get("pod_type", "workshop"),
                "resource_template": template,
                "notebook_path": t.get("notebook_repo", "https://github.com/ROCm/gpuaidev.git"),
                "unlimited_credits": True}
        if t.get("repo_subpath", "docs/notebooks"):
            body["repo_sub_path"] = t.get("repo_subpath", "docs/notebooks")
        if image:
            body["image"] = image
        r = self.call("POST", "notebooks", params={"uid": self.user}, json=body, timeout=120,
                      headers={"Idempotency-Key": key})
        if r.status_code not in (200, 201, 202):
            raise Failed(f"launch rejected: HTTP {r.status_code} — {r.text[:300]}")

    def handoff(self) -> dict:
        """One-time browser link: {url: .../_auth/start#ticket=…, direct_url: lab URL}."""
        for _ in range(10):
            r = self.call("POST", "notebooks/current/open", params=self.params)
            if r.status_code in (200, 201) and r.json().get("url") and r.json().get("direct_url"):
                return r.json()
            time.sleep(3)
        raise Failed(f"no browser handoff: HTTP {r.status_code} — {r.text[:200]}")


# ── Jupyter inside the instance ───────────────────────────────────────────────

class Lab:
    """Cookie session to the instance's Jupyter server, from a handoff."""

    def __init__(self, state: dict):
        self.state = state
        self.http = requests.Session()
        self.http.cookies.update(state.get("cookies", {}))

    @classmethod
    def authenticate(cls, state: dict, handoff: dict) -> "Lab":
        open_url, direct = urlparse(handoff["url"]), urlparse(handoff["direct_url"])
        ticket = open_url.fragment.removeprefix("ticket=")
        path = direct.path
        root = path[: path.index("/lab")] if "/lab" in path else path.rsplit("/", 1)[0]
        state.update(origin=f"{open_url.scheme}://{open_url.netloc}",
                     referer=handoff["direct_url"],
                     http_base=f"{direct.scheme}://{direct.netloc}{root.rstrip('/')}",
                     ws_base=f"{'wss' if direct.scheme == 'https' else 'ws'}://"
                             f"{direct.netloc}{root.rstrip('/')}")
        lab = cls(state)
        # The browser's JS POSTs the fragment's ticket to /_auth/start for a cookie.
        r = lab.http.post(f"{state['origin']}/_auth/start", data={"ticket": ticket}, timeout=40)
        if r.status_code not in (200, 302) or not lab.http.cookies:
            raise Failed(f"ticket exchange failed: HTTP {r.status_code}")
        state["cookies"] = lab.http.cookies.get_dict()
        deadline = time.time() + 300
        while lab.http.get(lab.api("contents"), timeout=30).status_code != 200:
            if time.time() > deadline:
                raise Failed("Jupyter API not reachable 300s after ready")
            time.sleep(4)
        return lab

    def api(self, path: str) -> str:
        return f"{self.state['http_base']}/api/{path.lstrip('/')}"

    def write_headers(self) -> dict:
        # The Front Door rejects REST writes without same-origin Origin + Referer.
        return {"Origin": self.state["origin"], "Referer": self.state["referer"]}

    def new_kernel(self) -> str:
        r = self.http.post(self.api("kernels"), json={"name": "python3"},
                           headers=self.write_headers(), timeout=60)
        if r.status_code not in (200, 201):
            raise Failed(f"kernel create failed: HTTP {r.status_code} — {r.text[:200]}")
        return r.json()["id"]

    def put_file(self, rel_path: str, data: bytes) -> None:
        r = self.http.put(self.api(f"contents/{rel_path}"), headers=self.write_headers(),
                          json={"type": "file", "format": "base64",
                                "content": base64.b64encode(data).decode()}, timeout=300)
        if r.status_code not in (200, 201):
            raise Failed(f"upload of {rel_path} failed: HTTP {r.status_code}")

    def get_file(self, rel_path: str) -> bytes:
        r = self.http.get(self.api(f"contents/{rel_path}"),
                          params={"type": "file", "format": "base64", "content": 1}, timeout=300)
        if r.status_code != 200:
            raise Failed(f"download of {rel_path} failed: HTTP {r.status_code}")
        return base64.b64decode(r.json()["content"])

    def run(self, code: str, out=sys.stdout, err=sys.stderr) -> str | None:
        """Execute `code` in the control kernel, streaming its output. Returns
        the kernel error (ename: evalue), or None."""
        session = uuid.uuid4().hex
        cookie = "; ".join(f"{k}={v}" for k, v in self.http.cookies.get_dict().items())
        # Kernel WebSocket: cookie yes, Origin no (the Front Door rejects it there).
        ws = websocket.create_connection(
            f"{self.state['ws_base']}/api/kernels/{self.state['kernel_id']}/channels"
            f"?session_id={session}", header=[f"Cookie: {cookie}"], timeout=60)
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "notebook-runner", "session": session,
                       "msg_type": "execute_request", "version": "5.3",
                       "date": datetime.now(timezone.utc).isoformat()},
            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}}))
        error, replied, idle = None, False, False
        ws.settimeout(30)
        try:
            while not (replied and idle):
                try:
                    reply = json.loads(ws.recv() or "{}")
                except websocket.WebSocketTimeoutException:
                    ws.ping()  # proxies reap silent sockets during long commands
                    continue
                if (reply.get("parent_header") or {}).get("msg_id") != msg_id:
                    continue
                kind, content = reply.get("msg_type"), reply.get("content") or {}
                if kind == "stream":
                    stream = err if content.get("name") == "stderr" else out
                    stream.write(content.get("text", ""))
                    stream.flush()
                elif kind == "error":
                    error = f"{content.get('ename')}: {content.get('evalue')}"
                elif kind == "execute_reply":
                    replied = True
                elif kind == "status" and content.get("execution_state") == "idle":
                    idle = True
        finally:
            ws.close()
        return error


# ── State and commands ────────────────────────────────────────────────────────

def state_path(target: dict) -> Path:
    return STATE_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', target['user_name'])}.json"


def load_state(target: dict) -> dict:
    p = state_path(target)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(target: dict, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = state_path(target)
    p.write_text(json.dumps(state))
    p.chmod(0o600)  # holds the instance's session cookie


def connect(target: dict, pod: str) -> tuple[Lab, dict]:
    """The Lab for `pod`, re-authenticating if the cookie has expired."""
    state = load_state(target)
    if state.get("pod") != pod or not state.get("kernel_id"):
        raise Failed(f"pod {pod} is not up (run up first)")
    lab = Lab(state)
    if lab.http.get(lab.api("contents"), timeout=30).status_code in (401, 403):
        lab = Lab.authenticate(state, Service(target).handoff())
        save_state(target, state)
    return lab, state


def shell(lab: Lab, state: dict, command: str, quiet: bool = False) -> int:
    """Run `command` with bash in the work directory; returns its exit code."""
    env = {**state.get("target_env", {}), **state.get("env", {}),
           **({"HF_TOKEN": os.environ["HF_TOKEN"]}
                                     if os.environ.get("HF_TOKEN") else {})}
    code = (
        "import os, subprocess, sys\n"
        f"_p = subprocess.Popen(['bash', '-lc', {json.dumps(command)}], cwd={json.dumps(WORKSPACE)},\n"
        f"    env={{**os.environ, **{json.dumps(env)}}}, stdout=subprocess.PIPE,\n"
        "    stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "_last = '\\n'\n"
        "for _line in _p.stdout:\n"
        "    print(_line, end='', flush=True)\n"
        "    _last = _line[-1:]\n"
        f"print(('' if _last == '\\n' else '\\n') + '{EXEC_MARKER}' + str(_p.wait()), flush=True)\n")

    class Capture:
        def __init__(self):
            self.buf, self.rc = "", None

        def write(self, text):
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line.startswith(EXEC_MARKER):
                    self.rc = int(line[len(EXEC_MARKER):])
                elif not quiet:
                    sys.stdout.write(line + "\n")

        def flush(self):
            sys.stdout.flush()

    cap = Capture()
    error = lab.run(code, out=cap)
    if cap.buf and not cap.buf.startswith(EXEC_MARKER) and not quiet:
        sys.stdout.write(cap.buf)
    if error:
        print(f"radeon_pod: kernel error: {error}", file=sys.stderr)
        return 255
    return cap.rc if cap.rc is not None else 255


def cmd_up(target: dict, args) -> int:
    svc = Service(target)
    state = load_state(target)
    if state.get("pod") != args.pod:
        # One instance per user: clear whatever is there (ours or not), then launch.
        svc.delete_and_wait()
        state = {"pod": args.pod, "launched_at": time.time(),
                 # The target's env (e.g. PIP_BREAK_SYSTEM_PACKAGES on images whose
                 # system Python is externally managed), then the notebook's.
                 "target_env": {k: str(v) for k, v in (target.get("env") or {}).items()},
                 "env": dict(e.split("=", 1) for e in args.env or [])}
        repo = (args.image or "").rsplit(":", 1)[0] if "/" in (args.image or "") else args.image
        image = (target.get("images") or {}).get(repo) or target.get("image")
        # A fresh key per launch: one derived from the pod name would make the API
        # replay an earlier, already-deleted launch of the same pod name.
        state["launch_key"] = uuid.uuid4().hex
        save_state(target, state)
        svc.launch(svc.template(args.gpus, args.memory_gb), image=image, key=state["launch_key"])
        save_state(target, state)
        print(f"launched instance for {target['user_name']} ({args.gpus} GPU, "
              f"image {image or 'pool default'})", flush=True)

    if not state.get("kernel_id"):
        deadline = time.time() + UP_WAIT_SECONDS
        while True:
            st = svc.status()
            if st.get("ready") or st.get("status") == "ready":
                break
            if st.get("status") in ("failed", "error"):
                print(f"FAILED: instance {st.get('status')}: {st.get('reason')} {st.get('detail')}")
                return EXIT_FAILED
            if st.get("reason"):
                state["last_reason"] = f"{st.get('reason')}: {st.get('detail') or ''}".strip()
                save_state(target, state)
            if st.get("status") == "not_found":
                # Radeon Global reaps an instance that stays unschedulable (no free
                # GPU in the pool) after about 30 seconds; say so, not just "gone".
                if state.get("last_reason", "").startswith("Unschedulable"):
                    print(f"UNSCHEDULABLE: {state['last_reason']}")
                    return EXIT_UNSCHEDULABLE
                # Right after a launch is accepted the instance may not exist yet.
                if time.time() - state["launched_at"] > LAUNCH_GRACE_SECONDS:
                    print("FAILED: instance disappeared before it was ready")
                    return EXIT_FAILED
            if (st.get("reason") == "Unschedulable"
                    and time.time() - state["launched_at"] > UNSCHEDULABLE_GRACE):
                print(f"UNSCHEDULABLE: {st.get('detail') or st.get('reason')}")
                return EXIT_UNSCHEDULABLE
            if time.time() >= deadline:
                print(f"Waiting for pod: {st.get('status')} {st.get('reason') or ''}".strip())
                return EXIT_WAITING
            time.sleep(5)
        lab = Lab.authenticate(state, svc.handoff())
        state["kernel_id"] = lab.new_kernel()
        root = bootstrap(lab)
        if root is None:
            print("FAILED: could not set up the /workspace work directory")
            return EXIT_FAILED
        state["xfer"] = f"{root}/{XFER}"
        save_state(target, state)
    lab, state = connect(target, args.pod)
    for path in args.push or []:
        name = Path(path).name
        lab.put_file(f"{XFER}/{name}", Path(path).read_bytes())
        if shell(lab, state, f"cp {state['xfer']}/{name} {WORKSPACE}/{name}", quiet=True):
            print(f"FAILED: could not copy {name} into {WORKSPACE}")
            return EXIT_FAILED
    print("READY", flush=True)
    return EXIT_READY


XFER = "nbr-xfer"


def bootstrap(lab: Lab) -> str | None:
    """Create /workspace and the transfer folder; return the Jupyter root's
    absolute path, found by writing a probe file through the Contents API."""
    probe = f"nbr-probe-{uuid.uuid4().hex}.txt"  # Jupyter refuses hidden names
    lab.put_file(probe, b"")
    out = Collect()
    error = lab.run(
        "import os, subprocess\n"
        f"_probe = {probe!r}\n"
        "_roots = [d for d in (os.getcwd(), os.path.expanduser('~'), '/workspace', '/root', '/')\n"
        "          if os.path.exists(os.path.join(d, _probe))]\n"
        "assert _roots, 'Jupyter root not found'\n"
        "os.remove(os.path.join(_roots[0], _probe))\n"
        f"os.makedirs(os.path.join(_roots[0], {XFER!r}), exist_ok=True)\n"
        "subprocess.run('mkdir -p /workspace 2>/dev/null || sudo mkdir -p -m 777 /workspace',"
        " shell=True)\n"
        "assert os.access('/workspace', os.W_OK), '/workspace is not writable'\n"
        "print('ROOT=' + os.path.realpath(_roots[0]))\n", out=out)
    match = re.search(r"^ROOT=(.*)$", out.text, re.MULTILINE)
    return None if error or not match else match.group(1)


class Collect:
    def __init__(self):
        self.text = ""

    def write(self, text):
        self.text += text

    def flush(self):
        pass


def cmd_exec(target: dict, args) -> int:
    lab, state = connect(target, args.pod)
    return shell(lab, state, args.command)


def cmd_pull(target: dict, args) -> int:
    lab, state = connect(target, args.pod)
    remote = args.remote
    if not remote.startswith(WORKSPACE + "/"):
        print(f"radeon_pod: pull only reads under {WORKSPACE}/", file=sys.stderr)
        return 2
    name = f"{uuid.uuid4().hex}-{Path(remote).name}"
    if shell(lab, state, f"cp {remote} {state['xfer']}/{name}", quiet=True):
        print(f"radeon_pod: no such file: {remote}", file=sys.stderr)
        return 1
    try:
        data = lab.get_file(f"{XFER}/{name}")
        shell(lab, state, f"rm -f {state['xfer']}/{name}", quiet=True)
    except Failed as e:
        print(f"radeon_pod: {e}", file=sys.stderr)
        return 1
    Path(args.local).parent.mkdir(parents=True, exist_ok=True)
    Path(args.local).write_bytes(data)
    return 0


def down(target: dict, pod: str) -> int:
    """Delete the instance this tool launched for `pod`."""
    state = load_state(target)
    if state.get("pod") not in (None, pod):
        return 0  # a later run owns the user's instance now
    Service(target).delete_and_wait()
    state_path(target).unlink(missing_ok=True)
    return 0


def load_target(name: str) -> dict:
    target = routing.load_config()["targets"].get(name)
    if not target:
        raise SystemExit(f"error: target '{name}' not found in {routing.CONFIG_PATH}")
    if target.get("type") != "radeon" or not target.get("user_name"):
        raise SystemExit(f"error: target '{name}' is not type: radeon with a user_name")
    return target


def main() -> int:
    p = argparse.ArgumentParser(description="Radeon Global instances for the Radeon runner.")
    p.add_argument("--target", required=True, help="runner.yaml target (type: radeon)")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up")
    p_up.add_argument("--pod", required=True)
    p_up.add_argument("--gpus", type=int, default=1)
    p_up.add_argument("--memory-gb", type=int, default=55)
    p_up.add_argument("--env", action="append", help="K=V to set for every command")
    p_up.add_argument("--push", action="append", help="local file to copy into /workspace")
    p_up.add_argument("--image", help="Docker repository (tag ignored), mapped by the target's images:")
    p_exec = sub.add_parser("exec")
    p_exec.add_argument("--pod", required=True)
    p_exec.add_argument("command")
    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--pod", required=True)
    p_pull.add_argument("remote")
    p_pull.add_argument("local")
    p_down = sub.add_parser("down")
    p_down.add_argument("--pod", required=True)
    args = p.parse_args()
    target = load_target(args.target)
    try:
        if args.cmd == "up":
            return cmd_up(target, args)
        if args.cmd == "exec":
            return cmd_exec(target, args)
        if args.cmd == "pull":
            return cmd_pull(target, args)
        return down(target, args.pod)
    except (Failed, requests.RequestException, websocket.WebSocketException, KeyError,
            ValueError) as exc:
        # `up` callers branch on the FAILED: prefix; keep it for every error.
        print(f"FAILED: {exc}", flush=True)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
