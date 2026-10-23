"""The credential broker: the only place an authoring job's secrets live.

An author's notebook is untrusted: its code runs on our GPUs, and the agent
that runs and annotates it reads it, so a prompt injection could make the
agent do anything its shell can. So an authoring job's container holds no
secret at all, only a random job key (NBR_JOB_KEY), and sits on an internal
Docker network whose one reachable host is this broker. The broker does, on
the job's behalf and with the real credentials, exactly what the job needs:

  /llm/...        the LLM gateway (the Claude CLI's ANTHROPIC_BASE_URL), with
                  the gateway headers added
  /tool/k8s       k8s_pod.py up/exec/pull/down on this job's own pod, on its
  /tool/radeon    target, with at most its GPUs (radeon_pod.py likewise)
  /hf/...         read-only Hugging Face API calls (env_check's gated-model
                  check), with the authors' Hugging Face token if there is one
  /dockerhub/...  read-only Docker Hub tag lookups (resolve_docker_image.py)

A key is valid only while its job runs: the worker writes the keys of running
jobs to jobs/.broker/keys.json (as SHA-256 hashes) and removes them when the
job ends. A stolen key is useless after the job, and from outside the job
network. /admin/cleanup lets the worker delete a cancelled job's pod.

What stays possible for a malicious job, by design: using its own pod and
the LLM until its timebox ends, and reading whatever the notebook's pod is
given (NBR_AUTHOR_HF_SECRET; see README, Authoring service).

Run (docker/run-broker.sh):
  uvicorn authoring.broker:app --host 0.0.0.0 --port 8790
"""

import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
KEYS_FILE = Path(os.environ.get("NBR_BROKER_KEYS")
                 or REPO_ROOT / "jobs" / ".broker" / "keys.json")
LLM_URL = (os.environ.get("NBR_UPSTREAM_ANTHROPIC_BASE_URL")
           or os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
HF_URL = "https://huggingface.co"
DOCKERHUB_URL = "https://hub.docker.com"
WORKSPACE = "/workspace"
TOOL_SCRIPTS = {"k8s": "k8s_pod.py", "radeon": "radeon_pod.py"}
VERBS = {"up", "exec", "pull", "down"}
EXIT_MARK = "\n__NBR_EXIT__ "
# Request headers never passed upstream: the job's key and hop-by-hop headers.
DROP = {"host", "x-api-key", "authorization", "content-length", "connection",
        "accept-encoding", "transfer-encoding"}

sys.path.insert(0, str(TOOLS))
import routing  # noqa: E402

app = FastAPI(title="notebook-runner credential broker", docs_url=None, redoc_url=None,
              openapi_url=None)


# ── who is asking ────────────────────────────────────────────────────────────

def job_for(request: Request) -> dict:
    """The running job whose key the request carries, or 401."""
    key = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        key = key or auth[7:].strip()
    if not key:
        raise HTTPException(401, "no job key")
    digest = hashlib.sha256(key.encode()).hexdigest()
    try:
        keys = json.loads(KEYS_FILE.read_text())
    except (OSError, ValueError):
        keys = {}
    for known, job in keys.items():
        if hmac.compare_digest(known, digest):
            return job
    raise HTTPException(401, "unknown or expired job key")


def gateway_headers() -> dict[str, str]:
    """The gateway's headers, from ANTHROPIC_CUSTOM_HEADERS ("Name: value" lines)."""
    headers = {}
    for line in (os.environ.get("ANTHROPIC_CUSTOM_HEADERS") or "").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip()] = value.strip()
    headers["x-api-key"] = os.environ.get("ANTHROPIC_API_KEY") or "dummy"
    return headers


def passthrough(upstream: requests.Response) -> StreamingResponse:
    keep = {k: v for k, v in upstream.headers.items()
            if k.lower() in ("content-type", "cache-control", "request-id", "x-request-id",
                             "anthropic-ratelimit-requests-remaining", "retry-after")}
    return StreamingResponse(upstream.iter_content(chunk_size=None), status_code=upstream.status_code,
                             headers=keep)


# ── the LLM gateway ──────────────────────────────────────────────────────────

@app.api_route("/llm/{path:path}", methods=["GET", "POST"])
async def llm(path: str, request: Request):
    job_for(request)
    if not LLM_URL:
        raise HTTPException(503, "the broker has no ANTHROPIC_BASE_URL")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
    headers.update(gateway_headers())
    upstream = requests.request(request.method, f"{LLM_URL}/{path}", params=request.query_params,
                                data=await request.body(), headers=headers, stream=True,
                                timeout=(15, 900))
    return passthrough(upstream)


# ── read-only lookups ────────────────────────────────────────────────────────

def _get(url: str, request: Request, headers: dict | None = None) -> Response:
    upstream = requests.request(request.method, url, params=request.query_params,
                                headers={"User-Agent": "notebook-runner-broker", **(headers or {})},
                                allow_redirects=False, timeout=30)
    keep = {k: v for k, v in upstream.headers.items()
            if k.lower() in ("content-type", "location", "etag", "x-repo-commit", "x-error-code")}
    return Response(upstream.content, status_code=upstream.status_code, headers=keep)


@app.api_route("/hf/{path:path}", methods=["GET", "HEAD"])
def hf(path: str, request: Request):
    job = job_for(request)
    token = os.environ.get("NBR_AUTHOR_HF_TOKEN") if job["kind"] != "regression" \
        else os.environ.get("HF_TOKEN")
    return _get(f"{HF_URL}/{path}", request,
                {"Authorization": f"Bearer {token}"} if token else None)


@app.get("/dockerhub/{path:path}")
def dockerhub(path: str, request: Request):
    job_for(request)
    return _get(f"{DOCKERHUB_URL}/{path}", request)


# ── the pod tools ────────────────────────────────────────────────────────────

def pod_allowed(pod: str, job: dict) -> bool:
    """run.py names a job's pods nbr-<notebook>-<v|f>-<job id> (k8s_pod.pod_name)."""
    return re.fullmatch(rf"nbr-[a-z0-9-]*-[vf]-{re.escape(job['id'].lower())}", pod) is not None


def option(argv: list[str], name: str) -> str | None:
    if name in argv[:-1]:
        return argv[argv.index(name) + 1]
    return None


def check(tool: str, argv: list[str], job: dict) -> str:
    """Refuse anything but this job's own pod, target and GPU count. Returns the verb."""
    if tool not in TOOL_SCRIPTS:
        raise HTTPException(404, "no such tool")
    if len(argv) < 3 or argv[0] != "--target" or argv[1] != job.get("target"):
        raise HTTPException(403, "this job may only use its own target")
    target = routing.load_config()["targets"].get(argv[1]) or {}
    if target.get("type") != tool:
        raise HTTPException(403, f"target {argv[1]} is not a {tool} target")
    verb = argv[2]
    if verb not in VERBS:
        raise HTTPException(403, f"{verb} is not allowed")
    pod = option(argv, "--pod")
    if not pod or not pod_allowed(pod, job):
        raise HTTPException(403, "this job may only use its own pod")
    if verb == "up":
        gpus = int(option(argv, "--gpus") or 1)
        if gpus > int(job.get("gpus") or 1):
            raise HTTPException(403, f"this job may use at most {job.get('gpus')} GPU(s)")
    if verb == "pull":
        remote = argv[-2]
        if not remote.startswith(WORKSPACE + "/") or ".." in remote.split("/"):
            raise HTTPException(403, f"pull only reads under {WORKSPACE}/")
    return verb


def tool_env(job: dict) -> dict:
    """The tool's environment: the real credentials, and the authors' HF token
    (or none) in place of the pass's for anything an author's notebook can read."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("NBR_BROKER")}
    env.pop("NBR_BROKER_URL", None)
    if job["kind"] != "regression":
        env.pop("HF_TOKEN", None)
        if os.environ.get("NBR_AUTHOR_HF_TOKEN"):
            env["HF_TOKEN"] = os.environ["NBR_AUTHOR_HF_TOKEN"]  # radeon_pod injects it
        env["NBR_HF_SECRET"] = os.environ.get("NBR_AUTHOR_HF_SECRET") or "hf-token-authors"
    return env


@app.post("/tool/{tool}")
async def tool(tool: str, request: Request):
    job = job_for(request)
    form = await request.form()
    try:
        argv = json.loads(form.get("argv") or "[]")
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    except (ValueError, AssertionError):
        raise HTTPException(400, "bad argv")
    verb = check(tool, argv, job)
    work = Path(tempfile.mkdtemp(prefix=f"broker-{job['id']}-"))
    # --push: the job sent the files; hand the tool their copies here.
    uploads = {Path(f.filename).name: f for f in form.getlist("push") if hasattr(f, "filename")}
    for i, a in enumerate(argv[:-1]):
        if a == "--push":
            name = Path(argv[i + 1]).name
            if name not in uploads:
                raise HTTPException(400, f"--push {name} was not uploaded")
            dest = work / name
            dest.write_bytes(await uploads[name].read())
            argv[i + 1] = str(dest)
    cmd = [sys.executable, str(TOOLS / TOOL_SCRIPTS[tool]), *argv]

    if verb == "pull":
        return JSONResponse(pull(tool, argv, job, work))

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=tool_env(job), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

    def stream():
        try:
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):
                yield chunk
            proc.wait()
            yield f"{EXIT_MARK}{proc.returncode}\n".encode()
        finally:
            if proc.poll() is None:
                proc.kill()
            for f in work.iterdir():
                f.unlink()
            work.rmdir()

    return StreamingResponse(stream(), media_type="text/plain")


def pull(tool: str, argv: list[str], job: dict, work: Path) -> dict:
    """Copy a file out of the job's pod and return it. On k8s this reads it
    with `cat` through exec, not `kubectl cp`: cp unpacks a tar the pod
    produces, and the pod is the author's."""
    pod, remote = option(argv, "--pod"), argv[-2]
    local = work / "pulled"
    try:
        if tool == "k8s":
            target = routing.load_config()["targets"][argv[1]]
            base = ["kubectl", *(["--context", target["context"]] if target.get("context") else []),
                    "-n", target.get("namespace", "notebook-runner")]  # as k8s_pod.Kubectl
            env = tool_env(job)
            with open(local, "wb") as out:
                proc = subprocess.run([*base, "exec", pod, "-c", "main", "--", "cat", remote],
                                      stdout=out, stderr=subprocess.PIPE, env=env, timeout=1800)
            want = subprocess.run([*base, "exec", pod, "-c", "main", "--", "sha256sum", remote],
                                  capture_output=True, text=True, env=env, timeout=300)
            rc, output = proc.returncode, proc.stderr.decode(errors="replace")
            if rc == 0 and want.returncode == 0 and \
                    want.stdout.split()[0] != hashlib.sha256(local.read_bytes()).hexdigest():
                rc, output = 1, f"FAILED: the copy of {remote} is incomplete\n"
        else:
            proc = subprocess.run([sys.executable, str(TOOLS / TOOL_SCRIPTS[tool]),
                                   *argv[:-1], str(local)], capture_output=True, text=True,
                                  env=tool_env(job), cwd=REPO_ROOT, timeout=1800)
            rc, output = proc.returncode, proc.stdout + proc.stderr
        data = base64.b64encode(local.read_bytes()).decode() if rc == 0 and local.is_file() else None
        return {"rc": rc, "output": output, "data": data}
    finally:
        for f in work.iterdir():
            f.unlink()
        work.rmdir()


# ── the worker's cleanup ─────────────────────────────────────────────────────

@app.post("/admin/cleanup/{job_id}")
def cleanup(job_id: str, request: Request):
    """Delete the pods a stopped job left (the worker calls this on cancel)."""
    admin = os.environ.get("NBR_BROKER_ADMIN_KEY") or ""
    given = (request.headers.get("authorization") or "")[7:]
    if not admin or not hmac.compare_digest(admin, given):
        raise HTTPException(401, "admin key required")
    if not re.fullmatch(r"[a-z][0-9a-f]+", job_id):
        raise HTTPException(400, "bad job id")
    out = []
    for name, target in routing.load_config()["targets"].items():
        if target.get("type") == "k8s":
            import k8s_pod
            proc = k8s_pod.Kubectl(target).run(
                "delete", "pods", "-l", f"{k8s_pod.APP_NAME}/run-id={job_id}",
                "--ignore-not-found", "--wait=false", check=False)
            out.append(f"{name}: {proc.stdout.strip() or proc.stderr.strip() or 'nothing to delete'}")
        elif target.get("type") == "radeon":
            import radeon_pod
            state = radeon_pod.load_state(target)
            if str(state.get("pod", "")).endswith(f"-{job_id}"):
                radeon_pod.down(target, state["pod"])
                out.append(f"{name}: deleted {state['pod']}")
    return {"done": out}


@app.get("/healthz")
def healthz():
    return "ok"
