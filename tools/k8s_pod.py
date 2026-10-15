"""
Provision and drive a per-notebook GPU pod for the Kubernetes runner.

The SSH runner runs every notebook through `docker run` on a GPU host. The k8s
runner instead gives each agent run its own pod, built from the image the agent
resolved, and runs papermill inside it via `kubectl exec`. This tool is the only
thing that talks to kubectl; run.py and the agent call it through the command
strings run.py puts in the runtime context (K8S_UP_CMD, POD_CMD, POD_PULL_CMD).

Usage:
  k8s_pod.py --target NAME up   --pod P --gpus N [--memory-gb G] [--env K=V ...]
                                [--push FILE ...] --image IMG
  k8s_pod.py --target NAME exec --pod P '<shell command>'
  k8s_pod.py --target NAME pull --pod P /workspace/x.ipynb local/x.ipynb
  k8s_pod.py --target NAME down --pod P
  k8s_pod.py --target NAME gc
  k8s_pod.py --target NAME up ... --dry-run     # print the pod spec, touch nothing

`up` is idempotent and returns within UP_WAIT_SECONDS so it fits under the
agent's 10-minute Bash cap. Its last line is one of:
  READY                      exit 0  pod running, --push files are in /workspace
  Waiting for pod: <reason>  exit 3  still pulling/scheduling: run it again
  FAILED: <reason>           exit 2  image pull or container start failed
  UNSCHEDULABLE: <reason>    exit 4  Pending past schedule_timeout_min

The target in runner.yaml (`type: k8s`) supplies context, namespace, node
selector, tolerations, GPU resource name and sizing; see runner.example.yaml.

One-time cluster setup (cluster admin):
  kubectl --context workshop apply -f tools/k8s/rbac.yaml
  # namespace-scoped kubeconfig for the CI host — never an admin kubeconfig,
  # since the agent can read whatever credential the runner holds:
  python3 tools/k8s_pod.py --target <k8s target> kubeconfig > ~/.kube/notebook-runner
  chmod 600 ~/.kube/notebook-runner
  # then set NBR_KUBECONFIG=~/.kube/notebook-runner in .env (docker/run-ci.sh
  # stages it) or export KUBECONFIG for workstation runs.
  # HF_TOKEN for every pod (read from stdin, so it stays out of argv):
  kubectl --context workshop -n notebook-runner create secret generic hf-token \
    --from-file=HF_TOKEN=/dev/stdin
"""

import argparse
import base64
import calendar
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing

APP_LABEL = "app.kubernetes.io/name"
APP_NAME = "notebook-runner"
SERVICE_ACCOUNT = "notebook-runner"
HF_SECRET = "hf-token"  # created once by an admin; see the module docstring
WORKSPACE = "/workspace"
UP_WAIT_SECONDS = 480
POLL_SECONDS = 10
KUBECTL_TIMEOUT = 60

# Container waiting reasons that will not resolve by waiting longer.
FATAL_WAIT_REASONS = {
    "InvalidImageName", "CreateContainerConfigError", "CreateContainerError",
    "CrashLoopBackOff",
}
# Pull failures are often transient (registry rate limits, network); only a
# definitive answer from the registry is fatal.
PULL_WAIT_REASONS = {"ErrImagePull", "ImagePullBackOff"}
PULL_FATAL_RE = re.compile(r"not found|manifest unknown|unauthorized|denied", re.IGNORECASE)

EXIT_READY, EXIT_FAILED, EXIT_WAITING, EXIT_UNSCHEDULABLE = 0, 2, 3, 4

# kubectl errors that mean "the API didn't answer", not "the API said no".
TRANSIENT_RE = re.compile(r"Unable to connect|i/o timeout|TLS handshake timeout|"
                          r"connection refused|connection reset", re.IGNORECASE)


class Kubectl:
    def __init__(self, server: dict):
        self.base = ["kubectl"]
        if server.get("context"):
            self.base += ["--context", server["context"]]
        self.base += ["-n", server.get("namespace", APP_NAME)]

    def run(self, *args: str, input: str | None = None, check: bool = True,
            timeout: int = KUBECTL_TIMEOUT) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [*self.base, *args], input=input, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"kubectl {args[0]} failed: {proc.stderr.strip()}")
        return proc

    def get_pod(self, pod: str, attempts: int = 4) -> dict | None:
        """Read is idempotent, so transient API failures are retried here."""
        for attempt in range(1, attempts + 1):
            proc = self.run("get", "pod", pod, "-o", "json", check=False)
            if proc.returncode == 0:
                return json.loads(proc.stdout)
            if "NotFound" in proc.stderr:
                return None
            if attempt < attempts and TRANSIENT_RE.search(proc.stderr):
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"kubectl get pod failed: {proc.stderr.strip()}")
        raise AssertionError("unreachable")


def load_target(name: str) -> dict:
    target = routing.load_config()["targets"].get(name)
    if not target:
        raise SystemExit(f"error: target '{name}' not found in {routing.CONFIG_PATH}")
    if target.get("type") != "k8s":
        raise SystemExit(f"error: target '{name}' is not type: k8s")
    return target


def pod_name(stem: str, phase: str, run_id: str) -> str:
    """DNS-1123 pod name: nbr-<stem>-<v|f>-<run_id>, at most 63 chars."""
    suffix = f"-{phase[0]}-{run_id}".lower()
    stem = re.sub(r"[^a-z0-9-]+", "-", stem.lower()).strip("-")
    stem = stem[: 63 - len("nbr-") - len(suffix)].rstrip("-")
    return re.sub(r"[^a-z0-9-]", "-", f"nbr-{stem}{suffix}")


def pod_spec(server: dict, pod: str, image: str, gpus: int, memory_gb: int = 64,
             env: dict[str, str] | None = None) -> dict:
    memory = f"{memory_gb}Gi"
    gpu = server.get("gpu_resource", "amd.com/gpu")
    run_id = pod.rsplit("-", 1)[-1]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod,
            "labels": {APP_LABEL: APP_NAME, f"{APP_NAME}/run-id": run_id},
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 5,
            # Backstop for pods leaked by a killed runner: the kubelet stops
            # them (freeing the GPUs) once this elapses; `gc` deletes them.
            "activeDeadlineSeconds": int(server.get("max_lifetime_hours", 12) * 3600),
            "nodeSelector": server.get("node_selector") or {},
            "tolerations": server.get("tolerations") or [],
            "containers": [{
                "name": "main",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                # Override the image entrypoint (vLLM/SGLang images start a
                # server); the agent drives everything through exec.
                "command": ["sleep", "infinity"],
                "workingDir": WORKSPACE,
                "env": [{"name": "HF_HOME", "value": f"{WORKSPACE}/hf"},
                        *({"name": k, "value": v} for k, v in (env or {}).items())],
                # optional: gated-model notebooks fail their own way without it.
                "envFrom": [{"secretRef": {"name": HF_SECRET, "optional": True}}],
                "resources": {
                    "requests": {gpu: str(gpus), "memory": memory},
                    "limits": {gpu: str(gpus), "memory": memory},
                },
                "volumeMounts": [
                    {"name": "workspace", "mountPath": WORKSPACE},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }],
            "volumes": [
                {"name": "workspace",
                 "emptyDir": {"sizeLimit": server.get("workspace_size", "300Gi")}},
                {"name": "dshm",
                 "emptyDir": {"medium": "Memory", "sizeLimit": server.get("shm_size", "8Gi")}},
            ],
        },
    }


def classify(pod_obj: dict) -> tuple[str, str]:
    """Return (state, reason): state is ready | waiting | failed | pending."""
    status = pod_obj.get("status") or {}
    phase = status.get("phase", "Unknown")
    if phase in ("Failed", "Succeeded"):
        return "failed", f"pod {phase.lower()}: {status.get('reason') or status.get('message') or ''}".strip()
    for cs in status.get("containerStatuses") or []:
        if cs.get("ready"):
            return "ready", "running"
        waiting = (cs.get("state") or {}).get("waiting") or {}
        reason = waiting.get("reason", "")
        message = waiting.get("message", "")
        if reason in FATAL_WAIT_REASONS or (
                reason in PULL_WAIT_REASONS and PULL_FATAL_RE.search(message)):
            return "failed", f"{reason}: {message}".strip()
        if reason:
            return "waiting", reason
    for cond in status.get("conditions") or []:
        if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
            return "pending", cond.get("message") or cond.get("reason") or "unscheduled"
    return "waiting", phase


def cmd_up(kc: Kubectl, server: dict, args) -> int:
    if not args.image:
        print("FAILED: --image is required", flush=True)
        return EXIT_FAILED
    env = dict(e.split("=", 1) for e in args.env or [])
    spec = pod_spec(server, args.pod, args.image, args.gpus, args.memory_gb, env)
    if args.dry_run:
        print(yaml.safe_dump(spec, sort_keys=False))
        return EXIT_READY

    pod = kc.get_pod(args.pod)
    if pod is None:
        kc.run("create", "-f", "-", input=json.dumps(spec))
        pod = kc.get_pod(args.pod)
        print(f"created pod {args.pod} (image {args.image}, {args.gpus} GPU)", flush=True)
    elif pod["spec"]["containers"][0]["image"] != args.image:
        print(f"FAILED: pod {args.pod} already exists with image "
              f"{pod['spec']['containers'][0]['image']}", flush=True)
        return EXIT_FAILED

    created = pod["metadata"]["creationTimestamp"]
    created_epoch = calendar.timegm(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))
    schedule_timeout = float(server.get("schedule_timeout_min", 45)) * 60
    deadline = time.time() + UP_WAIT_SECONDS

    while True:
        state, reason = classify(pod)
        if state == "ready":
            break
        if state == "failed":
            print(f"FAILED: {reason}", flush=True)
            return EXIT_FAILED
        if state == "pending" and time.time() - created_epoch > schedule_timeout:
            print(f"UNSCHEDULABLE: {reason}", flush=True)
            return EXIT_UNSCHEDULABLE
        if time.time() >= deadline:
            print(f"Waiting for pod: {reason}", flush=True)
            return EXIT_WAITING
        time.sleep(POLL_SECONDS)
        try:
            fresh = kc.get_pod(args.pod)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            # Transient API errors must not fail a run whose pod is fine;
            # keep the last state.
            print(f"warning: {exc}", file=sys.stderr, flush=True)
            continue
        if fresh is None:
            print("FAILED: pod disappeared while waiting", flush=True)
            return EXIT_FAILED
        pod = fresh

    for path in args.push or []:
        kc.run("cp", path, f"{args.pod}:{WORKSPACE}/{Path(path).name}", "-c", "main",
               timeout=300)
    print("READY", flush=True)
    return EXIT_READY


def cmd_exec(kc: Kubectl, args) -> int:
    # Same shape as `SSH_CMD '<cmd>'`: one shell string, output passed through.
    # No timeout: long waits are the agent's nohup+poll pattern, not one exec.
    proc = subprocess.run(
        [*kc.base, "exec", args.pod, "-c", "main", "--", "bash", "-c",
         f"cd {WORKSPACE} && {args.command}"],
        check=False,
    )
    return proc.returncode


def cmd_pull(kc: Kubectl, args) -> int:
    Path(args.local).parent.mkdir(parents=True, exist_ok=True)
    proc = kc.run("cp", f"{args.pod}:{args.remote}", args.local, "-c", "main",
                  check=False, timeout=300)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def down(kc: Kubectl, pod: str) -> int:
    proc = kc.run("delete", "pod", pod, "--ignore-not-found", "--wait=false",
                  check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def gc(kc: Kubectl) -> int:
    """Delete finished pods a killed runner left behind."""
    proc = kc.run("delete", "pods", "-l", f"{APP_LABEL}={APP_NAME}",
                  "--field-selector", "status.phase!=Running,status.phase!=Pending",
                  "--ignore-not-found", "--wait=false", check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def cmd_kubeconfig(kc: Kubectl, server: dict) -> int:
    """Print a kubeconfig for the namespace ServiceAccount, built from the
    long-lived token Secret in tools/k8s/rbac.yaml. Run with an admin context."""
    sec = json.loads(kc.run("get", "secret", f"{SERVICE_ACCOUNT}-token", "-o", "json").stdout)
    view = json.loads(subprocess.run(
        ["kubectl", "config", "view", "--raw", "--minify", "-o", "json",
         *(["--context", server["context"]] if server.get("context") else [])],
        capture_output=True, text=True, check=True).stdout)
    cluster = view["clusters"][0]["cluster"]
    name = server.get("context") or "cluster"
    cfg = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": name, "cluster": {
            "server": cluster["server"],
            "certificate-authority-data": sec["data"]["ca.crt"]}}],
        "users": [{"name": SERVICE_ACCOUNT, "user": {
            "token": base64.b64decode(sec["data"]["token"]).decode()}}],
        "contexts": [{"name": name, "context": {
            "cluster": name, "user": SERVICE_ACCOUNT,
            "namespace": server.get("namespace", APP_NAME)}}],
        "current-context": name,
    }
    print(yaml.safe_dump(cfg, sort_keys=False), end="")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Per-notebook GPU pods for the k8s runner.")
    p.add_argument("--target", required=True, help="runner.yaml target (type: k8s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up")
    p_up.add_argument("--pod", required=True)
    p_up.add_argument("--gpus", type=int, default=1)
    p_up.add_argument("--memory-gb", type=int, default=64)
    p_up.add_argument("--env", action="append", help="K=V to set in the container")
    p_up.add_argument("--push", action="append", help="local file to copy into /workspace")
    p_up.add_argument("--image")
    p_up.add_argument("--dry-run", action="store_true")

    p_exec = sub.add_parser("exec")
    p_exec.add_argument("--pod", required=True)
    p_exec.add_argument("command")

    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--pod", required=True)
    p_pull.add_argument("remote")
    p_pull.add_argument("local")

    p_down = sub.add_parser("down")
    p_down.add_argument("--pod", required=True)

    sub.add_parser("gc")
    sub.add_parser("kubeconfig")

    args = p.parse_args()
    server = load_target(args.target)
    kc = Kubectl(server)

    try:
        if args.cmd == "up":
            return cmd_up(kc, server, args)
        if args.cmd == "exec":
            return cmd_exec(kc, args)
        if args.cmd == "pull":
            return cmd_pull(kc, args)
        if args.cmd == "down":
            return down(kc, args.pod)
        if args.cmd == "gc":
            return gc(kc)
        return cmd_kubeconfig(kc, server)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        # `up` callers branch on the FAILED: prefix; keep it for every error.
        print(f"FAILED: {exc}", flush=True)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
