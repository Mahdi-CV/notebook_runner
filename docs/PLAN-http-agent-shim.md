# HTTP transport shim to replace SSH to the GPU node

**Status: implemented and verified, 2026-09-17.** This is the design doc,
kept for reference — every source file in `tools/http_ssh_shim/` and the
integration points in `run.py`/`docker/run-ci.sh` link back here.

## Context

`xcovdashdbd01` (the intended orchestrator host, on AMD's internal network,
`10.26.247.249`, no public IP) cannot reach the GPU node
(`134.199.202.42`, a DigitalOcean droplet) over SSH: AMD's egress proxy
completes the TCP handshake on port 22 but then silently drops all
subsequent traffic (confirmed reproducible: 0 bytes ever received, every
attempt times out at the banner-exchange stage). HTTPS to this destination
is refused outright in ~46ms — confirmed to be a block on the entire
`134.199.202.0/24` netblock (tested three other IPs in the range, all
refused identically), not specific to port 22 or this one IP. Reputable
HTTPS destinations by bare IP (`1.1.1.1`, `8.8.8.8`, GitHub's IP) all work
fine, so this is a category/reputation block on the DigitalOcean range,
not a generic "unknown TLS destination" policy — nothing app-layer can
route around it.

**Plain HTTP on port 80 is the only transport confirmed to pass cleanly**
end-to-end (real request/response round-trip in ~0.34s, no truncation, no
proxy interference). Caddy is already running on the GPU node listening on
`:80`. The user has explicitly chosen HTTP (not HTTPS) for now, given this
reality — and, since a static bearer token sent in the clear would be
both interceptable and replayable, has chosen **HMAC-signed requests**
over a bearer token: the shared secret never travels on the wire, only
a per-request signature bound to that specific request and a narrow
time window does, so a captured request can't be replayed or repurposed
even though the channel itself carries no confidentiality.

Separately, the network is confirmed **one-directional**: the GPU node has
no route back to AMD's internal `10.26.0.0/22` range (`Network is
unreachable`, not just filtered) and can't resolve AMD-internal DNS. So
`xcovdashdbd01` must always be the client; the GPU node is always the
server. There is no path for the GPU node to call back or push — this
rules out any webhook/callback design and confirms a plain synchronous
client-pull HTTP API is the only shape that fits.

## The real constraint: the agent runs SSH itself, autonomously

This is not just `run.py` making SSH calls — `run.py` builds a `SSH_CMD`
string and injects it into the **Claude Code agent's own system prompt**.
The agent then runs `ssh` itself via its Bash tool for the entire
multi-hour, multi-phase notebook test (every `docker run`, `papermill`,
health-poll, and cleanup command in `CLAUDE.md`/`CLAUDE_FIX.md` is
literally `Bash: <SSH_CMD> '<command>'`). Rewriting `CLAUDE.md` to teach
the agent a new remote-execution protocol would be invasive and risks
degrading its autonomous behavior in ways that are hard to fully test.

**Five independent SSH/SCP touchpoints**, all kept working identically
from the agent's/caller's point of view:

1. `run.py::scp_notebook()` — direct `ssh`+`scp` to stage the notebook +
   `preflight_patch.py` before the agent starts.
2. `run.py::run_env_check()` passes `--ssh-cmd "ssh ... user@host"` to
   `tools/env_check.py`, which shells out to it itself (`ssh_run()`) for
   GPU-count and disk-space checks.
3. `run.py::run_notebook()`/`run_fixer()` build `SSH_CMD`/`SSH_HF_CMD`
   strings, inject them into the agent's runtime context — **the agent
   itself executes these via its Bash tool** for the entire notebook
   lifecycle.
4. `tools/env_check.py` itself, as above.
5. `docker/run-ci.sh` staged a private key + `known_hosts` into the
   containerized agent so a containerized run could reach the GPU node
   the same way — now replaced with forwarding `AGENT_SECRET`/`AGENT_URL`
   instead (no key file, no `known_hosts` entry).

## Approach: a transport-transparent shim, not a rewrite

Replace the *destination* `ssh`/`scp` reach, not the *shape* of every call
site.

1. **`tools/http_ssh_shim/agent_server.py`** — a minimal stdlib-only HTTP
   server on the GPU node, behind the existing Caddy on `:80` (route
   `/agent/*`, alongside the pre-existing reverse-proxy to `:8888` for
   other work — undisturbed). Endpoints:
   - `POST /agent/exec` — `{cmd, timeout}` → `{exit_code, stdout, stderr}`.
     Direct equivalent of `ssh host 'cmd'`. Runs via
     `subprocess.run(cmd, shell=True, executable="/bin/bash", ...)` —
     **explicitly bash, not the default `/bin/sh`**, because
     `CLAUDE.md`'s own `nohup <cmd> & disown` backgrounding idiom needs a
     shell with a `disown` builtin; this gap was caught during
     verification (see below) and fixed.
   - `POST /agent/upload` — simple length-prefixed-JSON-header + raw-bytes
     framing (not real multipart — the shim controls both sides, so a
     multipart parser dependency wasn't worth it). Equivalent of
     `scp local host:path`.
   - `GET /agent/download?path=...` — equivalent of `scp host:path local`.
     Confirmed necessary during implementation: `CLAUDE_FIX.md` Step 4c
     retrieves `<nb>_fixed.ipynb` this exact way before cleanup deletes
     the remote copy.
   - **Auth: HMAC-SHA256 request signing**, covering method + path
     (including query string, so `/agent/download?path=X` can't be
     retargeted to a different path after signing) + body hash + a
     60-second timestamp window + a nonce replay cache. See
     `tools/http_ssh_shim/hmac_auth.py` for the exact signed-string
     format and header names.
   - `AGENT_SECRET` is a plain string; every caller (client and server)
     normalizes it to bytes via `.encode()` — **never `bytes.fromhex()`**,
     a distinction that caused every signature to mismatch during initial
     testing until traced down to two different byte representations of
     the "same" secret. `hmac_auth.py` now accepts `str | bytes` directly
     so this ambiguity can't recur in a future caller.
   - Long-running/backgrounded commands need no special server-side
     handling: `CLAUDE.md`'s existing `nohup ... & disown` + poll-via-
     `cat op.log` convention maps directly onto repeated stateless
     `/agent/exec` calls. Verified: a backgrounded command started in one
     call was still running and produced output visible to a later,
     independent call.

2. **`tools/http_ssh_shim/ssh` and `.../scp`** — drop-in wrapper scripts
   with the exact same argv shape as the real binaries
   (`ssh -o ... user@host 'cmd'`, `scp -o ... local user@host:path`, and
   the download form `scp -o ... user@host:path local`), translating to
   HMAC-signed HTTP calls instead of a real SSH connection.
   - `run.py` prepends `tools/http_ssh_shim/` to `$PATH` at import time
     (gated on `AGENT_URL`/`AGENT_SECRET` both being set — unset, real
     `ssh`/`scp` still resolve normally, so a future deployment where SSH
     actually works needs no code change). Every subprocess `run.py`
     spawns without an explicit `env=` inherits this PATH — including the
     `claude` CLI process, and in turn every command the Claude Code
     agent's own Bash tool executes. Verified end-to-end: importing
     `run.py` then spawning a bash subprocess with no `env=` override
     (mirroring `run.py`'s actual `claude` spawn) and running `ssh ...`
     from inside it reached the real GPU node through the shim.
   - `-o key=value` flags are accepted and silently ignored (they
     configure a real SSH connection that has no meaning over HTTP).

3. **`docker/run-ci.sh`**: the SSH-key staging/mount block (install,
   chown-to-uid-1001, `known_hosts` generation via `ssh-keyscan`) is
   replaced with a straight preflight check that `AGENT_URL`/
   `AGENT_SECRET` are set in `.env`, then forwarded via `-e` like every
   other credential. No key file, no staging, no chown dance.

4. **`.env` / `.env.example`**: `AGENT_URL` (`http://134.199.202.42/agent`)
   and `AGENT_SECRET` (a `secrets.token_hex(32)` string) alongside the
   existing `GPU_HOST`/`GPU_USER`/`GPU_SSH_KEY` (kept, documented as the
   fallback path if real SSH ever becomes reachable).

## What did NOT change

- `CLAUDE.md` / `CLAUDE_FIX.md` — zero edits. The agent keeps typing
  `Bash: <SSH_CMD> '<command>'` exactly as documented; `SSH_CMD` still
  literally starts with the string `ssh`, it's just resolving to the shim.
- `run.py`'s prompt-building / runtime-context injection — unchanged,
  since it only ever constructed a string; the string's meaning changes,
  not its shape.
- The playbook's `docker run -d ... & disown` / poll-with-`cat op.log`
  pattern — unchanged in the playbook text; made to actually work
  correctly server-side (see the bash-vs-sh fix above).

## Verification performed

All against the real deployed agent server on `134.199.202.42`, through
the real Caddy `:80` path (not a local mock):

1. **`ssh` shim exec** — real command execution, correct stdout/stderr/
   exit-code pass-through (tested 0, a specific nonzero code, and stderr
   content).
2. **`scp` shim upload** — multi-file upload matching `scp_notebook`'s
   exact two-source-one-destination-dir pattern; byte-exact content
   verified on the remote end.
3. **`scp` shim download** — the fixer's `<nb>_fixed.ipynb` retrieval
   pattern; byte-exact content verified locally after download.
4. **`tools/env_check.py` standalone**, invoked with a shim-resolved
   `--ssh-cmd`, against a real manifest entry — correctly reported the
   real remote GPU count (9) and real disk free space (603GB, matching
   independently-known state).
5. **Full process-tree inheritance** — imported `run.py` (module-level
   PATH prepend fires), then spawned a bash subprocess with no `env=`
   override (mirroring `run.py`'s actual `subprocess.Popen(["claude", ...])`
   call with no env override), and from inside that subprocess ran a
   real `ssh` command reaching the GPU node — confirming the exact
   `run.py` → `claude` → agent's-Bash-tool → `ssh` inheritance chain
   works, without needing a full multi-minute live agent session to prove
   it.
6. **Backgrounded command + poll**, the `CLAUDE.md` idiom exactly: started
   `nohup bash -c '...' < /dev/null > log 2>&1 & disown` in one `/agent/exec`
   call, polled immediately (correctly empty/not-yet), then polled again
   after completion (correctly showed output) — via two independent HTTP
   calls, proving state persists on the server side across stateless
   requests.
7. **HMAC negative security tests**, all confirmed against the live
   server: replay of an identical valid request rejected the second time;
   wrong secret rejected; missing auth headers rejected; tampered body
   (upload) rejected; tampered query path (download) rejected — all via
   generic 401s that don't reveal which check failed.
8. **Inside the real container image** (`notebook-runner:latest`, already
   built on the GPU node) — confirmed the shim scripts retain their `+x`
   bit through a `:ro` bind mount, run as the real non-root `agent`
   (uid 1001) user, and successfully complete an HMAC-signed HTTP round
   trip from inside the container to the agent server.

## Real bug found and fixed during verification

`agent_server.py`'s `_handle_exec` originally used bare
`subprocess.run(cmd, shell=True, ...)`, which on POSIX always invokes
`/bin/sh` regardless of the system's default login shell. This is
invisible for ordinary commands but broke `CLAUDE.md`'s own
`nohup <cmd> & disown` backgrounding idiom — POSIX `sh` has no `disown`
builtin (`disown: not found`, though the backgrounding itself still
technically worked via `nohup` alone). Fixed by passing
`executable="/bin/bash"` explicitly, matching what a real SSH session to
this host would actually invoke.

## Open item, not blocking

Long-lived HTTP connections / chunked transfer through AMD's egress for
*streaming* output were not empirically tested (only single-request/
response round trips). Since the implemented design needs no streaming
(every long-running command is background + poll, per `CLAUDE.md`'s
existing convention), this is not a gap — noted only in case a future
design wants live log tailing instead of poll-based `cat op.log`.
