"""
Report notebook regression results to GitHub as issues and pull requests.

The bridge between the regression agent's local result JSONs and the tutorial
authors on GitHub. It is the "report to authors / propose fixes" half of the
autonomous pipeline.

Philosophy (inherited from CLAUDE.md): health auditor, not a green-washer.
  - Content bugs that need a human            -> a GitHub ISSUE (diagnosis)
  - Mechanically-fixable bugs the fixer fixed -> a GitHub PR (proposal, never merged)
  - A notebook that now passes                -> CLOSE its open issue/PR
Nothing is auto-merged. Preflight scaffolding (gradio/audio/input skips) is
never shown to authors — PRs use the genuine `_fixed.ipynb` only.

Idempotency: each artifact carries a hidden fingerprint marker
(`<!-- agent-managed: <kind>:<stem> -->`). Re-runs update the existing artifact
instead of opening duplicates.

Modes:
  --mode audit       Reconcile issues/PRs from the latest result of every notebook.
  --mode pr-comment  Post ONE idempotent status comment on a PR summarising the
                     result of the notebooks changed in that PR (the PR trigger).

Safety:
  --dry-run is the DEFAULT. It computes the desired GitHub state and prints it
  (and writes previews to logs/github_preview/), calling `gh` for nothing.
  Pass --publish to actually create/update/close via the `gh` CLI.

Usage:
  # Preview what would be filed, against the current results (no repo needed):
  python3 tools/report_github.py --mode audit

  # Actually file/update/close against a repo (requires gh auth + access):
  python3 tools/report_github.py --mode audit --repo ORG/tutorials \
      --upstream-checkout /path/to/tutorials --publish

  # Preview the PR status comment for the notebooks changed in PR #42:
  python3 tools/report_github.py --mode pr-comment --pr-number 42 \
      --notebooks inference/foo.ipynb fine_tune/bar.ipynb

  # Post/update the single status comment on PR #42 (requires gh auth + access):
  python3 tools/report_github.py --mode pr-comment --pr-number 42 \
      --repo ORG/tutorials --notebooks inference/foo.ipynb --publish
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Reuse the dashboard's loaders so "latest result per notebook" is defined once.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import status as status_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = REPO_ROOT / "logs" / "github_preview"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

MANAGED = "agent-managed"  # marker prefix embedded in every artifact body

# Actions the reconciler can decide on per notebook.
CREATE_ISSUE = "CREATE_ISSUE"
UPDATE_ISSUE = "UPDATE_ISSUE"
CREATE_PR = "CREATE_PR"
UPDATE_PR = "UPDATE_PR"
CLOSE = "CLOSE"
NOOP = "NOOP"


# ── Desired-state computation (pure, no GitHub calls) ─────────────────────────

def marker(kind: str, stem: str) -> str:
    """Hidden fingerprint embedded in issue/PR bodies for dedup."""
    return f"<!-- {MANAGED}: {kind}:{stem} -->"


def _counts_as_pass(payload: dict, expected_partial: bool) -> bool:
    st = payload.get("status")
    return st == "pass" or (st == "partial" and expected_partial)


def split_issues(payload: dict) -> tuple[list, list]:
    """Return (pr_fixes, issue_problems).

    pr_fixes        — the validated fix entries (=> propose via PR). Source of
                      truth is the retrieved artifact + fixes[], NOT issue↔fix
                      cell-index correlation: the verify and fix agents do not
                      always index the same cell the same way, so we key off the
                      presence of a validated fix + a genuine-fix notebook.
    issue_problems  — issues still unresolved (=> diagnose via issue):
                      always needs_author/infra_blocked; plus auto_fixable ONLY
                      when no validated fix was produced for this notebook.

    Limitation: if a notebook has several auto_fixable issues and the fixer only
    validated some, we attribute all auto_fixable ones to the PR. The PR diff
    shows exactly what changed, so a reviewer sees the real scope.
    """
    issues = payload.get("issues") or []
    fixes = payload.get("fixes") or []
    have_artifact = bool(payload.get("fixed_notebook"))
    validated_fixes = [f for f in fixes if f.get("validated")]
    has_validated = bool(validated_fixes) and have_artifact

    pr_fixes = validated_fixes if has_validated else []

    issue_problems = []
    for iss in issues:
        fixability = iss.get("fixability")
        if fixability in ("needs_author", "infra_blocked"):
            issue_problems.append(iss)
        elif fixability == "auto_fixable" and not has_validated:
            issue_problems.append(iss)
        # auto_fixable + has_validated => considered covered by the PR
    return pr_fixes, issue_problems


def desired_state(rows_payloads: list[dict]) -> list[dict]:
    """Compute the desired GitHub artifact for each notebook.

    rows_payloads: list of {stem, rel_key, payload, expected_partial}
    Returns a list of plans: {stem, rel_key, want_issue, want_pr, pr_fixes,
    issue_problems, passes, payload}.
    """
    plans = []
    for r in rows_payloads:
        payload = r["payload"]
        passes = _counts_as_pass(payload, r["expected_partial"])
        pr_fixes, issue_problems = split_issues(payload)
        plans.append({
            "stem": r["stem"],
            "rel_key": r["rel_key"],
            "payload": payload,
            "passes": passes,
            # A validated fix is a proposal to make REGARDLESS of whether the
            # locally-fixed notebook now passes — upstream is still broken until
            # the PR merges. Issues only for unresolved problems on a notebook
            # that does not (yet) pass.
            "want_issue": bool(issue_problems) and not passes,
            "want_pr": bool(pr_fixes),
            "pr_fixes": pr_fixes,
            "issue_problems": issue_problems,
        })
    return plans


# ── Body rendering (mirrors upstream_prs/*.md structure) ──────────────────────

def _issue_lines(iss: dict) -> list[str]:
    out = [
        f"- **Error type:** `{iss.get('error_type', 'unknown')}`"
        f"  ·  **Cell:** {iss.get('cell_index', '?')}"
        f"  ·  **Fixability:** `{iss.get('fixability', 'unknown')}`",
    ]
    if iss.get("description"):
        out.append(f"  - **What broke:** {iss['description'].strip()}")
    if iss.get("proposed_fix"):
        out.append(f"  - **Suggested fix:** {iss['proposed_fix'].strip()}")
    return out


def render_issue_body(plan: dict) -> str:
    p = plan["payload"]
    lines = [
        marker("issue", plan["stem"]),
        f"## Regression report — `{plan['rel_key']}`",
        "",
        "_Filed automatically by the notebook regression agent. "
        "These issues require author judgment to fix._",
        "",
        f"**Status:** {p.get('status', 'fail')}",
    ]
    if p.get("docker_image_resolved"):
        lines.append(f"**Tested in:** `{p['docker_image_resolved']}`")
    if p.get("summary"):
        lines += ["", f"> {p['summary'].strip()}"]
    lines += ["", "### Issues needing author action", ""]
    for iss in plan["issue_problems"]:
        lines += _issue_lines(iss)
    if plan["pr_fixes"]:
        lines += ["", "_A separate PR proposes mechanical fixes for the "
                  "auto-fixable issues in this notebook._"]
    lines += ["", "---", "_Agent-managed: this issue is updated on each run and "
              "auto-closed when the notebook passes._"]
    return "\n".join(lines)


def _cell_diff(rel_key: str, fixed_path: str) -> list[str]:
    """Per-cell before/after for cells that changed between original and fixed."""
    orig_p = NOTEBOOKS_DIR / rel_key
    fixed_p = Path(fixed_path)
    if not orig_p.exists() or not fixed_p.exists():
        return ["_(diff unavailable — original or fixed notebook not found locally)_"]
    try:
        orig = json.load(open(orig_p))["cells"]
        fixed = json.load(open(fixed_p))["cells"]
    except (OSError, json.JSONDecodeError, KeyError):
        return ["_(diff unavailable — could not parse notebooks)_"]

    import difflib
    out = []
    for i in range(min(len(orig), len(fixed))):
        o = "".join(orig[i].get("source", []))
        f = "".join(fixed[i].get("source", []))
        if o != f:
            diff = difflib.unified_diff(
                o.splitlines(), f.splitlines(),
                lineterm="", n=3,
            )
            # drop the ---/+++ file headers; keep @@ hunks and +/-/context lines
            body = [ln for ln in diff if not ln.startswith(("---", "+++"))]
            out += [f"#### Cell {i}", "```diff", *body, "```", ""]
    return out or ["_(no cell-source differences detected)_"]


def render_pr_body(plan: dict) -> str:
    p = plan["payload"]
    lines = [
        marker("fix", plan["stem"]),
        f"## Proposed fix — `{plan['rel_key']}`",
        "",
        "_Opened automatically by the notebook regression agent. "
        "Each change below was validated by re-running the notebook on an "
        "AMD ROCm GPU. **Review before merging — this PR is a proposal, not "
        "an auto-merge.**_",
        "",
    ]
    if p.get("docker_image_resolved"):
        lines.append(f"**Validated in:** `{p['docker_image_resolved']}`")
    lines += ["", "### Fixes applied", ""]
    for fx in (p.get("fixes") or []):
        if not fx.get("validated"):
            continue
        lines.append(f"- **Cell {fx.get('cell_index', '?')}:** "
                     f"{fx.get('description', '').strip()}")
    lines += ["", "### Diff", ""]
    lines += _cell_diff(plan["rel_key"], p.get("fixed_notebook", ""))
    return "\n".join(lines)


# ── PR status comment (the PR trigger) ────────────────────────────────────────

def pr_comment_marker(pr_number: int) -> str:
    """Hidden fingerprint so the PR status comment is updated in place, not duplicated."""
    return f"<!-- {MANAGED}: pr-comment:{pr_number} -->"


_STATUS_EMOJI = {"pass": "✅", "partial": "🟡", "fail": "❌"}


def _pr_row_status(payload: dict, expected_partial: bool) -> str:
    """Human status label for the table: pass / partial / fail / (no result)."""
    st = payload.get("status")
    if st == "pass":
        return "pass"
    if st == "partial":
        return "pass" if expected_partial else "partial"
    if st == "fail":
        return "fail"
    return st or "unknown"


def _top_failure_reason(payload: dict) -> str:
    """The single most useful line to show authors for a failing notebook."""
    issues = payload.get("issues") or []
    if issues:
        first = issues[0]
        et = first.get("error_type", "error")
        desc = (first.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "…"
        return f"`{et}` — {desc}" if desc else f"`{et}`"
    summary = (payload.get("summary") or "").strip().replace("\n", " ")
    if len(summary) > 160:
        summary = summary[:157] + "…"
    return summary or "—"


def build_pr_comment_rows(changed: list[str], results_dir: Path) -> list[dict]:
    """For each changed notebook (rel path), pull its latest result payload.

    Returns rows: {rel_key, stem, payload_or_None, expected_partial}. Notebooks
    with no result JSON are reported explicitly as "no result" — never hidden.
    """
    manifest = status_mod.load_manifest()
    results = status_mod.load_results(results_dir)
    rows = []
    for rel in changed:
        stem = Path(rel).stem
        entry = manifest.get(rel) or manifest.get(Path(rel).name) or {}
        rec = results.get(stem)
        rows.append({
            "rel_key": rel,
            "stem": stem,
            "payload": rec["payload"] if rec else None,
            "expected_partial": entry.get("expected_result") == "partial",
        })
    return rows


def render_pr_comment(pr_number: int, rows: list[dict]) -> str:
    """One Markdown comment: header + per-notebook status table + failure detail."""
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["payload"] and _pr_row_status(r["payload"], r["expected_partial"]) == "pass")
    n_fail = sum(1 for r in rows if r["payload"] and _pr_row_status(r["payload"], r["expected_partial"]) == "fail")
    n_partial = sum(1 for r in rows if r["payload"] and _pr_row_status(r["payload"], r["expected_partial"]) == "partial")
    n_missing = sum(1 for r in rows if not r["payload"])

    lines = [
        pr_comment_marker(pr_number),
        "## Notebook regression — changed notebooks",
        "",
        "_Posted automatically by the notebook regression agent. Each changed "
        "notebook below was executed on an AMD ROCm GPU. This is a report, not a "
        "merge gate._",
        "",
        f"**Summary:** {n_pass}/{n_total} pass"
        + (f", {n_partial} partial" if n_partial else "")
        + (f", {n_fail} fail" if n_fail else "")
        + (f", {n_missing} no-result" if n_missing else ""),
        "",
        "| Notebook | Result | Detail |",
        "|----------|--------|--------|",
    ]
    for r in rows:
        if not r["payload"]:
            lines.append(f"| `{r['rel_key']}` | ⚠️ no result | _agent produced no result JSON_ |")
            continue
        st = _pr_row_status(r["payload"], r["expected_partial"])
        emoji = _STATUS_EMOJI.get(st, "❔")
        detail = "—" if st == "pass" else _top_failure_reason(r["payload"])
        # Escape pipes in detail so the table doesn't break.
        detail = detail.replace("|", "\\|")
        lines.append(f"| `{r['rel_key']}` | {emoji} {st} | {detail} |")

    lines += [
        "",
        "---",
        "_Agent-managed: this comment is updated in place on every push to the PR._",
    ]
    return "\n".join(lines)


def find_pr_comment(repo: str, pr_number: int) -> dict | None:
    """Find the existing agent-managed status comment on this PR via its marker."""
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments", "--paginate"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    try:
        comments = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    mk = pr_comment_marker(pr_number)
    for c in comments:
        if mk in (c.get("body") or ""):
            return c
    return None


def post_pr_comment(repo: str, pr_number: int, body: str) -> None:
    """Create or update (in place) the single agent-managed PR status comment."""
    existing = find_pr_comment(repo, pr_number)
    if existing:
        subprocess.run(
            ["gh", "api", "--method", "PATCH",
             f"repos/{repo}/issues/comments/{existing['id']}",
             "-f", f"body={body}"],
            check=False,
        )
    else:
        subprocess.run(
            ["gh", "pr", "comment", str(pr_number), "--repo", repo, "--body", body],
            check=False,
        )


def run_pr_comment(pr_number: int, changed: list[str], repo: str | None,
                   results_dir: Path, publish: bool) -> int:
    rows = build_pr_comment_rows(changed, results_dir)
    body = render_pr_comment(pr_number, rows)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview = PREVIEW_DIR / f"pr-{pr_number}.comment.md"
    preview.write_text(body)

    mode = "PUBLISH" if publish else "DRY-RUN"
    n_fail = sum(1 for r in rows if r["payload"]
                 and _pr_row_status(r["payload"], r["expected_partial"]) == "fail")
    print(f"\n[{mode}] pr-comment on PR #{pr_number} → {len(rows)} changed notebook(s), "
          f"{n_fail} failing")
    print(f"Preview written to {preview}")
    if publish:
        if not repo:
            print("error: --publish requires --repo OWNER/NAME", file=sys.stderr)
            return 2
        post_pr_comment(repo, pr_number, body)
        print(f"Posted/updated status comment on {repo}#{pr_number}")
    else:
        print("(dry-run — nothing sent to GitHub. Re-run with --repo ... --publish to post.)")
    return 0


# ── gh CLI layer (only touched when --publish) ────────────────────────────────

def gh_json(repo: str, args: list[str]) -> list:
    cmd = ["gh", *args, "--repo", repo, "--json", "number,state,body,title"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return []
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []


def find_managed(repo: str, kind: str, stem: str, is_pr: bool) -> dict | None:
    """Find an existing agent-managed issue/PR for this stem via the body marker."""
    listing = gh_json(repo, ["pr" if is_pr else "issue", "list",
                             "--state", "all", "--limit", "500"])
    mk = marker(kind, stem)
    for item in listing:
        if mk in (item.get("body") or ""):
            return item
    return None


# ── Reconcile (decide action per plan; execute only when publish) ─────────────

def reconcile(plans: list[dict], repo: str | None, upstream_checkout: str | None,
              publish: bool) -> list[dict]:
    decisions = []
    for plan in plans:
        stem = plan["stem"]

        # Issue side
        if plan["want_issue"]:
            existing = find_managed(repo, "issue", stem, is_pr=False) if (publish and repo) else None
            action = UPDATE_ISSUE if existing else CREATE_ISSUE
            decisions.append({"stem": stem, "kind": "issue", "action": action,
                              "plan": plan, "existing": existing})
        # PR side
        if plan["want_pr"]:
            existing = find_managed(repo, "fix", stem, is_pr=True) if (publish and repo) else None
            action = UPDATE_PR if existing else CREATE_PR
            decisions.append({"stem": stem, "kind": "pr", "action": action,
                              "plan": plan, "existing": existing})
        # Close side — a notebook that passes as published (no fix to propose,
        # no unresolved problems) with any open managed artifact.
        if (plan["passes"] and not plan["want_pr"] and not plan["want_issue"]
                and publish and repo):
            for kind, is_pr in (("issue", False), ("fix", True)):
                ex = find_managed(repo, kind, stem, is_pr=is_pr)
                if ex and ex.get("state") == "OPEN":
                    decisions.append({"stem": stem, "kind": kind, "action": CLOSE,
                                      "plan": plan, "existing": ex})

    if publish:
        for d in decisions:
            _execute(d, repo, upstream_checkout)
    return decisions


def _execute(d: dict, repo: str, upstream_checkout: str | None) -> None:
    """Perform the gh action. (PR creation requires upstream_checkout.)"""
    plan, action = d["plan"], d["action"]
    if action in (CREATE_ISSUE, UPDATE_ISSUE):
        body = render_issue_body(plan)
        title = f"[regression] {plan['rel_key']}"
        if action == CREATE_ISSUE:
            subprocess.run(["gh", "issue", "create", "--repo", repo,
                            "--title", title, "--body", body,
                            "--label", "regression"], check=False)
        else:
            subprocess.run(["gh", "issue", "edit", str(d["existing"]["number"]),
                            "--repo", repo, "--body", body], check=False)
            if d["existing"].get("state") != "OPEN":
                subprocess.run(["gh", "issue", "reopen",
                                str(d["existing"]["number"]), "--repo", repo], check=False)
    elif action == CLOSE:
        sub = "pr" if d["kind"] == "fix" else "issue"
        subprocess.run(["gh", sub, "close", str(d["existing"]["number"]),
                        "--repo", repo, "--comment",
                        "Notebook now passes regression — closing automatically."],
                       check=False)
    elif action in (CREATE_PR, UPDATE_PR):
        # PR creation needs a checkout to commit the patched notebook into a branch.
        # Left to the caller-provided upstream_checkout; printed if absent.
        if not upstream_checkout:
            sys.stderr.write(f"warning: --upstream-checkout required to open PR for "
                             f"{plan['stem']}; skipped\n")
            return
        _open_or_update_pr(d, repo, upstream_checkout)


def _open_or_update_pr(d: dict, repo: str, checkout: str) -> None:
    plan = d["plan"]
    stem = plan["stem"]
    branch = f"agent/fix/{stem}"
    co = Path(checkout)
    body = render_pr_body(plan)
    fixed = plan["payload"].get("fixed_notebook")
    target = co / plan["rel_key"]
    # Branch from default, copy patched notebook in, commit, push.
    subprocess.run(["git", "-C", str(co), "fetch", "origin"], check=False)
    subprocess.run(["git", "-C", str(co), "checkout", "-B", branch], check=False)
    if fixed and Path(fixed).exists() and target.parent.exists():
        target.write_text(Path(fixed).read_text())
    subprocess.run(["git", "-C", str(co), "add", plan["rel_key"]], check=False)
    subprocess.run(["git", "-C", str(co), "commit", "-m",
                    f"fix({stem}): apply validated regression fixes"], check=False)
    subprocess.run(["git", "-C", str(co), "push", "-f", "origin", branch], check=False)
    if d["action"] == CREATE_PR:
        subprocess.run(["gh", "pr", "create", "--repo", repo, "--head", branch,
                        "--title", f"fix: {plan['rel_key']} regression",
                        "--body", body, "--label", "agent-fix"], check=False)
    else:
        subprocess.run(["gh", "pr", "edit", str(d["existing"]["number"]),
                        "--repo", repo, "--body", body], check=False)


# ── Preview + summary ─────────────────────────────────────────────────────────

def write_previews(plans: list[dict]) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        if plan["want_issue"]:
            (PREVIEW_DIR / f"{plan['stem']}.issue.md").write_text(render_issue_body(plan))
        if plan["want_pr"]:
            (PREVIEW_DIR / f"{plan['stem']}.pr.md").write_text(render_pr_body(plan))


def load_rows_payloads(results_dir: Path) -> list[dict]:
    """Latest result payload per manifest notebook (skips never-tested / skipped)."""
    manifest = status_mod.load_manifest()
    results = status_mod.load_results(results_dir)
    out = []
    for key, entry in manifest.items():
        entry = entry or {}
        if entry.get("skip"):
            continue
        stem = Path(key).stem
        rec = results.get(stem)
        if rec is None:
            continue
        out.append({
            "stem": stem,
            "rel_key": key,
            "payload": rec["payload"],
            "expected_partial": entry.get("expected_result") == "partial",
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Report regression results to GitHub.")
    p.add_argument("--mode", choices=["audit", "pr-comment"], default="audit")
    p.add_argument("--repo", default=None, help="Target repo OWNER/NAME (required for --publish).")
    p.add_argument("--upstream-checkout", default=None,
                   help="Local checkout of the notebooks repo (required to open PRs).")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--pr-number", type=int, default=None,
                   help="PR number to comment on (required for --mode pr-comment).")
    p.add_argument("--notebooks", nargs="*", default=[],
                   help="Changed notebook rel-paths for --mode pr-comment.")
    p.add_argument("--publish", action="store_true",
                   help="Actually create/update/close via gh. Default is dry-run.")
    args = p.parse_args()

    if args.publish and not args.repo:
        print("error: --publish requires --repo OWNER/NAME", file=sys.stderr)
        return 2

    results_dir = Path(args.results_dir).resolve() if args.results_dir else (REPO_ROOT / "results")

    if args.mode == "pr-comment":
        if args.pr_number is None:
            print("error: --mode pr-comment requires --pr-number", file=sys.stderr)
            return 2
        if not args.notebooks:
            print("error: --mode pr-comment requires --notebooks <paths...>", file=sys.stderr)
            return 2
        return run_pr_comment(args.pr_number, args.notebooks, args.repo,
                              results_dir, args.publish)

    rows = load_rows_payloads(results_dir)
    plans = desired_state(rows)
    decisions = reconcile(plans, args.repo, args.upstream_checkout, args.publish)

    # Always write previews so the output is inspectable.
    write_previews(plans)

    # Summary
    n_issue = sum(1 for d in decisions if d["kind"] == "issue" and d["action"] != CLOSE)
    n_pr = sum(1 for d in decisions if d["kind"] == "pr" and d["action"] != CLOSE)
    n_close = sum(1 for d in decisions if d["action"] == CLOSE)
    mode = "PUBLISH" if args.publish else "DRY-RUN"
    print(f"\n[{mode}] audit over {len(plans)} tested notebook(s) → "
          f"{n_issue} issue(s), {n_pr} fix-PR(s), {n_close} close(s)")
    print(f"Previews written to {PREVIEW_DIR}\n")
    for d in decisions:
        plan = d["plan"]
        detail = ""
        if d["kind"] == "issue":
            detail = f"{len(plan['issue_problems'])} problem(s)"
        elif d["kind"] == "pr":
            detail = f"{len(plan['pr_fixes'])} validated fix(es)"
        print(f"  {d['action']:13} {d['kind']:5} {plan['rel_key']}  ({detail})")
    if not args.publish:
        print("\n(dry-run — nothing sent to GitHub. Re-run with --repo ... --publish to apply.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
