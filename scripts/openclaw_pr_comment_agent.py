#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise SystemExit("GITHUB_EVENT_PATH is not set")
    return json.loads(Path(event_path).read_text())


def is_pr_comment_event(event: dict) -> tuple[bool, int | None, str, str]:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    comment = event.get("comment", {})
    author = comment.get("user", {}).get("login", "")
    body = comment.get("body", "")

    if author.endswith("[bot]") or author in {"github-actions[bot]"}:
        return False, None, author, body
    if "[openclaw-skip]" in body.lower():
        return False, None, author, body

    if event_name == "issue_comment":
        issue = event.get("issue", {})
        if issue.get("pull_request"):
            return True, int(issue["number"]), author, body
        return False, None, author, body

    if event_name == "pull_request_review_comment":
        pr = event.get("pull_request", {})
        if pr.get("number"):
            return True, int(pr["number"]), author, body
        return False, None, author, body

    return False, None, author, body


def gh_json(args: list[str]) -> dict:
    result = run(["gh", *args, "--json", "title,body,headRefName,baseRefName,headRefOid,url,author"], cwd=REPO_ROOT)
    return json.loads(result.stdout)


def checkout_pr(pr_number: int, repo: str) -> None:
    run(["gh", "pr", "checkout", str(pr_number), "--repo", repo], cwd=REPO_ROOT)


def agent_prompt(pr_number: int, pr: dict, comment_author: str, comment_body: str, comment_url: str) -> str:
    return textwrap.dedent(
        f"""
        You are an autonomous OpenClaw coding agent responding to a GitHub PR comment.

        Repository: {os.environ.get('GITHUB_REPOSITORY', '')}
        PR #{pr_number}: {pr.get('title', '')}
        PR URL: {pr.get('url', '')}
        Base branch: {pr.get('baseRefName', '')}
        Head branch: {pr.get('headRefName', '')}
        Head SHA: {pr.get('headRefOid', '')}

        Comment author: {comment_author}
        Comment URL: {comment_url}
        Comment body:
        ---
        {comment_body}
        ---

        Your task:
        - Inspect the checked-out repository and the PR comment.
        - If code or docs need updating, make the minimum safe changes.
        - Run the most relevant validation you can.
        - Leave the repo in a commit-ready state.
        - If the comment is purely informational, explain the needed response clearly.

        Important:
        - Keep changes focused.
        - Avoid broad refactors unless the comment clearly requires them.
        - Prefer minimal, reviewable edits.
        """
    ).strip()


def git_status() -> str:
    result = run(["git", "status", "--short"], cwd=REPO_ROOT)
    return result.stdout.strip()


def current_branch() -> str:
    result = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT)
    return result.stdout.strip()


def maybe_commit_and_push(pr_number: int) -> str | None:
    if not git_status():
        return None
    run(["git", "config", "user.name", "openclaw-bot"], cwd=REPO_ROOT)
    run(["git", "config", "user.email", "openclaw-bot@users.noreply.github.com"], cwd=REPO_ROOT)
    run(["git", "add", "-A"], cwd=REPO_ROOT)
    run(["git", "commit", "-m", f"fix: address PR comment #{pr_number}"], cwd=REPO_ROOT)
    branch = current_branch()
    run(["git", "push", "origin", branch], cwd=REPO_ROOT)
    sha = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    return sha


def main() -> int:
    event = load_event()
    ok, pr_number, author, body = is_pr_comment_event(event)
    if not ok or pr_number is None:
        print("No action: not a PR comment event or it was skipped.")
        return 0

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        raise SystemExit("GITHUB_REPOSITORY is not set")

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        os.environ.setdefault("GH_TOKEN", token)
        os.environ.setdefault("GITHUB_TOKEN", token)

    checkout_pr(pr_number, repo)
    pr = gh_json(["pr", "view", str(pr_number), "--repo", repo])
    comment_url = event.get("comment", {}).get("html_url", "")
    prompt = agent_prompt(pr_number, pr, author, body, comment_url)

    agent_cmd = [
        "openclaw",
        "agent",
        "--agent",
        "main",
        "--local",
        "--thinking",
        "medium",
        "--timeout",
        "3600",
        "--json",
        "--message",
        prompt,
    ]
    agent_run = run(agent_cmd, cwd=REPO_ROOT)
    print(agent_run.stdout)
    if agent_run.stderr:
        print(agent_run.stderr, file=sys.stderr)

    commit_sha = maybe_commit_and_push(pr_number)
    if commit_sha:
        print(f"Committed and pushed agent changes at {commit_sha}")
    else:
        print("No file changes produced by the agent.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
