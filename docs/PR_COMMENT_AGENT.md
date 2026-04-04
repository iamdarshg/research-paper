# PR Comment Agent Hook

This repository includes a GitHub Actions workflow that can spin up a local OpenClaw agent whenever a pull request receives a comment.

## How it works

- The workflow listens to:
  - `issue_comment` events on PRs
  - `pull_request_review_comment` events
- It checks out the PR branch
- It calls the local OpenClaw CLI with the PR comment and PR context
- If the agent changes files, the hook commits and pushes those changes back to the branch

## Requirements

This is intended for a self-hosted runner with:

- `openclaw` installed and configured
- `gh` installed and authenticated
- access to the repository checkout

The workflow uses a self-hosted runner label set to:

- `self-hosted`
- `openclaw`

If your runner uses a different label, update `.github/workflows/pr-comment-agent.yml`.

## Safety controls

- Bot comments are ignored.
- Comments containing `[openclaw-skip]` are ignored.
- The agent runs on the checked-out PR branch and keeps changes focused.

## Manual use

You can also run the driver script locally after setting `GITHUB_EVENT_PATH`, `GITHUB_EVENT_NAME`, `GITHUB_REPOSITORY`, and `GH_TOKEN`.
