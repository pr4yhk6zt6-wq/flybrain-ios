#!/usr/bin/env bash
# Restore the git identity and the authenticated remote.
#
# The workspace snapshot deliberately excludes `.git/config` (it can carry
# credentials), so a fresh session starts with no remote and no author. This
# script puts both back. It reads the token from `.secrets/github_token`
# (gitignored) or from $GITHUB_TOKEN, and never writes it into the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="pr4yhk6zt6-wq/flybrain-ios"
TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "$TOKEN" && -f .secrets/github_token ]]; then
  TOKEN="$(tr -d '[:space:]' < .secrets/github_token)"
fi

git config user.name  "Arena Agent"
git config user.email "agent@arena.ai"

if [[ -n "$TOKEN" ]]; then
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://${REPO%%/*}:${TOKEN}@github.com/${REPO}.git"
  echo "remote set (authenticated)"
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/${REPO}.git"
  echo "remote set (unauthenticated — pushes will need a token)"
fi
git remote -v | sed 's/:[^@]*@/:***@/'
