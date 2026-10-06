#!/usr/bin/env bash
#
# Read this repository's CI without a browser.
#
#     bash tools/ci.sh list                 # the last few runs
#     bash tools/ci.sh steps <run-id>       # every step of every job
#     bash tools/ci.sh log   <job-id>       # the interesting lines of a job log
#     bash tools/ci.sh ipa   <run-id>       # the artifacts of a run
#     bash tools/ci.sh get   <artifact-id>  # download one artifact into /tmp/artz
#     bash tools/ci.sh cancel <run-id> ...  # stop runs a newer commit replaces
#
# The token, in this order: $FLYBRAIN_TOKEN, then `.github-token` beside the
# repository root (git-ignored — see .gitignore), and nothing else.
#
# It is deliberately *not* a default in this file: GitHub's push protection
# rejects a commit that carries one, and rightly — a script that reads CI is not
# worth a leaked credential. The file itself lives in git (a sandbox reset has
# lost the /tmp copy of it four times), the secret lives outside it.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN="${FLYBRAIN_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -s "$REPO_DIR/.github-token" ]; then
  TOKEN="$(tr -d '[:space:]' < "$REPO_DIR/.github-token")"
fi
if [ -z "$TOKEN" ]; then
  echo "no token: set FLYBRAIN_TOKEN, or write one to $REPO_DIR/.github-token" >&2
  echo "(it is git-ignored; the token is never committed)" >&2
  exit 2
fi
OWNER_REPO="pr4yhk6zt6-wq/flybrain-ios"
API="https://api.github.com/repos/$OWNER_REPO"
AUTH=(-H "Authorization: token $TOKEN")

case "${1:-}" in
  list)
    curl -sSL "${AUTH[@]}" "$API/actions/runs?per_page=8" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'workflow_runs' not in d:
    print('GitHub said:', str(d)[:300]); raise SystemExit(1)
for r in d['workflow_runs']:
    print(r['id'], r['head_sha'][:7], r['status'], r['conclusion'], r['created_at'])"
    ;;
  steps)
    curl -sSL "${AUTH[@]}" "$API/actions/runs/$2/jobs" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'jobs' not in d:
    print('GitHub said:', str(d)[:300]); raise SystemExit(1)
for j in d['jobs']:
    print('JOB', j['id'], j['status'], j['conclusion'])
    for s in j['steps']:
        cur = s['status'] != 'completed' or s['conclusion'] not in ('success', 'skipped')
        print('%s %2d %-45s %-11s %s' % ('>' if cur else ' ',
              s['number'], s['name'][:45], s['status'], s['conclusion']))"
    ;;
  log)
    curl -sSL "${AUTH[@]}" "$API/actions/jobs/$2/logs" -o /tmp/ci_job.log || true
    python3 - <<'PY'
import re
try:
    txt = open('/tmp/ci_job.log', errors='replace').read()
except OSError:
    print('no log downloaded'); raise SystemExit(1)
if '<Error>' in txt[:400]:
    print('the log is not available yet:', txt[:200].replace('\n', ' '))
    raise SystemExit(1)
txt = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', txt)
keys = ('error:', 'Executed ', 'FAIL ', 'skipped', 'Test Case', 'tests, ',
        'grep:', 'Error:', 'fatal')
for line in txt.split('\n'):
    s = line.strip()
    if any(k in s for k in keys) and 'appintents' not in s:
        print(s[:220])
PY
    ;;
  ipa)
    curl -sSL "${AUTH[@]}" "$API/actions/runs/$2/artifacts" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'artifacts' not in d:
    print('GitHub said:', str(d)[:300]); raise SystemExit(1)
for a in d['artifacts']:
    print(a['id'], a['name'], a['size_in_bytes'])"
    ;;
  get)
    rm -rf /tmp/artz && mkdir -p /tmp/artz
    curl -sSL "${AUTH[@]}" "$API/actions/artifacts/$2/zip" -o /tmp/art.zip
    (cd /tmp/artz && unzip -q -o /tmp/art.zip) && ls -lh /tmp/artz
    ;;
  cancel)
    for id in "${@:2}"; do
      curl -sSL -X POST "${AUTH[@]}" "$API/actions/runs/$id/cancel" >/dev/null
      echo "cancel requested: $id"
    done
    ;;
  *)
    sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 2
    ;;
esac
