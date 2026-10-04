#!/usr/bin/env bash
# Fetch the public BANC v888 brain-and-nerve-cord data. No authentication.
# Data: CC BY 4.0 — BANC / FlyWire Consortium, Lee Lab (Harvard), Zetta.ai.
set -euo pipefail

DEST="${1:-data/banc}"
CODEX="https://storage.googleapis.com/flywire-data/codex/data/banc/888"
LEELAB="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888"

mkdir -p "$DEST"
echo "Downloading BANC v888 -> $DEST  (~86 MB)"

fetch() {
  local url="$1" name="$2"
  if [[ -s "$DEST/$name" ]]; then
    echo "  skip   $name"
  else
    echo "  fetch  $name"
    curl -sS --retry 3 --fail -o "$DEST/$name" "$url/$name"
  fi
}

# NOTE: the BANC edge list is called connections_princeton.csv.gz, not
# connections.csv.gz as in the FAFB export.
fetch "$CODEX"  neurons.csv.gz
fetch "$CODEX"  connections_princeton.csv.gz
fetch "$LEELAB" banc_888_meta.feather

echo
ls -lh "$DEST"
echo "OK"
