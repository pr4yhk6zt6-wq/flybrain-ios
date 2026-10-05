#!/usr/bin/env bash
# Fetch the Janelia / Google DeepMind `flybody` fruit-fly body model.
# Apache-2.0. No authentication.
#
#   Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
#   Nature (2025).  https://github.com/TuragaLab/flybody
set -euo pipefail

DEST="${1:-data/flybody}"
URL="https://github.com/TuragaLab/flybody/archive/refs/heads/main.zip"

mkdir -p "$DEST"
if [[ -s "$DEST/flybody-main/flybody/fruitfly/assets/fruitfly.xml" ]]; then
  echo "  skip   flybody (already present)"
else
  echo "Downloading flybody -> $DEST  (~41 MB)"
  curl -sSL --retry 3 --fail -o "$DEST/flybody.zip" "$URL"
  (cd "$DEST" && unzip -q -o flybody.zip)
  rm -f "$DEST/flybody.zip"
fi
ls -d "$DEST/flybody-main/flybody/fruitfly/assets/fruitfly.xml"
echo OK
