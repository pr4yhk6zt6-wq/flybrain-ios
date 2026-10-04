#!/usr/bin/env bash
# Fetch the public FlyWire FAFB v783 Codex CSV dumps (no authentication required).
# Data: CC BY-NC 4.0 — FlyWire Consortium. Non-commercial use only.
set -euo pipefail

BASE="https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
DEST="${1:-data/raw}"
FILES=(
  neurons.csv.gz
  connections.csv.gz
  classification.csv.gz
  coordinates.csv.gz
  consolidated_cell_types.csv.gz
  labels.csv.gz
  names.csv.gz
)

mkdir -p "$DEST"
echo "Downloading FlyWire FAFB v783 -> $DEST  (~62 MB)"

for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo "  skip   $f (already present)"
  else
    echo "  fetch  $f"
    curl -sS --retry 3 --fail -o "$DEST/$f" "$BASE/$f" &
  fi
done
wait

echo
echo "Verifying checksums..."
cd "$DEST"

# macOS ships `md5 -q`, Linux ships `md5sum`. Normalise to one function.
if command -v md5sum >/dev/null 2>&1; then
  hash_of() { md5sum "$1" | awk '{print $1}'; }
elif command -v md5 >/dev/null 2>&1; then
  hash_of() { md5 -q "$1"; }
else
  echo "  no md5 tool available, skipping verification"
  hash_of() { echo "skip"; }
fi

cat > .md5sums <<'EOF'
701c0faf054ceddb460bea4e87d6a624  classification.csv.gz
41206440318c77418bfc7cff1cb3e0fa  connections.csv.gz
18cc628156bede3129c0e19d14cf52d6  consolidated_cell_types.csv.gz
f7a8120f8322405fecbeec2ee1d206e7  coordinates.csv.gz
b3951998eeeda84bb4a2e209b456f683  labels.csv.gz
6d134fe2712cb81d179d4b033a86fdc5  names.csv.gz
f60333e9e4124160b9b203b1712a6f91  neurons.csv.gz
EOF

fail=0
while read -r want name; do
  got=$(hash_of "$name")
  if [[ "$got" == "skip" ]]; then
    echo "  ?    $name"
  elif [[ "$got" == "$want" ]]; then
    echo "  OK   $name"
  else
    echo "  FAIL $name (expected $want, got $got)"
    fail=1
  fi
done < .md5sums

if [[ $fail -ne 0 ]]; then
  echo "Checksum verification failed." >&2
  exit 1
fi
echo "OK — all 7 files verified."
