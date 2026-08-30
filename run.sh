#!/usr/bin/env bash
#
# Travelers' Inn cancellation-risk pipeline — runner.
#
# Deliberately does NOT install dependencies: the assessment's GitHub Action
# installs requirements.txt itself before invoking this script, and doing it
# twice would be both slower and a spec violation.
#
# Usage:
#   bash run.sh                              # defaults from config.yaml
#   bash run.sh --set model.name=logistic    # any argument is forwarded
#   bash run.sh --compare-all                # every model in the registry
#
# Environment:
#   TI_DB_PATH=/path/to/hotel.db   use a database elsewhere
#   ALLOW_DOWNLOAD=0               never fetch; fail if the file is absent

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DB_PATH="${TI_DB_PATH:-data/hotel.db}"
DB_URL="https://techassessment.blob.core.windows.net/aiap25-assessment-data/hotel.db"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-1}"

# The dataset must not be committed (assessment rule), so in CI the file is
# absent and the pipeline would fail on a missing database. Fetching it on
# demand is what lets the GitHub Action actually exercise the pipeline. Local
# runs skip this entirely because the file already exists.
if [[ ! -f "$DB_PATH" ]]; then
  if [[ "$ALLOW_DOWNLOAD" != "1" ]]; then
    echo "ERROR: $DB_PATH not found and ALLOW_DOWNLOAD=0." >&2
    echo "       Download it from: $DB_URL" >&2
    exit 1
  fi
  echo "Database not found at $DB_PATH — downloading from the assessment source..."
  mkdir -p "$(dirname "$DB_PATH")"
  # Write to a temporary file first so an interrupted download cannot leave a
  # truncated database that later runs would treat as valid.
  TMP_DB="${DB_PATH}.partial"
  if ! curl --fail --silent --show-error --location --retry 3 \
            --output "$TMP_DB" "$DB_URL"; then
    rm -f "$TMP_DB"
    echo "ERROR: could not download the dataset from $DB_URL" >&2
    echo "       Download it manually and place it at $DB_PATH" >&2
    exit 1
  fi
  mv "$TMP_DB" "$DB_PATH"
  echo "Downloaded $(du -h "$DB_PATH" | cut -f1) to $DB_PATH"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python3

echo "Running pipeline with $($PYTHON_BIN --version)"
exec "$PYTHON_BIN" -m src.pipeline --config config.yaml "$@"
