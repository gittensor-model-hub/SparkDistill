#!/usr/bin/env bash
# Refresh datasets/canonical.json from the live HF mining dataset manifest.
#
#   scripts/update_canonical_pin.sh
#
# Run after a dataset registry merge republishes gittensor-model-hub/sparkproof-mining.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m eval.update_canonical_pin "$@"
