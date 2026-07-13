#!/usr/bin/env bash
# Filter raw teacher trajectories (drop empty/malformed rows and refusals)
# before folding them into SFT records.
#
#   scripts/filter_trajectories.sh --in data/processed/phase1_trajectories.jsonl --out data/processed/phase1_trajectories.filtered.jsonl [--min-response-chars 16 --dedupe-prompts]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run python -m teacher.filter "$@"
