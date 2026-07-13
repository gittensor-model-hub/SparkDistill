#!/usr/bin/env bash
# Summarize a teacher-trajectory jsonl (reasoning-capture rate, provider mix,
# length distributions, empty/malformed counts).
#
#   scripts/report_trajectories.sh --in data/processed/phase1_trajectories.jsonl [--out eval/results/trajectory_report.json]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run python -m teacher.report "$@"
