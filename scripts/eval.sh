#!/usr/bin/env bash
# Score a checkpoint against the quality benchmark basket, optionally
# comparing against the current frontier checkpoint's scores.
#
#   scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 [--compare-frontier] [--frontier-scores eval/results/frontier.json]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

checkpoint=""
compare_frontier=false
frontier_scores="eval/results/frontier.json"
extra_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --compare-frontier) compare_frontier=true; shift ;;
    --frontier-scores) frontier_scores="$2"; shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

if [ -z "$checkpoint" ]; then
  echo "usage: scripts/eval.sh --checkpoint <path> [--compare-frontier] [--frontier-scores <path>]" >&2
  exit 1
fi

candidate_scores="eval/results/candidate.json"
if [ "${#extra_args[@]}" -gt 0 ]; then
  uv run python -m eval.harness --checkpoint "$checkpoint" --out "$candidate_scores" "${extra_args[@]}"
else
  uv run python -m eval.harness --checkpoint "$checkpoint" --out "$candidate_scores"
fi

if [ "$compare_frontier" = true ]; then
  if [ ! -f "$frontier_scores" ]; then
    echo "no frontier scores found at $frontier_scores — run eval.harness on the current frontier checkpoint first" >&2
    exit 1
  fi
  uv run python -m eval.score --candidate "$candidate_scores" --frontier "$frontier_scores" --out eval/results/report.json
  cat eval/results/report.json
fi
