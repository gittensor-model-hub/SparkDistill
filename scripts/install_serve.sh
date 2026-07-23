#!/usr/bin/env bash
# Install the pinned vLLM serving stack for TritonBench evaluation.
#
#   scripts/install_serve.sh
#
# The eval claim is only comparable between miner and validator when both serve
# the checkpoint with the same stack, so the version is pinned here. vLLM's
# torch pin conflicts with the Axolotl training environment, so it lives in its
# own venv (~/.sparkdistill-serve by default, override with SPARKDISTILL_SERVE_VENV).
#
# Installs the official vLLM+cu129 GPU wheel (not the generic PyPI build that can
# pull CPU-only torch and JIT-compile for minutes at engine start). Hopper H100/H200
# and Blackwell B200/RTX PRO 6000 CC nodes both use cu129 on vLLM 0.25.0.
#
# torchcodec (vLLM's optional video support) is removed: it dlopens system
# ffmpeg libraries that CC VMs don't ship, and text-only kernel evaluation
# never needs it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VERSION="0.25.0"
VENV="${SPARKDISTILL_SERVE_VENV:-$HOME/.sparkdistill-serve}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run scripts/install.sh first" >&2
  exit 1
fi

WHEEL="$(cd "$ROOT" && uv run python -c "from eval.serve_stack import vllm_wheel_url; print(vllm_wheel_url(version='${VLLM_VERSION}'))")"
INDEX="$(cd "$ROOT" && uv run python -c "from eval.serve_stack import pytorch_extra_index_url; print(pytorch_extra_index_url())")"

echo ">>> creating serve venv at $VENV (vllm==${VLLM_VERSION}, wheel=$(basename "$WHEEL"))"
uv venv "$VENV" --python 3.12 --allow-existing
# ninja: vLLM JIT-compiles kernels at engine start and dies without it.
VIRTUAL_ENV="$VENV" uv pip install -q "$WHEEL" ninja --extra-index-url "$INDEX"
VIRTUAL_ENV="$VENV" uv pip uninstall -q torchcodec 2>/dev/null || true

"$VENV/bin/python" -c "import torch, vllm; print(f'  vllm: {vllm.__version__}'); print(f'  torch: {torch.__version__} (cuda {torch.version.cuda})')"

echo ""
echo "Serve a checkpoint for TritonBench eval (venv bin must be on PATH — vLLM's"
echo "engine JIT shells out to ninja by bare name):"
echo "  PATH=\"$VENV/bin:\$PATH\" vllm serve <checkpoint-dir> --served-model-name <name> --port 8000"
echo "Then:"
echo "  uv run python -m eval.triton_bench --checkpoint <checkpoint-dir> \\"
echo "      --endpoint http://127.0.0.1:8000/v1 --quick --out eval/results/triton.json"
