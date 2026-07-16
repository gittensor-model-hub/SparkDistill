"""Shared student-model serving for the eval harness.

Starting vLLM once and reusing the OpenAI-compatible endpoint across the full
benchmark basket avoids reloading the checkpoint for every lm-eval subprocess and
prevents TritonBench from spawning a second server.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class EvalSession:
    endpoint: str
    model_name: str


def served_model_name(model_path: str) -> str:
    return Path(model_path).name or model_path


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


@contextmanager
def eval_session(model_path: str, *, serve: bool = False) -> Iterator[EvalSession | None]:
    """Yield a shared endpoint for the basket, or None for the legacy HF path.

    Resolution order:
    1. ``SPARKDISTILL_STUDENT_ENDPOINT`` — use an already-running server.
    2. ``serve=True`` or ``SPARKDISTILL_EVAL_SERVE=1`` — start vLLM once.
    3. Otherwise yield None and callers fall back to per-benchmark HF loads.
    """
    model_name = served_model_name(model_path)
    env_endpoint = os.environ.get("SPARKDISTILL_STUDENT_ENDPOINT")
    if env_endpoint:
        yield EvalSession(env_endpoint, model_name)
        return

    if serve or _env_truthy("SPARKDISTILL_EVAL_SERVE"):
        from eval.triton_bench import serve_checkpoint

        with serve_checkpoint(model_path, served_model_name=model_name) as endpoint:
            yield EvalSession(endpoint, model_name)
        return

    yield None
