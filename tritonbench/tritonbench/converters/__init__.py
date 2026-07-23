"""Public converters API."""

from __future__ import annotations

from tritonbench.converters.difficulty_map import (
    CHANNEL_G,
    CHANNEL_T,
    LEVEL_DIRS,
    difficulty_to_level,
    infer_category,
    infer_required_patterns,
    infer_tags,
    level_dirname,
    slugify_id,
)
from tritonbench.converters.from_g_json import convert_g_channel, entry_to_problem as g_entry_to_problem
from tritonbench.converters.from_t_jsonl import convert_t_channel, entry_to_problem as t_entry_to_problem
from tritonbench.converters.generate_corpus import (
    BUGFIX_TEMPLATES,
    SEED_IDS,
    generate_corpus,
    synthetic_bugfix_problems,
    write_problem_yaml,
)
from tritonbench.converters.validate_problem import (
    ProblemSchemaError,
    load_and_validate_yaml,
    validate_corpus,
    validate_problem,
    validate_problem_strict,
)

__all__ = [
    "BUGFIX_TEMPLATES",
    "CHANNEL_G",
    "CHANNEL_T",
    "LEVEL_DIRS",
    "ProblemSchemaError",
    "SEED_IDS",
    "convert_g_channel",
    "convert_t_channel",
    "difficulty_to_level",
    "g_entry_to_problem",
    "generate_corpus",
    "infer_category",
    "infer_required_patterns",
    "infer_tags",
    "level_dirname",
    "load_and_validate_yaml",
    "slugify_id",
    "synthetic_bugfix_problems",
    "t_entry_to_problem",
    "validate_corpus",
    "validate_problem",
    "validate_problem_strict",
    "write_problem_yaml",
]
