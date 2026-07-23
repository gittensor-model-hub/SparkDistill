"""Distillation bridge: prompts, quality filter, SFT export, teacher data generation."""

from __future__ import annotations

from distill.data_generator import DataGenerator, MockTeacherClient, OpenAICompatibleTeacherClient
from distill.export_sft import (
    TrajectoryRecord,
    dedupe_by_prompt_hash,
    export_sft_records,
    load_jsonl,
    to_completion_record,
    to_messages_record,
)
from distill.prompt_builder import build_sft_user_content, build_system_prompt, build_teacher_prompt
from distill.quality_filter import FilterResult, QualityFilter

__all__ = [
    "DataGenerator",
    "FilterResult",
    "MockTeacherClient",
    "OpenAICompatibleTeacherClient",
    "QualityFilter",
    "TrajectoryRecord",
    "build_sft_user_content",
    "build_system_prompt",
    "build_teacher_prompt",
    "dedupe_by_prompt_hash",
    "export_sft_records",
    "load_jsonl",
    "to_completion_record",
    "to_messages_record",
]
