"""
utils.py
========
Shared helpers for Experiment A generation and evaluation.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import textstat
from datasets import load_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_eli5_samples(dataset_name: str, split: str, dataset_size: int):
    dataset = load_dataset(dataset_name, split=split)
    limit = min(dataset_size, len(dataset))
    return dataset.select(range(limit))


def extract_question(example: dict) -> str:
    question = example.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    title = example.get("title", "")
    return str(title).strip()


def extract_reference_answer(example: dict) -> str:
    answer = example.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    answers = example.get("answers") or {}
    texts = answers.get("text") or []
    if texts:
        first_answer = texts[0]
        if isinstance(first_answer, str) and first_answer.strip():
            return first_answer.strip()
        return str(first_answer).strip()
    return ""


def safe_textstat_score(score_fn, text: str, minimum: float = 0.01) -> float:
    try:
        value = float(score_fn(text))
    except Exception:
        return minimum
    if not math.isfinite(value):
        return minimum
    if value == 0.0:
        return minimum
    return value


def count_polysyllabic_percentage(text: str) -> float:
    words = [word for word in text.split() if word.strip()]
    if not words:
        return 0.0
    polysyllabic = sum(1 for word in words if textstat.syllable_count(word) >= 3)
    return (polysyllabic / len(words)) * 100.0


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes} B"
