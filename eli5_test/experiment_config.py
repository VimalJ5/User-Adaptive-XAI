"""
experiment_config.py
====================
Experiment A configuration for ELI5 constrained decoding runs.
"""

from __future__ import annotations

from pathlib import Path

ELI5_TEST_DIR = Path(__file__).resolve().parent

from local_settings import LAMBDA_MAP as CD_LAMBDA_MAP
from local_settings import LLM_MAX_NEW_TOKENS, MIN_NEW_TOKENS, NUM_BEAMS


CONFIG = {
    "seed": 42,
    "dataset_name": "sentence-transformers/eli5",
    "dataset_split": "train",
    "dataset_size": 10,
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "load_in_4bit": True,
    "checkpoint_interval": 1,
    "num_beams": 1,
    "min_new_tokens": MIN_NEW_TOKENS,
    "max_new_tokens": LLM_MAX_NEW_TOKENS,
    "bert_score_model": "distilbert-base-uncased",
    "bert_score_batch_size": 8,
    "results": {
        "raw": ELI5_TEST_DIR / "experiment_a_raw_results.csv",
        "checkpoint": ELI5_TEST_DIR / "experiment_a_results_checkpoint.csv",
        "final": ELI5_TEST_DIR / "experiment_a_results.csv",
        "summary": ELI5_TEST_DIR / "experiment_a_summary.csv",
    },
    "tiers": {
        "BEGINNER": {
            "alpha": float(CD_LAMBDA_MAP["BEGINNER"]),
            "system_prompt": "Explain the following in very simple terms, as if to a child:",
        },
        "INTERMEDIATE": {
            "alpha": float(CD_LAMBDA_MAP["INTERMEDIATE"]),
            "system_prompt": "Explain the following clearly for someone with general knowledge:",
        },
        "EXPERT": {
            "alpha": float(CD_LAMBDA_MAP["EXPERT"]),
            "system_prompt": "Explain the following with technical depth and precision:",
        },
    },
}
