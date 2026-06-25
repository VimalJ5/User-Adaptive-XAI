"""
local_settings.py
=================
Local constants and word lists for the standalone Experiment A pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path


ELI5_TEST_DIR = Path(__file__).resolve().parent
_WL_PATH = ELI5_TEST_DIR / "word_lists.json"

if not _WL_PATH.exists():
    raise FileNotFoundError(
        f"Missing local word list file at {_WL_PATH}."
    )

_wl = json.loads(_WL_PATH.read_text(encoding="utf-8"))

COMMON_WORDS: set = set(_wl["common_words"])
CLAUSE_MARKERS: set = set(_wl["clause_markers"])
DALE_CHALL_FAMILIAR: set = COMMON_WORDS | set(_wl["dale_chall_extra"])

# Generation defaults used by constrained decoding.
LLM_MAX_NEW_TOKENS = 180
MIN_NEW_TOKENS = 80
NUM_BEAMS = 4

# Readability/CD controls.
LAMBDA_MAP = {
    "BEGINNER": 5.0,
    "INTERMEDIATE": 0.5,
    "EXPERT": 0.0,
}

HARDNESS_WEIGHTS = {
    "length": 0.25,
    "polysyllabic": 0.25,
    "dale_chall": 0.20,
    "clause": 0.10,
    "syllable": 0.08,
    "sentence_len": 0.07,
    "char_per_word": 0.05,
}

POLYSYLLABIC_THRESHOLD = 3
SENTENCE_LEN_CAP = 25.0
CHAR_PER_WORD_CAP = 9.0
