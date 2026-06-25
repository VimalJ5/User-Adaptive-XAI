"""
config_eli5.py
==============
Configuration for Experiment A: Direct Verbalization with Constrained Decoding on ELI5.
This file satisfies all imports required by constrained_decoding.py.
"""

import json
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

WORD_LISTS_PATH = os.path.join(os.path.dirname(__file__), "word_lists.json")

with open(WORD_LISTS_PATH, "r") as f:
    _wl = json.load(f)

COMMON_WORDS       = set(_wl["common_words"])
CLAUSE_MARKERS     = set(_wl["clause_markers"])
DALE_CHALL_FAMILIAR = (
    set(_wl["common_words"])
    | set(_wl["dale_chall_extra"])
)

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

LLM_MODEL_NAME      = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MAX_NEW_TOKENS  = 180
MIN_NEW_TOKENS      = 80
NUM_BEAMS           = 4

# ─────────────────────────────────────────────
# Constrained decoding
# ─────────────────────────────────────────────

LAMBDA_MAP = {
    "BEGINNER":     5.0,
    "INTERMEDIATE": 0.5,
    "EXPERT":       0.0,
}

HARDNESS_WEIGHTS = {
    "length":       0.25,
    "polysyllabic": 0.25,
    "dale_chall":   0.20,
    "clause":       0.10,
    "syllable":     0.08,
    "sentence_len": 0.07,
    "char_per_word":0.05,
}

POLYSYLLABIC_THRESHOLD = 3
SENTENCE_LEN_CAP       = 25.0
CHAR_PER_WORD_CAP      = 9.0

HARDNESS_CAPS = {
    "length_words":           60.0,
    "polysyllabic_words":     10.0,
    "dale_chall_unfamiliar":   8.0,
    "clause_markers":          4.0,
    "avg_syllables_per_word":  4.0,
    "avg_sentence_length":    25.0,
    "avg_chars_per_word":      9.0,
}

# ─────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────

NUM_SAMPLES     = 10
CHECKPOINT_EVERY = 10

OUTPUT_DIR      = "results"
RESULTS_CSV     = os.path.join(OUTPUT_DIR, "experiment_a_results.csv")
SUMMARY_CSV     = os.path.join(OUTPUT_DIR, "experiment_a_summary.csv")

TIERS = ["BEGINNER", "INTERMEDIATE", "EXPERT"]

SYSTEM_PROMPTS = {
    "BEGINNER":     "You are a helpful assistant. Explain the following in very simple terms, as if explaining to a curious child with no background knowledge. Use short sentences and everyday words.",
    "INTERMEDIATE": "You are a helpful assistant. Explain the following clearly for someone with general knowledge. You may use some technical terms but keep the explanation accessible.",
    "EXPERT":       "You are a helpful assistant. Explain the following with technical depth and precision. Assume the reader has strong domain knowledge and can handle complex vocabulary.",
}
