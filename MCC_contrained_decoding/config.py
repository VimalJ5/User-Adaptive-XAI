"""
config.py
=========
Central configuration for the User-Adaptive XAI Pipeline.
Edit paths and hyperparameters here — nowhere else.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

# Directory that holds all intermediate/output files
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Intermediate checkpoint files (one per stage)
LIME_RESULTS_PATH      = OUTPUTS_DIR / "lr_exp.json"
ONTOLOGY_RESULTS_PATH  = OUTPUTS_DIR / "or_exp.json"
EXPLANATIONS_PATH      = OUTPUTS_DIR / "ex_exp_cd.json"
ANALYSIS_RESULTS_PATH  = OUTPUTS_DIR / "res_exp_cd.csv"

# Model / ontology paths  ← update these for your machine
CLASSIFIER_MODEL_PATH = (
    "C:/Users/vimal/OneDrive/Documents/Uni/BTP/"
    "User-Adaptive-XAI/Models/my_medical_model"
)
ONTOLOGY_PATH = (
    "C:/Users/vimal/OneDrive/Documents/Uni/BTP/"
    "User-Adaptive-XAI/Ontology/doid.owl"
)
NER_MODEL_NAME = "d4data/biomedical-ner-all"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ─────────────────────────────────────────────
# Classification label map
# ─────────────────────────────────────────────

LABEL_TO_CLASS = {
    "Class_0": "Neoplasms",
    "Class_1": "Digestive system diseases",
    "Class_2": "Nervous system diseases",
    "Class_3": "Cardiovascular diseases",
    "Class_4": "General pathological conditions",
}

CLASS_NAMES = list(LABEL_TO_CLASS.values())

# ─────────────────────────────────────────────
# LIME hyperparameters
# ─────────────────────────────────────────────

LIME_NUM_FEATURES = 6
LIME_NUM_SAMPLES  = 300

# ─────────────────────────────────────────────
# Ontology enrichment
# ─────────────────────────────────────────────

# How many top LIME features to look up in the ontology
TOP_LIME_FEATURES = 6

# ─────────────────────────────────────────────
# LLM generation
# ─────────────────────────────────────────────

LLM_MAX_NEW_TOKENS   = 180
LLM_TEMPERATURE      = 0.1
LLM_REPETITION_PENALTY = 1.1

# Enable constrained decoding path in Stage 3 generation.
USE_CONSTRAINED_DECODING = True

# Beam search settings for constrained decoding.
NUM_BEAMS = 4
MIN_NEW_TOKENS = 80

# λ values by audience.
LAMBDA_MAP = {
    "BEGINNER": 5.0,
    "INTERMEDIATE": 0.5,
    "EXPERT": 0.05,
}

LAMBDA_SWEEP_VALUES = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

# Hardness feature weights (must sum to 1.0).
HARDNESS_WEIGHTS = {
    "length": 0.40,
    "rare": 0.35,
    "clause": 0.15,
    "avglen": 0.10,
}

# Soft caps used to normalize features into [0, 1].
HARDNESS_CAPS = {
    "length_words": 60.0,
    "rare_words": 8.0,
    "clause_markers": 4.0,
    "avg_word_len": 8.0,
}

CLAUSE_MARKERS = {
    "which", "although", "whereas", "however", "nevertheless",
    "therefore", "moreover", "furthermore", "while", "though",
}

# Common words used for a lightweight rare-word heuristic.
COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "if", "in", "into", "is", "it",
    "its", "may", "more", "most", "of", "on", "or", "such", "than", "that",
    "the", "their", "there", "these", "this", "to", "was", "were", "will",
    "with", "without", "can", "could", "should", "would", "about", "over",
    "under", "between", "during", "after", "before", "also", "not", "no",
    "yes", "we", "our", "you", "your", "they", "them", "he", "she", "his",
    "her", "i", "me", "my", "mine", "do", "does", "did", "done", "because",
}

# Biomedical terms that should stay accessible (not treated as rare).
BIOMEDICAL_WHITELIST = {
    "blood", "body", "brain", "cancer", "cell", "cells", "disease", "dna",
    "gene", "genes", "heart", "immune", "infection", "kidney", "liver",
    "lung", "lungs", "medicine", "metabolism", "muscle", "nerves", "pain",
    "protein", "proteins", "risk", "symptom", "symptoms", "tissue", "tumor",
    "tumour", "virus", "viral", "bacteria", "bacterial", "inflammation",
}

# ─────────────────────────────────────────────
# Experiment parameters
# ─────────────────────────────────────────────

# Options: "BEGINNER" | "INTERMEDIATE" | "EXPERT"
USER_CATEGORY = "EXPERT"

# Options: "normal" | "full" | "one_parent" | "no_ontology"
ABLATION_MODE = "normal"
