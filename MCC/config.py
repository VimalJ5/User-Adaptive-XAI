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
LIME_RESULTS_PATH      = OUTPUTS_DIR / "lime_results.json"
ONTOLOGY_RESULTS_PATH  = OUTPUTS_DIR / "ontology_results.json"
EXPLANATIONS_PATH      = OUTPUTS_DIR / "explanations.json"
ANALYSIS_RESULTS_PATH  = OUTPUTS_DIR / "results.csv"

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

# ─────────────────────────────────────────────
# Experiment parameters
# ─────────────────────────────────────────────

# Options: "BEGINNER" | "INTERMEDIATE" | "EXPERT"
USER_CATEGORY = "EXPERT"

# Options: "normal" | "full" | "one_parent" | "no_ontology"
ABLATION_MODE = "normal"
