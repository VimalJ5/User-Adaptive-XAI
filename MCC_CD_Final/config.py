"""
config.py
=========
Central configuration for the User-Adaptive XAI Pipeline.
Edit paths and hyperparameters here — nowhere else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUICK-START: Scroll to EXPERIMENT RUN CONFIG below
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from pathlib import Path

# ─────────────────────────────────────────────
# Word lists (loaded from word_lists.json)
# ─────────────────────────────────────────────
_WL_PATH = Path(__file__).parent / "word_lists.json"
_wl = json.loads(_WL_PATH.read_text(encoding="utf-8"))

COMMON_WORDS: set        = set(_wl["common_words"])
BIOMEDICAL_WHITELIST: set = set(_wl["biomedical_whitelist"])
CLAUSE_MARKERS: set       = set(_wl["clause_markers"])
DALE_CHALL_FAMILIAR: set  = COMMON_WORDS | set(_wl["dale_chall_extra"])

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

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
# XAI method selection  ← change this to swap
# ─────────────────────────────────────────────
# Options: "LIME" | "IG"
#   LIME  — Local Interpretable Model-agnostic Explanations (default)
#   IG    — Integrated Gradients at embedding level (requires captum)
#
XAI_METHOD = "IG"

# Shared budget — LIME: num_samples  |  IG: ignored (50 steps fixed)
XAI_NUM_FEATURES = LIME_NUM_FEATURES   # top-k features passed to Stages 2–4
XAI_NUM_SAMPLES  = LIME_NUM_SAMPLES    # perturbation budget

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
    "EXPERT": 0.00,
}

LAMBDA_SWEEP_VALUES = [0.01, 0.05, 1.0, 3.0, 5.0, 7.5]

# Hardness feature weights (must sum to 1.0).
HARDNESS_WEIGHTS = {
    "length": 0.25,       # reduced — sentence_len now adds a separate length signal
    "polysyllabic": 0.25, # NEW: words >= POLYSYLLABIC_THRESHOLD syllables (SMOG/Fog signal)
    "dale_chall": 0.20,   # REPLACES "rare" — unfamiliar word rate (Dale-Chall signal)
    "clause": 0.10,
    "syllable": 0.08,     # kept but reduced — complemented by polysyllabic
    "sentence_len": 0.07, # NEW: avg words per sentence in prefix
    "char_per_word": 0.05,# NEW: ARI/Coleman-Liau signal
}

# Thresholds / soft caps for new signals.
POLYSYLLABIC_THRESHOLD = 3   # SMOG/Fog: words with >= this many syllables count as hard
SENTENCE_LEN_CAP = 25.0      # soft cap for avg sentence length normalization
CHAR_PER_WORD_CAP = 9.0      # soft cap for avg characters per word normalization

# Soft caps used to normalize features into [0, 1].
HARDNESS_CAPS = {
    "length_words": 60.0,
    "polysyllabic_words": 10.0,     # NEW
    "dale_chall_unfamiliar": 8.0,   # replaces rare_words
    "clause_markers": 4.0,
    "avg_syllables_per_word": 4.0,
    "avg_sentence_length": 25.0,    # NEW
    "avg_chars_per_word": 9.0,      # NEW
}




# ══════════════════════════════════════════════
# EXPERIMENT RUN CONFIG  ← edit this section
# ══════════════════════════════════════════════
#
# This is the only section you need to edit before
# running experiment.ipynb.

# ── Inputs ────────────────────────────────────
# List of medical abstracts to explain (one result row per entry).
# Set to None to load ALL non-empty lines from test_data.txt.
INPUT_TEXTS = [
    "Endometriosis associated with massive ascites and absence of pelvic peritoneum. Although massive ascites associated with endometriosis has been reported in rare cases, this patient was also noted to have massive destruction of the pelvic peritoneum. Failure of medical suppression necessitated total abdominal hysterectomy and bilateral salpingo-oophorectomy. Several months after surgery ascites resolved, possibly with reestablishment of the pelvic peritoneum. ",
    "Ultrasound-Doppler diagnosis of Budd-Chiari syndrome. We report a case of apparently idiopathic Budd-Chiari syndrome, diagnosed by ultrasound and Doppler sonography, in a patient with latent myeloproliferative disease. This case proves that Doppler sonography shows in the hepatic veins a flow pattern suggestive of partial thrombotic obstruction. Moreover, we suggest that the search for a latent myeloproliferative disorder, by means of the spontaneous erythroid colonies formation in culture of bone marrow or blood mononuclear cells, should be routinely included in the diagnostic evaluation of each case of hepatic vein thrombosis without other recognizable causes. ",
    "Neurogenic inflammation of the rat trachea: fate of neutrophils that adhere to venules. The goal of this study was to determine whether neutrophils that adhere to the vascular endothelium in association with neurogenic inflammation in the respiratory tract migrate out of the blood vessels or whether they detach and reenter the circulation. We also sought to determine whether the fate of the neutrophils is influenced by neutral endopeptidase (NEP), an enzyme that degrades the tachykinins that produce neurogenic inflammation. Neutrophils in the tracheal mucosa of anesthetized pathogen-free rats were examined 5 min or 4 h after neurogenic inflammation was produced by an injection of capsaicin (100 or 200 micrograms/kg iv). In whole mounts of these tracheae stained histochemically for myeloperoxidase, adherent intravascular neutrophils had a spherical or teardrop (regular) shape and migrating neutrophils had a polarized amoeboid (irregular) shape. The number of regular neutrophils in the tracheae was increased at both times, but the increase at 4 h was only half that present at 5 min. The reduction between 5 min and 4 h was not offset by an appreciable increase in the number of irregular neutrophils, unless NEP was inhibited by phosphoramidon. We interpret these results as indicating that the rapid adherence of neutrophils to the vascular endothelium after an injection of capsaicin is followed by a gradual reentry of the neutrophils into the circulation and comparatively little neutrophil migration. However, when the effect of the stimulus is increased and/or prolonged by inhibition of NEP, some of the adherent neutrophils migrate out of the vessels. Thus the activity of NEP can regulate both the magnitude of the neutrophil adherence and the fate of the adherent cells. ",
    "Aberrant regeneration in a case of syringobulbia: selective co-activation of abducens and facial nerves during saccades. A patient suffering from syringobulbia and syringomyelia exhibited a phasic contraction of the ipsilateral facial muscles, mainly the levator labii, whenever he looked to the left or right. Facial muscle twitches occurred exclusively with saccades. The selective co-activation of abducens and facial nerves is interpreted as the result of bilateral misrouting of regenerating neurons from the parapontine reticular formation to the facial nerve in the tegmentum pontis. ",
    "Germ cell tumor of testis in a patient with von Hippel-Lindau disease. Germ cell testicular tumor is a previously undescribed entity in association with von Hippel-Lindau disease. This case exemplifies the variety of pathologic entities encountered in von Hippel-Lindau disease and stresses the importance of thorough evaluation of the patient, as well as careful follow-up, to ensure early detection of potentially malignant lesions. ",
    "Failure of hepatitis B immunization in liver transplant recipients: results of a prospective trial. Twenty patients with advanced liver disease, in need of transplantation, were given three injections of 20 micrograms and three injections of 40 micrograms hepatitis B vaccine to see if an antibody response could be obtained. Only 20% of patients developed measurable anti-HBs. One who failed to develop anti-HBs developed chronic hepatitis B after exposure to her infected sexual partner. Type of liver disease in the native liver, age, sex, sexual preference, timing of immunization (before or after transplantation), and dosage of hepatitis B vaccine did not seem to explain the lack of immunologic response to hepatitis B vaccine. It is presumed that immunosuppression, both from the underlying disease and from immunosuppressive medications, best explains our findings. Liver transplantation patients infrequently benefit from hepatitis B vaccine. It is possible that other vaccines given to prevent viral and bacterial illness may also fail to elicit immunologic response in such patients. "
]

# Back-compat alias — not used by the notebook any more.
INPUT_TEXT = INPUT_TEXTS[0] if INPUT_TEXTS else None

# ── Audience ──────────────────────────────────
# Options: "BEGINNER" | "INTERMEDIATE" | "EXPERT"
USER_CATEGORY = "EXPERT"

# ── Ontology ablation ─────────────────────────
ABLATION_MODE = "normal"

# ── Constrained decoding ──────────────────────
# Already defined above; override here if needed.
# USE_CONSTRAINED_DECODING = True

# ── Experiment tag ────────────────────────────
# Used as a filename suffix for the saved CSV result.
# E.g. "expert_normal", "beginner_no_ontology", "ablation_v2"
EXPERIMENT_TAG = f"{USER_CATEGORY.lower()}_{ABLATION_MODE}_{XAI_METHOD.lower()}"

# ── Output path (auto-derived) ─────────────────
# The final CSV is saved to:  outputs/exp_<EXPERIMENT_TAG>.csv
EXPERIMENT_RESULTS_PATH = Path("Results_w_CD") / f"{EXPERIMENT_TAG}.csv"
