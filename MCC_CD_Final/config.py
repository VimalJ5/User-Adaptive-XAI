"""
config.py
=========
Central configuration for the User-Adaptive XAI Pipeline.
Edit paths and hyperparameters here — nowhere else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUICK-START: Scroll to EXPERIMENT RUN CONFIG below
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

# Directory that holds all intermediate/output files
OUTPUTS_DIR = Path("Pivot_OP")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Intermediate checkpoint files (one per stage)
LIME_RESULTS_PATH      = OUTPUTS_DIR / "lr.json"
ONTOLOGY_RESULTS_PATH  = OUTPUTS_DIR / "or.json"
EXPLANATIONS_PATH      = OUTPUTS_DIR / "ex_beg.json"
ANALYSIS_RESULTS_PATH  = OUTPUTS_DIR / "res_beg.csv"

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
LLM_MODEL_NAME = "microsoft/phi-2"

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
    "EXPERT": 0.05,
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
# Dale-Chall familiar-word set
# ─────────────────────────────────────────────
# Superset of COMMON_WORDS + BIOMEDICAL_WHITELIST plus a broad corpus of
# everyday English words (~500 seeds from the Dale-Chall 3,000-word list).
# A word that is NOT here and has len >= 2 is counted as "unfamiliar".
DALE_CHALL_FAMILIAR: set = (
    COMMON_WORDS
    # | BIOMEDICAL_WHITELIST
    | {
        # ── Articles / determiners / pronouns ──
        "all", "another", "any", "both", "each", "either", "every", "few",
        "many", "much", "neither", "none", "other", "own", "same", "several",
        "some", "us", "what", "which", "who", "whom", "whose",
        # ── Numbers (written out) ──
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand", "million", "billion", "first", "second", "third", "fourth",
        "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "once", "twice",
        # ── Colors ──
        "black", "blue", "brown", "gold", "gray", "green", "orange", "pink",
        "purple", "red", "silver", "tan", "white", "yellow",
        # ── Family / people ──
        "baby", "boy", "brother", "child", "children", "dad", "daughter",
        "family", "father", "friend", "girl", "grandma", "grandpa",
        "grandfather", "grandmother", "husband", "kid", "man", "men",
        "mom", "mother", "neighbor", "parent", "people", "person", "sister",
        "son", "teacher", "wife", "woman", "women",
        # ── Common verbs ──
        "act", "add", "allow", "ask", "back", "become", "bring", "build",
        "buy", "call", "came", "carry", "cause", "change", "check", "choose",
        "clean", "close", "come", "cook", "copy", "count", "cover", "create",
        "cut", "decide", "develop", "draw", "drink", "drive", "drop", "eat",
        "end", "enjoy", "enter", "explain", "fall", "feel", "fight", "fill",
        "find", "follow", "forget", "get", "give", "go", "grow", "happen",
        "hear", "help", "hit", "hold", "hope", "include", "keep", "kill",
        "know", "lead", "learn", "leave", "let", "like", "listen", "live",
        "look", "lose", "love", "make", "mean", "meet", "miss", "move",
        "need", "open", "play", "point", "put", "read", "remain", "remove",
        "run", "say", "see", "seem", "send", "set", "show", "sit", "sleep",
        "speak", "stand", "start", "stay", "stop", "study", "take", "talk",
        "teach", "tell", "think", "try", "turn", "use", "wait", "walk",
        "want", "watch", "work", "write",
        # ── Common adjectives ──
        "able", "afraid", "ago", "alive", "alone", "already", "always",
        "bad", "beautiful", "better", "big", "bright", "busy", "careful",
        "certain", "clear", "common", "dark", "dead", "dear", "deep",
        "different", "difficult", "early", "easy", "enough", "even", "ever",
        "fair", "far", "fast", "fine", "free", "full", "good", "great",
        "happy", "hard", "heavy", "high", "hot", "huge", "important",
        "large", "last", "late", "left", "light", "little", "long", "low",
        "main", "new", "nice", "normal", "now", "often", "only", "open",
        "poor", "possible", "pretty", "quick", "quiet", "ready", "real",
        "right", "round", "safe", "short", "simple", "slow", "small", "smart",
        "soft", "soon", "sorry", "special", "still", "strange", "strong",
        "sure", "sweet", "tall", "true", "warm", "wide", "wrong", "young",
        # ── Common nouns ──
        "age", "air", "animal", "answer", "area", "arm", "back", "ball",
        "bed", "bottom", "box", "break", "bus", "car", "care", "city",
        "class", "color", "corner", "country", "course", "cup", "day", "door",
        "end", "eye", "eyes", "face", "fact", "field", "fire", "floor",
        "food", "foot", "game", "ground", "group", "hand", "head", "home",
        "hour", "house", "idea", "job", "land", "letter", "life", "light",
        "line", "list", "look", "matter", "minute", "money", "month",
        "morning", "name", "nature", "night", "number", "order", "page",
        "part", "party", "place", "plan", "plant", "point", "power", "problem",
        "question", "reason", "road", "room", "school", "sea", "side", "size",
        "sky", "sound", "state", "story", "street", "sun", "table", "thing",
        "thought", "time", "top", "town", "tree", "turn", "type", "view",
        "voice", "water", "way", "week", "window", "word", "world", "year",
        # ── Common adverbs / prepositions / conjunctions ──
        "above", "across", "again", "almost", "along", "already", "although",
        "always", "around", "away", "back", "below", "close", "down", "else",
        "enough", "even", "ever", "here", "however", "instead", "just", "later",
        "maybe", "near", "never", "next", "now", "off", "once", "only", "out",
        "outside", "perhaps", "please", "quite", "rather", "since", "so",
        "somehow", "sometimes", "soon", "still", "then", "there", "through",
        "together", "too", "toward", "until", "up", "upon", "usually", "very",
        "well", "when", "where", "while", "yet",
        # ── Health / body basics (extending BIOMEDICAL_WHITELIST) ──
        "age", "arm", "arms", "back", "bone", "bones", "chest", "ear",
        "ears", "eat", "eye", "eyes", "face", "fat", "foot", "feet", "hair",
        "hand", "hands", "head", "hip", "jaw", "joint", "joints", "knee",
        "knees", "leg", "legs", "lip", "lips", "mouth", "neck", "nose",
        "organ", "organs", "rib", "ribs", "shoulder", "skin", "sleep", "spine",
        "stomach", "throat", "toe", "toes", "tooth", "teeth", "vein", "veins",
        "wrist", "healthy", "sick", "ill", "diet", "drug", "drugs", "dose",
        "test", "treat", "treatment", "doctor", "nurse", "patient", "surgery",
        "hospital", "care", "check", "study", "studies", "result", "results",
        "cause", "causes", "effect", "effects", "level", "levels", "rate",
        "rates", "type", "types", "sign", "signs", "stage", "stages",
        # ── Miscellaneous high-frequency words ──
        "able", "act", "action", "age", "air", "also", "amount", "base",
        "based", "basic", "best", "bit", "body", "both", "bring", "call",
        "case", "cases", "check", "clear", "close", "come", "common", "control",
        "current", "data", "date", "days", "decision", "develop", "different",
        "direct", "early", "example", "exist", "factor", "focus", "form",
        "found", "given", "goal", "health", "high", "hold", "human", "increase",
        "individual", "initial", "input", "issue", "key", "large", "later",
        "lead", "less", "likely", "link", "local", "lower", "major", "manage",
        "means", "medical", "method", "model", "often", "only", "output",
        "overall", "past", "patient", "patients", "period", "primary", "process",
        "provide", "range", "reach", "recent", "reduce", "related", "report",
        "research", "role", "run", "sample", "serve", "set", "share", "similar",
        "single", "specific", "step", "support", "system", "term", "test",
        "three", "total", "true", "understand", "unit", "value", "various",
        "well", "within",
    }
)

# ══════════════════════════════════════════════
# EXPERIMENT RUN CONFIG  ← edit this section
# ══════════════════════════════════════════════
#
# This is the only section you need to edit before
# running experiment.ipynb for a single-input run.

# ── Input ─────────────────────────────────────
# Paste the medical abstract you want to explain.
# Leave as None to pick the first line of test_data.txt.
INPUT_TEXT = "Endometriosis associated with massive ascites and absence of pelvic peritoneum. Although massive ascites associated with endometriosis has been reported in rare cases, this patient was also noted to have massive destruction of the pelvic peritoneum. Failure of medical suppression necessitated total abdominal hysterectomy and bilateral salpingo-oophorectomy. Several months after surgery ascites resolved, possibly with reestablishment of the pelvic peritoneum. "

# ── Audience ──────────────────────────────────
# Options: "BEGINNER" | "INTERMEDIATE" | "EXPERT"
USER_CATEGORY = "BEGINNER"

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
EXPERIMENT_RESULTS_PATH = Path("Results") / f"{EXPERIMENT_TAG}.csv"
