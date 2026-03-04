"""
pipeline_helpers.py
===================
Stateless helper functions used across the pipeline notebooks.

Sections
--------
    1. Entity merging          (Stage 1)
    2. LIME predictor factory  (Stage 1)
    3. Classifier prediction   (Stage 2)
    4. Prompt building         (Stage 3)
    5. Explanation generation  (Stage 3)
    6. Readability metrics     (Stage 4)
    7. Faithfulness metrics    (Stage 4)
    8. Checkpoint I/O          (all stages)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import textstat

from config import (
    LABEL_TO_CLASS,
    LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
    TOP_LIME_FEATURES,
)
from ontology_helpers import find_concept, select_ancestors


# ─────────────────────────────────────────────────────────────────────────────
# 1. Entity merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_entities(text: str, ner_pipe) -> str:
    """
    Replace spaces in multi-word biomedical entities with underscores so that
    LIME treats them as single tokens.

    Example
    -------
    "cardiac tamponade" → "cardiac_tamponade"
    """
    entities = ner_pipe(text)
    # Longest-first to avoid partial replacements
    entities.sort(key=lambda e: len(e["word"]), reverse=True)

    merged = text
    for ent in entities:
        word = ent["word"]
        if " " in word:
            fused   = word.replace(" ", "_")
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            merged  = pattern.sub(fused, merged)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIME predictor factory
# ─────────────────────────────────────────────────────────────────────────────

def make_lime_predictor(model, clf):
    """
    Return a batch predictor function compatible with LimeTextExplainer.

    Underscores in tokens are restored to spaces before classification so the
    underlying model sees natural text.

    Parameters
    ----------
    model : AutoModelForSequenceClassification  (used for label2id ordering)
    clf   : HuggingFace text-classification pipeline
    """
    def predictor(texts: list[str]) -> np.ndarray:
        cleaned = [t.replace("_", " ") for t in texts]
        results = clf(cleaned, truncation=True, max_length=512)
        probs = []
        for result in results:
            sorted_res = sorted(result, key=lambda x: model.config.label2id[x["label"]])
            probs.append([x["score"] for x in sorted_res])
        return np.array(probs)

    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classifier prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_class(text: str, clf) -> tuple[str, float]:
    """
    Run the classifier on *text* (truncated to 512 tokens).

    Returns
    -------
    (predicted_class_name, confidence)
    """
    result     = clf(text[:512], top_k=1)[0]
    class_name = LABEL_TO_CLASS.get(result["label"], result["label"])
    return class_name, round(float(result["score"]), 4)


def enrich_with_ontology(
    lime_features: list,
    ontology,
    user_category: str,
    ablation_mode: str,
) -> list[dict]:
    """
    Map LIME features to ontology ancestors.

    Parameters
    ----------
    lime_features  : list of [word, score] pairs (from Stage 1 JSON)
    ontology       : loaded owlready2 ontology
    user_category  : "BEGINNER" | "INTERMEDIATE" | "EXPERT"
    ablation_mode  : "normal" | "full" | "one_parent" | "no_ontology"

    Returns
    -------
    List of dicts:
        {"feature_word": str, "lime_score": float, "ancestors": list[str]}
    Only features that resolve to an ontology concept are included.
    """
    feature_data = []
    for word, score in lime_features[:TOP_LIME_FEATURES]:
        concept = find_concept(ontology, str(word))
        if concept is None:
            continue
        ancestors = select_ancestors(
            entity_concept=concept,
            user_category=user_category,
            ablation_mode=ablation_mode,
        )
        feature_data.append({
            "feature_word": word,
            "lime_score":   round(float(score), 6),
            "ancestors":    ancestors,
        })
    return feature_data


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prompt building
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an explanation generator for a biomedical XAI system.

You do NOT classify text.
You do NOT add external medical knowledge.
You do NOT infer physiology, causation, or mechanisms.

Your ONLY role:
Convert the structured explanation plan (LIME tokens + ontology ancestors +
model prediction) into a clear natural-language explanation.

Rules:
1. Use ONLY the provided ontology triples and model prediction.
2. NEVER add biomedical facts that are not explicitly listed.
3. If a concept has no relation to the predicted class, state this neutrally.
4. Beginner → broad/simple language.
5. Expert → technical terms and ontology hierarchy words.
6. DO NOT write with bullet points. Produce one coherent paragraph.
7. Be concise and strictly grounded.
""".strip()


def build_prompt(
    predicted_class: str,
    feature_data: list[dict],
    user_category: str,
) -> str:
    """
    Build the structured user prompt from ontology-enriched features.

    Parameters
    ----------
    predicted_class : e.g. "Cardiovascular diseases"
    feature_data    : list of {"feature_word", "ancestors", ...} dicts
    user_category   : "BEGINNER" | "INTERMEDIATE" | "EXPERT"
    """
    triple_lines = []
    for item in feature_data:
        word  = item["feature_word"]
        chain = " -> ".join(item["ancestors"]) if item["ancestors"] else "NONE"
        triple_lines.append(f"({word} → ancestor → {chain})")

    triples_block = "\n".join(triple_lines)
    token_list    = [item["feature_word"] for item in feature_data]

    return (
        "### INPUT\n"
        "text: [REDACTED FOR BREVITY]\n"
        f"Prediction: {predicted_class}\n\n"
        f"Salient tokens identified by LIME:\n{token_list}\n\n"
        f"Ontology triples (feature → relation → ancestor):\n{triples_block}\n\n"
        f"User type: {user_category}\n\n"
        "### OUTPUT REQUIREMENTS\n"
        "Write a single coherent explanation that:\n"
        "- States the prediction.\n"
        "- Uses ONLY the information above.\n"
        "- Maps each token to its listed ancestors.\n"
        "- If a token has no biomedical relevance to the prediction, state this without guessing.\n"
        "- For BEGINNER → use broad/simple language.\n"
        "- For EXPERT → include hierarchy terms like 'anatomical entity', 'subclass of', etc.\n"
        "- Do NOT add medical facts not listed in the triples.\n"
        "- Do NOT speculate, infer physiology, or make causal claims.\n\n"
        "### EXPLANATION:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Explanation generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_explanation(
    predicted_class: str,
    feature_data: list[dict],
    user_category: str,
    tokenizer,
    model,
) -> str:
    """
    Generate a natural-language explanation using the LLM.

    Returns a plain string explanation. Falls back to a canned message if
    feature_data is empty.
    """
    if not feature_data:
        return (
            f"The model predicted {predicted_class}, "
            "but no ontology-based features were available to explain this decision."
        )

    task_prompt = build_prompt(predicted_class, feature_data, user_category)

    full_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": task_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        do_sample=True,
        repetition_penalty=LLM_REPETITION_PENALTY,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Strip prompt prefix — keep only the generated explanation
    if "### EXPLANATION:" in decoded:
        return decoded.split("### EXPLANATION:")[-1].strip()
    if "assistant\n" in decoded:
        return decoded.split("assistant\n")[-1].strip()
    return decoded.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Readability metrics
# ─────────────────────────────────────────────────────────────────────────────

def readability_metrics(text: str) -> dict:
    """
    Compute standard readability scores for *text*.

    Returns
    -------
    dict with keys: flesch_reading_ease, flesch_kincaid_grade, smog_index
    """
    return {
        "flesch_reading_ease":  textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index":           textstat.smog_index(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Faithfulness metrics
# ─────────────────────────────────────────────────────────────────────────────

def lime_coverage(explanation: str, feature_data: list[dict]) -> float:
    """
    Fraction of LIME features (or their ancestors) mentioned in *explanation*.

    A feature 'counts' if its word OR any of its ancestors appears in the
    explanation text (case-insensitive substring match).
    """
    if not feature_data:
        return 0.0

    lower = explanation.lower()
    hits  = sum(
        1 for item in feature_data
        if item["feature_word"].lower() in lower
        or any(a.lower() in lower for a in item.get("ancestors", []))
    )
    return round(hits / len(feature_data), 4)


def ontology_hit_rate(feature_data: list[dict]) -> float:
    """
    Fraction of LIME features that were successfully mapped to ≥1 ontology ancestor.
    """
    if not feature_data:
        return 0.0
    hits = sum(1 for item in feature_data if item.get("ancestors"))
    return round(hits / len(feature_data), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(data: list[dict], path: Path) -> None:
    """Serialise *data* to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Checkpoint] Saved {len(data)} records → '{path}'")


def load_checkpoint(path: Path) -> list[dict]:
    """Load and return JSON data from *path*."""
    with open(path) as f:
        data = json.load(f)
    print(f"[Checkpoint] Loaded {len(data)} records ← '{path}'")
    return data


def checkpoint_exists(path: Path) -> bool:
    """Return True if *path* exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0
