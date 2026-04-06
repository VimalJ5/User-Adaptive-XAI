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
    USE_CONSTRAINED_DECODING,
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
You are a biomedical explanation assistant. Your job is to generate clear, accurate natural language explanations of why a machine learning model made a specific biomedical prediction. You are given:
- The model's predicted class
- Key tokens identified by LIME (local feature attribution) as influential in the prediction
- Ontology-derived ancestor chains for each token, showing its place in the biomedical concept hierarchy

Your explanations must be grounded strictly in the provided features and ontology context. Do not introduce facts, diseases, or concepts not present in the input.

Adapt your explanation style based on the user category:
- BEGINNER: Use plain, everyday language. Avoid technical jargon. Explain medical terms when they appear. Keep sentences short. The goal is comprehension, not completeness.
- INTERMEDIATE: Balance accessibility with domain accuracy. Define specialized terms briefly. Use medical vocabulary where helpful but not exclusively.
- EXPERT: Use precise clinical and biomedical terminology. Reference ontological relationships and mechanistic reasoning. Assume familiarity with standard medical vocabulary.

Always structure your explanation as a short, coherent paragraph (3–5 sentences). Do not use bullet points. Do not repeat the feature words mechanically — weave them naturally into the explanation.
""".strip()


def build_prompt(
    predicted_class: str,
    feature_data: list[dict],
    user_category: str,
) -> str:
    triple_lines = []
    for item in feature_data:
        word  = item["feature_word"]
        chain = " -> ".join(item["ancestors"]) if item["ancestors"] else "NONE"
        triple_lines.append(f"  - {word} (ontology path: {chain})")

    triples_block = "\n".join(triple_lines)
    token_list    = ", ".join(item["feature_word"] for item in feature_data)

    return (
        f"### INPUT\n"
        f"Predicted class: {predicted_class}\n"
        f"User category: {user_category}\n\n"
        f"Key features identified by the model (with ontology ancestry):\n"
        f"{triples_block}\n\n"
        f"Influential tokens: {token_list}\n\n"
        f"### TASK\n"
        f"Write a {user_category.lower()}-level explanation of why the model predicted '{predicted_class}'. "
        f"Use the ontology paths to reason about what each token represents conceptually. "
        f"Ground your explanation in the provided features — do not invent additional clinical details.\n\n"
        f"### EXPLANATION:"
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
    generator=None,
) -> str:
    """
    Generate a natural-language explanation using the LLM.
    Returns a plain string explanation. Falls back to a canned message if
    """
    if not feature_data:
        return (
            f"The model predicted {predicted_class}, "
            "but no ontology-based features were available to explain this decision."
        )

    task_prompt = build_prompt(predicted_class, feature_data, user_category)

    # Preferred path: constrained beam decoding by user category.
    if USE_CONSTRAINED_DECODING and generator is not None:
        return generator.generate(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=task_prompt,
            user_category=user_category,
        )

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
        repetition_penalty=LLM_REPETITION_PENALTY
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
    return round(float(hits) / len(feature_data), 4)


def ontology_hit_rate(feature_data: list[dict]) -> float:
    """
    Fraction of LIME features that were successfully mapped to ≥1 ontology ancestor.
    """
    if not feature_data:
        return 0.0
    hits = sum(1 for item in feature_data if item.get("ancestors"))
    return round(float(hits) / len(feature_data), 4)


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
