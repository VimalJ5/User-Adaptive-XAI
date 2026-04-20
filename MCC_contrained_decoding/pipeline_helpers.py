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
from typing import Optional
import textstat

from config import (
    LABEL_TO_CLASS,
    LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
    TOP_LIME_FEATURES,
    USE_CONSTRAINED_DECODING,
    XAI_METHOD,
    XAI_NUM_FEATURES,
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
# 2. XAI feature attribution  (Stage 1)
#    Supports LIME, SHAP, and Integrated Gradients (IG).
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


def run_lime(
    text: str,
    clf_model,
    clf_pipeline,
    class_names: list,
    num_features: int,
    num_samples: int,
) -> list:
    """
    Run LIME and return top-k [(word, score)] pairs for the predicted class.

    This is the original Stage-1 logic extracted into a reusable function.
    """
    from lime.lime_text import LimeTextExplainer

    predictor = make_lime_predictor(clf_model, clf_pipeline)
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(
        text, predictor, num_features=num_features, num_samples=num_samples
    )
    return exp.as_list()


def run_shap(
    text: str,
    clf_model,
    clf_pipeline,
    class_names: list,
    num_features: int,
    neval: int,
) -> list:
    """
    Run the SHAP Partition explainer (word-level masking) and return top-k
    [(word, score)] pairs for the model-predicted class.

    Requires: shap >= 0.41
    """
    import shap

    predictor = make_lime_predictor(clf_model, clf_pipeline)

    # Determine which class the model predicts so we take the right SHAP slice.
    probs = predictor([text])[0]
    pred_idx = int(np.argmax(probs))

    # Word-level masker: splits on any non-word character boundary.
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predictor, masker, output_names=list(class_names))
    shap_values = explainer([text], max_evals=neval, batch_size=20)

    # shap_values.values shape: (1, n_words, n_classes)
    vals = shap_values.values[0]
    word_attrs = vals[:, pred_idx] if vals.ndim == 2 else vals
    words = shap_values.data[0]

    pairs = [(str(w), float(s)) for w, s in zip(words, word_attrs)]
    # Sort by absolute attribution — both high-positive and high-negative are influential.
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:num_features]


def run_ig(
    text: str,
    clf_model,
    clf_pipeline,
    num_features: int,
    num_steps: int = 50,
) -> list:
    """
    Run Integrated Gradients (Captum) and return top-k [(word, score)] pairs.

    Attributions are computed at the token-embedding level, then aggregated
    to whole-word level by L2-norming per-token gradients and summing sub-word
    pieces. Works with both BPE (Ġ/▁ prefix) and WordPiece (## prefix) models.

    Requires: captum  (pip install captum)
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "captum is required for IG. Install with: pip install captum"
        ) from exc

    import torch

    clf_tokenizer = clf_pipeline.tokenizer
    device = next(clf_model.parameters()).device

    # Tokenize the input text.
    inputs = clf_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Determine predicted class index.
    clf_model.eval()
    with torch.no_grad():
        logits = clf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    pred_idx = int(logits.argmax(dim=-1).item())

    # Build embedding inputs and an all-zero baseline.
    embed_layer = clf_model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)          # (1, seq_len, hidden_dim)
    baseline = torch.zeros_like(input_embeds)

    def _forward(embeds):
        return clf_model(
            inputs_embeds=embeds, attention_mask=attention_mask
        ).logits[:, pred_idx]

    ig = IntegratedGradients(_forward)
    attrs = ig.attribute(input_embeds, baseline, n_steps=num_steps)

    # Per-token L2 norm over the embedding dimension → scalar score per token.
    token_scores = attrs.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    tokens = clf_tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    # Aggregate sub-word pieces back to whole words.
    special_toks = set(clf_tokenizer.all_special_tokens)
    word_scores: dict = {}
    current_word = None
    current_score = 0.0

    for tok, score in zip(tokens, token_scores):
        if not tok or tok in special_toks:
            continue
        score = float(score)
        # BPE / SentencePiece: leading Ġ or ▁ marks the start of a new word.
        if tok.startswith("\u0120") or tok.startswith("\u2581"):
            if current_word:
                word_scores[current_word] = (
                    word_scores.get(current_word, 0.0) + current_score
                )
            current_word = tok[1:].lower()
            current_score = score
        # BERT WordPiece: ## prefix marks a continuation sub-word.
        elif tok.startswith("##"):
            if current_word is None:
                current_word = tok[2:].lower()
                current_score = score
            else:
                current_word += tok[2:].lower()
                current_score += score
        else:
            if current_word:
                word_scores[current_word] = (
                    word_scores.get(current_word, 0.0) + current_score
                )
            current_word = tok.lower()
            current_score = score

    if current_word:
        word_scores[current_word] = (
            word_scores.get(current_word, 0.0) + current_score
        )

    # Filter single-character artefacts; sort highest attribution first.
    pairs = [(w, s) for w, s in word_scores.items() if len(w) >= 2]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:num_features]


def run_xai(
    method: str,
    text: str,
    clf_model,
    clf_pipeline,
    class_names: list,
    num_features: int,
    num_samples: int,
) -> list:
    """
    Dispatcher — calls the appropriate XAI attribution method.

    Parameters
    ----------
    method      : "LIME" | "SHAP" | "IG"  (case-insensitive)
    text        : raw input text
    clf_model   : AutoModelForSequenceClassification
    clf_pipeline: HuggingFace text-classification pipeline
    class_names : ordered list of class name strings
    num_features: number of top features to return
    num_samples : perturbation budget — LIME samples / SHAP max_evals;
                  ignored for IG (always uses 50 integration steps)

    Returns
    -------
    list of (word, score) tuples, length <= num_features
    """
    method = method.upper().strip()
    if method == "LIME":
        return run_lime(
            text, clf_model, clf_pipeline, class_names, num_features, num_samples
        )
    elif method == "SHAP":
        return run_shap(
            text, clf_model, clf_pipeline, class_names, num_features, num_samples
        )
    elif method == "IG":
        return run_ig(text, clf_model, clf_pipeline, num_features)
    else:
        raise ValueError(
            f"Unknown XAI method: {method!r}. "
            "Valid options are: 'LIME', 'SHAP', 'IG'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classifier prediction   (Stage 2)
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

_CLASS_DESCRIPTIONS = {
    "Neoplasms": (
        "Neoplasms are abnormal growths of tissue caused by uncontrolled cell "
        "division. They may be benign or malignant (cancerous) and are classified "
        "by the tissue or organ of origin."
    ),
    "Digestive system diseases": (
        "Diseases affecting the gastrointestinal tract, including the oesophagus, "
        "stomach, intestines, liver, pancreas, and gallbladder. These range from "
        "inflammatory conditions to motility disorders and malignancies."
    ),
    "Nervous system diseases": (
        "Disorders of the central and peripheral nervous system, including "
        "neurodegenerative diseases, epilepsy, stroke, neuropathies, and tumours "
        "of neural tissue."
    ),
    "Cardiovascular diseases": (
        "Diseases of the heart and blood vessels, including coronary artery disease, "
        "heart failure, arrhythmias, hypertension, and vascular disorders. "
        "A leading cause of global morbidity and mortality."
    ),
    "General pathological conditions": (
        "Broad pathological processes that cut across organ systems, including "
        "inflammation, fibrosis, cell death, metabolic dysregulation, and "
        "systemic responses to injury or disease."
    ),
}
 
 
# ── System prompt (unchanged from original — already well-written) ─────────────
 
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
 
 
# ── Prompt builder ────────────────────────────────────────────────────────────
 
def build_prompt(
    text: str,
    predicted_class: str,
    feature_data: list[dict],
    user_category: str,
    ontology_context: Optional[list[dict]] = None,
) -> str:
    """
    Construct the LLM instruction prompt for the MCC pipeline.
 
    Parameters
    ----------
    text            : original abstract or clinical text snippet
    predicted_class : e.g. "Cardiovascular diseases"
    feature_data    : list of dicts with keys "feature_word" and "ancestors"
                      (from LIME + ontology enrichment)
    user_category   : "BEGINNER", "INTERMEDIATE", or "EXPERT"
    ontology_context: optional richer ontology dicts (go_term, definition, ancestors)
                      if available from the ontology module; falls back to
                      feature_data ancestor chains if None
    """
 
    # ── Class grounding ───────────────────────────────────────────────────────
    class_desc = _CLASS_DESCRIPTIONS.get(predicted_class, "")
 
    # ── Audience instruction ──────────────────────────────────────────────────
    # Describes WHO the reader is — more actionable than a difficulty label alone
    if user_category == "BEGINNER":
        audience_instruction = (
            "The reader is a patient, carer, or curious non-specialist with no "
            "medical training. Use plain everyday language. Avoid jargon entirely "
            "or define it immediately when unavoidable. Prefer short sentences and "
            "concrete analogies. The goal is understanding, not completeness."
        )
    elif user_category == "INTERMEDIATE":
        audience_instruction = (
            "The reader is a healthcare professional or science graduate who is "
            "familiar with general medical concepts but may not specialise in this "
            "area. Balance accessibility with accuracy. Define specialised terms "
            "briefly. Use medical vocabulary where it aids precision."
        )
    else:  # EXPERT
        audience_instruction = (
            "The reader is a clinician or biomedical researcher with deep domain "
            "expertise. Use precise clinical and biomedical terminology. Reference "
            "pathophysiological mechanisms, ontological relationships, and "
            "established disease frameworks. Abbreviations are acceptable."
        )
 
    # ── LIME token block ──────────────────────────────────────────────────────
    top_tokens = ", ".join(
        f'"{item["feature_word"]}"'
        for item in feature_data[:8]
    )
 
    # ── Ontology context block ────────────────────────────────────────────────
    # Uses richer ontology_context dicts if provided, otherwise falls back to
    # the ancestor chains already present in feature_data
    onto_lines = []
 
    if ontology_context:
        for entry in ontology_context:
            if not entry.get("go_term") and not entry.get("ancestors"):
                continue
            word = entry.get("token", entry.get("feature_word", ""))
            term_name = (entry.get("go_term") or {}).get("name", "")
            term_def  = (entry.get("go_term") or {}).get("definition", "")
            anc_names = [a["name"] for a in entry.get("ancestors", [])[:3]]
            anc_str   = " → ".join(anc_names) if anc_names else "N/A"
            line = f'  • "{word}"'
            if term_name:
                line += f" → [{term_name}]"
            if term_def:
                line += f": {term_def[:120]}…"
            if anc_str != "N/A":
                line += f" | Concept hierarchy: {anc_str}"
            onto_lines.append(line)
    else:
        for item in feature_data:
            word  = item["feature_word"]
            chain = " → ".join(item["ancestors"]) if item.get("ancestors") else "N/A"
            onto_lines.append(f'  • "{word}" | Concept hierarchy: {chain}')
 
    onto_block = "\n".join(onto_lines) if onto_lines else "  (no ontology context available)"
 
    # ── Full prompt ───────────────────────────────────────────────────────────
    prompt = f"""A biomedical abstract was classified by a machine learning model. \
Your task is to explain WHY the model made this prediction to the specified reader.
 
---
TASK TYPE: Multi-class medical abstract classification
MODEL INPUT (abstract excerpt):
\"\"\"{text[:600]}\"\"\"
 
MODEL PREDICTION: {predicted_class}
WHAT THIS MEANS: {class_desc}
 
KEY TOKENS RESPONSIBLE (from {XAI_METHOD} feature attribution):
{top_tokens}
 
ONTOLOGY CONTEXT FOR EACH TOKEN:
{onto_block}
 
---
TARGET READER: {user_category}
READER DESCRIPTION: {audience_instruction}
 
---
INSTRUCTIONS:
1. Explain which words or phrases drove the model's prediction and WHY they \
are medically relevant to the predicted class.
2. Weave at least 3 of the key tokens naturally into your explanation — do \
not list them mechanically.
3. Use the ontology context to reason about what each token represents \
conceptually, not just literally.
4. Write 4 to 6 sentences as a single coherent paragraph. No bullet points.
5. Do NOT simply restate the abstract. Explain the model's reasoning.
6. Do NOT open with "The model predicted…" — write naturally from the reader's \
perspective.
 
Generate the explanation now:"""
 
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5. Explanation generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_explanation(
    text: str,
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

    task_prompt = build_prompt(text, predicted_class, feature_data, user_category)

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


# Generic alias — use this name in new code; lime_coverage kept for back-compat.
xai_coverage = lime_coverage


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
