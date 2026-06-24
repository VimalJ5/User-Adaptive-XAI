"""
pipeline_helpers.py
===================
Stateless helper functions used across the pipeline notebooks.

Sections
--------
    1. XAI feature attribution  (Stage 1)
    2. Classifier prediction    (Stage 1)
    3. Prompt building          (Stage 2)
    4. Explanation generation   (Stage 2)
    5. Readability metrics      (Stage 3)
    6. Faithfulness metrics     (Stage 3)
    7. Checkpoint I/O           (all stages)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import textstat

# Load prompts / class descriptions from the JSON sidecar file.
_PROMPTS = json.loads(
    (Path(__file__).parent / "prompts.json").read_text(encoding="utf-8")
)

from config import (
    LABEL_TO_CLASS,
    LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
    TASK_DESCRIPTION,
    TASK_NAME,
    USE_CONSTRAINED_DECODING,
    XAI_METHOD,
    XAI_NUM_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. XAI feature attribution  (Stage 1)
#    Supports LIME and Integrated Gradients (IG).
# ─────────────────────────────────────────────────────────────────────────────

def make_lime_predictor(model, clf):
    """
    Return a batch predictor function compatible with LimeTextExplainer.

    Parameters
    ----------
    model : AutoModelForSequenceClassification  (used for label2id ordering)
    clf   : HuggingFace text-classification pipeline
    """
    def predictor(texts: list[str]) -> np.ndarray:
        results = clf(texts, truncation=True, max_length=512)
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
    """
    from lime.lime_text import LimeTextExplainer

    predictor = make_lime_predictor(clf_model, clf_pipeline)
    explainer = LimeTextExplainer(class_names=list(class_names))
    exp = explainer.explain_instance(
        text, predictor, num_features=num_features, num_samples=num_samples
    )
    return exp.as_list()


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
    method      : "LIME" | "IG"  (case-insensitive)
    text        : raw input text
    clf_model   : AutoModelForSequenceClassification
    clf_pipeline: HuggingFace text-classification pipeline
    class_names : ordered list of class name strings
    num_features: number of top features to return
    num_samples : perturbation budget — LIME samples;
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
    elif method == "IG":
        return run_ig(text, clf_model, clf_pipeline, num_features)
    else:
        raise ValueError(
            f"Unknown XAI method: {method!r}. "
            "Valid options are: 'LIME', 'IG'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classifier prediction   (Stage 1)
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt building  (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_DESCRIPTIONS: dict = _PROMPTS["class_descriptions"]

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = _PROMPTS["system_prompt"]


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    text: str,
    predicted_class: str,
    xai_features: list,
    user_category: str,
    task_name: Optional[str] = None,
    task_description: Optional[str] = None,
) -> str:
    """
    Construct the LLM instruction prompt for the MCC pipeline.

    Parameters
    ----------
    text            : original abstract or clinical text snippet
    predicted_class : e.g. "Cardiovascular diseases"
    xai_features    : list of (word, score) tuples from XAI attribution
    user_category   : "BEGINNER", "INTERMEDIATE", or "EXPERT"
    task_name       : short human-readable task label
    task_description: optional detailed task context for the prompt
    """

    task_name = task_name or TASK_NAME
    task_description = task_description or TASK_DESCRIPTION

    # ── Class grounding ───────────────────────────────────────────────────────
    class_desc = _CLASS_DESCRIPTIONS.get(predicted_class, "")

    # ── Audience instruction ──────────────────────────────────────────────────
    if user_category == "BEGINNER":
        audience_instruction = (
            "The reader is a layman or non-specialist. Use plain everyday "
            "language. Avoid jargon entirely or define it immediately when "
            "unavoidable. Prefer short sentences and concrete analogies. The "
            "goal is understanding, not completeness."
        )
    elif user_category == "INTERMEDIATE":
        audience_instruction = (
            "The reader is a normal intermediate reader with some familiarity "
            "with the topic but not deep expertise. Balance accessibility with "
            "accuracy. Define specialised terms briefly. Use domain vocabulary "
            "where it aids precision."
        )
    else:  # EXPERT
        audience_instruction = (
            "The reader is a domain expert with deep subject knowledge. Use "
            "precise terminology and assume familiarity with established "
            "frameworks, mechanisms, and abbreviations that are standard for the "
            "task domain."
        )

    # ── XAI token block ───────────────────────────────────────────────────────
    token_lines = "\n".join(
        f'  "{word}" (score: {score:+.4f})'
        for word, score in xai_features
    )

    # ── Full prompt ───────────────────────────────────────────────────────────
    prompt = f"""A machine learning model was run on a {task_name} task. \
Your task is to explain WHY the model made this prediction to the specified reader.

---
TASK NAME: {task_name}
TASK DESCRIPTION: {task_description}
MODEL INPUT:
\"\"\"{text}\"\"\"

MODEL PREDICTION: {predicted_class}

KEY TOKENS RESPONSIBLE (from {XAI_METHOD} feature attribution):
{token_lines}
---
USER CATEGORY: {user_category}
AUDIENCE INSTRUCTION: {audience_instruction}

---
INSTRUCTIONS:
1. Explain which words or phrases drove the model's prediction and WHY they are relevant to the task and predicted label.
2. Do NOT simply restate the input. Explain the model's reasoning in the context of the task description.
3. Do NOT introduce facts, labels, or domain knowledge that are not supported by the input or task description.
4. Do NOT open with "The model predicted…" — write naturally from the reader's perspective.

Generate the explanation now:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4. Explanation generation  (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────


def generate_explanation(
    text: str,
    predicted_class: str,
    xai_features: list,
    user_category: str,
    tokenizer,
    model,
    generator=None,
    task_name: Optional[str] = None,
    task_description: Optional[str] = None,
) -> str:
    """
    Generate a natural-language explanation using the LLM.

    Parameters
    ----------
    text            : original input abstract
    predicted_class : classifier output label string
    xai_features    : list of (word, score) tuples from XAI attribution
    user_category   : "BEGINNER" | "INTERMEDIATE" | "EXPERT"
    tokenizer       : LLM tokenizer
    model           : LLM model
    generator       : optional ReadabilityBeamGenerator for constrained decoding
    task_name       : short human-readable task label
    task_description: optional detailed task context for the prompt

    Returns a plain string explanation.
    """
    if not xai_features:
        return (
            f"The model predicted {predicted_class}, "
            "but no XAI features were available to explain this decision."
        )

    task_prompt = build_prompt(
        text,
        predicted_class,
        xai_features,
        user_category,
        task_name=task_name,
        task_description=task_description,
    )

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
# 5. Readability metrics  (Stage 3)
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
# 6. Faithfulness metrics  (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def xai_coverage(explanation: str, xai_features: list) -> float:
    """
    Fraction of XAI feature words that appear in *explanation*.

    A feature 'counts' if its word appears in the explanation text
    (case-insensitive substring match).

    Parameters
    ----------
    explanation  : generated explanation string
    xai_features : list of (word, score) tuples from XAI attribution
    """
    if not xai_features:
        return 0.0

    lower = explanation.lower()
    hits  = sum(
        1 for word, _ in xai_features
        if word.lower() in lower
    )
    return round(float(hits) / len(xai_features), 4)


# Back-compat alias
lime_coverage = xai_coverage


# ─────────────────────────────────────────────────────────────────────────────
# 7. Checkpoint I/O
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
