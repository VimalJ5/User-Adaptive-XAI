"""
model_loaders.py
================
All model loading functions for the User-Adaptive XAI Pipeline.

Functions
---------
    load_classifier()   → (model, pipeline)
    load_llm()          → (tokenizer, model)
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from config import (
    CLASSIFIER_MODEL_PATH,
    CLASS_NAMES,
    LLM_MODEL_NAME,
)


def load_classifier():
    """
    Load the fine-tuned sequence classifier.

    Returns
    -------
    model : AutoModelForSequenceClassification
    clf   : HuggingFace text-classification pipeline (top_k=None)
    """
    print(f"[Loader] Loading classifier from '{CLASSIFIER_MODEL_PATH}' …")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
    device    = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
    )
    print("[Loader] Classifier ready.\n")
    return model, clf


def load_llm():
    """
    Load the Qwen instruction-tuned LLM for explanation generation.

    Returns
    -------
    tokenizer : AutoTokenizer
    model     : AutoModelForCausalLM  (float16, device_map='auto')
    """
    print(f"[Loader] Loading LLM '{LLM_MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[Loader] LLM ready.\n")
    return tokenizer, model
