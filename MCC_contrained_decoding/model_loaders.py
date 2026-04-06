"""
model_loaders.py
================
All model loading functions for the User-Adaptive XAI Pipeline.

Functions
---------
    load_classifier()       → (model, pipeline)
    load_ner_pipeline()     → pipeline
    load_llm()              → (tokenizer, model)
    load_ontology_model()   → ontology  (thin wrapper around ontology_helpers)
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from config import (
    CLASSIFIER_MODEL_PATH,
    CLASS_NAMES,
    LLM_MODEL_NAME,
    NER_MODEL_NAME,
    ONTOLOGY_PATH,
)
from ontology_helpers import load_ontology


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


def load_ner_pipeline():
    """
    Load the biomedical NER pipeline used for multi-word entity merging.

    Returns
    -------
    HuggingFace NER pipeline with aggregation_strategy='simple'
    """
    print(f"[Loader] Loading NER model '{NER_MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    print("[Loader] NER model ready.\n")
    return ner


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
    model     = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("[Loader] LLM ready.\n")
    return tokenizer, model


def load_ontology_model():
    """
    Load and return the OWL ontology (also pre-computes global stats).

    Returns
    -------
    owlready2 ontology object
    """
    return load_ontology(ONTOLOGY_PATH)
