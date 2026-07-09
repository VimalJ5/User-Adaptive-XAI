"""
xai.py
======
LIME-based token attribution for any HuggingFace text classification pipeline.

Returns a ranked list of (token, importance_score) pairs for a given input,
which are fed into the prompt builder in pipeline.py.
"""

from __future__ import annotations

import numpy as np
from lime.lime_text import LimeTextExplainer

from config import LIME_NUM_FEATURES, LIME_NUM_SAMPLES


class LIMEExplainer:
    """
    Wraps LimeTextExplainer for any HuggingFace text classification pipeline.

    Parameters
    ----------
    hf_pipeline : A transformers pipeline object (task="text-classification").
    class_names : List of label strings matching the model's output order.
    num_features: Top-K tokens to return (default from config).
    num_samples : LIME perturbation samples (default from config).
    """

    def __init__(
        self,
        hf_pipeline,
        class_names: list[str],
        num_features: int = LIME_NUM_FEATURES,
        num_samples: int = LIME_NUM_SAMPLES,
    ) -> None:
        self.pipeline = hf_pipeline
        self.class_names = class_names
        self.num_features = num_features
        self.num_samples = num_samples

        self.explainer = LimeTextExplainer(class_names=class_names)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Converts HF pipeline output to a probability matrix (N x num_classes).
        Handles both top-1 and all-labels output from the pipeline.
        """
        results = self.pipeline(texts, top_k=None)  # top_k=None -> all labels
        label_to_idx = {lbl: i for i, lbl in enumerate(self.class_names)}

        proba = np.zeros((len(texts), len(self.class_names)), dtype=np.float32)
        for i, result in enumerate(results):
            for item in result:
                idx = label_to_idx.get(item["label"])
                if idx is not None:
                    proba[i, idx] = item["score"]
        return proba

    def explain(
        self, text: str, predicted_label: str
    ) -> list[tuple[str, float]]:
        """
        Run LIME on a single input text for the predicted class.

        Parameters
        ----------
        text            : The raw input sentence.
        predicted_label : The label the classifier assigned (used to focus LIME).

        Returns
        -------
        List of (token, score) sorted by absolute importance (descending).
        Positive score  -> token supports the predicted label.
        Negative score -> token contradicts it.
        """
        label_idx = self.class_names.index(predicted_label)

        exp = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[label_idx],
        )

        raw = exp.as_list(label=label_idx)  # [(token, weight), ...]
        # Sort by absolute importance so the most influential tokens come first.
        sorted_attrs = sorted(raw, key=lambda x: abs(x[1]), reverse=True)
        return sorted_attrs