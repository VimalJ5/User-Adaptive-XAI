"""
steering_vector.py
===================
Activation-steering generation for user-adaptive XAI explanations (SV method).
Mirrors the notebook exactly:
  - load v_steering.npy -> scale it relative to hidden-state norm
  - forward hook at STEERING_LAYER subtracting alpha * v_scaled
  - sampled generation (temperature/top_p), same kwargs as the notebook
  - same prompt template and same output cleaning

Fully independent of constrained_decoding.py — does not import or touch it.
"""

from __future__ import annotations

import re
import numpy as np
import torch

from config import (
    STEERING_VECTOR_PATH,
    STEERING_LAYER,
    STEERING_SCALE_FRACTION,
    SV_MAX_NEW_TOKENS,
    SV_TEMPERATURE,
    SV_TOP_P,
    SV_REPETITION_PENALTY,
)


def make_prompt(sentence: str, label: str, attributions: list[tuple[str, float]]) -> str:
    """Exact prompt template from the steering-vector notebook."""
    lime_text = "\n".join(
        f"- {token}: {score:.3f}"
        for token, score in attributions
    )

    return f"""You are generating a faithful explanation of a sentiment classifier's prediction.

Input sentence:
{sentence}

Predicted label:
{label}

Model evidence (LIME feature importance):

{lime_text}

Explain why the model predicted "{label}" using the sentence and the model evidence above.

Your explanation should:
- explain the prediction rather than summarize the sentence,
- refer to the most relevant evidence when appropriate,
- not describe every highlighted word,
- not introduce external facts or assumptions,
- remain faithful to the input sentence.

Explanation:"""


def clean_output(text: str) -> str:
    """Exact cleaning logic from the notebook."""
    text = text.strip()

    if "Explanation:" in text:
        text = text.split("Explanation:", 1)[-1].strip()

    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class SteeringVectorGenerator:
    """
    Loads the steering vector once and scales it relative to the model's
    hidden-state norm (measured once at init — same as the notebook's
    measure_hidden_norm()). generate_with_steering() then runs generation
    for a given alpha, exactly like the notebook's function of the same name.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = STEERING_LAYER

        v = np.load(STEERING_VECTOR_PATH)
        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        self.v_torch = torch.tensor(v, dtype=model_dtype, device=device)

        hidden_norm = self._measure_hidden_norm()
        self.v_scaled = self.v_torch * (hidden_norm * STEERING_SCALE_FRACTION)

        print(
            f"  [SV] Steering vector ready | layer={self.layer_idx} "
            f"| hidden_norm={hidden_norm:.2f} "
            f"| scaled_norm={torch.norm(self.v_scaled).item():.2f}"
        )

    def _measure_hidden_norm(self) -> float:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer("Test sentence.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[self.layer_idx]
        return torch.norm(h, dim=-1).mean().item()

    def _make_hook(self, alpha: float):
        v_scaled = self.v_scaled

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
                steered = hidden_states - alpha * v_scaled
                return (steered, *rest)
            return output - alpha * v_scaled

        return hook

    def generate_with_steering(self, prompt: str, alpha: float = 0.0) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        hook = self.model.model.layers[self.layer_idx].register_forward_hook(
            self._make_hook(alpha)
        )

        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=SV_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=SV_TEMPERATURE,
                    top_p=SV_TOP_P,
                    repetition_penalty=SV_REPETITION_PENALTY,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            hook.remove()

        generated_tokens = output[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)