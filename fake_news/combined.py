"""
combined_decoding.py
=====================
Combined method (sv+cd): Steering Vectors control the vector space (a
forward hook at STEERING_LAYER shifts hidden states via alpha * v_scaled,
identical mechanism to steering_vector.py), while Constrained Decoding
controls the generation phase (a ReadabilityLogitsProcessor penalizes
hard/rare/long tokens during beam search, identical mechanism to
constrained_decoding.py).

Both run inside the SAME model.generate() call — the hook edits hidden
states during the forward pass, the logits processor edits the resulting
logits before token selection. These are compatible, non-overlapping stages.

Exposes generate_with_steering(prompt, alpha) — the SAME call signature as
SteeringVectorGenerator — so pipeline.py's existing run_sv() loop can use
this generator as a drop-in replacement with zero changes to that loop.
"""

from __future__ import annotations

import numpy as np
import torch

from config import (
    STEERING_VECTOR_PATH,
    STEERING_LAYER,
    STEERING_SCALE_FRACTION,
    STEERING_ALPHA_MAP,
    LAMBDA_MAP,
    NUM_BEAMS,
    LLM_MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
)
from constrained_decoding import ReadabilityLogitsProcessor


class CombinedGenerator:
    """
    SV at the vector space + CD at the generation phase, combined.

    Builds one steering vector (shared across levels, matching
    SteeringVectorGenerator) and THREE readability logits processors, one
    per level in STEERING_ALPHA_MAP, since CD's lambda is level-specific
    and this class must support being called with any of the three alphas
    interchangeably, exactly like SteeringVectorGenerator is.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = STEERING_LAYER
        self.num_beams = NUM_BEAMS

        # ---- SV component: identical to SteeringVectorGenerator ----
        v = np.load(STEERING_VECTOR_PATH)
        model_dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        self.v_torch = torch.tensor(v, dtype=model_dtype, device=device)

        hidden_norm = self._measure_hidden_norm()
        self.v_scaled = self.v_torch * (hidden_norm * STEERING_SCALE_FRACTION)

        # ---- CD component: one logits processor per level ----
        self._processors = {}
        for level in STEERING_ALPHA_MAP:
            lambda_value = float(LAMBDA_MAP.get(level.upper(), LAMBDA_MAP["INTERMEDIATE"]))
            self._processors[level] = ReadabilityLogitsProcessor(
                tokenizer=tokenizer,
                lambda_value=lambda_value,
                user_level=level.upper(),
                prompt_input_len=0,   # set per call
                eos_token_id=tokenizer.eos_token_id,
            )

        # reverse lookup: alpha value -> level name, so generate_with_steering(prompt, alpha)
        # can find the matching lambda without needing pipeline.py to pass the level explicitly
        self._alpha_to_level = {v: k for k, v in STEERING_ALPHA_MAP.items()}

        print(
            f"  [SV+CD] Ready | layer={self.layer_idx} | beams={self.num_beams} "
            f"| levels={list(STEERING_ALPHA_MAP.keys())}"
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

    def _resolve_processor(self, alpha: float) -> ReadabilityLogitsProcessor:
        """Finds the logits processor whose level matches this alpha value."""
        level = self._alpha_to_level.get(alpha)
        if level is None:
            # fallback: nearest alpha, in case of float rounding differences
            nearest = min(self._alpha_to_level, key=lambda a: abs(a - alpha))
            level = self._alpha_to_level[nearest]
        return self._processors[level]

    def generate_with_steering(self, prompt: str, alpha: float = 0.0) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        processor = self._resolve_processor(alpha)
        processor.prompt_input_len = prompt_len

        hook = self.model.model.layers[self.layer_idx].register_forward_hook(
            self._make_hook(alpha)
        )

        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=self.num_beams,
                    min_new_tokens=MIN_NEW_TOKENS,
                    max_new_tokens=LLM_MAX_NEW_TOKENS,
                    logits_processor=[processor],
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                )
        finally:
            hook.remove()

        generated_tokens = output[0][prompt_len:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded.strip()