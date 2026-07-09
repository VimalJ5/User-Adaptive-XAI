"""
constrained_decoding.py
=======================
Readability-constrained beam generation for user-adaptive XAI explanations.

Penalty strategy (two complementary signals):
  1. Token-intrinsic penalty  — precomputed at __init__ for every vocab token.
     Penalises rare/long/polysyllabic words regardless of position.
  2. Prefix-level pressure    — computed per beam per step from already-generated text.
     Grows as the response gets longer, nudging the model toward EOS.

Domain whitelist: domain-critical terms can be exempted from penalties at
EXPERT level and partially exempted at INTERMEDIATE level (see DOMAIN_WHITELIST
in config.py). Fully penalised at BEGINNER regardless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import LogitsProcessor
import syllables

from config import (
    CLAUSE_MARKERS,
    DALE_CHALL_FAMILIAR,
    DOMAIN_WHITELIST,
    HARDNESS_WEIGHTS,
    LAMBDA_MAP,
    LLM_MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    NUM_BEAMS,
    POLYSYLLABIC_THRESHOLD,
    SENTENCE_LEN_CAP,
    CHAR_PER_WORD_CAP,
    MAX_LENGTH_CAP,
    MAX_SYLLABLE_CAP,
)

_WORD_RE = re.compile(r"[A-Za-z]+")

# Whitelist exemption fractions per user level.
_WHITELIST_EXEMPTION = {
    "BEGINNER":     0.0,   # no exemption
    "INTERMEDIATE": 0.5,   # 50 % penalty reduction
    "EXPERT":       1.0,   # fully exempt (but lambda is 0 anyway)
}


def _count_syllables(word: str) -> int:
    return syllables.estimate(word)


@dataclass
class PrefixStats:
    word_count: int = 0
    sentence_count: int = 1


class ReadabilityLogitsProcessor(LogitsProcessor):
    """
    Subtract a readability penalty from beam logits at every generation step.

    Parameters
    ----------
    tokenizer       : HuggingFace tokenizer matching the LLM.
    lambda_value    : Penalty strength (0 = no penalty, higher = simpler output).
    user_level      : "BEGINNER" | "INTERMEDIATE" | "EXPERT" — controls whitelist exemption.
    prompt_input_len: Token length of the prompt (prefix to skip when reading generated text).
    eos_token_id    : Token id for <eos>; receives a bonus to encourage stopping.
    """

    def __init__(
        self,
        tokenizer,
        lambda_value: float,
        user_level: str,
        prompt_input_len: int,
        eos_token_id: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.lambda_value = float(max(0.0, lambda_value))
        self.user_level = user_level.upper()
        self.prompt_input_len = int(prompt_input_len)
        self.eos_token_id = eos_token_id
        self.whitelist_exemption = _WHITELIST_EXEMPTION.get(self.user_level, 0.0)

        # Unpack weights once.
        w = HARDNESS_WEIGHTS
        self.w_dale_chall    = w["dale_chall"]
        self.w_clause        = w["clause"]
        self.w_syllable      = w["syllable"]
        self.w_polysyllabic  = w["polysyllabic"]
        self.w_sentence_len  = w["sentence_len"]
        self.w_length        = w["length"]
        self.w_char_per_word = w["char_per_word"]

        # ------------------------------------------------------------------
        # Precompute per-token penalty vectors (token-intrinsic features).
        # ------------------------------------------------------------------
        vocab_size = len(tokenizer)
        self.token_dale_chall   = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_clause       = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_syllable     = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_polysyllabic = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_char_len     = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_is_wordish   = torch.zeros(vocab_size, dtype=torch.bool)

        for tok_id in range(vocab_size):
            raw_tok = tokenizer.convert_ids_to_tokens(tok_id)
            word, starts_new_word = self._clean_token(raw_tok)
            if not word or not _WORD_RE.fullmatch(word):
                continue

            self.token_is_wordish[tok_id] = True

            # Domain whitelist: compute effective exemption multiplier.
            in_whitelist = word in DOMAIN_WHITELIST
            exempt_fraction = self.whitelist_exemption if in_whitelist else 0.0
            penalty_scale = 1.0 - exempt_fraction  # 0.0 = fully exempt

            if starts_new_word and word in CLAUSE_MARKERS:
                self.token_clause[tok_id] = 1.0 * penalty_scale

            if starts_new_word and self._is_unfamiliar(word):
                self.token_dale_chall[tok_id] = 1.0 * penalty_scale

            syll = _count_syllables(word)
            self.token_syllable[tok_id] = min(syll / MAX_SYLLABLE_CAP, 1.0) * penalty_scale

            if syll >= POLYSYLLABIC_THRESHOLD:
                self.token_polysyllabic[tok_id] = 1.0 * penalty_scale

            self.token_char_len[tok_id] = min(len(word) / CHAR_PER_WORD_CAP, 1.0) * penalty_scale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_unfamiliar(self, word: str) -> bool:
        """True if word is not in the Dale-Chall familiar set."""
        return len(word) >= 2 and word not in DALE_CHALL_FAMILIAR

    def _clean_token(self, token: str) -> tuple[str, bool]:
        """
        Strip BPE/WordPiece prefix markers and return (cleaned_word, starts_new_word).
        Handles: Ġ (GPT-2/Qwen BPE), ▁ (SentencePiece), ## (BERT WordPiece).
        """
        if token is None:
            return "", False
        piece = token
        if piece.startswith("Ġ") or piece.startswith("▁"):
            return piece[1:].strip().lower(), True
        if piece.startswith("##"):
            return piece[2:].strip().lower(), False
        return piece.strip().lower(), False

    def _prefix_stats(self, input_ids_row: torch.Tensor) -> PrefixStats:
        """Decode generated tokens so far and return lightweight stats."""
        gen_ids = input_ids_row[self.prompt_input_len:]
        if gen_ids.numel() == 0:
            # Keep denominator safe before punctuation appears.
            return PrefixStats(word_count=0, sentence_count=1)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        word_count = len(_WORD_RE.findall(text))
        sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
        return PrefixStats(word_count=word_count, sentence_count=sentence_count)

    # ------------------------------------------------------------------
    # Core __call__
    # ------------------------------------------------------------------

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.lambda_value <= 0.0:
            return scores

        device = scores.device
        vocab_size = scores.size(-1)

        def _align(t: torch.Tensor, pad: float = 0.0) -> torch.Tensor:
            """Trim or pad a precomputed vector to match the live vocab size."""
            if t.numel() == vocab_size:
                return t.to(device)
            if t.numel() > vocab_size:
                return t[:vocab_size].to(device)
            padding = torch.full((vocab_size - t.numel(),), pad, dtype=t.dtype)
            return torch.cat([t, padding]).to(device)

        dale_chall   = _align(self.token_dale_chall)
        clause       = _align(self.token_clause)
        syllable     = _align(self.token_syllable)
        polysyllabic = _align(self.token_polysyllabic)
        char_len     = _align(self.token_char_len)
        wordish      = _align(self.token_is_wordish.float()).bool()

        # Token-intrinsic penalty (same for every beam row).
        token_penalty = self.lambda_value * (
            self.w_dale_chall    * dale_chall
            + self.w_clause        * clause
            + self.w_syllable      * syllable
            + self.w_polysyllabic  * polysyllabic
            + self.w_char_per_word * char_len
        )
        adjusted = scores - token_penalty.unsqueeze(0)  # broadcast over batch

        # EOS must exist and be in-range for length pressure to work.
        eos = self.eos_token_id
        if eos is None or not (0 <= int(eos) < vocab_size):
            return adjusted

        non_eos_wordish = torch.zeros_like(adjusted, dtype=torch.bool)
        for row_idx in range(input_ids.size(0)):
            non_eos_wordish[row_idx] = wordish.clone()
        non_eos_wordish[:, eos] = False

        # Prefix-level pressure: per beam, grows with output length.
        for row_idx in range(input_ids.size(0)):
            stats = self._prefix_stats(input_ids[row_idx])

            length_norm  = min(stats.word_count / MAX_LENGTH_CAP, 1.0)
            avg_sent_len = stats.word_count / max(stats.sentence_count, 1)
            sent_len_norm = min(avg_sent_len / SENTENCE_LEN_CAP, 1.0)

            if length_norm == 0 and sent_len_norm == 0:
                continue

            pressure = self.lambda_value * (
                self.w_length      * length_norm
                + self.w_sentence_len * sent_len_norm
            )
            adjusted[row_idx, non_eos_wordish[row_idx]] -= pressure
            adjusted[row_idx, eos] += pressure

        return adjusted


class ReadabilityBeamGenerator:
    """
    Thin wrapper around model.generate() that injects the ReadabilityLogitsProcessor.

    The processor is instantiated ONCE at init for the given user_level and
    reused across all generate() calls — avoiding repeated vocab-loop overhead.

    Usage
    -----
        gen = ReadabilityBeamGenerator(model, tokenizer, user_level="BEGINNER")
        explanation = gen.generate(system_prompt, task_prompt)
    """

    # Markers used by common chat-template formats to delimit assistant turns.
    _ASSISTANT_MARKERS = [
        "<|im_start|>assistant\n",   # Qwen / ChatML
        "assistant\n",               # generic
        "<|assistant|>",             # Phi
        "[/INST]",                   # Llama-2
    ]

    def __init__(
        self,
        model,
        tokenizer,
        user_level: str = "INTERMEDIATE",
        num_beams: int = NUM_BEAMS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.user_level = user_level.upper()
        self.num_beams = num_beams

        lambda_value = float(LAMBDA_MAP.get(self.user_level, LAMBDA_MAP["INTERMEDIATE"]))

        # Precompute once — reused for every sample in this run.
        # prompt_input_len is per-sample; updated via attribute before each call.
        self._base_processor = ReadabilityLogitsProcessor(
            tokenizer=tokenizer,
            lambda_value=lambda_value,
            user_level=self.user_level,
            prompt_input_len=0,        # updated per call, see generate()
            eos_token_id=tokenizer.eos_token_id,
        )
        print(f"  [CD] Processor ready for user_level={self.user_level}, lambda={lambda_value}")

    def _extract_response(self, decoded: str) -> str:
        """
        Robustly extract the assistant's reply from the decoded output.
        Tries known chat-template delimiters in order; falls back to the
        full decoded string if none match.
        """
        for marker in self._ASSISTANT_MARKERS:
            if marker in decoded:
                return decoded.split(marker)[-1].strip()
        return decoded.strip()

    def generate(
        self,
        system_prompt: str,
        task_prompt: str,
    ) -> str:
        """
        Generate a readability-constrained explanation for one sample.

        Parameters
        ----------
        system_prompt : Role/context instruction for the LLM.
        task_prompt   : The actual explanation request (includes XAI evidence).

        Returns
        -------
        str : The generated explanation (assistant turn only).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": task_prompt},
        ]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        # Update only the prompt length — all vocab penalty vectors stay cached.
        self._base_processor.prompt_input_len = prompt_len

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=self.num_beams,
                min_new_tokens=MIN_NEW_TOKENS,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                logits_processor=[self._base_processor],
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
            )

        generated_ids = outputs[0][prompt_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.strip()