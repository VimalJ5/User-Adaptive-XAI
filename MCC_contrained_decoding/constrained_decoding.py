"""
constrained_decoding.py
=======================
Readability-constrained beam generation for user-adaptive explanations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import LogitsProcessor

from config import (
    BIOMEDICAL_WHITELIST,
    CLAUSE_MARKERS,
    COMMON_WORDS,
    HARDNESS_CAPS,
    HARDNESS_WEIGHTS,
    LAMBDA_MAP,
    LLM_MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    NUM_BEAMS,
)


_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass
class PrefixStats:
    word_count: int = 0
    rare_count: int = 0
    clause_count: int = 0
    total_char_count: int = 0


class ReadabilityLogitsProcessor(LogitsProcessor):
    """
    Adjust beam scores with a readability hardness penalty.

    This processor uses a hybrid strategy:
    - Prefix features (length pressure) from the current beam text.
    - Token-intrinsic features (rare/clause/length proxy) from candidate tokens.
    """

    def __init__(
        self,
        tokenizer,
        lambda_value: float,
        prompt_input_len: int,
        eos_token_id: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.lambda_value = float(max(0.0, min(lambda_value, 0.5)))
        self.prompt_input_len = int(prompt_input_len)
        self.eos_token_id = eos_token_id

        self.w_length = HARDNESS_WEIGHTS["length"]
        self.w_rare = HARDNESS_WEIGHTS["rare"]
        self.w_clause = HARDNESS_WEIGHTS["clause"]
        self.w_avglen = HARDNESS_WEIGHTS["avglen"]

        self.length_cap = HARDNESS_CAPS["length_words"]

        vocab_size = len(tokenizer)
        self.token_rare = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_clause = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_avglen = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_is_wordish = torch.zeros(vocab_size, dtype=torch.bool)

        for tok_id in range(vocab_size):
            tok = tokenizer.convert_ids_to_tokens(tok_id)
            cleaned, starts_new_word = self._normalize_token_piece(tok)
            word = cleaned.lower()
            if not word:
                continue

            if _WORD_RE.fullmatch(word):
                self.token_is_wordish[tok_id] = True
                if starts_new_word and word in CLAUSE_MARKERS:
                    self.token_clause[tok_id] = 1.0
                if starts_new_word and self._is_rare_word(word):
                    self.token_rare[tok_id] = 1.0
                self.token_avglen[tok_id] = min(len(word) / HARDNESS_CAPS["avg_word_len"], 1.0)

    def _is_rare_word(self, word: str) -> bool:
        return (
            len(word) >= 5
            and word not in COMMON_WORDS
            and word not in BIOMEDICAL_WHITELIST
        )

    def _normalize_token_piece(self, token: str) -> tuple[str, bool]:
        starts_new_word = False
        piece = token

        if piece.startswith("Ġ") or piece.startswith("▁"):
            starts_new_word = True
            piece = piece[1:]
        elif piece.startswith("##"):
            starts_new_word = False
            piece = piece[2:]

        piece = piece.strip()
        return piece, starts_new_word

    def _extract_prefix_stats(self, input_ids_row: torch.Tensor) -> PrefixStats:
        gen_ids = input_ids_row[self.prompt_input_len :]
        if gen_ids.numel() == 0:
            return PrefixStats()

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        words = _WORD_RE.findall(text.lower())
        if not words:
            return PrefixStats()

        word_count = len(words)
        rare_count = sum(1 for w in words if self._is_rare_word(w))
        clause_count = sum(1 for w in words if w in CLAUSE_MARKERS)
        total_char_count = sum(len(w) for w in words)

        return PrefixStats(
            word_count=word_count,
            rare_count=rare_count,
            clause_count=clause_count,
            total_char_count=total_char_count,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.lambda_value <= 0.0:
            return scores

        device = scores.device
        scores_vocab = scores.size(-1)

        # Some model/tokenizer pairs can differ slightly in vocab size.
        # Align token feature vectors to the logits vocab to avoid shape mismatches.
        def _fit_vocab(t: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
            if t.numel() == scores_vocab:
                return t.to(device)
            if t.numel() > scores_vocab:
                return t[:scores_vocab].to(device)
            pad = torch.full((scores_vocab - t.numel(),), pad_value, dtype=t.dtype, device=device)
            return torch.cat([t.to(device), pad], dim=0)

        rare = _fit_vocab(self.token_rare, pad_value=0.0)
        clause = _fit_vocab(self.token_clause, pad_value=0.0)
        avglen = _fit_vocab(self.token_avglen, pad_value=0.0)
        wordish = _fit_vocab(self.token_is_wordish, pad_value=0.0).bool()

        token_penalty = self.lambda_value * (
            self.w_rare * rare + self.w_clause * clause + self.w_avglen * avglen
        )
        adjusted = scores - token_penalty.unsqueeze(0)

        if self.eos_token_id is None or not (0 <= int(self.eos_token_id) < scores_vocab):
            return adjusted

        non_eos_mask = torch.ones_like(adjusted, dtype=torch.bool)
        non_eos_mask[:, self.eos_token_id] = False

        for row_idx in range(input_ids.size(0)):
            stats = self._extract_prefix_stats(input_ids[row_idx])
            length_norm = min(stats.word_count / self.length_cap, 1.0)

            # Length pressure: as text gets long, make continuation slightly less attractive.
            if length_norm > 0:
                length_penalty = self.lambda_value * self.w_length * length_norm
                adjusted[row_idx, non_eos_mask[row_idx] & wordish] -= length_penalty
                adjusted[row_idx, self.eos_token_id] += length_penalty

        return adjusted


class ReadabilityBeamGenerator:
    """Wrapper around model.generate() with constrained decoding controls."""

    def __init__(self, model, tokenizer, num_beams: int = NUM_BEAMS):
        self.model = model
        self.tokenizer = tokenizer
        self.num_beams = num_beams

    def _resolve_lambda(self, user_category: str) -> float:
        if not user_category:
            return LAMBDA_MAP["EXPERT"]
        return float(LAMBDA_MAP.get(user_category.upper(), LAMBDA_MAP["EXPERT"]))

    def generate(
        self,
        system_prompt: str,
        task_prompt: str,
        user_category: str = "EXPERT",
    ) -> str:
        lambda_value = self._resolve_lambda(user_category)

        full_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        processor = ReadabilityLogitsProcessor(
            tokenizer=self.tokenizer,
            lambda_value=lambda_value,
            prompt_input_len=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=self.num_beams,
            min_new_tokens=MIN_NEW_TOKENS,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            logits_processor=[processor],
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### EXPLANATION:" in decoded:
            return decoded.split("### EXPLANATION:")[-1].strip()
        if "assistant\n" in decoded:
            return decoded.split("assistant\n")[-1].strip()
        return decoded.strip()
