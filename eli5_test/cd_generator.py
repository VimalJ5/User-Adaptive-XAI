"""
cd_generator.py
===============
Local constrained-decoding implementation for Experiment A.
"""

from __future__ import annotations

import gc
import re
import functools
from dataclasses import dataclass

import syllables
import torch
from transformers import LogitsProcessor

from local_settings import (
    CLAUSE_MARKERS,
    DALE_CHALL_FAMILIAR,
    HARDNESS_WEIGHTS,
    LAMBDA_MAP,
    LLM_MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    NUM_BEAMS,
    POLYSYLLABIC_THRESHOLD,
    SENTENCE_LEN_CAP,
    CHAR_PER_WORD_CAP,
)

_WORD_RE = re.compile(r"[A-Za-z]+")
MAX_LENGTH_CAP = 30
MAX_SYLLABLE_CAP = 4.0

@functools.lru_cache(maxsize=50000)
def _count_syllables(word: str) -> int:
    return syllables.estimate(word)

@functools.lru_cache(maxsize=50000)
def _is_dale_chall_unfamiliar(word: str) -> bool:
    return len(word) >= 2 and word not in DALE_CHALL_FAMILIAR

def _normalize_token_piece(token: str) -> tuple[str, bool]:
    starts_new_word = False
    piece = token
    if piece is None:
        return "", False
    if piece.startswith("Ġ") or piece.startswith("▁"):
        starts_new_word = True
        piece = piece[1:]
    elif piece.startswith("##"):
        piece = piece[2:]
    return piece.strip(), starts_new_word

@dataclass
class PrefixStats:
    word_count: int = 0
    dale_chall_unfamiliar_count: int = 0
    clause_count: int = 0
    total_syllable_count: int = 0
    polysyllabic_count: int = 0
    sentence_count: int = 0
    total_char_count: int = 0

class ReadabilityLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, lambda_value: float, prompt_input_len: int, eos_token_id: int | None,
                 token_dale_chall, token_clause, token_syllable, token_polysyllabic, token_char_len, token_is_wordish) -> None:
        self.tokenizer = tokenizer
        self.lambda_value = float(max(0.0, lambda_value))
        self.prompt_input_len = int(prompt_input_len)
        self.eos_token_id = eos_token_id

        self.w_length = HARDNESS_WEIGHTS["length"]
        self.w_dale_chall = HARDNESS_WEIGHTS["dale_chall"]
        self.w_clause = HARDNESS_WEIGHTS["clause"]
        self.w_syllable = HARDNESS_WEIGHTS["syllable"]
        self.w_polysyllabic = HARDNESS_WEIGHTS["polysyllabic"]
        self.w_sentence_len = HARDNESS_WEIGHTS["sentence_len"]
        self.w_char_per_word = HARDNESS_WEIGHTS["char_per_word"]

        self.length_cap = float(MAX_LENGTH_CAP)

        self.token_dale_chall = token_dale_chall
        self.token_clause = token_clause
        self.token_syllable = token_syllable
        self.token_polysyllabic = token_polysyllabic
        self.token_char_len = token_char_len
        self.token_is_wordish = token_is_wordish

    def _extract_prefix_stats(self, input_ids_row: torch.Tensor) -> PrefixStats:
        gen_ids = input_ids_row[self.prompt_input_len :]
        if gen_ids.numel() == 0:
            return PrefixStats()

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        words = _WORD_RE.findall(text.lower())
        if not words:
            return PrefixStats()

        word_count = len(words)
        dale_chall_unfamiliar_count = sum(1 for w in words if _is_dale_chall_unfamiliar(w))
        clause_count = sum(1 for w in words if w in CLAUSE_MARKERS)
        total_syllable_count = sum(_count_syllables(w) for w in words)
        polysyllabic_count = sum(1 for w in words if _count_syllables(w) >= POLYSYLLABIC_THRESHOLD)
        sentence_count = max(len(re.findall(r"[.!?]+", text)), 1)
        total_char_count = sum(len(w) for w in words)

        return PrefixStats(
            word_count=word_count,
            dale_chall_unfamiliar_count=dale_chall_unfamiliar_count,
            clause_count=clause_count,
            total_syllable_count=total_syllable_count,
            polysyllabic_count=polysyllabic_count,
            sentence_count=sentence_count,
            total_char_count=total_char_count,
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.lambda_value <= 0.0:
            return scores

        device = scores.device
        scores_vocab = scores.size(-1)

        def _fit_vocab(t: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
            if t.numel() == scores_vocab:
                return t
            if t.numel() > scores_vocab:
                return t[:scores_vocab]
            pad = torch.full((scores_vocab - t.numel(),), pad_value, dtype=t.dtype, device=device)
            return torch.cat([t, pad], dim=0)

        dale_chall = _fit_vocab(self.token_dale_chall, pad_value=0.0)
        clause = _fit_vocab(self.token_clause, pad_value=0.0)
        syllable = _fit_vocab(self.token_syllable, pad_value=0.0)
        polysyllabic = _fit_vocab(self.token_polysyllabic, pad_value=0.0)
        char_len = _fit_vocab(self.token_char_len, pad_value=0.0)
        wordish = _fit_vocab(self.token_is_wordish, pad_value=0.0).bool()

        token_penalty = self.lambda_value * (
            self.w_dale_chall * dale_chall
            + self.w_clause * clause
            + self.w_syllable * syllable
            + self.w_polysyllabic * polysyllabic
            + self.w_char_per_word * char_len
        )
        
        # Ensure the adjusted tensor strictly matches the input dtype (float16)
        adjusted = (scores - token_penalty.unsqueeze(0)).to(scores.dtype)

        if self.eos_token_id is None or not (0 <= int(self.eos_token_id) < scores_vocab):
            return adjusted

        non_eos_mask = torch.ones_like(adjusted, dtype=torch.bool)
        non_eos_mask[:, self.eos_token_id] = False

        for row_idx in range(input_ids.size(0)):
            stats = self._extract_prefix_stats(input_ids[row_idx])
            length_norm = min(stats.word_count / self.length_cap, 1.0)
            avg_sentence_len = stats.word_count / max(stats.sentence_count, 1)
            sentence_len_norm = min(avg_sentence_len / SENTENCE_LEN_CAP, 1.0)

            if length_norm > 0 or sentence_len_norm > 0:
                length_penalty = self.lambda_value * (
                    self.w_length * length_norm
                    + self.w_sentence_len * sentence_len_norm
                )
                adjusted[row_idx, non_eos_mask[row_idx] & wordish] -= length_penalty
                adjusted[row_idx, self.eos_token_id] += length_penalty

        return adjusted


class ReadabilityBeamGenerator:
    def __init__(self, model, tokenizer, num_beams: int = NUM_BEAMS):
        self.model = model
        self.tokenizer = tokenizer
        self.num_beams = num_beams

        vocab_size = len(tokenizer)
        device = model.device

        token_dale_chall = torch.zeros(vocab_size, dtype=torch.float32)
        token_clause = torch.zeros(vocab_size, dtype=torch.float32)
        token_syllable = torch.zeros(vocab_size, dtype=torch.float32)
        token_polysyllabic = torch.zeros(vocab_size, dtype=torch.float32)
        token_char_len = torch.zeros(vocab_size, dtype=torch.float32)
        token_is_wordish = torch.zeros(vocab_size, dtype=torch.bool)

        for tok_id in range(vocab_size):
            tok = tokenizer.convert_ids_to_tokens(tok_id)
            cleaned, starts_new_word = _normalize_token_piece(tok)
            word = cleaned.lower()
            if not word:
                continue

            if _WORD_RE.fullmatch(word):
                token_is_wordish[tok_id] = True
                if starts_new_word and word in CLAUSE_MARKERS:
                    token_clause[tok_id] = 1.0
                if starts_new_word and _is_dale_chall_unfamiliar(word):
                    token_dale_chall[tok_id] = 1.0
                syllable_count = _count_syllables(word)
                token_syllable[tok_id] = min(syllable_count / MAX_SYLLABLE_CAP, 1.0)
                if syllable_count >= POLYSYLLABIC_THRESHOLD:
                    token_polysyllabic[tok_id] = 1.0
                token_char_len[tok_id] = min(len(word) / CHAR_PER_WORD_CAP, 1.0)

        # Cast tensors to float16 to match the model's dtype and prevent segfaults
        self.token_dale_chall = token_dale_chall.to(dtype=torch.float16, device=device)
        self.token_clause = token_clause.to(dtype=torch.float16, device=device)
        self.token_syllable = token_syllable.to(dtype=torch.float16, device=device)
        self.token_polysyllabic = token_polysyllabic.to(dtype=torch.float16, device=device)
        self.token_char_len = token_char_len.to(dtype=torch.float16, device=device)
        self.token_is_wordish = token_is_wordish.to(device)

    def _resolve_lambda(self, user_category: str) -> float:
        if not user_category:
            return LAMBDA_MAP["EXPERT"]
        return float(LAMBDA_MAP.get(user_category.upper(), LAMBDA_MAP["EXPERT"]))

    def generate(self, system_prompt: str, task_prompt: str, user_category: str = "EXPERT") -> str:
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
            token_dale_chall=self.token_dale_chall,
            token_clause=self.token_clause,
            token_syllable=self.token_syllable,
            token_polysyllabic=self.token_polysyllabic,
            token_char_len=self.token_char_len,
            token_is_wordish=self.token_is_wordish
        )

        try:
            with torch.inference_mode():
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
        finally:
            del inputs
            del processor
            if "outputs" in locals():
                del outputs
            gc.collect()