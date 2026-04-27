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
import syllables

from config import (
    BIOMEDICAL_WHITELIST,
    CLAUSE_MARKERS,
    COMMON_WORDS,
    DALE_CHALL_FAMILIAR,
    HARDNESS_CAPS,
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
_VOWEL_RE = re.compile(r"[aeiou]+", re.IGNORECASE)  # vowel-cluster syllable heuristic

# Tightened normalization caps for biomedical text.
MAX_LENGTH_CAP = 30
RARE_CAP = 4
CLAUSE_CAP = 3
MAX_SYLLABLE_CAP = 4.0  # avg syllables/word cap (biomedical words can be 4–6 syllables)

# def _count_syllables(word: str) -> int:
#     """Lightweight vowel-cluster syllable counter.

#     Rules (same as CMU-based approximations):
#     - Count contiguous vowel groups as one syllable each.
#     - Ensure a minimum of 1 syllable per word.
#     - Subtract silent trailing 'e' (e.g., 'care' -> 2 not 3).
#     """
#     word = word.lower().strip()
#     if not word:
#         return 0
#     count = len(_VOWEL_RE.findall(word))
#     # Adjust for silent trailing 'e'
#     if word.endswith("e") and count > 1:
#         count -= 1
#     return max(1, count)

def _count_syllables(word: str) -> int:
    return syllables.estimate(word)



@dataclass
class PrefixStats:
    word_count: int = 0
    dale_chall_unfamiliar_count: int = 0   # replaces rare_count
    clause_count: int = 0
    total_syllable_count: int = 0
    polysyllabic_count: int = 0            # NEW: words >= POLYSYLLABIC_THRESHOLD syllables
    sentence_count: int = 0               # NEW: for avg sentence length
    total_char_count: int = 0             # NEW: for avg char/word (ARI signal)


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

        vocab_size = len(tokenizer)
        self.token_dale_chall = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_clause = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_syllable = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_polysyllabic = torch.zeros(vocab_size, dtype=torch.float32)
        self.token_char_len = torch.zeros(vocab_size, dtype=torch.float32)
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
                if starts_new_word and self._is_dale_chall_unfamiliar(word):
                    self.token_dale_chall[tok_id] = 1.0
                syll = _count_syllables(word)
                # Syllable count per word, normalized to [0, 1] by cap.
                self.token_syllable[tok_id] = min(
                    syll / MAX_SYLLABLE_CAP, 1.0
                )
                # Polysyllabic flag (SMOG/Fog signal).
                if syll >= POLYSYLLABIC_THRESHOLD:
                    self.token_polysyllabic[tok_id] = 1.0
                # Avg char length per word signal (ARI/Coleman-Liau).
                self.token_char_len[tok_id] = min(
                    len(word) / CHAR_PER_WORD_CAP, 1.0
                )

    def _is_dale_chall_unfamiliar(self, word: str) -> bool:
        """Return True if the word is not in the Dale-Chall familiar-word set."""
        return len(word) >= 2 and word not in DALE_CHALL_FAMILIAR

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
        dale_chall_unfamiliar_count = sum(
            1 for w in words if self._is_dale_chall_unfamiliar(w)
        )
        clause_count = sum(1 for w in words if w in CLAUSE_MARKERS)
        total_syllable_count = sum(_count_syllables(w) for w in words)
        polysyllabic_count = sum(
            1 for w in words if _count_syllables(w) >= POLYSYLLABIC_THRESHOLD
        )
        sentence_count = max(
            len(re.findall(r'[.!?]+', text)), 1
        )
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

        # Some model/tokenizer pairs can differ slightly in vocab size.
        # Align token feature vectors to the logits vocab to avoid shape mismatches.
        def _fit_vocab(t: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
            if t.numel() == scores_vocab:
                return t.to(device)
            if t.numel() > scores_vocab:
                return t[:scores_vocab].to(device)
            pad = torch.full((scores_vocab - t.numel(),), pad_value, dtype=t.dtype, device=device)
            return torch.cat([t.to(device), pad], dim=0)

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
        adjusted = scores - token_penalty.unsqueeze(0)

        if self.eos_token_id is None or not (0 <= int(self.eos_token_id) < scores_vocab):
            return adjusted

        non_eos_mask = torch.ones_like(adjusted, dtype=torch.bool)
        non_eos_mask[:, self.eos_token_id] = False

        for row_idx in range(input_ids.size(0)):
            stats = self._extract_prefix_stats(input_ids[row_idx])
            length_norm = min(stats.word_count / self.length_cap, 1.0)

            avg_sentence_len = stats.word_count / max(stats.sentence_count, 1)
            sentence_len_norm = min(avg_sentence_len / SENTENCE_LEN_CAP, 1.0)

            # Length + sentence-length pressure: nudge toward EOS as text grows.
            if length_norm > 0 or sentence_len_norm > 0:
                length_penalty = self.lambda_value * (
                    self.w_length * length_norm
                    + self.w_sentence_len * sentence_len_norm
                )
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
