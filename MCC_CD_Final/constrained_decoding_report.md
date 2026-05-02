# Constrained Decoding for User-Adaptive Biomedical Explanation Generation

## Overview

This document describes the **readability-constrained beam decoding** mechanism implemented in the User-Adaptive XAI pipeline (`MCC_CD_Final`). The goal of constrained decoding is to steer a language model's generation process at inference time — without any retraining — so that the vocabulary complexity, sentence structure, and overall text length of the produced explanation adapt automatically to the background of the end user. This is a core component of the adaptive explainability system, which targets three audience tiers: **BEGINNER** (patients / lay users), **INTERMEDIATE** (healthcare professionals), and **EXPERT** (clinicians / biomedical researchers).

---

## Motivation

Standard language model generation (greedy decoding or temperature-based sampling) optimises for fluency and faithfulness to the prompt, but has no explicit mechanism to regulate lexical difficulty. In a biomedical XAI context, an explanation generated for a clinician will be far too technical for a patient, and vice versa. Two broad approaches exist to address this:

1. **Fine-tuning / prompt-only adaptation** — insert audience instructions into the prompt text. Simple but brittle; the model may ignore or inconsistently follow style instructions.
2. **Inference-time logit manipulation (constrained decoding)** — directly modify the probability distribution over the vocabulary at every generation step to penalise tokens that violate readability constraints.

This pipeline adopts approach 2, implemented as a custom `LogitsProcessor` integrated with the Hugging Face `generate()` API. This approach is model-agnostic, requires no gradient updates, and operates on any causal language model at inference time.

---

## Architecture and Integration

The constrained decoding subsystem is implemented across two files:

| File | Role |
|---|---|
| `constrained_decoding.py` | Core logic: `ReadabilityLogitsProcessor`, `ReadabilityBeamGenerator` |
| `config.py` | All hyperparameters, λ values, weights, word lists |
| `pipeline_helpers.py` | Integration point: `generate_explanation()` dispatches to constrained or standard path |
| `word_lists.json` | External vocabulary data: Dale-Chall word list, clause markers, biomedical whitelist |

The pipeline flow is:

```
Input Text
    │
    ▼
Stage 1: XAI Feature Attribution (LIME / IG)
    │  → top-k (word, score) pairs
    ▼
Stage 2: Classifier Prediction + Ontology Enrichment
    │  → predicted class, ancestor chains per feature
    ▼
Stage 3: Prompt Construction + Constrained Beam Generation  ◄── CD operates here
    │  → natural-language explanation
    ▼
Stage 4: Readability + Faithfulness Metrics
```

When `USE_CONSTRAINED_DECODING = True` in `config.py`, the `generate_explanation()` function delegates generation to a `ReadabilityBeamGenerator` instance rather than calling `model.generate()` directly. If the flag is `False`, or if no generator object is supplied, the pipeline falls back to standard temperature-based sampling.

---

## The `ReadabilityLogitsProcessor`

The central class is `ReadabilityLogitsProcessor`, which implements the Hugging Face `LogitsProcessor` protocol. It is called by the model at every decoding step — once per token position — and receives the full matrix of raw log-probability scores (logits) over the vocabulary. It returns a modified score matrix that penalises tokens associated with linguistic difficulty.

### Initialisation: Vocabulary-Level Feature Precomputation

At construction time, the processor iterates over every token in the tokenizer's vocabulary (typically ~32 000 tokens for Qwen2.5) and precomputes five feature vectors:

```python
self.token_dale_chall    # float32 tensor: 1.0 if token is Dale-Chall unfamiliar
self.token_clause        # float32 tensor: 1.0 if token is a clause marker
self.token_syllable      # float32 tensor: normalised syllable count ∈ [0, 1]
self.token_polysyllabic  # float32 tensor: 1.0 if token has ≥ 3 syllables
self.token_char_len      # float32 tensor: normalised character length ∈ [0, 1]
self.token_is_wordish    # bool tensor: True if token is a pure alphabetic word
```

This precomputation is done **once** (not per generation step), making the per-step penalty application a simple vectorised subtraction — O(vocab_size) but fully parallelised on GPU.

#### Token Normalisation

Because different model families use different tokenisation schemes, the processor normalises token pieces before classifying them:

- **BPE / SentencePiece** (e.g., Qwen, LLaMA): leading `Ġ` or `▁` characters indicate a new-word boundary. These are stripped; the token is tagged `starts_new_word = True`.
- **WordPiece** (e.g., BERT): `##` prefix indicates a subword continuation. These are stripped; `starts_new_word = False`.

Only tokens that (after stripping) match `[A-Za-z]+` fully are treated as "word-like" (`token_is_wordish`). Punctuation, numbers, and special tokens are explicitly excluded from penalisation.

---

## Hardness Features and Heuristics

The processor computes seven distinct hardness signals, each motivated by an established readability framework.

### 1. Dale-Chall Unfamiliarity

**Source**: Dale-Chall Readability Formula (Dale & Chall, 1948).

The Dale-Chall formula defines difficulty as the proportion of words not found in a list of ~3 000 words familiar to typical 4th-grade readers. This pipeline extends that list with additional common vocabulary, biomedical layperson terms, and function words (stored in `word_lists.json` as `common_words` + `dale_chall_extra`, merged into `DALE_CHALL_FAMILIAR`).

A token is flagged as **Dale-Chall unfamiliar** if:
- It is alphabetic and ≥ 2 characters long, AND
- It does not appear in `DALE_CHALL_FAMILIAR`.

This is the primary lexical difficulty signal and receives a weight of **0.20** in the hardness composite.

```python
def _is_dale_chall_unfamiliar(self, word: str) -> bool:
    return len(word) >= 2 and word not in DALE_CHALL_FAMILIAR
```

### 2. Syllable Count (Normalised)

**Source**: Flesch Reading Ease / Flesch-Kincaid Grade Level (Flesch, 1948; Kincaid et al., 1975).

Syllable count per word is the strongest single predictor of readability in Flesch-type formulae. Syllables are estimated using the `syllables` library (Jurafsky-style vowel-cluster algorithm). The raw syllable count is normalised by a soft cap of **4.0** (clamped to `[0, 1]`), reflecting that biomedical terms commonly reach 4–6 syllables.

Weight in hardness composite: **0.08**.

```python
self.token_syllable[tok_id] = min(syll / MAX_SYLLABLE_CAP, 1.0)
```

### 3. Polysyllabic Flag (SMOG / Gunning Fog Signal)

**Source**: SMOG Index (McLaughlin, 1969), Gunning Fog Index (Gunning, 1952).

Both SMOG and Fog grade texts by the fraction of words with 3 or more syllables (termed "polysyllabic words"). A binary flag is set for any token whose estimated syllable count meets or exceeds `POLYSYLLABIC_THRESHOLD = 3`. This is treated as a separate, higher-weight signal from the raw syllable count because polysyllabic words cause disproportionate comprehension difficulty.

Weight in hardness composite: **0.25** (the highest individual weight).

```python
if syll >= POLYSYLLABIC_THRESHOLD:
    self.token_polysyllabic[tok_id] = 1.0
```

### 4. Clause Marker Density

**Source**: Syntactic complexity theory (Miller & Kintsch, 1980).

Subordinate clauses and connective constructions increase syntactic load and working memory demand. The `CLAUSE_MARKERS` set (stored in `word_lists.json`) contains ten high-frequency clause-introducing words: `which, although, whereas, however, nevertheless, therefore, moreover, furthermore, while, though`.

A token is penalised as a clause marker only if it `starts_new_word` (to avoid penalising subword fragments of longer terms).

Weight in hardness composite: **0.10**.

### 5. Average Sentence Length (Prefix-Level Feature)

**Source**: Flesch-Kincaid Grade Level; Coleman-Liau Index.

Sentence length is a prefix-level feature, meaning it is computed from the text already generated (the "prefix"), not from the candidate token. At each decoding step, the number of sentence-ending punctuation marks (`[.!?]+`) is counted in the prefix to estimate sentence count. The ratio `words / sentence_count` gives the average sentence length, normalised by `SENTENCE_LEN_CAP = 25.0`.

This signal applies a **length pressure** that increases the EOS (end-of-sentence) token's score and decreases non-EOS word tokens' scores as sentences grow longer, nudging the model toward ending sentences sooner.

Weight in hardness composite: **0.07**.

### 6. Text Length Pressure

**Source**: Design heuristic for conciseness.

A raw word-count-based length normalisation discourages unnecessarily long explanations. As the generated prefix grows toward `MAX_LENGTH_CAP = 30` words per "segment", a growing penalty is applied to non-EOS tokens and a symmetric bonus is added to the EOS token. This creates a soft length cap that biases the model toward finishing within a target word budget without a hard cutoff.

Weight in hardness composite: **0.25**.

```python
length_norm = min(stats.word_count / self.length_cap, 1.0)
...
adjusted[row_idx, non_eos_mask[row_idx] & wordish] -= length_penalty
adjusted[row_idx, self.eos_token_id] += length_penalty
```

### 7. Average Characters Per Word (ARI / Coleman-Liau Signal)

**Source**: Automated Readability Index (Senter & Smith, 1967); Coleman-Liau Index.

Both ARI and Coleman-Liau use character count per word as a proxy for lexical complexity, as longer words tend to be rarer and more morphologically complex. Token character length is normalised by `CHAR_PER_WORD_CAP = 9.0` (clamped to `[0, 1]`).

Weight in hardness composite: **0.05** (lowest individual weight; complementary signal).

---

## The Composite Hardness Penalty

At each decoding step, the seven features are combined into a single scalar penalty per token using a weighted sum. The weights are defined in `config.py` as `HARDNESS_WEIGHTS` and sum to 1.0:

| Feature | Weight | Formula Type |
|---|---|---|
| `polysyllabic` | 0.25 | Token-level (SMOG/Fog signal) |
| `length` | 0.25 | Prefix-level (conciseness pressure) |
| `dale_chall` | 0.20 | Token-level (Dale-Chall formula) |
| `clause` | 0.10 | Token-level (syntactic complexity) |
| `syllable` | 0.08 | Token-level (Flesch formula) |
| `sentence_len` | 0.07 | Prefix-level (sentence-level Flesch-Kincaid) |
| `char_per_word` | 0.05 | Token-level (ARI/Coleman-Liau signal) |

The raw logit scores are modified as:

```
adjusted_score(token) = raw_score(token) - λ × hardness(token)
```

Where `hardness(token)` is the weighted sum of all applicable features, and **λ (lambda)** is the audience-specific penalty strength.

---

## The λ (Lambda) Parameter: Audience Adaptation

The lambda parameter is the single most important control knob. It scales the entire penalty, allowing the degree of readability enforcement to be tuned per audience tier without any change to the model weights:

| Audience | λ value | Effect |
|---|---|---|
| `BEGINNER` | **5.0** | Strong penalty — dramatically suppresses complex vocabulary, long sentences, and polysyllabic tokens |
| `INTERMEDIATE` | **0.5** | Moderate penalty — modest vocabulary simplification |
| `EXPERT` | **0.0** | No penalty — standard beam search, no readability constraint |

When `λ = 0`, the processor is a no-op (returns scores unchanged immediately). The expert tier is therefore computationally free.

```python
LAMBDA_MAP = {
    "BEGINNER":     5.0,
    "INTERMEDIATE": 0.5,
    "EXPERT":       0.00,
}
```

The appropriate λ is resolved automatically from the `user_category` string passed to `ReadabilityBeamGenerator.generate()`.

---

## Beam Search Configuration

The constrained decoder is paired with beam search rather than greedy or sampling-based decoding. This is a deliberate design choice: beam search maintains multiple candidate sequences simultaneously, allowing the readability penalty to eliminate entire branches that accumulate too many hard tokens — not just the single next token.

```python
outputs = self.model.generate(
    **inputs,
    do_sample=False,          # deterministic beam search
    num_beams=4,              # 4 parallel beam candidates
    min_new_tokens=80,        # ensures a minimum explanation length
    max_new_tokens=180,       # caps output at 180 tokens
    logits_processor=[processor],
    repetition_penalty=1.05,  # light repetition suppression
    no_repeat_ngram_size=3,   # forbids exact 3-gram repetition
)
```

Key parameters (all configurable in `config.py`):

| Parameter | Value | Rationale |
|---|---|---|
| `NUM_BEAMS` | 4 | Balances generation quality vs. compute cost |
| `MIN_NEW_TOKENS` | 80 | Ensures substantive explanations (not truncated) |
| `LLM_MAX_NEW_TOKENS` | 180 | Prevents excessively long outputs |
| `repetition_penalty` | 1.05 | Mild penalty avoids verbatim repetition |
| `no_repeat_ngram_size` | 3 | Blocks 3-gram repeats |

---

## Prefix Statistics: Decoding-Time Monitoring

For the length and sentence-length features, the processor decodes the already-generated token sequence (the "prefix") at each step and extracts live statistics. This is handled by `_extract_prefix_stats()`:

```python
@dataclass
class PrefixStats:
    word_count: int               # total words generated so far
    dale_chall_unfamiliar_count: int
    clause_count: int
    total_syllable_count: int
    polysyllabic_count: int
    sentence_count: int           # estimated from [.!?]+ occurrences
    total_char_count: int
```

This prefix decoding adds a small overhead per beam per step, but enables genuinely context-sensitive length pressure — the penalty grows as the explanation becomes longer, rather than being a flat per-token constant.

---

## Vocabulary Alignment Safeguard

Some model–tokenizer pairs have a slight mismatch between the tokenizer's reported vocabulary size and the actual logits dimension produced at generation time. The processor handles this gracefully by padding or truncating its precomputed feature tensors to match the actual `scores.size(-1)`:

```python
def _fit_vocab(t: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    if t.numel() == scores_vocab:
        return t.to(device)
    if t.numel() > scores_vocab:
        return t[:scores_vocab].to(device)
    pad = torch.full((scores_vocab - t.numel(),), pad_value, ...)
    return torch.cat([t.to(device), pad], dim=0)
```

---

## Interaction with the Prompt

The constrained decoding mechanism operates **independently** of the prompt content. The system prompt (`prompts.json`) already instructs the model to adapt its language to the audience tier. The logit processor then acts as a second, hard enforcement layer: even if the model's base generation tendency ignores the audience instruction, the penalty will suppress complex tokens. This dual-channel design (soft prompt + hard penalty) improves robustness compared to relying on either mechanism alone.

---

## Ablation Toggle

The `USE_CONSTRAINED_DECODING` flag in `config.py` allows the constrained path to be disabled for ablation studies without changing any other pipeline logic:

```python
USE_CONSTRAINED_DECODING = True   # set to False for standard generation baseline
```

When disabled, `generate_explanation()` falls back to `model.generate()` with temperature sampling (`do_sample=True`, `temperature=0.1`). This produces the ablation baseline used for comparing readability metrics (FRE, FKGL, SMOG) and faithfulness metrics (XAI coverage, ontology hit rate) between constrained and unconstrained conditions.

---

## Readability Metrics Used for Evaluation

The generated explanations are evaluated against the following standard readability metrics (computed in `pipeline_helpers.py` via the `textstat` library):

| Metric | What it Measures | Target for BEGINNER |
|---|---|---|
| **Flesch Reading Ease (FRE)** | Syllable-weighted ease score (0–100; higher = easier) | ≥ 60 |
| **Flesch-Kincaid Grade Level (FKGL)** | US school grade level equivalent | ≤ 8 |
| **SMOG Index** | Years of education needed to understand | ≤ 8 |

These metrics directly mirror the hardness signals used during constrained decoding (syllable counts → FRE/FKGL/SMOG), making the evaluation internally consistent with the generation constraints.

---

## Design Decisions and Tradeoffs

### Why Not Fine-Tune?

Fine-tuning a language model to produce audience-specific outputs requires labelled data (parallel corpora at each difficulty level), significant compute, and a separate checkpoint per audience tier. The constrained decoding approach requires none of these: a single model checkpoint serves all three audience tiers, differentiated only by the λ value.

### Why Multiple Readability Signals?

No single readability formula captures all dimensions of linguistic complexity. The composite hardness score draws from four independent formula families (Flesch, Dale-Chall, SMOG/Fog, ARI) to avoid over-indexing on any one signal. For example, the Dale-Chall signal catches semantically unfamiliar words that may be short (e.g., "nevi", "coxa"), which the syllable-based signals would underweight.

### Why Token-Level AND Prefix-Level Features?

Token-level features (Dale-Chall, syllable, polysyllabic, clause, char_len) operate prospectively: they penalise complex tokens before they are chosen. Prefix-level features (length, sentence_len) operate retrospectively: they measure what has already been generated and apply growing pressure toward termination. Together they provide both forward-looking vocabulary control and backward-looking structural control.

### Why Beam Search (Not Sampling)?

Under temperature sampling, even a strong penalty may be overcome by a high-probability complex token. Beam search makes the penalty more effective because it evaluates entire trajectories: a beam that accumulated several unpenalised complex tokens will have a cumulatively lower score and is likely to be pruned in favour of beams that consistently selected simpler vocabulary.

### EOS Token Handling

The EOS token is explicitly excluded from the vocabulary penalty and instead receives a **bonus** proportional to length pressure. This means the model is actively nudged toward ending the explanation as the prefix length grows, rather than being passively allowed to terminate. Without this, the beam search could prefer to continue generating (adding more content to increase raw log-probability) even when the explanation is already of sufficient length.

---

## Summary of Key Hyperparameters

| Parameter | Value | Location |
|---|---|---|
| `LAMBDA_MAP["BEGINNER"]` | 5.0 | `config.py` |
| `LAMBDA_MAP["INTERMEDIATE"]` | 0.5 | `config.py` |
| `LAMBDA_MAP["EXPERT"]` | 0.0 | `config.py` |
| `NUM_BEAMS` | 4 | `config.py` |
| `MIN_NEW_TOKENS` | 80 | `config.py` |
| `LLM_MAX_NEW_TOKENS` | 180 | `config.py` |
| `POLYSYLLABIC_THRESHOLD` | 3 (syllables) | `config.py` |
| `SENTENCE_LEN_CAP` | 25.0 (words) | `config.py` |
| `CHAR_PER_WORD_CAP` | 9.0 (chars) | `config.py` |
| `MAX_LENGTH_CAP` (internal) | 30 (words) | `constrained_decoding.py` |
| `MAX_SYLLABLE_CAP` (internal) | 4.0 | `constrained_decoding.py` |
| `HARDNESS_WEIGHTS["polysyllabic"]` | 0.25 | `config.py` |
| `HARDNESS_WEIGHTS["length"]` | 0.25 | `config.py` |
| `HARDNESS_WEIGHTS["dale_chall"]` | 0.20 | `config.py` |
| `HARDNESS_WEIGHTS["clause"]` | 0.10 | `config.py` |
| `HARDNESS_WEIGHTS["syllable"]` | 0.08 | `config.py` |
| `HARDNESS_WEIGHTS["sentence_len"]` | 0.07 | `config.py` |
| `HARDNESS_WEIGHTS["char_per_word"]` | 0.05 | `config.py` |

---

## References

- Dale, E., & Chall, J. S. (1948). A formula for predicting readability. *Educational Research Bulletin*, 27(1), 11–28.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221–233.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of New Readability Formulas for Navy Enlisted Personnel*. Naval Air Station Memphis.
- McLaughlin, G. H. (1969). SMOG grading — a new readability formula. *Journal of Reading*, 12(8), 639–646.
- Miller, J. R., & Kintsch, W. (1980). Readability and recall of short prose passages. *Journal of Experimental Psychology: Human Learning and Memory*, 6(4), 335–354.
- Senter, R. J., & Smith, E. A. (1967). *Automated Readability Index*. Wright-Patterson Air Force Base Technical Report.
- Hugging Face. (2024). *LogitsProcessor* API Documentation. https://huggingface.co/docs/transformers/internal/generation_utils
