# Readability-Constrained Beam Search
## Theory, Design Decisions, and Integration into the User-Adaptive XAI Pipeline

---

## 1. What This Is and Why It Matters

Standard language model generation picks the highest-probability tokens at each step. The model has no notion of *who is reading* the output. It produces text at whatever complexity its training distribution implies — which for a biomedical LLM means dense, technical language by default.

The approach your professor proposes solves this at **inference time**, without any retraining. Instead of changing the model, you change the *scoring function* that beam search uses to select between candidate continuations:

```
score(y₁:t | x) = log P(y₁:t | x) − λ · C_hard(y₁:t)
```

- `log P(y₁:t | x)` — the model's own probability for the prefix so far (unchanged)
- `C_hard(y₁:t)` — a hardness penalty you define based on surface features of the text
- `λ` — a scalar dial: high λ = aggressive simplification, low λ = near-standard generation

The model still drives *what* is said (content, facts, faithfulness). λ controls *how* it is said (complexity, vocabulary, sentence structure). This maps perfectly onto your `user_category` input — BEGINNER gets high λ, EXPERT gets low λ, and the same model weights serve all users at runtime.

---

## 2. How Beam Search Works (and Where the Penalty Goes)

Standard beam search maintains B candidate sequences (beams) at each step. At every token position it:

1. Extends each beam by every possible next token
2. Scores each extension with `log P(token | beam_prefix)`
3. Keeps only the top-B scoring sequences

Your modification inserts the hardness penalty at step 2:

```
Standard:    score = log P(token | prefix)
Modified:    score = log P(token | prefix) − λ · ΔC_hard(token, prefix)
```

Where `ΔC_hard(token, prefix)` is the **incremental hardness** added by appending this token to the current prefix. A token like "lipopolysaccharide" that is long, rare, and rare-looking contributes a large ΔC_hard and gets penalised. A token like "cells" contributes near-zero ΔC_hard and is unaffected.

In HuggingFace Transformers, this is implemented as a `LogitsProcessor` — a hook that receives the full logits tensor before selection and modifies it in-place. No changes to the model architecture or weights are needed.

---

## 3. The Hardness Penalty Function

Your professor recommends starting with 2–3 features. The full recommended set is:

### Feature 1 — Sentence/Prefix Length

```
length_feature = current_word_count / MAX_LENGTH_CAP
```

Longer prefixes are harder to parse. This feature creates a gentle pressure toward shorter sentences by penalising tokens that extend already-long sequences.

**Why it works:** Flesch Reading Ease is partly a function of average sentence length. Penalising length indirectly improves FRE.

### Feature 2 — Rare Word Penalty

```
rare_feature = rare_word_count_in_prefix / RARE_CAP
```

A word is "rare" if it does not appear in a common English word frequency list.

**Critical biomedical caveat:** General frequency lists will flag almost all domain terms — "apoptosis", "carcinoma", "angiogenesis" — as rare. For biomedical XAI, you need a **two-tier definition**:
- Rare AND not in a biomedical accessibility whitelist → penalise
- Rare BUT in the whitelist (e.g. "cancer", "cell", "immune", "DNA") → do not penalise

This distinction separates accessible biomedical vocabulary from genuine expert jargon.

### Feature 3 — Clause Marker Penalty

```
clause_feature = clause_marker_count_in_prefix / CLAUSE_CAP
```

Clause markers ("which", "although", "whereas", "however", "nevertheless") introduce subordinate clauses and syntactic complexity. Their presence correlates with harder Dale-Chall scores.

### Feature 4 — Average Word Length (optional)

```
avglen_feature = (total_char_count / word_count) / MAX_WORD_LEN_CAP
```

Longer words are harder. This is the second major component of Flesch Reading Ease (syllables per word ≈ chars per word as a proxy).

### Combined Penalty

```
C_hard = w₁·length_feature + w₂·rare_feature + w₃·clause_feature + w₄·avglen_feature
```

Where w₁ + w₂ + w₃ + w₄ = 1.0. A reasonable starting point:

| Weight | Value | Rationale |
|--------|-------|-----------|
| w₁ (length) | 0.40 | Strongest single predictor of FRE |
| w₂ (rare words) | 0.35 | Vocabulary difficulty drives Dale-Chall |
| w₃ (clause markers) | 0.15 | Syntactic complexity, secondary signal |
| w₄ (avg word length) | 0.10 | Correlated with rare words, lower weight |

---

## 4. Normalisation — The Most Critical Detail

**This is where most implementations fail.**

Log probabilities live in roughly [−10, 0]. Raw hardness features live in completely different ranges — a 40-word sentence gives length=40, a 6-rare-word prefix gives rare=6. Without normalisation:

```
λ = 0.3, raw_length = 40  →  penalty = 0.3 × 40 = 12
```

This subtracts 12 from every log probability, completely overriding the model's signal. The output degenerates to single words or nonsense.

The solution is to normalise every feature to [0, 1] using soft caps before combining:

```python
length_norm = min(word_count / 60,   1.0)   # cap at 60 words
rare_norm   = min(rare_count  / 8,   1.0)   # cap at 8 rare words
clause_norm = min(clause_count / 4,  1.0)   # cap at 4 clause markers
avglen_norm = min(avg_char_len / 8.0, 1.0)  # cap at 8 chars/word
```

After normalisation, `C_hard` always lives in [0, 1], and λ = 0.3 means *"subtract at most 0.3 from log probabilities"* — a meaningful, interpretable quantity.

---

## 5. Incremental Computation

Beam search calls the scoring function at every token position for every beam. With prefix lengths up to 300 tokens and beam size 4, this is ~1200 scoring calls per generation. Recomputing hardness from scratch each time would be O(t²) overall.

The efficient solution is to maintain a running **HardnessState** per beam that updates in O(1) per token:

```python
# At each step, when token is appended:
if new_word_completed:
    word_count      += 1
    total_char_len  += len(word)
    
    if word not in COMMON_WORDS and word not in BIO_WHITELIST:
        rare_count  += 1
    
    if word in CLAUSE_MARKERS:
        clause_count += 1
```

The only complexity is handling **subword tokenization** correctly. Models like Qwen and BioBART tokenize "apoptosis" as multiple tokens ("ap", "##optosis" or "Ġap", "optosis"). The state tracker must assemble subword tokens into complete words before applying feature logic — checking each fragment independently would produce incorrect rare-word counts.

---

## 6. The λ Parameter — Target Values

λ = 0 recovers standard beam search exactly. As λ increases, the output simplifies. Recommended starting values:

| User Category | λ | Expected FRE | Expected Dale-Chall |
|---------------|---|-------------|---------------------|
| BEGINNER | 0.30 | 55–85 | ≤ 7.5 |
| INTERMEDIATE | 0.15 | 35–55 | 7.5–8.5 |
| EXPERT | 0.05 | −10–35 | ≥ 8.5 |

These are **starting points, not final values**. The actual λ needed depends on your base model's default complexity. Qwen2.5-1.5B-Instruct tends to generate moderately technical text; a model pre-trained heavily on PubMed may need higher λ to reach BEGINNER targets.

**Tuning procedure:**
1. Generate at λ = {0.0, 0.1, 0.2, 0.3, 0.5} on 10 test examples
2. Plot FRE vs λ — expect a roughly monotonic increase
3. Read off the λ value where FRE crosses into each user category's target bracket
4. Use those λ values for evaluation

---

## 7. Failure Modes and Guards

**Degenerate short outputs:** The penalty rewards brevity, so beam search may prefer outputs that end after one sentence. Guard: set `min_new_tokens = 80` (roughly 2–3 sentences). This is presented as optional in the professor's notes but is practically mandatory.

**Fluency degradation at high λ:** If λ is too large, the model avoids rare words so aggressively that outputs become ungrammatical or vague. Guard: cap λ at 0.5 and prefer soft normalisation. If outputs start losing coherence, reduce λ and add more training data instead.

**Domain term suppression:** Without a biomedical whitelist, the rare word penalty will suppress "apoptosis", "proliferation", and "metastasis" even in BEGINNER outputs, where they should appear with explanation. The whitelist is the domain-specific contribution of this implementation.

**Scale mismatch across features:** If one feature (e.g. rare words) regularly saturates at 1.0 while others stay near 0.1, it will dominate regardless of weight settings. Monitor per-feature values on your test set and adjust caps accordingly.

---

## 8. Relationship to the Rest of the Pipeline

This approach sits at **Module 4** of your existing pipeline — replacing or augmenting the plain `model.generate()` call that currently produces explanations.

```
Module 1: Task Model Prediction
Module 2: LIME Feature Extraction
Module 3: Ontology Enrichment + Difficulty Scoring
Module 4: Explanation Generation  ← constrained decoding goes here
Module 5: Readability / Faithfulness Evaluation
```

The inputs to Module 4 do not change — it still receives the model prediction, LIME tokens, ontology context, and user category. The only change is that the LLM's `generate()` call is wrapped with the `LogitsProcessor` parameterised by the appropriate λ for the given user category.

This means the constrained decoding approach is **additive** — it can be combined with GRPO training (use a GRPO-trained model as the base, then apply constrained decoding at inference) or used standalone with the base instruction-tuned model.

---

## 9. Integration into the Existing MCC/HoC Pipeline

### Files to modify

**`config.py`** — add λ values per user category:

```python
# Constrained decoding λ values per user category
LAMBDA_MAP = {
    "BEGINNER":     0.30,
    "INTERMEDIATE": 0.15,
    "EXPERT":       0.05,
}

# Beam search settings
NUM_BEAMS      = 4
MIN_NEW_TOKENS = 80
```

**`model_loaders.py`** — no changes needed. The constrained decoding module wraps whatever model is already loaded.

**`pipeline_helpers.py`** — replace the `generate_explanation()` function body. Currently it calls the OpenAI API. For local constrained decoding, it instead calls `ReadabilityBeamGenerator.generate()`. Keep the function signature identical so `03_llm.ipynb` requires no changes:

```python
# Before (OpenAI API call):
def generate_explanation(prompt: str) -> str:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content.strip()

# After (constrained local generation):
def generate_explanation(
    prompt: str,
    user_category: str = "EXPERT",
    generator = None,       # ReadabilityBeamGenerator instance, passed in
) -> str:
    if generator is None:
        # fallback to API if no local generator provided
        return _api_generate(prompt)
    return generator.generate(prompt, user_category=user_category)
```

**`03_llm.ipynb`** — add one cell before the generation loop to initialise the generator, then pass it through:

```python
# Add this cell after model loading:
from constrained_decoding import ReadabilityBeamGenerator
generator = ReadabilityBeamGenerator(model, tokenizer, num_beams=config.NUM_BEAMS)

# In the generation loop, change:
explanation = ph.generate_explanation(prompt)
# to:
explanation = ph.generate_explanation(prompt, user_category=user_cat, generator=generator)
```

**`constrained_decoding.py`** — new file, sits alongside `config.py`, `model_loaders.py`, `pipeline_helpers.py` at the project root.


---

## 10. What to Report

The key result your professor expects is a table showing that different λ values produce measurably different readability scores on the same inputs:

| Condition | FRE (mean) | Dale-Chall (mean) | % in BEGINNER bracket | % in EXPERT bracket |
|-----------|------------|--------------------|-----------------------|---------------------|
| λ = 0.00 (baseline) | — | — | — | — |
| λ = 0.05 (EXPERT) | — | — | — | — |
| λ = 0.15 (INTERMEDIATE) | — | — | — | — |
| λ = 0.30 (BEGINNER) | — | — | — | — |

A secondary ablation table showing the contribution of individual features:

| Feature set | FRE gap (BEGINNER−EXPERT) | Dale-Chall gap |
|-------------|--------------------------|----------------|
| Length only | — | — |
| Length + rare words | — | — |
| Full (+ clause + avg len) | — | — |

These two tables, with qualitative examples showing the same input rendered at different complexity levels, constitute the complete empirical result for this component.
