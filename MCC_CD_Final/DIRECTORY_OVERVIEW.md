# MCC_CD_Final — Directory Overview

**Project**: User-Adaptive Explainable AI (XAI) for Multi-Class Medical Abstract Classification (MCC)  
**Purpose**: A full end-to-end pipeline that classifies biomedical abstracts, generates feature attributions, enriches them with an OWL ontology, and produces natural-language explanations tailored to a reader's expertise level — with or without **Constrained Decoding (CD)** at generation time.

---

## Architecture at a Glance

```
Input Abstract
     │
     ▼
[Stage 1] XAI Feature Attribution  (LIME or Integrated Gradients)
     │   pipeline_helpers.run_xai()
     ▼
[Stage 2] Ontology Enrichment  →  Concept hierarchy per feature word
     │   pipeline_helpers.enrich_with_ontology() + ontology_helpers
     ▼
[Stage 3] LLM Explanation Generation  (Qwen2.5-1.5B-Instruct)
     │   pipeline_helpers.generate_explanation()
     │   ├── Standard greedy/sampling  (USE_CONSTRAINED_DECODING = False)
     │   └── Readability-Constrained Beam Search  (USE_CONSTRAINED_DECODING = True)
     │         constrained_decoding.ReadabilityBeamGenerator
     ▼
[Stage 4] Metrics  →  FRE, FKGL, SMOG, LIME coverage, ontology hit-rate
     │   pipeline_helpers.readability_metrics() / lime_coverage()
     ▼
Results CSV  (Results_w_CD/ or Results_wo_CD/)
```

---

## File-by-File Reference

### Core Python Modules

| File | Role |
|------|------|
| [`config.py`](#configpy) | Central configuration hub — all paths, hyperparameters, and experiment knobs live here |
| [`model_loaders.py`](#model_loaderspy) | Loads all models: classifier, NER pipeline, LLM, ontology |
| [`pipeline_helpers.py`](#pipeline_helperspy) | Stateless helper functions spanning all four pipeline stages |
| [`ontology_helpers.py`](#ontology_helperspy) | OWL ontology loading, concept lookup, and user-adaptive ancestor selection |
| [`constrained_decoding.py`](#constrained_decodingpy) | Readability-penalised beam search via a custom `LogitsProcessor` |

---

### `config.py`
**Central configuration** — the single file to edit before any run.

Key sections:
- **Word lists** — loads `word_lists.json` into Python sets (`COMMON_WORDS`, `DALE_CHALL_FAMILIAR`, `CLAUSE_MARKERS`, `BIOMEDICAL_WHITELIST`).
- **Paths** — local paths to the fine-tuned classifier, Disease Ontology (DOID) OWL file, NER model name, and LLM name.
- **Label map** — maps `Class_0`…`Class_4` to disease category strings (Neoplasms, Digestive, Nervous, Cardiovascular, General Pathological).
- **LIME / IG hyperparameters** — `LIME_NUM_FEATURES`, `LIME_NUM_SAMPLES`, `XAI_METHOD` (switch between `"LIME"` and `"IG"`).
- **LLM generation** — `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, `LLM_REPETITION_PENALTY`.
- **Constrained Decoding** — `USE_CONSTRAINED_DECODING`, `NUM_BEAMS`, `MIN_NEW_TOKENS`, `LAMBDA_MAP` (per-audience penalty strength), `HARDNESS_WEIGHTS` and `HARDNESS_CAPS` (readability feature coefficients).
- **Experiment Run Config** *(bottom section)* — `INPUT_TEXTS`, `USER_CATEGORY`, `ABLATION_MODE`, `EXPERIMENT_TAG`, and the auto-derived `EXPERIMENT_RESULTS_PATH`.

---

### `model_loaders.py`
**Model initialisation** — four thin loader functions used at notebook startup.

| Function | Returns | Notes |
|----------|---------|-------|
| `load_classifier()` | `(model, clf_pipeline)` | Fine-tuned `AutoModelForSequenceClassification`; GPU-aware |
| `load_ner_pipeline()` | `ner_pipeline` | `d4data/biomedical-ner-all` with `aggregation_strategy="simple"` |
| `load_llm()` | `(tokenizer, model)` | `Qwen2.5-1.5B-Instruct`; tries 4-bit BitsAndBytes quantisation, falls back to fp16 |
| `load_ontology_model()` | `ontology` | Thin wrapper over `ontology_helpers.load_ontology()` |

---

### `pipeline_helpers.py`
**Pipeline logic** — the largest module; covers all four pipeline stages.

| Section | Key Functions | Description |
|---------|---------------|-------------|
| **1. Entity merging** | `merge_entities()` | Replaces spaces in multi-word NER entities with underscores so LIME treats them as single tokens |
| **2. XAI attribution** | `run_lime()`, `run_ig()`, `run_xai()` | `run_xai()` is the dispatcher — calls LIME or Integrated Gradients depending on `XAI_METHOD`; IG uses Captum at embedding level |
| **3. Classification** | `predict_class()`, `enrich_with_ontology()` | Classifies text and maps LIME features to ontology ancestors per the selected ablation mode |
| **4. Prompt building** | `build_prompt()` | Constructs the structured LLM instruction prompt; injects class description, XAI token list, ontology ancestor chains, and audience-adaptive instructions |
| **5. Generation** | `generate_explanation()` | Orchestrates LLM generation; routes to `ReadabilityBeamGenerator` (constrained) or standard sampling |
| **6. Readability** | `readability_metrics()` | Returns FRE, FKGL, SMOG via `textstat` |
| **7. Faithfulness** | `lime_coverage()` / `xai_coverage()`, `ontology_hit_rate()` | Measures how many XAI features appear in the explanation; fraction of features that resolved to an ontology concept |
| **8. Checkpoint I/O** | `save_checkpoint()`, `load_checkpoint()`, `checkpoint_exists()` | JSON serialisation for intermediate results |

---

### `ontology_helpers.py`
**Ontology interface** — wraps `owlready2` for the Disease Ontology (DOID).

| Function | Description |
|----------|-------------|
| `load_ontology(owl_path)` | Loads OWL file, pre-computes `_MAX_DEPTH`, `_TOTAL_CLASSES`, `_OBJECT_PROPERTIES` for scoring |
| `find_concept(ontology, label)` | Three-tier search: owlready2 label → case-insensitive scan → direct name lookup |
| `get_ancestors(concept)` | Returns ancestors sorted general → specific (by depth), excluding `owl:Thing` |
| `calculate_suitability_score(concept, user_category)` | Scores a concept on a [0,1] scale using **specificity** (depth) and **popularity** (incoming properties); formula differs per audience tier |
| `select_ancestors(concept, user_category, ablation_mode)` | Picks the top-k ancestors most suited to the audience; supports ablation modes: `"normal"`, `"full"`, `"one_parent"`, `"no_ontology"` |

**Suitability scoring formulas:**
- **Beginner**: `0.6 × popularity + 0.4 × (1 − specificity)` — prefers popular, general concepts
- **Expert**: `specificity` — prefers deep, specific concepts
- **Intermediate**: `1 − |specificity − 0.5|` — prefers mid-level concepts

---

### `constrained_decoding.py`
**Readability-constrained generation** — implements token-level hardness penalties during beam search.

| Class / Function | Description |
|-----------------|-------------|
| `PrefixStats` | Dataclass accumulating per-beam text statistics: word count, Dale-Chall unfamiliar words, clause markers, syllables, polysyllabic words, sentence count, character count |
| `ReadabilityLogitsProcessor` | Custom `LogitsProcessor`; penalises logit scores for hard tokens using a weighted combination of 7 features (see below) |
| `ReadabilityBeamGenerator` | Wraps `model.generate()` with beam search + the processor; resolves λ from `LAMBDA_MAP` based on `user_category` |

**Hardness features (weights from `config.py`):**

| Feature | Signal | Weight (default) |
|---------|--------|-----------------|
| `dale_chall` | Word not in Dale-Chall familiar list | 0.20 |
| `clause` | Clause connective (sub-clause complexity) | 0.10 |
| `syllable` | Avg syllables per word (normalised) | 0.08 |
| `polysyllabic` | Word ≥ 3 syllables (SMOG/Fog signal) | 0.25 |
| `char_per_word` | Avg characters per word (ARI signal) | 0.05 |
| `length` | Prefix word count (encourages EOS as text grows) | 0.25 |
| `sentence_len` | Avg words per sentence | 0.07 |

**λ values** (penalty strength): `BEGINNER = 5.0`, `INTERMEDIATE = 0.5`, `EXPERT = 0.00` (no penalty).

---

### Data & Configuration Files

| File | Role |
|------|------|
| [`prompts.json`](#promptsjson) | Stores the LLM system prompt and per-class natural-language descriptions used in prompt building |
| [`word_lists.json`](#word_listsjson) | JSON arrays for `common_words`, `dale_chall_extra`, `biomedical_whitelist`, `clause_markers`; loaded by `config.py` |
| [`test_data.txt`](#test_datatxt) | Corpus of biomedical abstracts used as batch inputs; one abstract per non-empty line (~606 KB) |

#### `prompts.json`
- **`system_prompt`**: Instructs the LLM to act as a biomedical explanation assistant, stay grounded in provided features, and adapt style per audience tier.
- **`class_descriptions`**: One-sentence–to–two-sentence descriptions for each of the five disease categories, injected into every task prompt.

#### `word_lists.json`
Externalised vocabulary lists to avoid hardcoding large sets in Python:
- `common_words` — high-frequency everyday English words (readability baseline)
- `dale_chall_extra` — supplementary familiar words extending the Dale-Chall list
- `biomedical_whitelist` — domain-specific terms that should not be penalised as "hard"
- `clause_markers` — conjunctions/connectives that signal clause complexity

#### `test_data.txt`
Raw medical abstract corpus used for batch experimentation. When `INPUT_TEXTS = None` in `config.py`, the pipeline loads all non-empty lines from this file.

---

### Notebooks

| File | Location | Role |
|------|----------|------|
| [`experiment.ipynb`](experiment.ipynb) | Root | **Main experiment notebook** — runs the full pipeline for all inputs in `INPUT_TEXTS`, produces a results CSV |
| [`scripts/compare_results.ipynb`](scripts/compare_results.ipynb) | `scripts/` | **Interactive analysis notebook** — visualises and compares FRE/FKGL between With-CD and Without-CD result sets |

---

### Scripts

| File | Location | Role |
|------|----------|------|
| [`scripts/compare_results.py`](scripts/compare_results.py) | `scripts/` | CLI script — prints a formatted table comparing average FRE and FKGL between `Results_w_CD` and `Results_wo_CD` for Beginner and Expert audiences |
| [`scripts/_generate_charts.py`](scripts/_generate_charts.py) | `scripts/` | Headless chart generator — produces four matplotlib comparison plots (grouped bar, per-sample FRE line, per-sample FKGL line, delta bar) and saves them as PNG files |
| [`scripts/_patch_notebook.py`](scripts/_patch_notebook.py) | `scripts/` | Utility script for patching/updating the `compare_results.ipynb` notebook programmatically |

---

### Result Directories

| Directory | Contents |
|-----------|----------|
| `Results_w_CD/` | CSVs produced with Constrained Decoding enabled; also stores the four comparison chart PNGs |
| `Results_wo_CD/` | CSVs produced with standard generation (no Constrained Decoding) |

**Naming convention for CSVs**: `{audience}_{ablation_mode}_{xai_method}.csv`  
*Example*: `beginner_normal_lime.csv`, `expert_normal_ig.csv`

**Chart PNGs** (stored in `Results_w_CD/`):

| File | Description |
|------|-------------|
| `cd_comparison_bar.png` | Grouped bar chart — average FRE & FKGL per audience, With CD vs Without CD |
| `cd_comparison_fre_line.png` | Per-sample FRE line chart for Beginner and Expert |
| `cd_comparison_fkgl_line.png` | Per-sample FKGL line chart for Beginner and Expert |
| `cd_comparison_delta.png` | Per-sample delta bar (Beginner only) — improvement from CD |

---

### Documentation

| File | Role |
|------|------|
| [`constrained_decoding_report.md`](constrained_decoding_report.md) | Detailed technical report on the constrained decoding design, experiments, and readability results |
| [`results.tex`](results.tex) | LaTeX results section — includes readability metric tables and figure placeholders for the BTP report |

---

## Key Configuration Toggles (Quick Reference)

```python
# config.py

XAI_METHOD = "LIME"          # or "IG"
USER_CATEGORY = "BEGINNER"   # "BEGINNER" | "INTERMEDIATE" | "EXPERT"
ABLATION_MODE = "normal"     # "normal" | "full" | "one_parent" | "no_ontology"
USE_CONSTRAINED_DECODING = True   # False → standard sampling

# Results go to:
EXPERIMENT_RESULTS_PATH = Results_w_CD / f"{USER_CATEGORY.lower()}_{ABLATION_MODE}_{XAI_METHOD.lower()}.csv"
```

---

## Dependencies (Key Packages)

| Package | Purpose |
|---------|---------|
| `transformers` | Classifier, NER, LLM loading and generation |
| `torch` | Model inference and gradient computation |
| `lime` | LIME text explainer (Stage 1, LIME path) |
| `captum` | Integrated Gradients attribution (Stage 1, IG path) |
| `owlready2` | OWL ontology loading and traversal |
| `textstat` | Readability metric computation (FRE, FKGL, SMOG) |
| `syllables` | Syllable counting for hardness features |
| `bitsandbytes` | 4-bit LLM quantisation (optional, falls back to fp16) |
| `matplotlib` / `numpy` | Chart generation in analysis scripts |
