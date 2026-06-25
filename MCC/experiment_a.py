"""
experiment_a.py
===============
Experiment A: Direct Verbalization with Constrained Decoding on ELI5.

Pipeline per sample:
    ELI5 question -> Qwen2.5 + Constrained Decoding (3 tiers) -> Generated answer
    -> Evaluated against ELI5 reference answer on readability + semantic similarity

Run:
    python experiment_a.py
"""

import os
import re
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import textstat

# ── seed ──────────────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── import config & CD ────────────────────────────────────────────────────────
# constrained_decoding.py imports from "config" — we point that name at config_eli5
sys.path.insert(0, os.path.dirname(__file__))
import config_eli5 as config          # noqa: E402  (used by constrained_decoding via alias)
sys.modules["config"] = config        # make `from config import ...` resolve to config_eli5

from constrained_decoding import ReadabilityBeamGenerator   # noqa: E402

from config_eli5 import (
    LLM_MODEL_NAME,
    NUM_SAMPLES,
    CHECKPOINT_EVERY,
    OUTPUT_DIR,
    RESULTS_CSV,
    SUMMARY_CSV,
    TIERS,
    SYSTEM_PROMPTS,
)

# ── VRAM check ────────────────────────────────────────────────────────────────
print("\n── GPU info ──────────────────────────────────────────────────────")
if torch.cuda.is_available():
    print(f"  Device : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM   : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
else:
    print("  WARNING: CUDA not available — running on CPU (will be very slow)")
print("──────────────────────────────────────────────────────────────────\n")

# ── output directory ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load dataset ──────────────────────────────────────────────────────────────
print("Loading ELI5 dataset...")
dataset = load_dataset(
    "dany0407/eli5_category",
    split="train"
)
dataset = dataset.select(range(NUM_SAMPLES))
print(f"  Loaded {len(dataset)} samples.\n")

def get_reference(sample: dict) -> str:
    """Return the first non-empty reference answer.
    eli5 stores answers as a dict with a 'text' list, or directly as a list.
    """
    answers = sample.get("answers", {})
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, list):
        texts = answers
    else:
        texts = []
    for a in texts:
        if isinstance(a, str) and a.strip():
            return a.strip()
    return ""

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading {LLM_MODEL_NAME} in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

free_after, total = torch.cuda.mem_get_info(0)
used_gb = (total - free_after) / 1e9
print(f"  Model loaded. VRAM used: {used_gb:.2f} GB\n")

# ── build generator ───────────────────────────────────────────────────────────
generator = ReadabilityBeamGenerator(model=model, tokenizer=tokenizer)

# ── metric helpers ────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[A-Za-z]+")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def safe_textstat(fn, text: str, fallback: float = 0.0) -> float:
    """Call a textstat function; return fallback if text is too short."""
    try:
        val = fn(text)
        return float(val) if val is not None else fallback
    except Exception:
        return fallback

def polysyllabic_pct(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    poly = sum(1 for w in words if textstat.syllable_count(w) >= 3)
    return round(100.0 * poly / len(words), 2)

def compute_readability(text: str) -> dict:
    return {
        "fk_grade":    safe_textstat(textstat.flesch_kincaid_grade, text),
        "smog":        safe_textstat(textstat.smog_index, text),
        "gunning_fog": safe_textstat(textstat.gunning_fog, text),
        "dale_chall":  safe_textstat(textstat.dale_chall_readability_score, text),
        "poly_pct":    polysyllabic_pct(text),
    }

def compute_rouge_l(generated: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    scores = rouge.score(reference, generated)
    return round(scores["rougeL"].fmeasure, 4)

# ── main loop ─────────────────────────────────────────────────────────────────
all_results = []

# Load checkpoint if a partial run exists
if os.path.exists(RESULTS_CSV):
    existing = pd.read_csv(RESULTS_CSV)
    all_results = existing.to_dict("records")
    done_keys = {(r["sample_id"], r["tier"]) for r in all_results}
    print(f"  Resuming from checkpoint: {len(all_results)} rows already saved.\n")
else:
    done_keys = set()

print("Starting generation loop...\n")

for tier in TIERS:
    system_prompt = SYSTEM_PROMPTS[tier]
    print(f"── Tier: {tier} {'─'*50}")

    for idx in tqdm(range(NUM_SAMPLES), desc=tier, unit="sample"):

        if (idx, tier) in done_keys:
            continue

        sample     = dataset[idx]
        question   = sample["title"].strip()
        reference  = get_reference(sample)

        # generate
        try:
            answer = generator.generate(
                system_prompt=system_prompt,
                task_prompt=question,
                user_category=tier,
            )
        except Exception as e:
            answer = f"[GENERATION ERROR: {e}]"

        # readability
        read = compute_readability(answer)

        # rouge-L
        rouge_l = compute_rouge_l(answer, reference)

        all_results.append({
            "sample_id":    idx,
            "tier":         tier,
            "question":     question,
            "generated":    answer,
            "reference":    reference,
            "fk_grade":     read["fk_grade"],
            "smog":         read["smog"],
            "gunning_fog":  read["gunning_fog"],
            "dale_chall":   read["dale_chall"],
            "poly_pct":     read["poly_pct"],
            "rouge_l":      rouge_l,
        })

        # checkpoint every N samples
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)

    # save after each tier completes
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
    print(f"  Tier {tier} complete. Results saved.\n")

# ── BERTScore (batched per tier, run once at end) ─────────────────────────────
print("Computing BERTScore (this may take a few minutes)...")
df = pd.DataFrame(all_results)

bertscore_records = []
for tier in TIERS:
    sub = df[df["tier"] == tier].copy()
    gens = sub["generated"].tolist()
    refs = sub["reference"].tolist()

    # skip rows with empty references
    valid = [(g, r) for g, r in zip(gens, refs) if r.strip()]
    if not valid:
        for _ in sub.itertuples():
            bertscore_records.append({"P": 0.0, "R": 0.0, "F1": 0.0})
        continue

    v_gens, v_refs = zip(*valid)
    P, R, F1 = bert_score(
        list(v_gens),
        list(v_refs),
        model_type="distilbert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    vi = 0
    for _, row in sub.iterrows():
        if row["reference"].strip():
            bertscore_records.append({
                "P":  round(P[vi].item(), 4),
                "R":  round(R[vi].item(), 4),
                "F1": round(F1[vi].item(), 4),
            })
            vi += 1
        else:
            bertscore_records.append({"P": 0.0, "R": 0.0, "F1": 0.0})

df["bertscore_p"]  = [r["P"]  for r in bertscore_records]
df["bertscore_r"]  = [r["R"]  for r in bertscore_records]
df["bertscore_f1"] = [r["F1"] for r in bertscore_records]

df.to_csv(RESULTS_CSV, index=False)
print(f"  Full results saved to {RESULTS_CSV}\n")

# ── summary table ─────────────────────────────────────────────────────────────
metric_cols = [
    "fk_grade", "smog", "gunning_fog", "dale_chall",
    "poly_pct", "rouge_l",
    "bertscore_p", "bertscore_r", "bertscore_f1",
]

summary_rows = []
for tier in TIERS:
    sub = df[df["tier"] == tier][metric_cols]
    row = {"tier": tier}
    for col in metric_cols:
        row[f"{col}_mean"] = round(sub[col].mean(), 4)
        row[f"{col}_std"]  = round(sub[col].std(), 4)
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_CSV, index=False)

print("── Summary ───────────────────────────────────────────────────────")
print(summary.to_string(index=False))
print(f"\nSummary saved to {SUMMARY_CSV}")
print("── Done ──────────────────────────────────────────────────────────\n")