"""
pipeline.py
===========
End-to-end pipeline: dataset -> classifier -> LIME -> LLM explanation.
Two explanation methods, selected via config.EXPLANATION_METHOD:
  - "cd": constrained decoding (unchanged, original implementation)
  - "sv": activation steering (new, parallel path)

All per-sample console output is also written to LOG_FILE_PATH.

Run:
    python pipeline.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from combined import CombinedGenerator   # NEW: sv+cd combined method
import textstat
import torch
from datasets import load_dataset
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_SPLIT,
    LLM_LOCAL_PATH,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TASK_LABELS,
    TASK_MODEL_NAME,
    EXPLANATION_METHOD,
    STEERING_ALPHA_MAP,
    LIME_JSON_PATH,
)
from constrained_decoding import ReadabilityBeamGenerator
from steering_vector import SteeringVectorGenerator, make_prompt, clean_output
from xai import LIMEExplainer

USER_LEVEL = "BEGINNER"  # "BEGINNER" | "INTERMEDIATE" | "ADVANCED"  (CD only)

if EXPLANATION_METHOD == "cd":
    OUTPUT_FILE = Path(f"results_{USER_LEVEL.lower()}.json")
elif EXPLANATION_METHOD == "sv_cd":
    OUTPUT_FILE = Path("results_svcd.json")  
else:
    OUTPUT_FILE = Path("results_finsent_sv.json")


# ---------------------------------------------------------------------------
# Logging helper — prints to console AND writes to LOG_FILE_PATH
# ---------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg)

def _load_done_sentences(path: Path) -> set[str]:
    """Returns the set of sentences already present in an existing results file."""
    if not Path(path).exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    return {rec["sentence"] for rec in existing if "sentence" in rec}

# ---------------------------------------------------------------------------
# Prompt builder (CD) — UNCHANGED
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant that explains financial sentiment predictions "
    "in plain, accurate language. Use only the evidence provided. "
    "Do not speculate beyond what the evidence shows."
)


def build_task_prompt(
    sentence: str,
    predicted_label: str,
    confidence: float,
    attributions: list[tuple[str, float]],
) -> str:
    """
    Constructs the user-turn prompt passed to the LLM.
    Attributions are formatted as a ranked evidence list.
    """
    positive_tokens = [t for t, s in attributions if s > 0]
    negative_tokens = [t for t, s in attributions if s < 0]

    evidence_lines = []
    for token, score in attributions[:6]:   # top 6 is enough context
        direction = "supports" if score > 0 else "contradicts"
        evidence_lines.append(f'  - "{token}" ({direction} the prediction, strength {abs(score):.3f})')

    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  - No strong evidence found."

    return (
        f"Sentence: \"{sentence}\"\n\n"
        f"A financial sentiment model predicted: {predicted_label.upper()} "
        f"(confidence: {confidence:.1%}).\n\n"
        f"Key words that influenced this prediction:\n{evidence_block}\n\n"
        f"Explain in 2-4 sentences why the model made this prediction, "
        f"referring to the key words above."
    )



# ---------------------------------------------------------------------------
# Result dataclass (CD) — UNCHANGED
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    idx: int
    sentence: str
    true_label: str
    predicted_label: str
    confidence: float
    attributions: list[tuple[str, float]]
    user_level: str
    explanation: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class XAIVerbalizationPipeline:

    def __init__(self):
        if EXPLANATION_METHOD == "cd":
            # ---- everything below is byte-for-byte your original CD init ----
            print("[1/4] Loading task classifier (FinBERT from HuggingFace)...")
            self.classifier = hf_pipeline(
                "text-classification",
                model=TASK_MODEL_NAME,
                top_k=None,
                device=0 if torch.cuda.is_available() else -1,
            )

            print("[2/4] Setting up LIME explainer...")
            self.lime = LIMEExplainer(
                hf_pipeline=self.classifier,
                class_names=TASK_LABELS,
            )

            print("[3/4] Running classification + LIME on all samples first...")
            self.samples = self._load_samples()
            self.precomputed = self._run_classify_and_lime()

            # Free FinBERT from GPU before loading LLM.
            del self.classifier
            self.lime.pipeline = None
            torch.cuda.empty_cache()
            print("    [GPU] FinBERT freed from VRAM.")

            print(f"[4/4] Loading LLM for verbalization (local) | user_level={USER_LEVEL}...")

            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_PATH)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_LOCAL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.llm.eval()
            print(f"    [GPU] LLM loaded on: {next(self.llm.parameters()).device}")
            self.generator = ReadabilityBeamGenerator(
                self.llm, self.tokenizer, user_level=USER_LEVEL
            )
            # ---- end of original CD init ----

        else:  # EXPLANATION_METHOD == "sv" or "sv_cd" — same precomputed-LIME + float16 phi-2 setup
            print("[1/3] Loading precomputed LIME attributions from JSON...")
            self.samples, self.precomputed = self._load_precomputed_lime()

            print("[2/3] Loading LLM (phi-2, float16) for steering-vector generation...")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_PATH)
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_LOCAL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.llm.eval()
            print(f"    [GPU] LLM loaded on: {next(self.llm.parameters()).device}")

            print("[3/3] Preparing generator...")
            if EXPLANATION_METHOD == "sv_cd":
                self.sv_generator = CombinedGenerator(self.llm, self.tokenizer)   # NEW: sv+cd combined
            else:
                self.sv_generator = SteeringVectorGenerator(self.llm, self.tokenizer)

    # -----------------------------------------------------------------
    # CD-only data loading — UNCHANGED
    # -----------------------------------------------------------------

    def _load_samples(self) -> list[dict]:
        ds = self._load_phrasebank_dataset()

        # Financial PhraseBank: fields are "sentence" and "label" (0/1/2 int).
        # Label mapping for sentences_allagree: 0=negative, 1=neutral, 2=positive
        label_map = {0: "negative", 1: "neutral", 2: "positive"}

        records = [
            {
                "sentence": row["sentence"],
                "true_label": label_map[row["label"]],
            }
            for row in ds
            if isinstance(row.get("sentence"), str) and row["sentence"].strip()
        ]

        random.seed(RANDOM_SEED)
        return random.sample(records, min(SAMPLE_SIZE, len(records)))

    def _load_phrasebank_dataset(self):
        """
        Loads Financial PhraseBank with fallbacks for datasets versions that no
        longer support script-based datasets on the Hub.
        """
        candidates = [
            (DATASET_NAME, DATASET_CONFIG),
            ("warwickai/financial_phrasebank_mirror", None),
            ("atrost/financial_phrasebank", None),
        ]

        last_error = None
        for name, config in candidates:
            try:
                if config is None:
                    ds = load_dataset(name, split=DATASET_SPLIT)
                else:
                    ds = load_dataset(name, config)[DATASET_SPLIT]

                if name != DATASET_NAME:
                    print(f"[dataset] Using fallback dataset source: '{name}'")
                return ds
            except Exception as e:  # noqa: BLE001
                last_error = e
                continue

        raise RuntimeError(
            "Failed to load Financial PhraseBank from all known sources. "
            "Please check internet access and Hugging Face availability."
        ) from last_error

    def _classify(self, sentence: str) -> tuple[str, float]:
        """Run the classifier and return (predicted_label, confidence)."""
        raw = self.classifier(sentence, top_k=None)

        if isinstance(raw, dict):
            candidates = [raw]
        elif isinstance(raw, list):
            if not raw:
                raise RuntimeError("Classifier returned an empty result list.")

            first = raw[0]
            if isinstance(first, dict):
                candidates = raw
            elif isinstance(first, list):
                if not first:
                    raise RuntimeError("Classifier returned an empty nested result list.")
                candidates = first
            else:
                raise TypeError(f"Unexpected classifier output item type: {type(first)}")
        else:
            raise TypeError(f"Unexpected classifier output type: {type(raw)}")

        best = max(candidates, key=lambda x: float(x.get("score", 0.0)))
        label = str(best.get("label", "")).lower()
        confidence = float(best.get("score", 0.0))
        return label, confidence

    def _run_classify_and_lime(self) -> list[dict]:
        """Run classification and LIME for all samples while FinBERT is on GPU."""
        precomputed = []
        for i, sample in enumerate(self.samples):
            sentence = sample["sentence"]
            print(f"  [{i+1}/{len(self.samples)}] Classifying + LIME: {sentence[:60]}...")
            pred_label, confidence = self._classify(sentence)
            attributions = self.lime.explain(sentence, pred_label)
            precomputed.append({
                "pred_label": pred_label,
                "confidence": confidence,
                "attributions": attributions,
            })
        return precomputed

    # -----------------------------------------------------------------
    # SV-only data loading (new)
    # -----------------------------------------------------------------

    def _load_precomputed_lime(self) -> tuple[list[dict], list[dict]]:
        """Loads all precomputed LIME records in the exact order they appear in the JSON."""
        with open(LIME_JSON_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        samples = []
        precomputed = []
        for rec in records:
            samples.append({
                "sentence": rec["sentence"],
                "true_label": rec["true_label"],
            })
            precomputed.append({
                "pred_label": rec["predicted_label"],
                "confidence": rec["confidence"],
                "attributions": [(a["token"], a["score"]) for a in rec["attributions"]],
            })

        return samples, precomputed

    # -----------------------------------------------------------------
    # CD run/save — logic UNCHANGED, print() swapped for log()
    # -----------------------------------------------------------------

    def run(self) -> list[SampleResult]:
        results = []

        for i, sample in enumerate(self.samples):
            sentence = sample["sentence"]
            true_label = sample["true_label"]
            pre = self.precomputed[i]

            log(f"\n--- Sample {i+1}/{len(self.samples)} ---")
            log(f"  Sentence : {sentence}")
            log(f"  Predicted: {pre['pred_label']} ({pre['confidence']:.1%}), True: {true_label}")
            log(f"  Top tokens: {[t for t, _ in pre['attributions'][:5]]}")

            task_prompt = build_task_prompt(
                sentence, pre["pred_label"], pre["confidence"], pre["attributions"]
            )

            log(f"  Generating [{USER_LEVEL}] explanation...")
            explanation = self.generator.generate(
                system_prompt=SYSTEM_PROMPT,
                task_prompt=task_prompt,
            )
            log(f"    -> {explanation[:120]}...")

            results.append(SampleResult(
                idx=i,
                sentence=sentence,
                true_label=true_label,
                predicted_label=pre["pred_label"],
                confidence=pre["confidence"],
                attributions=pre["attributions"],
                user_level=USER_LEVEL,
                explanation=explanation,
            ))

        return results

    def save(self, results: list[SampleResult]) -> None:
        serializable = []
        for r in results:
            d = asdict(r)
            d["attributions"] = [
                {"token": t, "score": float(s)} for t, s in r.attributions
            ]
            serializable.append(d)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        log(f"\nResults saved to {OUTPUT_FILE}")

    # -----------------------------------------------------------------
    # SV run/save — logic UNCHANGED, print() swapped for log()
    # -----------------------------------------------------------------

    def _append_results(self, rows: list[dict]) -> None:
        """Reads the current output file (if any), appends new rows, writes it back."""
        existing = []
        if OUTPUT_FILE.exists():
            try:
                with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.extend(rows)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def run_sv(self) -> None:
        done_sentences = _load_done_sentences(OUTPUT_FILE)
        processed_this_run = 0

        for i, sample in enumerate(self.samples):
            sentence = sample["sentence"]

            if sentence in done_sentences:
                continue

            if SAMPLE_SIZE and processed_this_run >= SAMPLE_SIZE:
                break

            true_label = sample["true_label"]
            pre = self.precomputed[i]
            pred_label = pre["pred_label"]
            attributions = pre["attributions"]

            print(f"Sample {i}")

            prompt = make_prompt(sentence, pred_label, attributions)
            sample_rows = []

            for level, alpha in STEERING_ALPHA_MAP.items():
                raw_output = self.sv_generator.generate_with_steering(prompt, alpha=alpha)
                cleaned = clean_output(raw_output)
                fre = textstat.flesch_reading_ease(cleaned)
                fkgl = textstat.flesch_kincaid_grade(cleaned)

                sample_rows.append({
                    "sample": i,
                    "sentence": sentence,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": pre["confidence"],
                    "level": level,
                    "alpha": alpha,
                    "output": cleaned,
                    "fre": fre,
                    "fkgl": fkgl,
                })

                print(f"  {level.capitalize()} done")

            self._append_results(sample_rows)
            done_sentences.add(sentence)
            processed_this_run += 1

        print(f"\nDone. {processed_this_run} new sample(s) processed.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipe = XAIVerbalizationPipeline()

    if EXPLANATION_METHOD == "cd":
        results = pipe.run()
        pipe.save(results)
        log(f"\nDone. {len(results)} samples processed.")
    else:
        pipe.run_sv()   # saving now happens incrementally inside run_sv()