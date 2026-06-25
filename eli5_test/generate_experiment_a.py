"""
generate_experiment_a.py
========================
Stage 1 of Experiment A: generate answers for ELI5 using the existing
constrained decoding implementation.
"""

from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from config import CONFIG
from model_loader import build_cd_generator, load_qwen_model
from utils import (
    ensure_parent_dir,
    extract_question,
    extract_reference_answer,
    load_eli5_samples,
    seed_everything,
)


def main() -> None:
    seed_everything(CONFIG["seed"])

    samples = load_eli5_samples(
        dataset_name=CONFIG["dataset_name"],
        split=CONFIG["dataset_split"],
        dataset_size=CONFIG["dataset_size"],
    )

    tokenizer, model = load_qwen_model()
    generator = build_cd_generator(model=model, tokenizer=tokenizer)

    raw_rows = []
    checkpoint_path = CONFIG["results"]["checkpoint"]
    raw_results_path = CONFIG["results"]["raw"]
    ensure_parent_dir(raw_results_path)

    for sample_index, example in enumerate(tqdm(samples, desc="Samples", total=len(samples), position=0)):
        question = extract_question(example)
        reference_answer = extract_reference_answer(example)

        tier_iter = tqdm(
            CONFIG["tiers"].items(),
            desc=f"Sample {sample_index + 1}/{len(samples)}",
            total=len(CONFIG["tiers"]),
            leave=False,
            position=1,
        )
        for tier_name, tier_config in tier_iter:
            tier_iter.set_postfix_str(tier_name)
            generated_answer = generator.generate(
                system_prompt=tier_config["system_prompt"],
                task_prompt=question,
                user_category=tier_name,
            )
            raw_rows.append(
                {
                    "sample_id": sample_index,
                    "tier": tier_name,
                    "question": question,
                    "generated_answer": generated_answer,
                    "reference_answer": reference_answer,
                }
            )

        if (sample_index + 1) % CONFIG["checkpoint_interval"] == 0:
            raw_df = pd.DataFrame(raw_rows)
            raw_df.to_csv(checkpoint_path, index=False)
            print(f"[Checkpoint] Saved {len(raw_df)} rows to {checkpoint_path}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_results_path, index=False)
    raw_df.to_csv(checkpoint_path, index=False)
    print(f"[Done] Raw generation saved to {raw_results_path}")
    print(f"[Done] Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
