"""
evaluate_experiment_a.py
========================
Stage 2 of Experiment A: compute readability, semantic similarity,
and constraint-effectiveness metrics on the generated answers.
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd
import torch
import textstat
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from config import CONFIG
from utils import (
    count_polysyllabic_percentage,
    ensure_parent_dir,
    safe_textstat_score,
)


METRIC_COLUMNS = [
    "fk_grade",
    "smog",
    "gunning_fog",
    "dale_chall",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "rouge_l",
    "polysyllabic_pct",
]


def compute_text_readability_metrics(text: str) -> dict[str, float]:
    return {
        "fk_grade": safe_textstat_score(textstat.flesch_kincaid_grade, text),
        "smog": safe_textstat_score(textstat.smog_index, text),
        "gunning_fog": safe_textstat_score(textstat.gunning_fog, text),
        "dale_chall": safe_textstat_score(textstat.dale_chall_readability_score, text),
    }


def compute_rouge_l(predictions: Iterable[str], references: Iterable[str]) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values = []
    for prediction, reference in zip(predictions, references):
        score = scorer.score(reference or "", prediction or "")["rougeL"].fmeasure
        values.append(float(score))
    return values


def format_mean_std(mean: float, std: float) -> str:
    if math.isnan(std):
        std = 0.0
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    raw_results_path = CONFIG["results"]["raw"]
    final_results_path = CONFIG["results"]["final"]
    summary_path = CONFIG["results"]["summary"]

    if not raw_results_path.exists():
        raise FileNotFoundError(
            f"Missing raw generation output: {raw_results_path}. Run generate_experiment_a.py first."
        )

    df = pd.read_csv(raw_results_path)
    expected_rows = CONFIG["dataset_size"] * len(CONFIG["tiers"])
    if len(df) != expected_rows:
        print(f"[Warn] Expected {expected_rows} rows, found {len(df)} rows in {raw_results_path}")

    enriched_frames = []
    summary_rows = []

    for tier_name in CONFIG["tiers"].keys():
        tier_df = df[df["tier"] == tier_name].copy().reset_index(drop=True)
        if tier_df.empty:
            continue

        print(f"[Eval] Computing metrics for {tier_name} ({len(tier_df)} samples)")

        readability_cache = tier_df["generated_answer"].apply(lambda text: compute_text_readability_metrics(str(text)))
        tier_df["fk_grade"] = readability_cache.apply(lambda item: item["fk_grade"])
        tier_df["smog"] = readability_cache.apply(lambda item: item["smog"])
        tier_df["gunning_fog"] = readability_cache.apply(lambda item: item["gunning_fog"])
        tier_df["dale_chall"] = readability_cache.apply(lambda item: item["dale_chall"])
        tier_df["polysyllabic_pct"] = tier_df["generated_answer"].apply(
            lambda text: count_polysyllabic_percentage(str(text))
        )

        rouge_values = compute_rouge_l(
            tier_df["generated_answer"].fillna("").tolist(),
            tier_df["reference_answer"].fillna("").tolist(),
        )
        tier_df["rouge_l"] = rouge_values

        bert_p, bert_r, bert_f1 = bert_score(
            tier_df["generated_answer"].fillna("").tolist(),
            tier_df["reference_answer"].fillna("").tolist(),
            model_type=CONFIG["bert_score_model"],
            lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=CONFIG["bert_score_batch_size"],
            verbose=True,
            rescale_with_baseline=False,
        )

        tier_df["bertscore_p"] = bert_p.detach().cpu().tolist()
        tier_df["bertscore_r"] = bert_r.detach().cpu().tolist()
        tier_df["bertscore_f1"] = bert_f1.detach().cpu().tolist()

        enriched_frames.append(tier_df)

        tier_summary = {"tier": tier_name}
        for column in METRIC_COLUMNS:
            mean_value = float(tier_df[column].mean())
            std_value = float(tier_df[column].std())
            tier_summary[column] = format_mean_std(mean_value, std_value)
            tier_summary[f"{column}_mean"] = mean_value
            tier_summary[f"{column}_std"] = std_value
        summary_rows.append(tier_summary)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_df = pd.concat(enriched_frames, ignore_index=True)
    final_df = final_df[
        [
            "sample_id",
            "tier",
            "question",
            "generated_answer",
            "reference_answer",
            *METRIC_COLUMNS,
        ]
    ]

    ensure_parent_dir(final_results_path)
    final_df.to_csv(final_results_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_display_df = summary_df[["tier", *METRIC_COLUMNS]].copy()
    summary_display_df.to_csv(summary_path, index=False)

    console_rows = []
    for _, row in summary_df.iterrows():
        pretty_row = {"tier": row["tier"]}
        for column in METRIC_COLUMNS:
            pretty_row[column] = row[column]
        console_rows.append(pretty_row)

    print("\n[Summary] Mean ± std by tier")
    print(pd.DataFrame(console_rows).to_string(index=False))
    print(f"\n[Done] Final results saved to {final_results_path}")
    print(f"[Done] Summary saved to {summary_path}")
    print("[Note] Average token hardness is not reported because the existing CD implementation does not expose per-step h(t).")


if __name__ == "__main__":
    main()
