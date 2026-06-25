# Experiment A: ELI5 Constrained Decoding

This folder contains a small, modular pipeline for Experiment A on ELI5:

- `generate_experiment_a.py` runs Qwen2.5 with the existing constrained decoding logic and saves raw generations.
- `evaluate_experiment_a.py` computes readability, BERTScore, ROUGE-L, and polysyllabic percentage on the saved generations.
- `run_experiment_a.py` runs generation first, then evaluation.

## Run

From the repository root:

```powershell
python .\eli5_test\run_experiment_a.py
```

You can also run the stages separately:

```powershell
python .\eli5_test\generate_experiment_a.py
python .\eli5_test\evaluate_experiment_a.py
```

## Outputs

- `experiment_a_raw_results.csv` - raw generations from all three tiers.
- `experiment_a_results.csv` - final evaluated results.
- `experiment_a_summary.csv` - per-tier mean ± std summary.
- `experiment_a_results_checkpoint.csv` - checkpointed generation output saved every 10 samples.

## Notes

- The pipeline uses CUDA and 4-bit Qwen2.5 loading.
- The existing constrained decoding implementation is reused as-is from `MCC/constrained_decoding.py`.