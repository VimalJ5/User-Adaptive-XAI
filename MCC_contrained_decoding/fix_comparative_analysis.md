# Agent Instructions — Fix Comparative Analysis Pipeline

## Context

The constrained decoding pipeline is functional but producing near-zero effect
sizes due to miscalibrated parameters. The notebook
`05_comparative_analysis.ipynb` shows:

- BEGINNER FRE delta: **+1.7** (target bracket: 55–85, current mean: 21.7)
- EXPERT FRE delta: **+6.9** (right direction, still far from targets)
- BEGINNER / EXPERT gap: only **7 FRE points** (should be 40+)
- Dale-Chall missing from analysis entirely

Four fixes are required. Do them in order — each one builds on the previous.

---

## Fix 1 — Tighten Normalisation Caps in `constrained_decoding.py`

**Why:** The current caps (MAX_LENGTH_CAP=60, RARE_CAP=8) are calibrated for
general English. Biomedical sentences are information-dense — 30 words with 4
rare terms is already hard for a beginner. With loose caps, `C_hard` rarely
exceeds 0.3, so the penalty has almost no force.

**What to change:** Find the constants at the top of `constrained_decoding.py`
and replace:

```python
# BEFORE
MAX_LENGTH_CAP   = 60
RARE_CAP         = 8
CLAUSE_CAP       = 4
MAX_WORD_LEN_CAP = 8.0

# AFTER
MAX_LENGTH_CAP   = 30
RARE_CAP         = 4
CLAUSE_CAP       = 3
MAX_WORD_LEN_CAP = 7.0
```

No other changes to this file.

---

## Fix 2 — Increase λ Values in `config.py`

**Why:** At λ=0.30 for BEGINNER, the penalty only moved FRE by +1.7 points.
The base model has a very strong biomedical prior — λ must be large enough to
overcome it. The gap between BEGINNER and EXPERT λ must also be large enough
to produce visibly different text.

**What to change:** Find `LAMBDA_MAP` in `config.py` and replace:

```python
# BEFORE
LAMBDA_MAP = {
    "BEGINNER":     0.30,
    "INTERMEDIATE": 0.15,
    "EXPERT":       0.05,
}

# AFTER
LAMBDA_MAP = {
    "BEGINNER":     1.5,
    "INTERMEDIATE": 0.5,
    "EXPERT":       0.05,
}
```

Also add a sweep list directly below it for Fix 3 to use:

```python
LAMBDA_SWEEP_VALUES = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
```

---

## Fix 3 — Add λ Sweep Cell to `05_comparative_analysis.ipynb`

**Why:** The correct λ for your specific model is unknown until empirically
measured. A sweep on 10 samples takes minutes and tells you exactly where FRE
crosses into each target bracket. This cell must run before the main analysis
so the operating λ can be confirmed.

**What to do:** Insert a new cell immediately after the imports cell (after
cell 2, before the "Load the four result files" markdown cell). The cell must:

1. Load the model and tokenizer using `model_loaders.load_classifier()` — or
   whichever load function your pipeline uses for the generation model
2. Initialise `ReadabilityBeamGenerator` from `constrained_decoding.py`
3. Take the first 10 rows from `results_beginner.csv` as test prompts
4. For each λ in `config.LAMBDA_SWEEP_VALUES`, generate outputs and compute
   FRE and Dale-Chall using `textstat`
5. Print a table: λ | mean FRE | mean Dale-Chall | % rows with FRE ≥ 55
6. Plot FRE vs λ as a line chart with horizontal bands showing the BEGINNER
   target bracket (55–85) and EXPERT target bracket (−10–35)
7. Print the recommended operating λ — the lowest λ where mean FRE ≥ 55

The cell should be clearly labelled:

```python
# ── λ SWEEP — run this first to confirm operating parameters ──────────────
```

Do not regenerate the full 200-row dataset during this cell. 10 rows is
sufficient to calibrate.

---

## Fix 4 — Add Dale-Chall and Bracket Conformance to `05_comparative_analysis.ipynb`

**Why:** The target brackets are defined on FRE **and** Dale-Chall together.
The current analysis only uses FRE, FKGL, and SMOG. Dale-Chall is more
sensitive to vocabulary difficulty in biomedical text than FRE alone.
Bracket conformance (% of rows inside the target) is the headline metric
your professor expects.

**What to change:** Four edits inside the existing notebook cells:

### 4a — Add dale-chall to metric_cols (cell 4, the load cell)

```python
# BEFORE
metric_cols = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "lime_coverage",
    "ontology_hit_rate",
]

# AFTER
metric_cols = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "dale_chall",
    "smog_index",
    "lime_coverage",
    "ontology_hit_rate",
]
```

Also add `"dale_chall"` to `expected_columns` in the same cell.

### 4b — Add bracket conformance columns immediately after `pd.concat`

Add these lines right after `df = pd.concat(frames, ignore_index=True)`:

```python
# Bracket conformance flags — these are the headline evaluation metrics
BEGINNER_BRACKET_FRE = (55, 85)
EXPERT_BRACKET_FRE   = (-10, 35)
BEGINNER_BRACKET_DC  = (0.0, 7.5)
EXPERT_BRACKET_DC    = (8.5, 20.0)

def in_bracket(row):
    cat = row.get("user_category", "")
    fre = row.get("flesch_reading_ease", float("nan"))
    dc  = row.get("dale_chall", float("nan"))
    if cat == "BEGINNER":
        return (
            BEGINNER_BRACKET_FRE[0] <= fre <= BEGINNER_BRACKET_FRE[1]
            and BEGINNER_BRACKET_DC[0] <= dc <= BEGINNER_BRACKET_DC[1]
        )
    elif cat == "EXPERT":
        return (
            EXPERT_BRACKET_FRE[0] <= fre <= EXPERT_BRACKET_FRE[1]
            and EXPERT_BRACKET_DC[0] <= dc <= EXPERT_BRACKET_DC[1]
        )
    return False

df["in_target_bracket"] = df.apply(in_bracket, axis=1)
```

### 4c — Add bracket conformance to the aggregate summary (cell 6)

```python
# BEFORE
summary_by_file = (
    df.groupby("experiment")[metric_cols + ["explanation_word_count"]]
    .mean()
    .round(3)
)

# AFTER
summary_by_file = (
    df.groupby("experiment")[metric_cols + ["explanation_word_count", "in_target_bracket"]]
    .mean()
    .round(3)
)
summary_by_file = summary_by_file.rename(columns={"in_target_bracket": "pct_in_bracket"})
summary_by_file["pct_in_bracket"] = (summary_by_file["pct_in_bracket"] * 100).round(1)
```

### 4d — Add a bracket conformance bar chart as the final plot

Add a new cell at the end of the notebook (before the save cell):

```python
# ── Bracket conformance — the headline result ─────────────────────────────
bracket_summary = (
    df.groupby(["user_category", "generation_mode"])["in_target_bracket"]
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
    .rename(columns={"in_target_bracket": "pct_in_bracket"})
)

fig, ax = plt.subplots(figsize=(8, 5))
colors = {
    ("BEGINNER", "normal"):               "#aec6e8",
    ("BEGINNER", "constrained_decoding"): "#1f77b4",
    ("EXPERT",   "normal"):               "#f5c08a",
    ("EXPERT",   "constrained_decoding"): "#dd8452",
}
x_positions = {"BEGINNER": 0, "EXPERT": 1}
bar_width = 0.35

for _, row in bracket_summary.iterrows():
    cat, mode, pct = row["user_category"], row["generation_mode"], row["pct_in_bracket"]
    offset = -bar_width / 2 if mode == "normal" else bar_width / 2
    x = x_positions[cat] + offset
    color = colors.get((cat, mode), "#888888")
    bar = ax.bar(x, pct, width=bar_width, color=color,
                 label=f"{cat} / {mode.replace('_', ' ')}")
    ax.text(x, pct + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()))
ax.set_ylabel("% Explanations Within Target Readability Bracket")
ax.set_title("Bracket Conformance: Normal vs Constrained Decoding")
ax.set_ylim(0, 110)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/bracket_conformance.png", dpi=150)
plt.show()
print("[✓] Saved outputs/bracket_conformance.png")
```

---

## Running Instructions

```
1.  Edit constrained_decoding.py
    → Apply Fix 1 (tighten caps)

2.  Edit config.py
    → Apply Fix 2 (new LAMBDA_MAP + LAMBDA_SWEEP_VALUES)

3.  Open 05_comparative_analysis.ipynb

4.  Insert Fix 3 sweep cell after the imports cell
    → Run that cell alone first
    → Read the printed table — confirm FRE ≥ 55 is reached at some λ
    → If FRE never reaches 55 even at λ=3.0, increase RARE_CAP tighter
      (try MAX_LENGTH_CAP=20, RARE_CAP=3) and re-run

5.  Apply Fix 4 edits across the four locations in the notebook

6.  Regenerate the four CSV files (results_beginner.csv, results_beginner_CD.csv,
    results_expert.csv, results_expert_CD.csv) using the updated λ values
    → Run whichever notebook/script produces these files in your pipeline
    → This is the step that takes the most time

7.  Run 05_comparative_analysis.ipynb top to bottom (Kernel → Restart & Run All)

8.  Check the output of the delta summary table
    Target: BEGINNER FRE delta ≥ 30, EXPERT FRE delta ≤ 5
    Target: bracket_conformance.png shows visible difference between
            normal and constrained bars for BEGINNER

9.  If BEGINNER bracket conformance is still < 20%:
    → Increase LAMBDA_MAP["BEGINNER"] to 2.5 and repeat from step 6
    → Each increment of +0.5 to λ should move mean FRE by ~8–12 points
      based on the sweep results from step 4
```
