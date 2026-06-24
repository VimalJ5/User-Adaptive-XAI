"""
compare_results.py
==================
Compare average Flesch Reading Ease (FRE) and Flesch-Kincaid Grade Level (FKGL)
between Results_old_prompt1 (with Constrained Decoding) and Results_wo_CD
(without Constrained Decoding), for BEGINNER and EXPERT audiences.

Run from the MCC_CD_Final directory:
    python scripts/compare_results.py
"""

import csv
import io
import sys
from pathlib import Path

# Force UTF-8 stdout so the script works on Windows regardless of locale.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent.parent  # MCC_CD_Final/

FOLDERS = {
    "With CD":    BASE / "Results_old_prompt1",
    "Without CD": BASE / "Results_wo_CD",
}

AUDIENCES = ["beginner", "expert"]
METRICS = ["flesch_reading_ease", "flesch_kincaid_grade"]
METRIC_LABELS = {
    "flesch_reading_ease":  "Flesch Reading Ease (FRE)",
    "flesch_kincaid_grade": "Flesch-Kincaid Grade (FKGL)",
}
METRIC_NOTES = {
    "flesch_reading_ease":  "higher = easier to read",
    "flesch_kincaid_grade": "lower  = simpler grade level",
}


def read_avg(csv_path: Path) -> dict:
    """Return {metric: average_value, 'n': sample_count} for a CSV file."""
    totals = {m: 0.0 for m in METRICS}
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                for m in METRICS:
                    totals[m] += float(row[m])
                n += 1
            except (ValueError, KeyError):
                continue
    if n == 0:
        return {m: float("nan") for m in METRICS} | {"n": 0}
    return {m: round(totals[m] / n, 4) for m in METRICS} | {"n": n}


def direction(metric: str, delta: float) -> str:
    """Return a short tag showing whether the delta is good or bad."""
    if metric == "flesch_reading_ease":   # higher is better
        return "[BETTER]" if delta > 0 else ("[worse]" if delta < 0 else "[same]")
    else:                                  # FKGL: lower is better
        return "[worse]" if delta > 0 else ("[BETTER]" if delta < 0 else "[same]")


def main():
    # Collect results -----------------------------------------------------------
    data = {}
    for audience in AUDIENCES:
        data[audience] = {}
        for label, folder in FOLDERS.items():
            csv_path = folder / f"{audience}_normal_lime.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing file: {csv_path}")
                data[audience][label] = {m: float("nan") for m in METRICS} | {"n": 0}
            else:
                data[audience][label] = read_avg(csv_path)

    # Pretty-print --------------------------------------------------------------
    W = 72
    SEP  = "-" * W
    SEP2 = "=" * W

    print()
    print(SEP2)
    print("  CONSTRAINED DECODING vs. STANDARD GENERATION")
    print("  Metrics: Flesch Reading Ease (FRE) | Flesch-Kincaid Grade (FKGL)")
    print(SEP2)

    for audience in AUDIENCES:
        with_cd    = data[audience]["With CD"]
        without_cd = data[audience]["Without CD"]

        print()
        print(f"  AUDIENCE : {audience.upper()}")
        print(f"  Samples  : With CD = {with_cd['n']}, Without CD = {without_cd['n']}")
        print(SEP)
        print(f"  {'Metric':<35} {'With CD':>9} {'No CD':>9} {'Delta':>9}  Result")
        print(SEP)

        for m in METRICS:
            wc  = with_cd[m]
            noc = without_cd[m]
            delta = round(wc - noc, 4) if (wc == wc and noc == noc) else float("nan")
            tag = direction(m, delta)
            label = METRIC_LABELS[m]
            note  = METRIC_NOTES[m]
            print(f"  {label:<35} {wc:>9.4f} {noc:>9.4f} {delta:>+9.4f}  {tag}")
            print(f"  {'    (' + note + ')' :<35}")

        print(SEP)

    # Cross-audience summary ----------------------------------------------------
    print()
    print(SEP2)
    print("  CROSS-AUDIENCE COMPARISON")
    print(SEP2)

    for config_label, folder_label in [("With CD", "With CD"), ("Without CD", "Without CD")]:
        print()
        print(f"  Config: {config_label}")
        print(SEP)
        print(f"  {'Metric':<35} {'Beginner':>10} {'Expert':>10} {'Delta (B-E)':>12}")
        print(SEP)
        for m in METRICS:
            beg = data["beginner"][folder_label][m]
            exp = data["expert"][folder_label][m]
            delta = round(beg - exp, 4) if (beg == beg and exp == exp) else float("nan")
            print(f"  {METRIC_LABELS[m]:<35} {beg:>10.4f} {exp:>10.4f} {delta:>+12.4f}")
        print(SEP)

    print()


if __name__ == "__main__":
    main()
