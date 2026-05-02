"""Quick smoke-test: load data + save all three charts. No display needed."""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless / no GUI
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.parent   # MCC_CD_Final

FOLDERS = {
    "With CD":    BASE / "Results_old_prompt1",
    "Without CD": BASE / "Results_wo_CD",
}
AUDIENCES = ["beginner", "expert"]
METRICS   = ["flesch_reading_ease", "flesch_kincaid_grade"]
M_LABELS  = ["Flesch Reading Ease (FRE)", "Flesch-Kincaid Grade (FKGL)"]

COL_CD    = "#4C72B0"
COL_NO_CD = "#DD8452"

# ── Load ─────────────────────────────────────────────────────────────────────

def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({m: float(row[m]) for m in METRICS})
            except (ValueError, KeyError):
                pass
    return rows

data = {}
for aud in AUDIENCES:
    data[aud] = {}
    for lbl, folder in FOLDERS.items():
        rows = read_rows(folder / f"{aud}_normal_lime.csv")
        avgs = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        data[aud][lbl] = {"rows": rows, "avg": avgs, "n": len(rows)}

for aud in AUDIENCES:
    for lbl in FOLDERS:
        n = data[aud][lbl]["n"]
        a = data[aud][lbl]["avg"]
        fre  = a["flesch_reading_ease"]
        fkgl = a["flesch_kincaid_grade"]
        print(f"{aud.upper():12s} | {lbl:11s} | n={n} | FRE={fre:.2f} | FKGL={fkgl:.2f}")

OUT = BASE / "Results_old_prompt1"

# ── Chart 1: Grouped bar ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Constrained Decoding vs. Standard Generation\nAverage Readability Metrics (n=6 per condition)",
    fontsize=13, fontweight="bold", y=1.01
)
x      = np.arange(len(AUDIENCES))
width  = 0.35
labels = [a.capitalize() for a in AUDIENCES]

for i, metric in enumerate(METRICS):
    ax = axes[i]
    vcd    = [data[a]["With CD"]["avg"][metric]    for a in AUDIENCES]
    vnoc   = [data[a]["Without CD"]["avg"][metric] for a in AUDIENCES]
    b1 = ax.bar(x - width/2, vcd,  width, label="With CD",    color=COL_CD,    alpha=0.88, zorder=3)
    b2 = ax.bar(x + width/2, vnoc, width, label="Without CD", color=COL_NO_CD, alpha=0.88, zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(M_LABELS[i], fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "flesch_reading_ease":
        ax.set_ylabel("Score (higher = easier)", fontsize=10)
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel("Grade level (lower = simpler)", fontsize=10)
        ax.set_ylim(0, 22)

plt.tight_layout()
plt.savefig(OUT / "cd_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cd_comparison_bar.png")

# ── Chart 2: Per-sample FRE line ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Per-Sample Flesch Reading Ease — With CD vs. Without CD",
             fontsize=12, fontweight="bold")
for i, aud in enumerate(AUDIENCES):
    ax = axes[i]
    rc  = data[aud]["With CD"]["rows"]
    rnc = data[aud]["Without CD"]["rows"]
    n   = max(len(rc), len(rnc))
    xs  = list(range(1, n + 1))
    ax.plot(xs[:len(rc)],  [r["flesch_reading_ease"] for r in rc],  "o-",  color=COL_CD,    linewidth=2, markersize=7,
            label=f"With CD (avg={data[aud]['With CD']['avg']['flesch_reading_ease']:.1f})")
    ax.plot(xs[:len(rnc)], [r["flesch_reading_ease"] for r in rnc], "s--", color=COL_NO_CD, linewidth=2, markersize=7,
            label=f"Without CD (avg={data[aud]['Without CD']['avg']['flesch_reading_ease']:.1f})")
    ax.set_title(f"{aud.upper()} audience", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("FRE score", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "cd_comparison_fre_line.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cd_comparison_fre_line.png")

# ── Chart 3: Per-sample FKGL line ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Per-Sample Flesch-Kincaid Grade — With CD vs. Without CD",
             fontsize=12, fontweight="bold")
for i, aud in enumerate(AUDIENCES):
    ax = axes[i]
    rc  = data[aud]["With CD"]["rows"]
    rnc = data[aud]["Without CD"]["rows"]
    n   = max(len(rc), len(rnc))
    xs  = list(range(1, n + 1))
    ax.plot(xs[:len(rc)],  [r["flesch_kincaid_grade"] for r in rc],  "o-",  color=COL_CD,    linewidth=2, markersize=7,
            label=f"With CD (avg={data[aud]['With CD']['avg']['flesch_kincaid_grade']:.1f})")
    ax.plot(xs[:len(rnc)], [r["flesch_kincaid_grade"] for r in rnc], "s--", color=COL_NO_CD, linewidth=2, markersize=7,
            label=f"Without CD (avg={data[aud]['Without CD']['avg']['flesch_kincaid_grade']:.1f})")
    ax.set_title(f"{aud.upper()} audience", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Grade level", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "cd_comparison_fkgl_line.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cd_comparison_fkgl_line.png")

# ── Chart 4: Delta bar (BEGINNER only) ────────────────────────────────────────

beg_rc  = data["beginner"]["With CD"]["rows"]
beg_rnc = data["beginner"]["Without CD"]["rows"]
n = min(len(beg_rc), len(beg_rnc))
xs = list(range(1, n + 1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Per-Sample Readability Improvement from Constrained Decoding\n"
    "BEGINNER audience  |  Delta = (With CD) - (Without CD)",
    fontsize=11, fontweight="bold"
)
notes  = ["positive = improvement", "negative = improvement"]
for i, metric in enumerate(METRICS):
    ax = axes[i]
    deltas = [beg_rc[j][metric] - beg_rnc[j][metric] for j in range(n)]
    good   = "#2ecc71"
    bad    = "#e74c3c"
    if metric == "flesch_reading_ease":
        colors = [good if d > 0 else bad for d in deltas]
    else:
        colors = [good if d < 0 else bad for d in deltas]
    bars = ax.bar(xs, deltas, color=colors, alpha=0.85, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    avg_d = sum(deltas) / n
    ax.axhline(avg_d, color="navy", linewidth=1.5, linestyle="--",
               label=f"Mean delta = {avg_d:+.2f}")
    for bar, d in zip(bars, deltas):
        yoff = max(abs(d) * 0.05, 0.3) * (1 if d >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width()/2, d + yoff, f"{d:+.1f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_title(f"{M_LABELS[i]}\n({notes[i]})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel("Delta", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "cd_comparison_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cd_comparison_delta.png")

print("\nAll charts generated successfully.")
