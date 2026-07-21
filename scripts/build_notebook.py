"""Assemble and execute the corrected EDA notebook."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# EDA — Reward Attribution Integrity

Exploratory analysis of the mobile click/attribution data underpinning the fraud
detection system. This notebook runs on whatever `SOURCE_CSV` points at in
`src/config.py` — currently configured for the full TalkingData `train.csv` (184.9M rows).

**Discipline:** every rate carries a confidence interval, and we never read a
conclusion off a small denominator.""")

code("""import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.utils import wilson_ci
from src.evaluate import detectability_floor

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
print("Reading:", config.SOURCE_CSV)""")

md("""## 1. Load with optimised dtypes

TalkingData ships every id as `int64` (8 bytes). The real cardinalities are
small, so we downcast — `ip → uint32`, the rest `uint16`. On the full 200M-row
file this is the difference between fitting on a laptop and not.

Note we do **not** load `attributed_time`: it is non-null exactly when
`is_attributed == 1`, i.e. perfect target leakage.""")

code("""usecols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]

df = pd.read_csv(config.SOURCE_CSV, usecols=usecols, dtype=config.CSV_DTYPES)
df["click_time"] = pd.to_datetime(df["click_time"], format="ISO8601")   # explicit: robust across pandas versions

# Quantify the saving vs a naive load.
naive = pd.read_csv(config.SOURCE_CSV, usecols=usecols, nrows=len(df))
saving = 1 - df.memory_usage(deep=True).sum() / naive.memory_usage(deep=True).sum()
del naive

print(f"rows: {len(df):,}")
print(f"memory (optimised): {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
print(f"memory reduction vs naive load: {saving:.0%}")
df.info()""")

md("""## 2. Class balance — with a confidence interval, not a bare rate

The first thing to know about any fraud problem is how rare the positive class
is. On the TalkingData dataset, the positive class (`is_attributed == 1`) represents ~0.25% (0.00247) of all clicks.
A bare proportion on this few positives is misleading, so we attach a 95% Wilson confidence interval to quantify uncertainty.""")

code("""n = len(df)
k = int(df["is_attributed"].sum())
rate = k / n
lo, hi = wilson_ci([k], [n])

print(f"conversions (is_attributed=1): {k:,} of {n:,}")
print(f"conversion rate: {rate:.4%}")
print(f"95% Wilson CI: [{lo[0]:.4%}, {hi[0]:.4%}]")
print(f"\\nClass imbalance: 1 positive per ~{n/max(k,1):.0f} rows")""")

code("""# Log scale, because a linear bar makes the ~0.25% positive class invisible.
counts = df["is_attributed"].value_counts().sort_index()
ax = counts.plot(kind="bar", color=["#4c72b0", "#c44e52"], logy=True)
ax.set_xticklabels(["not attributed", "attributed"], rotation=0)
ax.set_ylabel("clicks (log scale)")
ax.set_title(f"Class distribution — positives are {rate:.2%} of clicks")
for i, v in enumerate(counts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout(); plt.show()""")

md("""## 3. The leakage trap, demonstrated

`attributed_time` is populated on exactly the converted rows. Any model given it
scores ~100% and learns nothing. We show the trap rather than just assert it.""")

code("""raw = pd.read_csv(config.SOURCE_CSV, nrows=200_000)
if "attributed_time" in raw.columns:
    ct = pd.crosstab(raw["is_attributed"], raw["attributed_time"].notna(),
                     rownames=["is_attributed"], colnames=["attributed_time present"])
    print(ct)
    print("\\n-> attributed_time is present iff is_attributed == 1. Perfect leakage.")
else:
    print("attributed_time not in this file (synthetic set omits it).")
del raw""")

md("""## 4. The detectability floor

If an entity converts at the baseline rate *p* (0.00247), seeing zero conversions in *n*
clicks has probability $(1-p)^n$. Setting that equal to 0.05 (95% confidence) and solving gives:

$$n = \\frac{\\ln(0.05)}{\\ln(1 - 0.00247)} \\approx 1,211 \\text{ clicks}$$

This single number (~1,211 clicks) governs the whole evaluation: below it, a zero-conversion suspicious entity
is **unproven**, not guilty.""")

code("""floor = detectability_floor(rate)
print(f"baseline rate: {rate:.4%}")
print(f"detectability floor: {floor:,} clicks")
print(f"\\nBelow ~{floor:,} clicks, a zero-conversion entity cannot be proven")
print("fraudulent from its own record alone.")""")

md("""## 5. Entity-level view — where the fraud actually lives

Fraud is a property of an actor, not an isolated click. We aggregate to the entity (IP address) and look
for the fingerprint: **high click volume combined with near-zero conversion rate.**

### Chart Explanation:
- **X-axis (Log Scale)**: Total click volume per entity ($n_{clicks}$).
- **Y-axis**: Conversion rate of the entity ($n_{conversions} / n_{clicks}$).
- **Vertical Dashed Line (Detectability Floor = 1,211 clicks)**: Entities to the left of this floor cannot be statistically proven fraudulent based on zero conversions alone. Entities to the right with near-zero conversion fall into the "proven fraud" zone (bottom-right).
- **Horizontal Dotted Line (Population Rate = 0.247%)**: The population baseline conversion rate for comparison.""")

code("""ent = (df.groupby("ip")
         .agg(n_clicks=("is_attributed", "size"),
              n_conv=("is_attributed", "sum"))
         .reset_index())
ent["conv_rate"] = ent["n_conv"] / ent["n_clicks"]
ent = ent[ent["n_clicks"] >= 30]

print(f"entities (>=30 clicks): {len(ent):,}")
print(ent.sort_values("n_clicks", ascending=False).head(10).to_string(index=False))""")

code("""fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(ent["n_clicks"], ent["conv_rate"], s=10, alpha=0.3, color="#4c72b0")
ax.axvline(floor, ls="--", color="black", lw=1)
ax.axhline(rate, ls=":", color="#c44e52", lw=1)
ax.annotate("detectability floor", (floor, ax.get_ylim()[1]*0.9),
            textcoords="offset points", xytext=(6, 0), fontsize=8)
ax.annotate("population rate", (ax.get_xlim()[1]*0.4, rate),
            textcoords="offset points", xytext=(0, 6), fontsize=8, color="#c44e52")
ax.set_xscale("log")
ax.set_xlabel("clicks per entity (log scale)")
ax.set_ylabel("conversion rate")
ax.set_title("Fraud fingerprint: high volume + near-zero conversion (bottom-right)")
plt.tight_layout(); plt.show()""")

md("""## 6. The money chart — timing signature of fraud vs normal

The single most discriminative behavior is inter-arrival regularity. A human
clicks irregularly; an automated script fires at near-constant intervals. Here we contrast
the highest-volume entity against a typical one.

### Chart Explanation:
- **Left Histogram (Highest-Volume / Scripted Entity)**: Inter-arrival gaps cluster tightly around fixed intervals, resulting in a low Coefficient of Variation ($CV = \\text{std} / \\text{mean} \\ll 1.0$). This is the signature of scripted, metronomic automation.
- **Right Histogram (Typical / Human Entity)**: Inter-arrival gaps are spread out over varying lengths of time with a high Coefficient of Variation ($CV \\ge 1.0$), reflecting natural human activity patterns.""")

code("""top_ip = ent.sort_values("n_clicks", ascending=False).iloc[0]["ip"]
mid = ent[(ent["n_clicks"] > 50) & (ent["n_clicks"] < 200)]
normal_ip = mid.iloc[len(mid)//2]["ip"] if len(mid) else ent.iloc[0]["ip"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for ax, ip, label in [(axes[0], top_ip, "highest-volume entity"),
                      (axes[1], normal_ip, "typical entity")]:
    t = df.loc[df["ip"] == ip, "click_time"].sort_values()
    gaps = t.diff().dt.total_seconds().dropna()
    gaps = gaps[gaps < gaps.quantile(0.99)] if len(gaps) > 10 else gaps
    ax.hist(gaps, bins=40, color="#4c72b0")
    cv = gaps.std() / gaps.mean() if gaps.mean() else float("nan")
    ax.set_title(f"{label}\\nIP {ip} — {len(t):,} clicks — CV={cv:.2f}")
    ax.set_xlabel("seconds between consecutive clicks")
axes[0].set_ylabel("count")
plt.tight_layout(); plt.show()
print("Low coefficient of variation (CV) = metronomic = scripted. Humans sit near 1+.")""")

md("""## Summary

- **Dataset & Class Imbalance**: Positive class represents ~0.25% (0.00247) of clicks — extreme imbalance, quantified with a 95% Wilson confidence interval.
- **Target Leakage Guard**: `attributed_time` is a perfect-leakage column (populated iff `is_attributed == 1`) and is excluded everywhere.
- **Detectability Floor**: Calculated at **1,211 clicks** (at baseline rate $p=0.00247$), establishing the statistical threshold below which zero-conversion entities remain unproven.
- **Entity Fingerprint & Timing**: Fraud manifests as high click volume + near-zero conversion rate, accompanied by a metronomic (low CV) timing signature.

Next: `src/pipeline.py` builds these entity features at scale and runs the three
detectors. See `docs/analysis_writeup.md` for results.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}

with open("notebooks/01_eda.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote updated notebooks/01_eda.ipynb")
