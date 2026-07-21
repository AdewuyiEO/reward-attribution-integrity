import json
from pathlib import Path

nb_path = Path("notebooks/01_eda.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        
        # 1. Title / Header
        if "Exploratory analysis of the mobile click/attribution data" in src:
            cell["source"] = [
                "# EDA — Reward Attribution Integrity\n",
                "\n",
                "Exploratory analysis of the mobile click/attribution data underpinning the fraud\n",
                "detection system. This notebook runs on whatever `SOURCE_CSV` points at in\n",
                "`src/config.py` — currently configured for the full TalkingData `train.csv` (184.9M rows).\n",
                "\n",
                "**Discipline:** every rate carries a confidence interval, and we never read a\n",
                "conclusion off a small denominator."
            ]
            
        # 2. Class Balance
        elif "Class balance — with a confidence interval" in src:
            cell["source"] = [
                "## 2. Class balance — with a confidence interval, not a bare rate\n",
                "\n",
                "The first thing to know about any fraud problem is how rare the positive class\n",
                "is. On the TalkingData dataset, the positive class (`is_attributed == 1`) represents ~0.25% (0.00247) of all clicks.\n",
                "A bare proportion on this few positives is misleading, so we attach a 95% Wilson confidence interval to quantify uncertainty."
            ]
            
        # 3. Detectability floor
        elif "The detectability floor" in src:
            cell["source"] = [
                "## 4. The detectability floor\n",
                "\n",
                "If an entity converts at the baseline rate *p* (0.00247), seeing zero conversions in *n*\n",
                "clicks has probability $(1-p)^n$. Setting that equal to 0.05 (95% confidence) and solving gives:\n",
                "\n",
                "$$n = \\frac{\\ln(0.05)}{\\ln(1 - 0.00247)} \\approx 1,211 \\text{ clicks}$$\n",
                "\n",
                "This single number (~1,211 clicks) governs the whole evaluation: below it, a zero-conversion suspicious entity\n",
                "is **unproven**, not guilty."
            ]
            
        # 4. Entity-level view & Chart explanation
        elif "Entity-level view" in src:
            cell["source"] = [
                "## 5. Entity-level view — where the fraud actually lives\n",
                "\n",
                "Fraud is a property of an actor, not an isolated click. We aggregate to the entity (IP address) and look\n",
                "for the fingerprint: **high click volume combined with near-zero conversion rate.**\n",
                "\n",
                "### Chart Explanation:\n",
                "- **X-axis (Log Scale)**: Total click volume per entity ($n_{clicks}$).\n",
                "- **Y-axis**: Conversion rate of the entity ($n_{conversions} / n_{clicks}$).\n",
                "- **Vertical Dashed Line (Detectability Floor = 1,211 clicks)**: Entities to the left of this floor cannot be statistically proven fraudulent based on zero conversions alone. Entities to the right with near-zero conversion fall into the \"proven fraud\" zone (bottom-right).\n",
                "- **Horizontal Dotted Line (Population Rate = 0.247%)**: The population baseline conversion rate for comparison."
            ]

        # 5. Timing signature & Chart explanation
        elif "timing signature of fraud vs normal" in src:
            cell["source"] = [
                "## 6. The money chart — timing signature of fraud vs normal\n",
                "\n",
                "The single most discriminative behavior is inter-arrival regularity. A human\n",
                "clicks irregularly; an automated script fires at near-constant intervals. Here we contrast\n",
                "the highest-volume entity against a typical one.\n",
                "\n",
                "### Chart Explanation:\n",
                "- **Left Histogram (Highest-Volume / Scripted Entity)**: Inter-arrival gaps cluster tightly around fixed intervals, resulting in a low Coefficient of Variation ($CV = \\text{std} / \\text{mean} \\ll 1.0$). This is the signature of scripted, metronomic automation.\n",
                "- **Right Histogram (Typical / Human Entity)**: Inter-arrival gaps are spread out over varying lengths of time with a high Coefficient of Variation ($CV \\ge 1.0$), reflecting natural human activity patterns."
            ]

        # 6. Summary
        elif "## Summary" in src:
            cell["source"] = [
                "## Summary\n",
                "\n",
                "- **Dataset & Class Imbalance**: Positive class represents ~0.25% (0.00247) of clicks — extreme imbalance, quantified with a 95% Wilson confidence interval.\n",
                "- **Target Leakage Guard**: `attributed_time` is a perfect-leakage column (populated iff `is_attributed == 1`) and is excluded everywhere.\n",
                "- **Detectability Floor**: Calculated at **1,211 clicks** (at baseline rate $p=0.00247$), establishing the statistical threshold below which zero-conversion entities remain unproven.\n",
                "- **Entity Fingerprint & Timing**: Fraud manifests as high click volume + near-zero conversion rate, accompanied by a metronomic (low CV) timing signature.\n",
                "\n",
                "Next: `src/pipeline.py` builds these entity features at scale and runs the three\n",
                "detectors. See `docs/analysis_writeup.md` for results."
            ]

    elif cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "# Log scale, because a linear bar makes the 0.2% positive class invisible." in src:
            cell["source"] = [
                line.replace(
                    "# Log scale, because a linear bar makes the 0.2% positive class invisible.",
                    "# Log scale, because a linear bar makes the ~0.25% positive class invisible."
                ) for line in cell["source"]
            ]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebooks/01_eda.ipynb markdown and chart explanations!")
