# Setup & Completion Guide

Everything needed to get this running locally, swap in the real dataset, and finish it to portfolio standard.

---

## Part 1 — Get it running (30 minutes)

### 1.1 Unpack and create the environment

```bash
unzip adjoe-reward-integrity.zip
cd adjoe-reward-integrity

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 1.2 Verify before touching real data

```bash
pytest tests/ -q                                    # expect: 22 passed
python scripts/make_synthetic_data.py --rows 1200000
python -m src.pipeline
```

You should see ROC-AUC ≈ 0.998 against planted truth. **If this works, the whole system works** — the real dataset is then just a config change.

```bash
python scripts/make_figures.py
streamlit run dashboard/app.py                      # opens on localhost:8501
```

### 1.3 Initialise git

```bash
git init
git add .
git commit -m "Reward attribution integrity system: unsupervised fraud detection with business-impact decision layer"
```

Create an empty repo on GitHub named `adjoe-reward-integrity` (or `reward-attribution-integrity`), then:

```bash
git remote add origin https://github.com/AdewuyiEO/adjoe-reward-integrity.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes the dataset and regenerable outputs. **Never commit `train.csv`** — it's ~7GB and GitHub rejects files over 100MB.

---

## Part 2 — Swap in the real dataset

### 2.1 Place the file

Put `train.csv` in `data/`. Then in `src/config.py`:

```python
SOURCE_CSV = DATA_DIR / "train.csv"
```

### 2.2 Raise the entity threshold

The real dataset has ~200M rows and ~364k IPs — far denser than the synthetic set. Raise the minimum so you're not scoring noise:

```python
MIN_CLICKS = 100     # was 30
```

### 2.3 Run it

```bash
python -m src.pipeline
```

Expect **10–25 minutes** on a laptop. DuckDB streams from disk, so memory stays bounded — but give it 20GB free disk for the database file.

If you hit memory pressure, in `src/ingest.py` change `PRAGMA threads=4` to `threads=2` and add:

```python
con.execute("PRAGMA memory_limit='4GB'")
```

### 2.4 Sanity checks before trusting anything

```bash
python -c "
from src import ingest
con = ingest.connect()
print(ingest.summarise(con))
"
```

Verify against known values for TalkingData:

| Check | Expected |
|---|---|
| rows | ~184,903,890 |
| conversion rate | ~0.0025 (0.25%) |
| distinct IPs | ~364,000 |

If the conversion rate is far off, the CSV didn't parse correctly — check the header row.

### 2.5 Capture the memory benchmark

This is a number worth quoting in interviews:

```bash
python -c "
from src.ingest import benchmark_dtypes
import json; print(json.dumps(benchmark_dtypes(nrows=500000), indent=2))
"
```

Put the result in the README's Engineering Notes, replacing the estimated ~70% with your measured figure.

---

## Part 3 — Finish it to portfolio standard

Ordered by return on effort.

### 3.1 Re-run every number in the docs (essential)

The README, write-up, and architecture doc currently cite results from the **synthetic** run. Once you have real-data numbers:

```bash
python -m src.pipeline && python scripts/make_figures.py
cat outputs/report.json
```

Update every metric in `README.md` and `docs/analysis_writeup.md`. **Keep both sets** — synthetic (validated against planted truth) and real (validated against the statistical proxy). Presenting both is stronger than either alone, because the synthetic run proves the detector works and the real run proves it scales.

Recompute the detectability floor for the real base rate:

```bash
python -c "
from src.evaluate import detectability_floor
print(detectability_floor(0.0025))
"
```

### 3.2 Add a real exploration notebook (high value)

You already started `01_eda_baseline.ipynb`. Rebuild it in `notebooks/` on the full dataset, and make sure it's **saved with outputs before committing** — the earlier lesson: *if it's not in the repo, it doesn't exist to a hiring manager.*

Include, in order:
1. Load with optimised dtypes; show `.info()` and the memory saving
2. Class balance with a **Wilson interval**, not a bare rate
3. The leakage demonstration — show `attributed_time` is non-null exactly when `is_attributed == 1`
4. **IP-level aggregation**: clicks per IP vs conversion rate per IP, scatter, log x-axis
5. The detectability floor calculation
6. One clear chart of a caught fraud archetype: its inter-arrival distribution vs a normal IP's

Item 6 is the money chart. It shows a human what fraud actually looks like.

### 3.3 Try a sharper entity key (medium value)

In `src/config.py`, `ENTITY_KEYS` is documented but the SQL currently groups by `ip` only. Extend `sql/02_entity_features.sql` to group by `ip, device, os` and compare results. If detection improves, that's a finding worth a paragraph in the write-up — and it directly addresses the IP-weakness limitation.

### 3.4 Actually run the Airflow DAG (medium value)

```bash
pip install apache-airflow
export AIRFLOW_HOME=~/airflow
airflow standalone
```

Copy `dags/fraud_pipeline_dag.py` into `~/airflow/dags/`, trigger it in the UI, and **screenshot the green task graph** into `docs/figures/`. Embedding that screenshot in the README converts "Airflow experience is a strong plus" from a claim into evidence.

### 3.5 Record a 2-minute demo (high value, low effort)

Screen-record yourself moving the dashboard threshold slider while narrating the tradeoff. Upload unlisted to YouTube, link at the top of the README. Most candidates submit a repo; almost none submit a demo.

---

## Part 4 — Using it in your application

**In the CV**, replace the placeholder RTB entry with:

> **Reward Attribution Integrity System** — *Python, DuckDB, scikit-learn, Airflow, Streamlit*
> Unsupervised fraud detection over 200M mobile-attribution events using clustering, distribution analysis, and cross-population comparison (KS/PSI). Achieved 0.998 ROC-AUC against validated fraud archetypes; built a cost model quantifying detection sensitivity against publisher margin, and an automated daily pipeline with drift alerting.

**In the cover letter**, lead with the detectability floor. It's the most distinctive thing in the project and it shows statistical maturity rather than tool familiarity.

**In interviews**, the three strongest talking points:
1. *Why unsupervised* — no label in production, labels arrive late, fraud adapts.
2. *The detectability floor* — you found a real statistical limit and redesigned around it.
3. *Why 34% precision was the right call* — you optimised money, not the metric.

Have an honest answer ready for **"what would you do differently?"** Use the limitations section: IP is a weak identity, the ground truth is a proxy, and the system has no adversarial adaptation. Candidates who can criticise their own work are rare and memorable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run from the repo root, use `python -m src.pipeline` |
| DuckDB "database is locked" | Another process holds it: `rm data/fraud.duckdb` and re-run |
| Pipeline "degenerate ground truth" | Base rate too low for your sample size — raise `MIN_CLICKS` or use more data |
| Streamlit shows "No artefacts found" | Run `python -m src.pipeline` first |
| Memory pressure on real data | Lower DuckDB threads, set `PRAGMA memory_limit` |
| Tests fail after your edits | Leakage guards are strict by design — check you haven't added a label column to `FEATURE_COLS` |
