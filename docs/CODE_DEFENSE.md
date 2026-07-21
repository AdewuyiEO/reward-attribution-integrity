# Code Defense — Every Part, Explained

A module-by-module walkthrough of the whole system, written so you can defend any line of it in a technical interview. Each section explains **what the code does**, **why it was built that way**, and **the questions an interviewer will ask** — with answers.

Read this once end to end, then re-read the "interviewer asks" boxes the night before.

---

## The 30-second version (memorise this)

> "It's an unsupervised fraud-detection system for reward attribution. Three detectors — clustering, distribution analysis, and cross-population comparison — score each entity on behaviour alone, never on conversion labels. An evidence gate keyed to a statistical detectability floor stops low-volume entities masquerading as confident fraud. A cost model turns the scores into a threshold decision denominated in money, not accuracy. It runs on an Airflow pipeline with drift alerting and a Streamlit monitor."

Every claim in that paragraph is defended below.

---

## 1. `src/config.py` — the control panel

**What it does.** Single source of truth for paths, column dtypes, the entity definition, detector weights, and the two cost-model assumptions.

**Why it's built this way.** Two things a stakeholder or interviewer will challenge are the **dtype choices** and the **cost assumptions**. Both live here, in the open, rather than scattered through the code. `SOURCE_CSV` is the one line that switches between synthetic and real data — nothing else changes.

> **Interviewer asks: "Why hard-code these dtypes?"**
> Because TalkingData's ids have small, known cardinalities — `ip` maxes around 364k (fits `uint32`), the rest under 1,000 (fit `uint16`). DuckDB and pandas both default to 64-bit, which wastes 4–6 bytes per value per row. Across 200M rows that's the difference between fitting on a laptop and not. It's the cheapest optimisation in the project.

> **Interviewer asks: "Why is false-positive cost higher than payout?"**
> It's a deliberate ethical and business choice. Wrongly blocking a real user costs their lifetime value plus trust damage plus a support ticket plus a possible publisher escalation — more than the ~$2 of letting one fraudulent install through. If I made them equal, the optimiser would recommend a scorched-earth threshold. Putting the number in config means anyone can challenge my $5 estimate and see how the recommendation moves.

---

## 2. `src/ingest.py` — getting 200M rows onto a laptop

**What it does.** Streams the CSV into a typed DuckDB table, summarises it, and benchmarks the memory saving from downcasting.

**Why DuckDB and not pandas.** This is the key defensible decision. `pd.read_csv` on the full file would try to materialise ~7GB in RAM and die. DuckDB reads the CSV **out-of-core** — it streams from disk and does the `GROUP BY` aggregation without ever holding the whole file in memory. pandas only ever touches the small per-entity *result* (tens of thousands of rows), never the 200M raw ones.

**The leakage barrier.** `attributed_time` is excluded at load time — it never enters the pipeline. It's non-null in exactly the rows where `is_attributed = 1`, so any model given it scores near-perfect AUC and has learned nothing.

> **Interviewer asks: "Walk me through what happens to memory when this runs on the full file."**
> DuckDB opens a read stream on the CSV, parses in batches, casts each id column to the small type, and builds the aggregation incrementally. Peak memory is bounded by the working set of the aggregation, not the file size. The only large object that ever reaches Python is the entity table, which is small. I benchmarked the pandas dtype saving separately — about 70% — because that's the concrete number to quote, but the real scale win is never loading the raw rows into pandas at all.

> **Interviewer asks: "What breaks first at 10× the data?"**
> The DuckDB database file on disk, not memory. I'd add `PRAGMA memory_limit`, keep threads modest to bound the spill buffers, and if it still strained I'd partition by day and aggregate incrementally, since entity features are additive across time windows.

---

## 3. `sql/02_entity_features.sql` — where the signal is built

**What it does.** Aggregates clicks to the **entity** level (IP), computing volume, diversity, timing, and a 24-bin hour-of-day histogram. The `LAG()` window function computes the gap between consecutive clicks per entity.

**Why entity-level.** A single click is a few categorical ids and a timestamp — almost no signal. An *actor* producing 40,000 clicks at 38-second intervals with zero installs is unmistakable. Fraud is a property of behaviour over time, so the unit of analysis must be the actor.

**The features that matter, and why:**
- `cv_gap` (coefficient of variation of inter-arrival time) — the single strongest signal. A script firing every ~40s has CV near zero; humans are irregular (CV ≥ 1). Hard to fake without deliberately adding jitter.
- `burst_rate` / `rapid_rate` — share of sub-second / sub-10s gaps. Click injection fires hundreds of clicks in seconds.
- hour histogram (`h00`–`h23`) — feeds entropy and cross-population comparison. Humans sleep; bots don't.

**The leakage discipline.** Every column is behaviour-only. A comment and a unit test both enforce that `is_attributed` never appears here.

> **Interviewer asks: "Why compute this in SQL instead of pandas?"**
> Two reasons. Performance — the aggregation runs where the data lives, out-of-core, instead of pulling rows into Python. And correctness — window functions like `LAG()` over `PARTITION BY ip ORDER BY click_time` express "gap between an entity's consecutive clicks" directly and unambiguously. Doing that in pandas means a sort and a groupby-shift that's slower and easier to get wrong.

> **Interviewer asks: "Your entity key is IP. Isn't that weak?"**
> Yes, and I say so in the limitations. Carrier-grade NAT hides thousands of real users behind one IP; an attacker can rotate through thousands of IPs. `ip+device+os` is a sharper key and it's supported in config. But the fundamental identity problem is why production anti-fraud leans on device-integrity signals — rooting, emulator detection, SDK attestation — which this dataset doesn't contain. I'd never claim IP alone is sufficient.

---

## 4. The three detectors

The design principle: **each detector is blind in a different place, and the blind spots don't overlap.** One detector is one point of failure an adversary only has to beat once. I proved they're complementary — pairwise score correlations are 0.21, 0.10, and −0.22, effectively independent.

### 4a. `clustering.py`

**What it does.** KMeans distance-to-nearest-centroid (outliers are far from every centroid) plus DBSCAN, whose noise points are isolated oddities and whose small dense clusters are candidate fraud *rings*.

**Key preprocessing.** `log1p` on skewed volume features (else distance is just "click count"), then `RobustScaler`.

> **Interviewer asks: "Why RobustScaler, not StandardScaler?"**
> Because the data is contaminated by what I'm hunting. A 200,000-click farm inflates the mean and standard deviation, which compresses every normal entity toward the centre — the fraud hides inside its own effect on the scaling. RobustScaler uses median and IQR, which have a ~50% breakdown point, so extreme entities can't distort the scale.

> **Interviewer asks: "How did you pick the number of clusters / DBSCAN eps?"**
> They're in the function signature and I treat them as tunable, not sacred. KMeans k scales with the entity count; DBSCAN eps I set from the scaled-distance distribution. In production I'd tune eps against a small labelled validation set and monitor cluster stability over time — a sudden change in cluster structure is itself a signal.

### 4b. `distribution.py`

**What it does.** Ignores volume, scores the *shape* of behaviour: inter-arrival regularity (inverted — low variance is the anomaly), diurnal entropy (too-uniform = no sleep cycle), burstiness, and narrow targeting. Uses **robust z-scores** (median/MAD).

**The inversion is the subtle bit.** For `cv_gap`, *low* values are suspicious, so the detector negates the z-score. Regularity is the tell.

> **Interviewer asks: "Why median and MAD instead of mean and std here too?"**
> Same reason as the scaler — the fraud contaminates the baseline. If I computed the mean regularity across all entities, the bots drag it, and they end up looking closer to "normal" than they should. Median/MAD stay put under contamination.

> **Interviewer asks: "Only positive deviations count — why?"**
> I clip the z-scores at zero before combining, because I only care about entities anomalous in the *fraudulent* direction. An entity that's unusually irregular or unusually human isn't fraud; ignoring the negative tail keeps the score focused.

### 4c. `cross_population.py`

**What it does.** Asks the sharpest question — not "is this entity unusual?" but "does it behave like the population?" Uses the **KS statistic** (max CDF gap vs the population hour profile), **PSI** (the standard drift metric — <0.1 stable, 0.1–0.25 moderate, >0.25 major), and a robust volume z-score. Shrinks scores for low-volume entities.

**Why it's adaptive.** The baseline is recomputed every run, so when genuine traffic shifts (a holiday, a new region), the reference moves with it. A hard-coded rule ("flag any IP over 1,000 clicks") can't do that and floods the queue with false positives the first time traffic changes.

> **Interviewer asks: "Why PSI? Isn't that a credit-risk metric?"**
> Exactly why I used it — it's the industry-standard measure of distributional divergence, and its thresholds are ones a risk stakeholder already reads fluently. Reporting "PSI 0.4, major divergence" lands with a fraud or risk team in a way a raw KS number doesn't.

> **Interviewer asks: "What about small entities diverging by chance?"**
> That's the shrinkage term — `min(n_clicks / shrink_at, 1)`. A 30-click entity diverges from any baseline by luck, so its evidence is damped. This is the same idea as the detectability floor and the confidence intervals: never over-read a small denominator.

---

## 5. `src/detectors/ensemble.py` — combining, and the evidence gate

**What it does.** Rank-normalises each detector's score to [0,1], blends by configured weights into `fraud_score`, attaches an **evidence gate**, and generates reason codes.

**Why rank-normalise before blending.** The raw outputs are a Euclidean distance, a weighted z-score, and a PSI — incomparable scales. Averaging them raw lets whichever has the widest numeric range silently dominate. Ranking first makes the weights mean what they say.

### The evidence gate (the fix)

**The problem it solves.** `fraud_score` measures "how anomalous is this behaviour?" — and a 115-click IP can genuinely look extreme. But the detectability floor says that below ~1,760 clicks we can't prove a single entity is fraudulent from its own record. Ranking it as a top hit contradicts our own statistics.

**The distinction that resolves it:**
- **Individual evidence** (distribution, cross-population) judges one entity in isolation → subject to the floor.
- **Collective evidence** (DBSCAN ring membership) is a claim about *many* entities behaving identically → the coordination doesn't depend on any single entity's volume, so rings are **not** demoted.

**The mechanism:**
```
volume_confidence = min(n_clicks / floor, 1)     for isolated entities
                  = 1                             for ring members
fraud_priority    = fraud_score * volume_confidence
evidence_tier     ∈ {proven, ring (collective), suspected, unproven}
```
We rank and action on `fraud_priority`, not `fraud_score`. This turns the detectability floor from a caveat in the README into a visible, enforced property of the output.

> **Interviewer asks: "So you just suppress low-volume entities? Won't you miss distributed fraud that stays under the radar?"**
> That's exactly the risk, and it's why the ring exception exists. A fleet of 500 low-volume bots is suspicious *because they move together* — that's collective evidence the clustering detector captures, and those entities keep full priority regardless of individual volume. What I demote is the *isolated* low-volume anomaly, where the only evidence is one entity's own record, which the statistics say I can't trust. So it's not "ignore small entities," it's "require the right kind of evidence for the claim I'm making."

> **Interviewer asks: "Isn't using the base rate to set the floor a form of leakage?"**
> No. The floor is derived from the *global* conversion rate — a single population constant — not from any individual entity's label. I use it to set a volume threshold, which is a property of the data distribution, not of any entity's outcome. The per-entity labels never touch the detectors. I documented that distinction explicitly.

**Reason codes.** Every flag carries plain-language evidence ("metronomic click timing"; "LOW EVIDENCE: below detectability floor, treat as watchlist"). A bare score of 0.94 is unactionable and undisputable; reason codes make both an analyst decision and a partner appeal possible.

---

## 6. `src/evaluate.py` — honest scoring without a fraud label

**What it does.** Builds a ground-truth *proxy* (entities converting far below baseline, via a one-sided binomial test with Benjamini-Hochberg FDR correction), computes the detectability floor, and reports ROC-AUC / PR-AUC / precision-at-k.

**Why a proxy, stated as a proxy.** TalkingData has no `is_fraud` column, only `is_attributed`. "Converts far below baseline" correlates with fraud but isn't fraud, and I say so. Overclaiming here is exactly what a technical interviewer probes.

**Why a binomial test, not "conversion rate == 0".** An IP with 40 clicks and zero installs is unremarkable at a 0.2% base rate (you'd expect 0.08 conversions). An IP with 40,000 clicks and zero installs is essentially impossible by chance. The binomial test encodes that distinction; a naive rate threshold doesn't.

**Why Benjamini-Hochberg.** One test per entity means thousands of simultaneous tests. At raw α=0.01 across 20,000 entities you'd expect ~200 false discoveries by chance. BH controls the false *discovery* rate — the difference between "I ran a test" and "I ran a defensible screen."

**The detectability floor.** `n = ln(α) / ln(1 − p)`. At a 0.17% base rate, ~1,760 clicks before zero conversions is statistically surprising. Below it, an entity is *unproven*, not innocent.

> **Interviewer asks: "Your PR-AUC is 0.97 on synthetic but 0.58 on the proxy. Isn't that bad?"**
> It's the most honest number in the project. The proxy can only recognise fraud *above* the detectability floor, so it identifies about half the planted fraud. The gap between the two isn't the detector failing — it's the detector finding fraud the proxy can't prove. Reporting only the 0.97 would hide that, so I report both and explain the gap.

---

## 7. `src/cost_model.py` — turning scores into a decision

**What it does.** Sweeps every threshold; at each, computes fraud payouts prevented, false-positive cost, and net value; recommends the argmax **subject to a guardrail** that ≤1% of legitimate entities may be blocked.

**Why net value, not fraud caught.** A model measured on fraud caught "wins" by blocking everyone. The right metric is denominated in the outcome you care about — money — with the two assumptions exposed in config.

**Why the guardrail.** Pure net-value maximisation quietly accepts a high false-positive rate when fraud volumes are large. Capping legitimate-user impact keeps the recommendation defensible to a publisher and encodes "real users are not collateral damage." If no threshold satisfies the cap, the code says so rather than silently relaxing it.

> **Interviewer asks: "Your recommended precision is only 34%. That's terrible."**
> It's correct here, by design. At that operating point the fraud caught is high-volume and expensive, while the false positives are low-volume and cheap — the cost model weighs dollars, not counts. Optimising precision would mean catching less of what actually costs money. The stakeholder statement reports it in money, and the guardrail guarantees I'm affecting under 1% of real users while doing it.

---

## 8. `dags/fraud_pipeline_dag.py` — automation and monitoring

**What it does.** Runs ingest → features → score → evaluate → cost → spike-detection → dashboard daily, with retries. Logic lives in `src/`; the DAG is a thin wrapper, so everything stays testable outside Airflow. State passes through parquet on disk, not XCom (XCom is for small metadata, not a scored table).

**The alert that matters most.** `detect_fraud_spike` alerts on both a spike (>1.5× baseline) *and* an implausible drop (<0.5× baseline). A silent drop to zero flags is almost always a broken pipeline, not a quiet day — and it's the alert most systems forget to build.

> **Interviewer asks: "Why not put the logic in the DAG?"**
> Because then it's only runnable inside Airflow and can't be unit-tested. Thin tasks wrapping importable functions means I test the fraud logic directly and the DAG just orchestrates. It also makes local development trivial — I run `python -m src.pipeline` without an Airflow instance.

---

## 9. `dashboard/app.py` — data visibility

**What it does.** A Streamlit monitor whose threshold slider moves every number together — fraud caught, legitimate impact, net value — plus a flagged-entity table with reason codes and evidence tiers, and a tier filter.

**Why it exists.** The JD asks for "data visibility solutions with cross-team value" and findings that "allow for easy decisions." The slider turns an abstract precision/recall tradeoff into a ten-second business conversation. It flags on `fraud_priority`, not raw score, so the evidence gate is visible in the product.

---

## 10. `tests/` — why the leakage guards are the important ones

22 tests. Most check statistical correctness (Wilson intervals, PSI, KS, BH-FDR) and detector behaviour. **Three are leakage guards** that fail the build if any label column ever reaches the feature set — because a leaked label silently invalidates every number in the project, and that's the one error you can't recover from after the fact.

> **Interviewer asks: "What would you test that you haven't?"**
> Property-based tests on the detectors (e.g. scores must be monotonic in anomaly strength), a golden-file test on the full pipeline output, and a data-contract test that fails if the input schema drifts. And I'd add drift tests on the population baseline itself.

---

## The one-liner for each anticipated hard question

- **"This is a stretch for your experience."** → "The project does the job of the role: unsupervised anomaly detection, a business-impact decision layer, and an automated pipeline. I'd rather show the work than claim the years."
- **"What would you do differently?"** → "Sharper entity key using device signals; a supervised layer once confirmed cases accumulate; streaming detection for the cheap signals; and adversarial testing, because everything here is beatable by an attacker who adds jitter."
- **"What's the weakest part?"** → "The ground truth is a proxy and IP is a weak identity. Both are in the limitations section, because a fraud system whose limits you can't state is one you shouldn't trust."
