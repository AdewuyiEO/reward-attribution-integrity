# Architecture & Design Tradeoffs

> *"You own the systems, not just your slice. You can speak to the full picture of the systems and flows that you work with: the architecture, the tradeoffs, and why things were built the way they were."*

This document is the answer to that. It describes what was built, what a production version would look like, and — most importantly — **why each choice was made and what it costs.**

---

## 1. The problem in one paragraph

A rewarded-advertising platform pays real value to users for genuine engagement. Fraud converts that value exchange into a leak: click farms, bot fleets, and click injection manufacture engagement that was never real, and every fraudulent reward paid is both a direct loss and a publisher-margin problem. The system's job is to decide, before a reward is honoured, whether an actor is authentic — and to do it without denying real users the rewards they earned.

Both failure directions are expensive, which is the entire design tension:

| Failure | Cost |
|---|---|
| Fraud let through (FN) | Payout leaked, publisher margin eroded, advertiser trust damaged |
| Real user blocked (FP) | User churns, support cost, partner escalation, reputational harm |

**Neither error is free, and they are not symmetric.** The cost model encodes an explicit judgement that a false positive is more expensive than a false negative.

---

## 2. Pipeline as built

```
data/train.csv (200M rows)
      │
      ▼
┌─────────────────┐   typed load, streaming, no full-file pandas read
│  DuckDB ingest  │   ids downcast int64 → uint32/uint16
└────────┬────────┘   attributed_time excluded at load (leakage barrier)
         ▼
┌─────────────────┐   window functions (LAG) for inter-arrival gaps
│  SQL feature    │   24-bin hour histogram per entity
│     layer       │   GROUP BY entity, behaviour-only columns
└────────┬────────┘
         ▼
┌────────────────────────────────────────────────┐
│  Three detectors, run independently             │
│   (a) clustering          KMeans + DBSCAN       │
│   (b) distribution        robust z on timing    │
│   (c) cross-population    KS + PSI vs baseline  │
└────────┬───────────────────────────────────────┘
         ▼
┌─────────────────┐   rank-normalise → weighted blend → reason codes
│    ensemble     │
└────────┬────────┘
         ▼
┌─────────────────┐   binomial deficit test + Benjamini-Hochberg FDR
│   evaluation    │   (labels used here ONLY, never upstream)
└────────┬────────┘
         ▼
┌─────────────────┐   threshold sweep → net value → guardrailed recommendation
│   cost model    │
└────────┬────────┘
         ▼
   Airflow DAG (daily)  →  Streamlit dashboard  +  spike/drop alerting
```

---

## 3. Key design decisions and their tradeoffs

### 3.1 Entity-level, not click-level

**Decision.** The unit of analysis is the actor (IP, or IP+device+OS), not the individual click.

**Why.** A single click carries almost no signal — it is a handful of categorical ids and a timestamp. An *actor* producing 40,000 clicks at 38-second intervals with zero installs is unmistakable. Fraud is a property of behaviour over time.

**Cost.** Latency. Entity aggregates require history, so this cannot be computed from the incoming request alone (see §4). It also means new actors have no profile — a cold-start blind spot that a determined attacker can exploit by continuously rotating identifiers.

### 3.2 Unsupervised, with labels held back

**Decision.** Detectors see behaviour only. `is_attributed` is used exclusively for evaluation.

**Why.** Three reasons, in order of importance:
1. **There is no fraud label in production.** Nobody hands you `is_fraud`. A supervised model trained on a proxy learns the proxy, not fraud.
2. **Labels arrive late.** Conversion is observed hours or days after the click; a real-time defence cannot wait.
3. **Fraud adapts.** A supervised model trained on last quarter's fraud is blind to a technique invented this week. Distributional detectors flag "unlike the population" regardless of technique.

**Cost.** Lower precision than a supervised model would achieve *on known fraud*, and no ability to learn from confirmed cases. A mature system runs both: unsupervised for novelty, supervised for known patterns.

### 3.3 Three detectors instead of one

**Decision.** Ensemble of clustering, distribution analysis, and cross-population comparison.

**Why.** Each is blind in a different place, and the blind spots do not overlap:

| Detector | Catches | Misses |
|---|---|---|
| Clustering | Coordinated fleets behaving identically | A lone sophisticated actor |
| Distribution | Machine-like timing signatures | Fraud that deliberately adds human-like jitter |
| Cross-population | Actors unlike the population, adaptively | Fraud that mimics the population profile |

Evidence from the validation run supports this: the distribution detector catches metronomic bots perfectly, while the click-farm archetype — high volume but with randomised timing — is caught primarily by the clustering and volume signals.

**Cost.** Three components mean three sets of assumptions to maintain, and weights that need periodic revisiting. Simplicity was traded for coverage.

### 3.4 Rank-normalisation before blending

**Decision.** Every detector score is converted to a [0,1] rank before the weighted sum.

**Why.** The raw outputs are incomparable — a Euclidean distance, a robust z-score, a PSI. A naive average would let whichever has the widest numeric range silently dominate, making the configured weights meaningless.

**Cost.** Rank-normalisation discards magnitude. An entity ten times more extreme than the runner-up ranks merely first. Where absolute severity matters, raw scores are retained in the output table.

### 3.5 Robust statistics throughout

**Decision.** Median/MAD instead of mean/standard deviation; RobustScaler instead of StandardScaler.

**Why.** This is the subtle one. The data is *contaminated by the very thing we are looking for*. A click farm with 200,000 clicks drags the mean and inflates the standard deviation, which compresses everyone else and **hides the fraud inside its own influence on the baseline**. Median and MAD have roughly a 50% breakdown point and stay stable under contamination.

**Cost.** Slightly less statistical efficiency on clean data. An easy trade.

### 3.6 Shrinkage for low-volume entities

**Decision.** Cross-population scores are damped by `min(n_clicks / 100, 1)`.

**Why.** An entity with 30 clicks will diverge from any baseline by chance. Without shrinkage the top of the flag list fills with sparse, unprovable entities and the analyst reviewing it loses trust in the system within a week.

**Cost.** Genuinely fraudulent low-volume actors are under-weighted — a deliberate concession, since a low-volume actor is by definition doing limited damage.

---

## 4. What production would look like

The batch pipeline here is the **modelling and calibration layer**. Deciding in real time requires splitting the work:

```
                 ┌──────────────────────────────────────┐
  ad request ───▶│  edge service (Go)                    │
                 │  entity key → KV lookup → decision    │  ~1-5 ms
                 └───────────────┬──────────────────────┘
                                 │ reads
                                 ▼
                 ┌──────────────────────────────────────┐
                 │  entity reputation store (DynamoDB)  │
                 │  {entity: score, reasons, expires}   │
                 └───────────────▲──────────────────────┘
                                 │ writes
                 ┌───────────────┴──────────────────────┐
                 │  batch scoring (this repo, Airflow)  │  hourly / daily
                 │  reads event stream → recomputes      │
                 └───────────────▲──────────────────────┘
                                 │
                 ┌───────────────┴──────────────────────┐
                 │  event stream (Kafka / SQS)          │  billions/day
                 └──────────────────────────────────────┘
```

**The central tradeoff: freshness vs latency.**

The heavy statistics — clustering, PSI, distribution fitting — cannot run inside a 200ms ad request. The resolution is to **precompute reputation offline and serve it as a key-value lookup**, which turns an expensive computation into a single-digit-millisecond read.

The price is staleness: a brand-new attacker is invisible until the next batch cycle. Three mitigations, in increasing cost:

1. **Tiered cadence** — cheap counters (clicks per entity per minute) updated in near-real-time via stream processing; expensive distributional statistics on the daily batch.
2. **Velocity rules at the edge** — hard caps that need no history and stop the crudest high-volume attacks immediately.
3. **Streaming detection** — windowed aggregations in Flink/Kafka Streams. Highest engineering cost; deploy only where the fraud loss justifies it.

**Why a KV store rather than a relational DB at the edge:** the access pattern is a single-key point lookup at enormous QPS with no joins. That is exactly the shape DynamoDB is built for, and the property that makes single-digit-millisecond reads achievable. The cost is that any analytical question ("show me all entities flagged for burst behaviour last week") is impossible against that store — hence the separate analytical layer.

---

## 5. Monitoring the monitor

A detector that silently stops working is worse than no detector, because the team stops looking. The DAG therefore tracks the flagged rate over a trailing window and alerts on **both** directions:

- **Spike** (> 1.5× baseline) — a genuine attack, or a broken feature upstream.
- **Drop** (< 0.5× baseline) — almost always a pipeline failure, not a quiet day. This is the alert most systems forget to build.

Further drift checks worth adding: PSI of the population baseline against the previous week (has "normal" itself shifted?), feature null rates, and score-distribution stability.

---

## 6. Known limitations

Stated plainly, because pretending otherwise is worse than the limitations themselves.

1. **IP is an imperfect entity key.** Carrier-grade NAT puts thousands of legitimate users behind one IP; conversely, one attacker can rotate through thousands of IPs. `ip+device+os` is a sharper key and is supported in config, but the fundamental identity problem is unsolved here. Production systems lean on device integrity signals (rooting, emulator detection, SDK attestation) precisely because network identity is weak.

2. **The ground truth is a proxy, not truth.** "Converts far below baseline" correlates with fraud but is not fraud. Some legitimate traffic converts poorly.

3. **A detectability floor exists.** At a ~0.2% conversion rate, roughly 1,500 clicks are needed before zero conversions is statistically surprising. Low-volume fraud is *unproven*, not absent.

4. **No adversarial adaptation.** A determined attacker who reads this design can defeat it by adding jitter and mimicking the diurnal curve. Real defence is an arms race requiring device-level signals the dataset does not contain.

5. **Static data.** The dataset covers a few days, so genuine longitudinal drift cannot be measured.
