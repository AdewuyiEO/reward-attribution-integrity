# How to Catch a Click Farm Without Punishing Real Users

*An unsupervised approach to reward-attribution integrity.*

---

## The problem is not "detect fraud"

It is easy to catch fraud. Block every IP with more than a thousand clicks and you will catch a great deal of it — along with every corporate NAT gateway, every university campus, and every mobile carrier proxy in your traffic. You will also spend the next quarter answering escalations from publishers whose legitimate users you locked out.

The real problem has two sides. Roughly a quarter of global mobile ad spend is estimated lost to fraudulent activity, so doing nothing is expensive. But a rewarded-advertising platform runs on a promise — engage, and you get paid — and a false positive breaks that promise for someone who kept their side of the bargain. **Both errors cost money, and they are not symmetric.**

So the question this project answers is not "can we detect fraud" but: *how sensitive should detection be, and how do we know?*

---

## Why unsupervised

The obvious approach is a supervised classifier. Label the fraud, train a model, ship it. The obvious approach is wrong here, for three reasons.

**There is no label.** Production data has no `is_fraud` column. It has conversion outcomes, which are a *proxy* for fraud and a lossy one.

**Labels arrive too late.** A conversion is observed hours or days after the click. A defence that has to wait for the label cannot act before the reward is paid.

**Fraud adapts.** A supervised model trained on last quarter's techniques is blind to one invented this week. That is the failure mode that matters most, because attackers are actively looking for it.

The alternative is to model what *normal* looks like and flag deviation. So the detectors here see **behaviour only** — click timing, app and channel diversity, hourly rhythm. They never see conversions. Labels are held back entirely for evaluation, which makes the evaluation an honest out-of-sample test rather than a circular one.

That constraint is enforced structurally, not by discipline: `attributed_time` is dropped at load time, and a unit test fails the build if a label column ever appears in the feature set.

*(A note on `attributed_time`: it is non-null in exactly the rows where `is_attributed = 1`. Any model given it scores near-perfect AUC and has learned nothing at all. It is the most inviting trap in this dataset.)*

---

## Three ways of being abnormal

A single detector has a single blind spot, and an adversary only needs to find it once. This system runs three, chosen because they fail in different places.

**Clustering.** Fraud is manufactured, and manufactured things are uniform. A click farm runs one script across its fleet, so its entities collapse into a dense region of behaviour space while human traffic stays diffuse. KMeans distance-to-centroid finds outliers; DBSCAN finds the tight clusters that suggest a coordinated ring.

**Distribution analysis.** Ignore volume and look at the shape of behaviour. A script firing every forty seconds produces a coefficient of variation near zero — humans are irregular, and this is the single most discriminative signal in the set, because faking it requires deliberately adding jitter. Diurnal entropy catches the rest: humans sleep, server-side bots do not.

**Cross-population comparison.** The sharpest framing: not "is this entity unusual?" but "does it behave like the population it claims to belong to?" A KS statistic and PSI measure each entity's hourly profile against the global baseline. Because the baseline is recomputed every run, the detector adapts — when genuine traffic shifts for a holiday or a new region, the reference moves with it. A hard-coded rule cannot do that, and will flood the queue with false positives the first time traffic changes.

**Do they actually add anything?** That claim is testable, so I tested it. Pairwise correlations between the three scores are **0.21, 0.10, and −0.22** — effectively independent. They are not three views of the same signal; they detect different things. The archetype results confirm it: metronomic bots and burst injection are caught perfectly by timing analysis, while high-volume click farms with randomised timing are caught primarily by clustering and volume.

---

## Robustness, or: the data is contaminated by what you are hunting

One subtle decision runs through the whole codebase. Every statistic is **robust** — median and MAD instead of mean and standard deviation, RobustScaler instead of StandardScaler.

The reason is not fastidiousness. A click farm with 200,000 clicks drags the mean upward and inflates the standard deviation. That compresses every normal entity toward the centre and **the fraud hides inside its own effect on the baseline**. Robust estimators have roughly a 50% breakdown point, so contamination cannot move them. You give up a little efficiency on clean data. The data is not clean.

---

## The detectability floor

Here is the finding I did not expect, and the one I would lead with in a conversation about this project.

Evaluation needs ground truth, and the best available proxy is "converts far below the population baseline". Formally: a one-sided binomial test per entity, with Benjamini-Hochberg correction across all of them — because running twenty thousand simultaneous tests at α = 0.01 manufactures two hundred false discoveries by construction.

When I first ran it on the small synthetic development set, the proxy returned **zero positives**. Not a bug. The probability of an entity seeing zero conversions purely by chance is `(1 − p)^n`. Solving for the point where that becomes surprising:

```
synthetic (p = 0.0017):  n = ln(0.05) / ln(1 − 0.0017) ≈ 1,760 clicks
real TalkingData (p = 0.00247):  n = ln(0.05) / ln(1 − 0.00247) ≈ 1,211 clicks
```

**Below that floor, a zero-conversion entity is not innocent — it is unproven.** There is no statistical basis to condemn it, no matter how suspicious it looks. On the small synthetic sample almost every entity fell below the floor, so the proxy found nothing; on the full 185M-row real dataset there is enough volume above the floor for the proxy to work. That single number reframes the whole evaluation, and it is why the ensemble now demotes low-volume isolated entities via an evidence gate (`fraud_priority`) rather than trusting a high anomaly score alone.

It also happens to be the same lesson as never reading a conversion rate off a small denominator — a mistake I made earlier in this project on an hour-of-day chart, where a dramatic-looking spike turned out to be three conversions with a 95% confidence interval spanning nearly the entire plausible range. The fix and the floor are the same idea applied twice.

---

## Results

Validated against planted fraud in a synthetic dataset built to the real schema (1.2M clicks, 19,203 entities, 0.17% conversion rate):

| Metric | Value |
|---|---|
| ROC-AUC | **0.998** |
| PR-AUC | **0.974** |
| Recall in top 1% of scores | **97.6%** |
| Recall — scripted bots | 100% |
| Recall — burst / click injection | 100% |
| Recall — click farms | 92.9% |

Against the statistical label proxy on the same run: ROC-AUC **0.996**, PR-AUC **0.579**. The gap between the two PR-AUCs is itself informative — the proxy only identifies entities above the detectability floor, so it recognises about half the planted fraud. **The detector is finding fraud the proxy cannot prove.** Reporting only the flattering number would hide that.

### On the real 185M-row TalkingData dataset

The synthetic numbers above validate the *mechanism* against known ground truth. On the real data there are no planted labels, only the statistical proxy, and the picture is honestly harder:

| Metric (real, proxy-evaluated) | Value |
|---|---|
| ROC-AUC | **0.735** |
| Entities blocked @ recommended threshold | **573** |
| Precision / recall @ threshold | **7% / 1%** |
| Legitimate entities affected | **0.72%** |
| Estimated net value protected | **~$792** |

Two things must be said plainly about these numbers. First, they are *lower* than the synthetic run because the proxy is a weak, noisy stand-in for truth on real data — a proxy that can only see above-floor fraud will report low recall by construction, not because the detector missed. Second, the ROC-AUC of 0.735 means the ranking is clearly better than chance but far from the near-perfect synthetic figure; on genuinely unlabelled data with a heavy-tailed entity distribution, that is a realistic result, and inflating it would be dishonest. The right read is: the mechanism is proven on synthetic ground truth, and it produces a plausible, defensible ranking on real data that a fraud analyst could triage from — not a solved problem.

---

## The decision, in money

A score is not a decision. Turning one into the other requires two assumptions, stated in the open where a stakeholder can argue with them:

- payout per install: **$2.00**
- cost of blocking a real user: **$5.00** (lost lifetime value plus trust damage)

The second figure exceeds the first deliberately. That asymmetry is the ethical core of the model: wrongly denying a real user their reward costs more than letting one fraudulent install through. Without it, the optimiser recommends scorched earth.

Sweeping every threshold and computing net value gives the operating point — subject to a hard guardrail that no more than 1% of legitimate entities may be blocked, because pure value maximisation will quietly accept ugly collateral damage when fraud volumes are large.

> **Recommended threshold (synthetic run): 0.98.** Blocks 58 entities, catching **95% of estimated fraudulent traffic at 34% precision**, while affecting **0.20% of legitimate entities** — well inside the 1% guardrail. On the real dataset the same threshold blocks **573 entities** at **0.72% legitimate impact**, protecting an estimated **~$792** of net value — still inside the guardrail, on messier data.

Note that precision of 34% is *acceptable here by design*. At this operating point the fraud caught is high-volume and expensive, while the false positives are low-volume and cheap. Optimising precision instead would have meant catching less of what actually costs money. **The right metric is the one denominated in the outcome you care about**, and precision is not it.

---

## What I would do next

1. **Sharper entity keys.** IP is weak — carrier NAT hides thousands of real users behind one address, and an attacker can rotate through thousands of addresses. Device-level integrity signals are what production systems actually lean on.
2. **A supervised layer alongside.** Once confirmed cases accumulate, run both: unsupervised for novelty, supervised for known patterns.
3. **Streaming detection** for the cheap signals, so a new attacker is not invisible until the next batch cycle.
4. **Adversarial testing.** Every technique here is defeatable by an attacker who adds jitter and mimics the diurnal curve. Assuming otherwise is how defences rot.

---

## Honest limitations

The ground truth is a proxy, not truth. The dataset covers a few days, so real drift cannot be measured. IP is an imperfect identity. And the detectability floor means low-volume fraud is invisible to the evaluation — present, but unprovable.

Every one of these is documented in `docs/architecture.md` rather than buried, because a fraud system whose limits you cannot state is a fraud system you should not trust.
