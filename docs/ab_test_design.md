# Experiment Design: Rolling Out a New Fraud Threshold

> *"Statistically grounded. You can reason confidently about experimentation, setting up tests, interpreting results, and applying core statistical concepts."*

The cost model recommends an operating threshold from historical data. That is a *hypothesis*, not a decision. This document specifies how to test it in production without breaking things.

---

## 1. Hypothesis

**H₀:** Enabling the new detector at the recommended threshold produces no change in net platform value versus the current rule-based defence.

**H₁:** It increases net value — fraudulent payouts prevented exceed the cost of additional false positives.

Note the metric is **net value, not fraud caught**. A test measuring only fraud blocked would "succeed" by blocking everything.

---

## 2. Randomisation unit

**Unit: the entity (IP, or IP+device+OS) — not the click, not the user session.**

Why this matters more than it looks:

- **Clicks are not independent.** Clicks from one actor are heavily correlated. Randomising by click splits a single fraudster across both arms, contaminating control with treatment and biasing the effect toward zero.
- **The intervention acts on entities.** We block actors, so the unit of randomisation must match the unit of treatment.
- **Consistency of experience.** A user must not be blocked on one click and allowed on the next.

Assignment: `hash(entity_id + salt) % 100`, deterministic and stable across days. A fixed salt per experiment prevents carryover from previous tests.

**Caveat to state openly:** entities are not perfectly independent either. A single fraud operator controlling 500 IPs lands partly in each arm, and the network structure violates SUTVA. With many independent operators this averages out; with few large ones, variance is understated. Mitigation: cluster-robust standard errors on operator-like groupings where they can be inferred.

---

## 3. Metrics

### Primary
**Net value per entity** = payouts prevented − false-positive cost, using the same assumptions as the cost model, so the experiment tests the *same* quantity that was optimised.

### Secondary
- Fraudulent payout rate (payouts to entities the proxy flags)
- Detection precision and recall against the label proxy
- Total payout volume

### Guardrails — any breach stops the rollout
| Guardrail | Threshold | Rationale |
|---|---|---|
| Legitimate entities blocked | ≤ 1% | The fairness constraint from the cost model |
| D7 retention of affected real users | no significant drop | Detects churn the cost model only estimates |
| Publisher-level block rate | no publisher > 3× median | Catches a rule that quietly destroys one partner |
| Support ticket rate | ≤ 1.2× baseline | Real-world proxy for user harm |
| p99 decision latency | ≤ current | The lookup must not slow the ad path |

The publisher-level guardrail exists because **aggregate metrics hide concentrated harm**. A change that is neutral platform-wide can be catastrophic for one partner, and that is the call you receive.

---

## 4. Power analysis

Detecting a relative change δ in a metric with baseline mean μ and standard deviation σ, at 80% power and α = 0.05 (two-sided):

```
n per arm ≈ 16 · σ² / (δ · μ)²
```

Worked example for the primary metric, using observed dispersion:

| Detectable effect | Entities per arm |
|---|---|
| 20% | ~6,300 |
| 10% | ~25,000 |
| 5% | ~100,000 |

**Run length** = (2 × n) / (new entities per day). Fill these in from your own traffic before starting — computing power *after* a null result is how teams talk themselves into false conclusions.

Net value is heavy-tailed (a few huge entities dominate), so σ is large and the naive formula understates the requirement. Two mitigations: **winsorise** the metric at the 99th percentile before testing, and report **bootstrap confidence intervals** rather than relying on normal-theory standard errors.

---

## 5. Rollout plan

Staged, with a hard gate at each step:

| Stage | Traffic | Duration | Gate to proceed |
|---|---|---|---|
| 0. Shadow | 100% scored, 0% blocked | 1 week | Flag rate and reason codes look sane on manual review |
| 1. Canary | 5% | 1 week | No guardrail breach |
| 2. Ramp | 50% | 2 weeks | Primary metric positive, no guardrail breach |
| 3. Full | 100% | — | Sustained effect; holdback retained |

**Stage 0 is non-negotiable.** Shadow mode scores traffic without acting on it, so a catastrophic misconfiguration is caught with zero user impact. It costs a week and it is the cheapest insurance in the plan.

**Keep a permanent 1% holdback** after full rollout. Without it, you can never again measure what the system is worth, and you cannot detect silent degradation as fraud adapts.

---

## 6. Analysis and interpretation

- **Do not peek.** Fixed horizon set in advance, or a sequential design (always-valid p-values / group-sequential boundaries) if early stopping is required. Repeatedly checking a fixed-horizon test inflates the false-positive rate far above the nominal 5%.
- **Pre-register** the primary metric, guardrails, and run length before launch.
- **Multiple comparisons:** guardrails are tested simultaneously; apply Benjamini-Hochberg across the secondary set. The primary metric is not adjusted.
- **Novelty effects:** fraudsters *react*. An effect that decays over the test is not noise — it is adaptation, and it changes the long-run value estimate. Plot the effect over time rather than reading a single pooled number.
- **Heterogeneity:** segment by publisher, geography, and traffic source before concluding. A neutral average can conceal a large gain and a large harm cancelling out.

---

## 7. Decision rule, agreed in advance

| Outcome | Action |
|---|---|
| Primary positive, no guardrail breach | Ship |
| Primary positive, guardrail breached | Do not ship. Lower threshold, re-test |
| Primary flat, guardrails clean | Do not ship. Complexity without benefit is a cost |
| Primary negative | Roll back, investigate which archetype drives the loss |

Writing this table **before** the data arrives is the entire point. Deciding afterwards is how a team rationalises whatever number it happens to see.
