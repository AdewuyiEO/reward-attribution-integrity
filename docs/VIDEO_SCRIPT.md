# Demo Video Script (~2 minutes)

**Goal:** show a working system *and* statistical maturity in 120 seconds. Most candidates submit a repo; a narrated demo of you reasoning about tradeoffs is what makes you memorable.

**Setup before recording**
- Pipeline already run on the **real** dataset (`python -m src.pipeline` finished).
- Dashboard open: `streamlit run dashboard/app.py`, browser full-screen.
- Close other tabs, silence notifications, use a clean desktop.
- Record at 1080p. Screen + mic. One take is fine; you can re-record any 20-second block.
- Speak slightly slower than feels natural. Pauses read as confidence.

**Tone:** you're a colleague walking a teammate through what you built, not a salesperson. Understated and precise beats hyped.

---

## The script

### [0:00–0:15] — Frame the problem
> "This is a reward-attribution integrity system. The problem: around a quarter of mobile ad spend is lost to fraud — click farms, fake installs, bots. On a rewarded platform, every fraudulent install pays out real money and erodes publisher margin. So I built a system to catch it — without punishing real users."

*On screen: the dashboard title and the three headline metrics.*

### [0:15–0:35] — The core design choice
> "The key decision: it's unsupervised. It never uses conversion labels to detect. Three detectors score each entity on behaviour alone — clustering finds coordinated fleets, distribution analysis catches machine-like timing, and cross-population comparison flags entities that don't behave like the population. I do that because in production there's no fraud label, labels arrive days late, and fraud adapts faster than a supervised model can retrain."

*On screen: scroll slowly past the "Sensitivity vs business impact" charts.*

### [0:35–1:05] — The slider (the star moment)
> "Here's the part I care about most. A score isn't a decision. This slider is the operating threshold."

*Action: drag the slider LOW.*
> "Push sensitivity up and I catch more fraud — but watch the legitimate-users-affected number climb. That's real people denied rewards they earned."

*Action: drag the slider HIGH, then back to the recommended point.*
> "Push it down and I barely touch real users but miss fraud. The cost model picks between them using money, not intuition — it maximises net value subject to a hard cap of one percent of legitimate users blocked. It lands here, at the recommended threshold."

*On screen: the recommended threshold and the net-value peak.*

### [1:05–1:35] — The finding that shows depth
> "The most interesting thing I found is a statistical limit. At a 0.2% conversion rate, an entity needs about 1,760 clicks before zero conversions is even surprising. Below that, a suspicious-looking entity is *unproven*, not guilty."

*Action: point to the evidence-tier column / filter.*
> "So the system tags every flag with an evidence tier. A 115-click IP might look extremely anomalous, but it gets demoted to 'unproven' — a watchlist, not a top hit — unless it's part of a detected ring, where the coordination across many entities is evidence that doesn't depend on any single one's volume. That way the system polices its own confidence."

*On screen: filter to 'unproven', then to 'proven', so the difference is visible.*

### [1:35–1:55] — Close on rigour and honesty
> "Every flag carries plain-language reasons, so an analyst can act and a partner can appeal. It runs on an automated daily pipeline with drift alerting. And the whole thing is honest about its limits — the ground truth is a proxy, IP is a weak identity, and a determined attacker who adds jitter can beat it. Those are all documented, because a fraud system whose limits you can't state is one you shouldn't trust."

### [1:55–2:00] — Sign off
> "Code and full write-up are in the repo. Thanks for watching."

---

## After recording
- Trim dead air at the ends.
- Upload to YouTube as **Unlisted**.
- Put the link at the very top of the README: `**▶ [2-min demo](your-link)**`.
- Optional: pin it in your application message — "here's a 2-minute walkthrough."

## What NOT to do
- Don't read the code line by line. Show behaviour, not source.
- Don't oversell ("cutting-edge", "state-of-the-art"). The restraint is the signal.
- Don't hide the limitations — naming them is the most senior thing in the whole video.
- Don't exceed ~2:15. If you run long, cut the intro, not the slider.
