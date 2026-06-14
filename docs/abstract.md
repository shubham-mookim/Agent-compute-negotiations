# When Intelligence Tiers Invert: Emergent Market Dynamics in Decentralized Compute Negotiation

**Author:** Shubham Mookim

*Draft research writeup — findings summary, not yet a finished paper. See "Open Issues" at the end for what still needs hardening before submission anywhere.*

---

## Abstract

We present a simulation framework for studying decentralized compute resource negotiation between autonomous agents operating at three intelligence tiers: rule-based heuristics (Tier 1), tabular Q-learning (Tier 2), and large language model-powered reasoning (Tier 3). Agents negotiate bilaterally for GPU hours, CPU cycles, and memory allocations without central coordination, using a protocol of requests, counter-offers, accepts, and rejects. Across 9 experiments with over 10,000 statistically controlled trials, we report several counter-intuitive findings. First, we identify a sharp detection threshold at approximately 30% defection rate for reputation-based cheater isolation, below which dishonest agents are virtually undetectable even with gossip-based collaborative reputation sharing. Second, we demonstrate an intelligence tier inversion: Tier 1 greedy heuristics accumulate more wealth than both Tier 2 RL and Tier 3 LLM agents, challenging the assumption that more sophisticated reasoning yields better economic outcomes. Third, we show that LLM agents break negotiation deadlocks impossible for rule-based agents (achieving 100% deal rate against greedy opponents where rule-based agents achieve 0%) but are systematically exploited by adaptive learning agents, overpaying by 67.5% relative to fair market price. Fourth, we find that coalition effectiveness depends on internal negotiation strategy rather than coalition size, with greedy-strategy cartels extracting lower prices than solo fair-strategy providers. These findings contribute to the growing literature on LLM economic behavior and have implications for the design of decentralized compute markets such as Akash Network and io.net.

**Keywords:** multi-agent negotiation, compute resource allocation, LLM agents, reinforcement learning, reputation systems, coalition formation, decentralized markets

---

## 1. Introduction

The rise of GPU-intensive AI workloads has created acute demand for compute resources, driving interest in decentralized allocation mechanisms that bypass centralized cloud providers. Commercial platforms (Akash Network, Render Network, io.net) already implement peer-to-peer compute markets, but lack formal analysis of equilibrium properties, trust dynamics, and agent behavior under varying intelligence levels.

We address this gap with a simulation framework where agents negotiate bilaterally over abstract compute resources. Our key contribution is the systematic comparison of three intelligence tiers within the same market environment:

- **Tier 1 (Rule-based):** Five hand-coded strategies — Greedy, Fair, Patient, Adaptive, and Broker — representing common negotiation heuristics.
- **Tier 2 (RL):** Tabular Q-learning agent with a 60-state space encoding message type, partner reputation, price ratio, and urgency. Learns purely from negotiation experience.
- **Tier 3 (LLM):** GPT-4o-mini powered agent that receives full negotiation context (budget, resources, reputation, deal history) and responds with structured JSON decisions.

## 2. Key Results

**Finding 1: The 30% Detection Threshold.** Across 500-trial sweeps of cheater defection rates (1%–100%), we observe a sharp phase transition in reputation-based detection. Below 20%, cheaters maintain positive reputation (0.57–0.94) and are virtually undetectable. Between 20–50%, detection rises steeply (slope = 4.70). Gossip-based collaborative reputation improves detection from 0% to 16% at 10% cheat rate (Cohen's d = 0.94) but does not close the gap. Paradoxically, adaptive cheaters that adjust their rate based on rejection feedback overshoot the threshold and get caught more often than fixed-rate cheaters.

**Finding 2: Intelligence Tier Inversion.** In head-to-head wealth comparisons across 200+ trials:

| Rank | Agent | Tier | Wealth Change |
|------|-------|------|---------------|
| 1 | Greedy | T1 | +49.0 |
| 2 | Patient | T1 | -0.8 |
| 3 | RL (Q-learning) | T2 | -2.6 |
| 4 | LLM (buyer role) | T3 | -17.0% below fair |
| 5 | Fair | T1 | -5.0 |
| 6 | Adaptive | T1 | -16.5 |

Higher intelligence does not guarantee better economic outcomes. The simplest strategy (Greedy) dominates wealth accumulation because it refuses unfavorable deals, while LLM agents accept too eagerly (all deals close in round 1 with no counter-offering).

**Finding 3: LLM Breaks Deadlocks but Gets Exploited.** LLM buyers achieve 100% deal rate against Greedy sellers, where all rule-based agents achieve 0%. The LLM's flexible reasoning satisfies demands that rigid heuristics cannot. However, Adaptive agents learn to exploit the LLM's generosity, extracting 67.5% premium over fair price. The LLM does not learn within a single session — it lacks the across-negotiation memory that RL agents build.

**Finding 4: Coalition Strategy Dominance.** Buyer coalitions that pool demand paradoxically hurt their members (exhausting budgets faster). Seller cartels using greedy strategy extract lower prices than solo fair-strategy providers, because greedy negotiations deadlock. Coalition benefit is flat across sizes 2–5. Free-riders are detected with 100% accuracy within 10 rounds via contribution-ratio monitoring.

## 3. Related Work

Our work connects to NegotiationArena (2024), which found LLMs fail at Nash equilibrium strategies, and AgenticPay (2025), which studied LLM payment negotiation. We extend these by introducing the cross-tier comparison and the compute-resource domain. Our detection threshold finding relates to evolutionary game theory work on partial defectors (Nowak & Sigmund, 1998) and the FIRE trust model (Huynh et al., 2006). The coalition findings extend classical coalition formation theory (Shapley, 1953) to the compute market setting.

## 4. Implications and Future Work

Our intelligence tier inversion has practical implications: deploying LLM agents as autonomous compute buyers without guardrails risks systematic overpayment. The 30% detection threshold suggests decentralized compute markets need mechanisms beyond local reputation (possibly cryptographic attestation or stake-based commitment) to handle subtle dishonesty.

Future work includes bluffing experiments (can LLMs deceive about urgency?), multi-round LLM memory (does context fix overpaying?), and cross-model comparison (GPT vs Claude vs Gemini).

---

## References

1. Rubinstein, A. (1982). Perfect equilibrium in a bargaining model. *Econometrica*.
2. Shapley, L.S. (1953). A value for n-person games. *Contributions to the Theory of Games*.
3. Huynh, T.D., Jennings, N.R., & Shadbolt, N. (2006). An integrated trust and reputation model. *AAMAS*.
4. NegotiationArena (2024). How well can LLMs negotiate? *arXiv:2402.05863*.
5. AgenticPay (2025). Multi-agent LLM negotiation system. *arXiv:2602.06008*.
6. Agent Workflow for Negotiation Games (2024). *arXiv:2411.05990*.
7. x402-RAM (2025). Game-theoretic resource allocation for decentralized compute markets.

---

## Open Issues (What Still Needs Hardening)

This is a candid list of weaknesses a reviewer would raise. None are fatal, but they must be addressed before this is a defensible paper rather than a promising prototype.

1. **The wealth metric rewards non-participation.** "Wealth delta" goes up for agents that refuse to trade (Greedy, Patient hold their budget). This is the single biggest threat to the headline "tier inversion" finding — Greedy may "win" simply by abstaining, not by negotiating better. **Fix:** define a proper utility that credits *needs satisfied* (compute actually acquired at or below reservation price), then re-run the tier comparison. The inversion finding only survives if it holds under a needs-weighted metric.

2. **Several "findings" may be implementation artifacts.** "Greedy and Patient deadlock at 0%", "all deals close in round 1" — these could be properties of how the five strategies were hand-coded, not general phenomena. **Fix:** parameter-sweep the strategy constants (greed_factor, patience, etc.) and show the qualitative findings are stable across the parameter space, not knife-edge.

3. **LLM sample size is small and single-model.** 10 trials, one model (GPT-4o-mini), one temperature, one resource scenario. The 67.5% overpayment and the deadlock-breaking are striking but underpowered. **Fix:** 50+ trials, multiple scenarios, and at least one second model to show the effect isn't GPT-4o-mini-specific.

4. **No theoretical baseline.** We never compare observed prices to the Rubinstein alternating-offers equilibrium or a Nash bargaining solution. Without it, "fair price = total units" is an assumption, not a benchmark. **Fix:** compute the game-theoretic equilibrium for the two-agent case and measure deviation per tier.

5. **Abstract units, no external validity.** Resources are dimensionless; there is no grounding in real GPU pricing or real workloads. This caps the claims to "in simulation." **Fix:** either own the limitation explicitly in scope, or calibrate one experiment against real spot-market data (e.g., Akash/io.net pricing).

6. **Reputation update rule is hand-tuned.** The 30% detection threshold depends on the specific reputation increment/decrement values. **Fix:** show the threshold's sensitivity to the update rule, or derive it analytically.

**Bottom line:** Findings 1 (detection threshold) and 3 (LLM breaks deadlock) are the most defensible and novel. The tier-inversion (Finding 2) is the most exciting but the most fragile — it lives or dies on fixing the utility metric.
