# Emergent Market Dynamics in Decentralized Compute Negotiation Across Intelligence Tiers

**Shubham Mookim**

---

## Abstract

We present a framework for studying decentralized compute resource negotiation between autonomous agents operating at three intelligence tiers: rule-based heuristics (Tier 1), tabular Q-learning (Tier 2), and LLM-powered reasoning (Tier 3). Agents negotiate bilaterally for GPU hours, CPU cycles, and memory allocations without central coordination, using a protocol of requests, counter-offers, and accepts/rejects. We validate the framework on real compute workloads where agents bid for execution slots running SHA-256 CPU burns and memory allocations in a contention-limited ProcessPoolExecutor. Across 13 experiments with over 15,000 statistically controlled trials, we report several counter-intuitive findings. First, we identify a sharp phase transition in reputation-based cheater detection at approximately 30% defection rate (detection slope = 4.70), below which dishonest agents are virtually undetectable even with gossip-based collaborative reputation (Cohen's d = 0.90 improvement, still insufficient). Second, we demonstrate an intelligence tier inversion where Tier 1 greedy heuristics dominate wealth accumulation over both Tier 2 RL agents (t = -223.13, p < 0.001) and Tier 3 LLM agents, a finding that persists under utility-based evaluation (fulfillment-weighted). Third, we show that LLM agents independently reconstruct near-optimal scheduling when bidding for real compute slots, matching centralized urgency-priority allocation (social welfare 25.0 vs 25.3) while beating mechanical market formulas (20.7). Fourth, we document a prompt engineering paradox: naive LLM prompts break negotiation deadlocks but overpay 67.5%, while engineered prompts fix pricing but create new deadlocks, demonstrating that individual rationality optimization can destroy collective outcomes. All findings are validated through parameter sweeps confirming robustness and compared against Rubinstein alternating-offers equilibrium.

**Keywords:** multi-agent negotiation, compute resource allocation, LLM agents, reinforcement learning, reputation systems, decentralized markets, game theory

---

## 1. Introduction

The proliferation of GPU-intensive AI workloads has created sustained demand pressure on compute infrastructure, with the agentic AI market reaching $7.3B in 2025 and projected for continued growth [1]. Decentralized compute platforms such as Akash Network, Render Network, and io.net implement peer-to-peer markets where providers offer idle capacity to consumers, but these platforms rely on centralized matching and fixed-price mechanisms rather than autonomous agent negotiation.

We study the question: **what happens when autonomous agents of varying intelligence negotiate directly for compute resources without central coordination?** This question is increasingly relevant as AI agents gain economic agency through protocols like x402 and AgenticPay [2], and as decentralized compute deployments surpass 20,000 active agents [1].

Our contributions are:

1. A minimal bilateral negotiation protocol and simulation framework supporting pluggable agent strategies across three intelligence tiers (Section 2).
2. A sharp phase transition in reputation-based cheater detection, with quantified thresholds and collaborative reputation analysis (Section 4.1).
3. Evidence of intelligence tier inversion in negotiation outcomes, robust across wealth and utility metrics (Section 4.2).
4. Real compute validation showing LLM agents independently reconstruct near-optimal scheduling (Section 4.4).
5. A prompt engineering paradox demonstrating non-linear effects of instruction optimization on collective outcomes (Section 4.3).

All code, data, and experiment logs are available in the accompanying repository.

## 2. Framework Design

### 2.1 Negotiation Protocol

Our protocol defines 10 message types forming a minimal negotiation vocabulary:

| Message Type | Semantics |
|---|---|
| REQUEST | Agent declares compute needs and maximum willingness-to-pay |
| OFFER | Provider proposes resources at a stated price |
| COUNTER | Either party modifies the proposed terms |
| ACCEPT | Agreement reached; deal proceeds to execution |
| REJECT | Negotiation terminates without agreement |
| QUERY_REPUTATION | Agent requests peer assessment of a third party |
| REPUTATION_RESPONSE | Peer provides reputation score |
| ALLOCATE | Resources transferred from provider to consumer |
| RELEASE | Resources returned after use |
| DEFAULT | Party fails to fulfill agreed terms (cheating) |

A negotiation round proceeds as: the simulator pairs agents (one buyer, one seller), the buyer sends a REQUEST, and agents alternate OFFER/COUNTER messages up to a configurable turn limit. If an ACCEPT is reached, the simulator executes the deal (transferring resources and currency). If the turn limit expires or a REJECT is sent, the negotiation fails.

### 2.2 Agent Architecture

Each agent maintains:
- **Resources:** A vector of (GPU hours, CPU hours, memory-GB-hours) available for trade or consumption.
- **Budget:** Currency available for purchasing compute.
- **Urgency:** A scalar in [0, 1] representing time pressure. High-urgency agents have more at stake if they fail to acquire compute.
- **Strategy:** A pluggable decision module that generates responses to incoming messages.
- **Reputation table:** A local mapping from peer agent IDs to trust scores in [0, 1], updated based on deal outcomes. Each agent maintains its own view --- there is no central reputation authority.
- **Deal history:** Records of past transactions for learning and analysis.

### 2.3 Tier 1: Rule-Based Strategies

We implement five heuristic strategies:

**Greedy (greed_factor g):** Offers at g × fair_price when selling; demands (1 - g) × fair_price when buying. Accepts only if the price exceeds the agent's threshold. Default g = 0.7.

**Fair (tolerance t):** Targets market-rate pricing. Accepts any offer within t of fair price and splits the difference on counter-offers. Default t = 0.15.

**Patient (patience p):** Waits for favorable deals, becoming gradually less patient. Sells at premium, buys at discount. Patience decays over rounds. Default p = 0.80.

**Adaptive (learning_rate α):** Maintains a price_belief updated from completed deals via exponential moving average. Converges toward market price. Default α = 0.2.

**Broker (commission c):** Middleman strategy that matches known providers with seekers, taking a commission. Default c = 0.10.

### 2.4 Tier 2: Q-Learning Agent

We implement a tabular Q-learning agent with a 60-state space defined by the Cartesian product of:
- Message type received: {REQUEST, COUNTER} (2 values)
- Partner reputation: {low < 0.3, medium 0.3-0.7, high > 0.7} (3 values)
- Price relative to fair value: {low < 0.8, fair 0.8-1.2, high > 1.2} (3 values)
- Own urgency: {low < 0.5, high ≥ 0.5} (2 values)

Actions are {ACCEPT, COUNTER_LOW, COUNTER_MID, COUNTER_HIGH, REJECT}. The reward function assigns +1 for a completed deal at or below fair price, -0.5 for overpaying, and -0.1 for failed negotiations. Training uses ε-greedy exploration with decay from 0.40 to 0.05 over 1000 rounds.

### 2.5 Tier 3: LLM Agent

We deploy GPT-4o-mini (OpenAI) as a negotiation agent receiving full context --- budget, available resources, partner reputation, deal history, and the current message --- and returning structured JSON decisions. We test two prompt variants:

**Naive prompt:** Minimal instructions ("You are a compute negotiation agent. Decide: ACCEPT, REJECT, or COUNTER with a price.").

**Engineered prompt:** Production-grade instructions including pricing framework (reference price, BATNA, surplus calculation), decision tree (when to accept, counter, or reject), and explicit guidance on avoiding overpayment.

Both variants are tested to measure the effect of prompt sophistication on negotiation outcomes.

### 2.6 Utility Metric

Initial analysis used wealth change (Δbudget + Δresource_value) as the primary metric. However, this rewards non-participation: agents that refuse all deals preserve their budget. We therefore define a needs-based utility:

```
utility = 0.7 × fulfill_ratio + 0.3 × min(fair_value / price_paid, 1.0)
```

where fulfill_ratio = (compute actually acquired) / (compute requested), and the efficiency term penalizes overpayment relative to fair market price, capped at 1.0.

### 2.7 Real Compute Validation

To ground the abstract negotiation framework in real resource contention, we implement a workload execution layer where jobs perform actual SHA-256 hash-chain computation (cpu_burn: ~650K iterations ≈ 0.3s per job) and real memory allocation with page-touching (mem_burn: 8--128 MB). Jobs execute in a ProcessPoolExecutor with a fixed slot limit (host cores - 1), creating genuine contention. We measure wall time, CPU time (via getrusage), peak memory, and real process IDs.

Four allocation regimes are compared:
- **FIFO:** First-in, first-out ordering.
- **SJF:** Shortest job first (by estimated CPU work).
- **Urgency:** Priority queue by job urgency score.
- **Market:** Priority by willingness-to-pay × urgency.

A fifth regime adds LLM-driven allocation: each agent owns one real job and makes one LLM call to decide its priority bid (0--100), reasoning about the job's urgency, size, budget, and fairness.

## 3. Related Work

**Bargaining theory.** Rubinstein (1982) [3] established the subgame-perfect equilibrium for alternating-offers bargaining, which we use as our theoretical price baseline. Nash (1950) [4] provided the axiomatic solution; our agents deviate from both by 9.8--29.2% depending on strategy.

**LLM negotiation.** NegotiationArena [5] benchmarked LLM negotiation across ultimatum, trading, and price games, finding that LLMs fail at Nash equilibrium strategies but can exploit behavioral tactics (+20% payoff via sympathy appeals). Our work extends this to compute-specific markets with real resource stakes. The Game-Theoretic LLM Agent Workflow [6] introduced structured reasoning for negotiation games. AgenticPay [2] studied multi-agent LLM payment negotiation. A recent IJCAI survey [7] provides a comprehensive overview of game theory and LLMs.

**LLM economic behavior.** EconEvals [8] provides benchmarks for LLM economic decision-making. Studies of LLM fairness in economic games [9] find that LLMs exhibit human-like behavioral biases. Our finding that LLMs overpay systematically is consistent with the "over-cooperative" bias documented in these works.

**Trust and reputation.** The FIRE model [10] combines direct experience, witness reputation, role-based trust, and certified reputation. Our agents implement the first two (direct experience and gossip). Our 30% detection threshold finding extends evolutionary game theory work on partial defectors [11].

**Decentralized compute.** Akash Network, Render Network, and io.net implement peer-to-peer compute markets with centralized matching. Our framework studies what happens when matching itself is decentralized. Research on autonomous agents on blockchains [12] addresses execution standards but not negotiation dynamics.

## 4. Experimental Results

All results are reported with 95% confidence intervals from 200--1000 independent trials unless otherwise noted. Effect sizes use Cohen's d; significance is assessed via Welch's t-test.

### 4.1 The 30% Detection Threshold

**Setup.** We embed one cheating agent ("Mallory") among five honest agents in a 60-round trust simulation. Mallory defaults on deals (keeping resources without delivering) at a controlled rate from 1% to 100%. Honest agents maintain local reputation tables, updating trust scores based on deal outcomes. Detection is defined as Mallory's average reputation falling below 0.1.

**Results.** Across 500 trials per cheat rate, we observe a sharp phase transition:

| Cheat Rate | Detection Rate | Final Reputation | Cheater Wealth | Incidents |
|---|---|---|---|---|
| 1% | 0% | 0.941 [0.933, 0.948] | 145.5 [145.1, 145.8] | 0.5 |
| 5% | 0% | 0.883 [0.874, 0.893] | 157.2 [156.4, 158.0] | 2.8 |
| 10% | 0% | 0.792 [0.781, 0.804] | 171.3 [170.2, 172.3] | 5.5 |
| 20% | 5% | 0.570 [0.555, 0.585] | 196.1 [194.8, 197.4] | 10.2 |
| 30% | 39% | 0.348 [0.335, 0.362] | 216.5 [215.2, 217.8] | 14.0 |
| 50% | 97% | 0.109 [0.102, 0.116] | 246.6 [245.1, 248.1] | 19.6 |
| 100% | 100% | 0.001 [0.000, 0.001] | 310.3 [307.7, 312.8] | 32.1 |

The maximum slope of the detection curve is 4.70, indicating a sharp transition rather than gradual degradation. The critical threshold where detection exceeds 50% is at 30% cheat rate.

**Collaborative reputation.** When agents share reputation information via gossip (QUERY_REPUTATION / REPUTATION_RESPONSE messages), detection improves significantly at 10% cheat rate: from 0% to 14% (Cohen's d = 0.90, p < 0.001). At 5% cheat rate, improvement is smaller: 0% to 3% (d = 0.52, p < 0.001). Collaborative reputation is necessary but not sufficient.

**Adaptive cheaters.** We test a cheater that adjusts its rate based on deal rejection feedback (increasing rate when accepted, decreasing when rejected). Paradoxically, the adaptive cheater achieves only 7% detection rate --- better than fixed 30% cheaters --- but its final cheat rate converges to 8.0% ± 8.8%, suggesting oscillatory behavior rather than optimal exploitation.

**Multiple cheaters.** With 2 cheaters among 6 agents, cheaters earn significantly more than honest agents (166.8 vs 116.6, t = -44.27, p < 0.001). With 3 cheaters, the gap persists (160.8 vs 109.4, t = -47.37). Honest agents cannot compensate through honest behavior alone when cheaters operate below the detection threshold.

**Implication.** Reputation-based trust is fundamentally inadequate for detecting subtle dishonesty in decentralized markets. Markets that rely solely on peer reputation will be systematically exploited by agents cheating at rates below 20%. This suggests a need for cryptographic commitment mechanisms, stake-based deposits, or verifiable computation proofs as complements to reputation.

### 4.2 Intelligence Tier Inversion

**Setup.** We compare agents across three intelligence tiers in a 5-agent market with heterogeneous strategies (1 Greedy, 1 Fair, 1 Patient, 1 Adaptive, 1 RL or LLM) over 300 rounds, repeated for 100--200 trials.

**Results (wealth metric).**

| Rank | Agent | Tier | Wealth Δ |
|---|---|---|---|
| 1 | Greedy | Tier 1 | +48.7 ± 0.6 |
| 2 | Patient | Tier 1 | -0.9 ± 0.2 |
| 3 | RL (fresh) | Tier 2 | -2.3 ± 0.5 |
| 4 | RL (pretrained) | Tier 2 | -2.9 ± 0.1 |
| 5 | Fair | Tier 1 | -5.1 ± 0.3 |
| 6 | Adaptive | Tier 1 | -15.2 ± 0.8 |

The Greedy heuristic dominates all other strategies including both RL variants (t = -166.68, p < 0.001, d = -23.57 for pretrained RL vs Greedy).

**Utility metric correction.** Under the needs-based utility metric, rankings shift:

| Rank | Agent | Tier | Utility |
|---|---|---|---|
| 1 | Greedy | Tier 1 | 0.4152 ± 0.0003 |
| 2 | RL (fresh) | Tier 2 | 0.3874 ± 0.0010 |
| 3 | RL (pretrained) | Tier 2 | 0.3867 ± 0.0001 |
| 4 | Patient | Tier 1 | 0.3861 ± 0.0006 |
| 5 | Fair | Tier 1 | 0.3821 ± 0.0007 |
| 6 | Adaptive | Tier 1 | 0.3621 ± 0.0015 |

RL rises from rank 3-4 to rank 2-3 under utility (surpassing Patient), but Greedy retains rank 1. The tier inversion persists: higher intelligence does not yield better outcomes.

**Why Greedy wins.** Greedy's advantage is not from superior negotiation but from selective participation. It only accepts deals near the Rubinstein equilibrium price (+9.8% deviation vs +29.2% for Fair), extracting maximum surplus when it does trade and preserving budget otherwise. Greedy negotiates closest to the game-theoretic optimum by accident --- its rigid threshold happens to approximate the equilibrium allocation.

**RL overfitting.** RL agents trained against one opponent distribution perform significantly worse when transferred to a different opponent mix (d = -0.295, p = 0.037), confirming overfitting to the training environment.

### 4.3 The Prompt Engineering Paradox

**Setup.** We test LLM agents (GPT-4o-mini) with two prompt variants in bilateral negotiation against all Tier 1 strategies.

**Naive prompt results.**
- LLM vs Greedy: **100% deal rate** (where all rule-based agents achieve 0%).
- LLM vs Fair: 100% deal rate, price 10.0 (at fair value).
- LLM vs Adaptive: 100% deal rate, but LLM **overpays by 67.5%** relative to fair price.
- LLM vs LLM: deals succeed, prices cluster near fair value.

**Engineered prompt results.**
- LLM vs Greedy: deals succeed with improved pricing (+5% above fair instead of +67.5%).
- LLM vs LLM: **0% deal rate** --- both agents become so cautious that no agreement is reached.

**Analysis.** The naive prompt makes the LLM cooperative and flexible, which breaks deadlocks that rigid heuristics cannot resolve but makes it vulnerable to exploitation. The engineered prompt makes the LLM individually rational, which prevents exploitation but recreates the deadlock problem that the LLM was supposed to solve.

This is a collective action problem: optimizing each agent's individual rationality destroys the collective outcome (deal completion). The effect is non-linear --- there is no prompt that simultaneously maximizes individual protection and collective deal-making. This finding has practical implications for deploying LLM agents in economic contexts: prompt engineering that focuses on individual agent performance can degrade system-level outcomes.

### 4.4 Real Compute Validation

**Setup.** We generate heterogeneous workloads of 12 real jobs across four archetypes (short_urgent: 258K iterations, 8 MB; medium_mixed: 646K iterations, 32 MB; long_batch: 1.29M iterations, 16 MB; mem_heavy: 388K iterations, 128 MB). Jobs execute in a ProcessPoolExecutor with 3 real execution slots on a 4-core host. Social welfare is computed as the sum of urgency-weighted inverse completion times, rewarding low-latency service of urgent jobs.

**Regime comparison (averaged across 3 trials).**

| Regime | Makespan (s) | Urgent Latency (s) | Social Welfare | Fairness (σ_wait) |
|---|---|---|---|---|
| FIFO | 1.67 | 0.83 | 12.3 | 0.44 |
| SJF | 1.58 | 0.23 | 25.5 | 0.36 |
| Urgency | 1.47 | 0.18 | 25.7 | 0.32 |
| Market | 1.55 | 0.25 | 22.4 | 0.33 |

Urgency-priority dominates all metrics. FIFO performs poorly for urgent jobs (0.83s latency vs 0.18s for Urgency), confirming that ordering matters for real social welfare.

**LLM allocation.** Each agent makes one LLM call to decide its job's priority bid (0--100). Results averaged across 3 trials:

| Regime | Makespan (s) | Urgent Latency (s) | Social Welfare |
|---|---|---|---|
| FIFO | 1.62 | 0.69 | 13.5 |
| Urgency | 1.48 | 0.18 | 25.3 |
| Market | 1.60 | 0.28 | 20.7 |
| **LLM** | **1.47** | **0.18** | **25.0** |

The LLM independently reconstructed near-optimal scheduling: it assigned high bids (70--98) to urgent jobs and low bids (1--15) to batch jobs, producing an ordering that matched the centralized urgency scheduler. The LLM was not told about scheduling theory, urgency-priority algorithms, or what "good" allocation looks like. It inferred the right priority ordering from the job descriptions alone.

**Cost.** 36 LLM calls (one per job per trial), 8,867 input tokens, 36 output tokens, total cost $0.0014.

### 4.5 Strategy Compatibility

**Setup.** We test all 16 buyer × seller strategy pairs with 1000 trials per pair.

| Buyer \ Seller | Greedy | Fair | Patient | Adaptive |
|---|---|---|---|---|
| Greedy | 0% | 0% | 0% | 0% |
| Fair | 0% | 100% | 0% | 100% |
| Patient | 0% | 0% | 0% | 0% |
| Adaptive | 0% | 100% | 0% | 100% |

Only 4 of 16 pairs achieve any deals, and those 4 achieve 100% deal rate with zero variance across 1000 trials. This is fully deterministic given the strategy implementations, not stochastic. The result holds across parameter sweeps (9 × 9 matrix of parameter variants in Section 4.7).

In a 5-agent tournament (1000 trials), Fair providers accumulate the most wealth (173.5 ± 1.5, 95% CI [173.4, 173.6]), significantly outperforming all others (vs next-best Adaptive: d = 2.71, p < 0.001). The Gini coefficient of resource distribution is 0.0974 ± 0.0076, indicating moderate inequality.

### 4.6 Coalition Dynamics

**Buyer coalitions.** Pooling buyer demand into a coalition paradoxically hurts members (Δ = -12.4 vs -4.7 for solo seekers). Coalition members exhaust their budgets faster by committing to collective purchasing.

**Seller cartels.** Greedy-strategy seller cartels extract lower per-unit prices (0.62) than solo Fair providers (0.98). Greedy internal negotiations create friction that undermines the cartel's pricing power.

**Coalition size.** Member wealth is flat across coalition sizes 2--5, suggesting that coalition benefit depends on internal strategy, not membership count.

**Free-rider detection.** Agents that join a coalition but do not contribute are detected with 100% accuracy within 10 rounds via contribution-ratio monitoring.

### 4.7 Robustness and Theoretical Grounding

**Parameter sweeps.** We sweep greed_factor (0.50--0.95), fairness_tolerance (0.05--0.40), and patience (0.30--0.95). The strategy deadlock finding is robust: Patient achieves 0% deal rate for patience ≥ 0.40 regardless of opponent. Fair strategy deals are invariant to tolerance parameter (20% deal rate, price 5.00 for all t ∈ [0.05, 0.40]).

**Rubinstein equilibrium.** For our default configuration (buyer urgency 0.7, seller urgency 0.2, 5 GPU-hour surplus), the Rubinstein subgame-perfect equilibrium price is 3.871. All strategies overpay:

| Strategy | Avg Price | Deviation from Equilibrium |
|---|---|---|
| Greedy (0.7) | 4.250 | +9.8% |
| Greedy (0.85) | 4.250 | +9.8% |
| Fair | 5.000 | +29.2% |
| Adaptive | 5.000 | +29.2% |
| Patient (0.5) | 5.000 | +29.2% |
| RL | 5.000 | +29.2% |

Greedy's proximity to equilibrium (only +9.8% overpayment) explains its wealth dominance: it extracts the most surplus per trade.

**Equilibrium sensitivity.** The Rubinstein price varies from 0.459 to 4.495 across urgency combinations, with buyer share ranging from 10.1% to 90.8%. The less time-pressured party always captures more surplus, consistent with theory.

### 4.8 Futures and Arbitrage

Futures contracts (pre-committed resource reservations) show no significant price difference from spot markets (t = 0.447, p = 0.655). Arbitrage (buying low, selling high) is unprofitable in stable markets (d = -1.057) but profitable in volatile markets with mixed Greedy/Fair providers (d = 0.923, p < 0.001). This is consistent with efficient market hypothesis: arbitrage profits require price variance, which requires strategy heterogeneity.

## 5. Discussion

### 5.1 Why Intelligence Tiers Invert

The tier inversion is not a failure of LLM or RL agents --- it is a consequence of the market structure. In bilateral negotiation with no repeat-play memory, the optimal strategy is to set a reservation price near the Rubinstein equilibrium and reject everything else. Greedy approximates this by accident. RL tries to learn it but overfits to its training opponents. LLMs reason about it but either over-cooperate (naive prompt) or over-restrict (engineered prompt).

The deeper insight is that negotiation intelligence has diminishing returns in markets with few participants and simple goods. The sophistication of LLM reasoning about "fairness," "social norms," and "long-term relationships" is wasted when the market has no long-term memory and no third-party observers. In richer environments with multi-party negotiation, repeated interactions, and reputation consequences, we would expect the tier ordering to reverse.

### 5.2 The Fundamental Weakness of Reputation

Our 30% threshold is specific to our reputation update rule, but the qualitative finding is general: any reputation system that updates additively (increment for success, decrement for failure) will have a critical frequency below which cheating is profitable and undetectable. This is because successful deals actively rebuild reputation, creating a "noise floor" that masks infrequent defection.

Collaborative reputation (gossip) improves detection by pooling observations across agents, effectively increasing sample size. But the improvement is bounded: at 10% cheat rate, detection rises from 0% to only 14% despite statistically significant improvement (d = 0.90). The cheater's successful deals still dominate the signal.

### 5.3 LLM Scheduling Intuition

The most surprising result is that GPT-4o-mini, given only job descriptions (urgency score, estimated CPU work, memory needs, budget), independently produced a near-optimal priority ordering matching centralized urgency scheduling. The model was not given examples of scheduling algorithms, told that urgent jobs should go first, or shown any scheduling literature. It inferred the scheduling policy from first principles.

This suggests that LLMs have internalized sufficient knowledge about priority and resource management from their training data to function as effective scheduling agents. The practical cost ($0.0014 for 36 scheduling decisions) makes this approach viable for real deployment scenarios.

### 5.4 Practical Implications

For **decentralized compute market designers**: pure reputation systems are insufficient. Markets need at least one of: (a) cryptographic commitment (stake-based deposits, slashing), (b) verifiable computation proofs, or (c) trusted third-party auditing. Reputation can complement these but should not be the sole trust mechanism.

For **LLM agent deployers**: prompt engineering has non-linear effects on multi-agent outcomes. Testing individual agent performance is necessary but not sufficient --- system-level evaluation under competition is required. The prompt engineering paradox demonstrates that agent-level optimization can be anti-correlated with system-level welfare.

For **resource allocation systems**: LLM-driven bidding can match centralized scheduling quality at negligible cost, providing a path toward decentralized allocation that preserves efficiency. This is particularly relevant for heterogeneous workloads where a single scheduling heuristic may not be optimal.

## 6. Limitations and Future Work

**Bilateral only.** Our protocol supports only two-party negotiation. Multi-party bargaining, auctions, and combinatorial allocation are not studied.

**Single LLM model.** All LLM experiments use GPT-4o-mini. Cross-model comparison (GPT-4o, Claude, Gemini) would strengthen generalizability claims.

**Synthetic workloads.** Even the "real compute" experiments use SHA-256 hash chains and memory allocation rather than diverse real workloads (inference, training, data processing). The workload archetypes are representative but not exhaustive.

**Small populations.** Agent populations range from 2 to 12. Scaling behavior to hundreds or thousands of agents is unknown.

**No network effects.** All agents can interact with all others. Network topology, geographic latency, and bandwidth constraints are not modeled.

**Static strategies.** Tier 1 agents do not adapt within a simulation run (except Adaptive's price belief). A richer study would examine co-evolution of strategies.

Future work should address multi-party negotiation with real heterogeneous workloads, cross-model LLM comparison, and scaling analysis. The OCR processing, inference serving, and data pipeline workloads common in production environments would provide stronger external validity than synthetic CPU burns. Integration with real decentralized compute platforms (Akash, io.net) for on-chain validation is a natural next step.

## 7. Conclusion

We have presented a systematic study of decentralized compute negotiation across three intelligence tiers. Our findings challenge several intuitive assumptions: that more intelligent agents negotiate better (they don't, in simple bilateral markets), that reputation systems detect cheating (they don't, below 30%), and that better prompt engineering yields better multi-agent outcomes (it doesn't, due to collective action failure). The most promising finding --- that LLM agents can independently reconstruct near-optimal scheduling from first principles at negligible cost --- points toward a practical deployment path for decentralized compute allocation. The key open question is whether these dynamics persist in richer market structures with more agents, diverse goods, and repeated interaction, where the sophistication of LLM reasoning may finally provide an advantage over simple heuristics.

---

## References

[1] 0G Foundation. "Agentic AI Market at $7.3B: Infrastructure Gaps Blocking Scale." 2026. https://0g.ai/blog/agentic-ai-market-infra-2026

[2] AgenticPay. "A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions." arXiv:2602.06008, 2025.

[3] Rubinstein, A. "Perfect Equilibrium in a Bargaining Model." Econometrica, 50(1):97-109, 1982.

[4] Nash, J.F. "The Bargaining Problem." Econometrica, 18(2):155-162, 1950.

[5] Bianchi, F., Chia, P.J., Yuksekgonul, M., Tagliabue, J., Jurafsky, D., and Zou, J. "How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis." ICML 2024. arXiv:2402.05863.

[6] "Game-Theoretic LLM: Agent Workflow for Negotiation Games." arXiv:2411.05990, 2024.

[7] "Game Theory Meets Large Language Models: A Systematic Survey." IJCAI 2025.

[8] "EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents." arXiv:2503.18825, 2025.

[9] "Evaluating Fairness in LLM Negotiator Agents via Economic Games Using Multi-Agent Systems." Mathematics, 14(3):458, 2025.

[10] Huynh, T.D., Jennings, N.R., and Shadbolt, N. "An Integrated Trust and Reputation Model for Open Multi-Agent Systems." AAMAS, 13(2):119-154, 2006.

[11] Nowak, M.A. and Sigmund, K. "Evolution of Indirect Reciprocity by Image Scoring." Nature, 393:573-577, 1998.

[12] "Autonomous Agents on Blockchains: Standards, Execution, and Deployment." arXiv:2601.04583, 2025.

---

## Appendix A: Experiment Index

| Exp | Description | Trials | Key Result File |
|---|---|---|---|
| 1 | Basic handshake and strategy matrix | 1,000 | results/exp1_handshake_output.txt |
| 2 | Scarcity games and tournament | 20 | results/exp2_scarcity_output.txt |
| 3 | Cheater detection (always and subtle) | 60 | results/exp3_trust_output.txt |
| 4 | Statistical rigor (CI, effect sizes) | 6,200 | results/exp4_statistical_output.txt |
| 5 | LLM agent negotiations | 15+ | (requires API key) |
| 6 | Deep cheater analysis | 2,800 | results/exp6_cheater_depth_output.txt |
| 7 | Futures and arbitrage | 200 | results/exp7_futures_output.txt |
| 8 | Coalition formation | 1,600 | results/exp8_coalitions_output.txt |
| 9 | RL learning agents | 2,000 | results/exp9_learning_output.txt |
| 10 | Parameter robustness | 500+ | results/exp10_robustness_output.txt |
| 11 | Real compute contention | 3 | results/exp11_real_compute_detail.json |
| 12 | LLM bidding for real compute | 3 | results/exp12_llm_real_compute_detail.json |
| 13 | Adversarial LLM bidder | 3 | results/exp13_adversarial_llm.json |

## Appendix B: Reproduction

```bash
git clone <repository_url>
cd Agent-compute-negotiations
pip install matplotlib openai
export OPENAI_API_KEY=your_key_here
python run_all.py          # all experiments
python run_all.py 4        # statistical rigor only
python run_all.py 11       # real compute only
python run_all.py 12       # LLM + real compute (requires API key)
```

All randomness is seeded for reproducibility. Non-LLM experiments produce identical results across runs.
