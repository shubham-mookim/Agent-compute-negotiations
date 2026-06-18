# Emergent Market Dynamics in Decentralized Compute Negotiation Across Agent Intelligence Tiers

**Shubham Mookim**

---

## Abstract

The rise of autonomous AI agents with economic agency motivates a foundational question: when agents of differing intelligence negotiate directly for compute resources without central coordination, what market dynamics emerge? We present a framework for studying decentralized compute resource negotiation between agents operating at three intelligence tiers — rule-based heuristics (Tier 1), tabular Q-learning (Tier 2), and large language model (LLM) reasoning (Tier 3) — and validate it on *real* compute workloads in which agents bid for execution slots running genuine SHA-256 CPU burns and memory allocations under hardware-enforced contention. Across 15 experiments comprising more than 15,000 statistically controlled trials, we report five principal findings. **(1)** Reputation-based cheater detection exhibits a sharp phase transition at approximately a 30% defection rate (maximum detection slope 4.70); below 20%, dishonest agents are effectively invisible, and gossip-based collaborative reputation improves detection only marginally (0%→14% at a 10% cheat rate, Cohen's *d* = 0.90, still far from reliable). **(2)** Negotiation outcomes exhibit an *intelligence tier inversion*: a simple greedy heuristic dominates wealth accumulation over both RL (*t* = −166.7, *p* < 0.001) and LLM agents, and this inversion persists under a needs-based utility metric. **(3)** LLM agents independently reconstruct near-optimal scheduling when bidding for real compute, matching a centralized urgency scheduler (social welfare 25.0 vs 25.3); the effect is *not* model-specific — it holds across GPT-4o-mini and GPT-4o, and the larger model slightly *exceeds* the centralized baseline (+6%). **(4)** Prompt engineering has non-linear, sometimes adversarial effects on collective outcomes: naive prompts break deadlocks but overpay 67.5%, while individually rational engineered prompts re-introduce deadlock (0% LLM-vs-LLM deal rate). **(5)** LLM-based allocation is vulnerable to adversarial prompt manipulation: a single selfish agent jumps from last to first priority, reducing social welfare 8.3%. A scaling study (4–40 agents) shows the qualitative findings are stable but the strategy ranking shifts — Adaptive overtakes Fair on utility at scale — and all claims are grounded against the Rubinstein subgame-perfect equilibrium. We discuss implications for the design of decentralized compute markets (e.g., Akash, io.net) and the deployment of economically autonomous LLM agents.

**Keywords:** multi-agent negotiation, compute resource allocation, LLM agents, reinforcement learning, reputation systems, decentralized markets, mechanism design, game theory

---

## 1. Introduction

### 1.1 Motivation

The demand for compute has become a defining economic constraint of the AI era. The agentic AI market reached an estimated \$7.3B in 2025, and autonomous agent deployments on decentralized infrastructure surpassed 20,000 by early 2026 [1]. Simultaneously, AI agents are acquiring genuine economic agency: payment protocols such as x402 and frameworks such as AgenticPay [2] let agents transact on behalf of users, and decentralized compute marketplaces (Akash Network, Render Network, io.net) already match idle GPU supply with demand at scale.

These two trends are converging toward a regime in which *autonomous agents negotiate for compute with other autonomous agents*. Yet today's decentralized compute platforms retain a centralized core: a matching engine and a fixed or oracle-determined price. The properties of a market in which the *matching and pricing themselves* are decentralized — emerging from bilateral bargaining between self-interested agents — remain largely unstudied.

### 1.2 The Gap

Prior work on LLM negotiation (NegotiationArena [5], the Game-Theoretic LLM Agent Workflow [6]) studies LLMs against LLMs or humans in abstract bargaining games. Work on decentralized compute studies infrastructure and incentive layers but not agent-level negotiation dynamics. Work on multi-agent reinforcement learning studies emergent behavior but rarely in economic markets with heterogeneous agent intelligence. No existing study, to our knowledge, (i) places rule-based, RL, and LLM agents in a *single* compute market, (ii) measures the resulting dynamics against a formal bargaining-theoretic benchmark, and (iii) grounds the abstract market in *real* hardware contention.

### 1.3 Research Questions

We organize our investigation around six research questions:

- **RQ1 (Compatibility):** Which agent strategies can successfully close deals, and is deal-formation an emergent property or a knife-edge artifact?
- **RQ2 (Equilibrium):** How closely do agents of each tier approach the Rubinstein subgame-perfect equilibrium and the Nash bargaining solution?
- **RQ3 (Tier comparison):** Does higher agent intelligence yield better economic outcomes?
- **RQ4 (Trust):** Can a decentralized reputation system detect and isolate dishonest agents, and at what defection rate does it fail?
- **RQ5 (Real allocation):** When agents bid for *real* compute slots, does decentralized LLM-driven allocation match centralized scheduling, and is it robust to manipulation?
- **RQ6 (Scale & model):** Do the findings persist as the population grows and as the underlying LLM changes?

### 1.4 Contributions

Our contributions are:

1. A formal model and open simulation framework for decentralized bilateral compute negotiation supporting pluggable agent strategies across three intelligence tiers (Sections 4–5), validated against the Rubinstein and Nash bargaining benchmarks.
2. The identification of a sharp **30% detection threshold** in reputation-based trust, with a quantified phase transition and an analysis of why collaborative reputation is necessary but insufficient (Section 7.4).
3. Empirical evidence of **intelligence tier inversion** — simpler agents outperform sophisticated ones in bilateral markets — robust across wealth and utility metrics and explained via proximity to the bargaining equilibrium (Section 7.3).
4. A **real-compute validation** in which LLM agents reconstruct near-optimal scheduling from first principles, shown to be model-general (GPT-4o-mini and GPT-4o) yet vulnerable to adversarial prompt manipulation (Sections 7.5–7.6).
5. A **prompt-engineering paradox**: individually rational prompt optimization can destroy system-level welfare (Section 7.7).
6. A **scaling and cross-model study** establishing the external validity envelope of the above (Section 7.9), with all code, prompts, seeds, and per-trial result artifacts released for reproduction.

### 1.5 Organization

Section 2 develops formal preliminaries; Section 3 surveys related work; Section 4 formalizes the decentralized compute market; Section 5 specifies the agent tiers and strategies with pseudocode; Section 6 details experimental methodology; Section 7 presents results organized by research question; Section 8 discusses mechanisms; Section 9 analyzes threats to validity; Section 10 addresses broader impact and ethics; Sections 11 and 12 cover limitations and future work; Section 13 concludes.

---

## 2. Preliminaries

We adopt standard notation from non-cooperative and cooperative game theory. Let $N = \{1, \dots, n\}$ denote a set of agents.

**Definition 1 (Normal-form game).** A normal-form game is a triple $\langle N, \{A_i\}_{i \in N}, \{U_i\}_{i \in N} \rangle$, where $A_i$ is agent $i$'s action set and $U_i : \prod_j A_j \to \mathbb{R}$ is its payoff over strategy profiles $a = (a_1, \dots, a_n)$. We write $a_{-i}$ for the profile of all agents except $i$.

**Definition 2 (Nash equilibrium).** A profile $a^\ast$ is a Nash equilibrium if for every agent $i$ and every action $a_i \in A_i$, $U_i(a_i^\ast, a_{-i}^\ast) \ge U_i(a_i, a_{-i}^\ast)$.

**Definition 3 (Extensive-form game and subgame-perfect equilibrium).** An extensive-form game adds a sequential structure of histories $h$. A strategy profile is a *subgame-perfect equilibrium* (SPE) if it induces a Nash equilibrium in every subgame, computed by backward induction via $V_i(h) = \max_{a \in A(h)} V_i(h \cdot a)$.

**Definition 4 (Pareto optimality).** An allocation is Pareto optimal if no alternative makes every agent weakly better off and at least one strictly better off.

**Definition 5 (Individual rationality).** An outcome is individually rational if each agent receives at least its *disagreement payoff* $d_i$ — the utility of walking away with no deal.

**Definition 6 (Nash bargaining solution).** For a two-player bargaining problem $(F, d)$ with convex feasible set $F \subseteq \mathbb{R}^2$ and disagreement point $d = (d_1, d_2)$, the Nash bargaining solution (NBS) is the unique maximizer of the *Nash product*:
$$
(u^\ast, v^\ast) = \arg\max_{(u,v) \in F,\; u \ge d_1,\; v \ge d_2} (u - d_1)(v - d_2),
$$
which uniquely satisfies Pareto efficiency, symmetry, scale invariance, and independence of irrelevant alternatives. We use the NBS as a *normative* benchmark for cooperative efficiency.

**Definition 7 (Rubinstein alternating-offers SPE).** In the alternating-offers bargaining model, two players split a surplus of size $S$ over discrete periods $t = 0, 1, 2, \dots$. Player 1 proposes $(x, S - x)$; player 2 accepts or counters in the next period. With per-period discount factors $\delta_1, \delta_2 \in (0,1)$ encoding impatience, the game has a unique SPE with immediate agreement, in which player 1's equilibrium share is
$$
x^\ast = \frac{1 - \delta_2}{1 - \delta_1 \delta_2}\, S.
$$
The less patient party (lower $\delta$) captures a smaller share. We use this SPE as the *positive* (predictive) price benchmark.

In our setting, agent urgency $u_i \in [0,1]$ maps to impatience via $\delta_i = 1 - \kappa\, u_i$ for a calibration constant $\kappa$: a more urgent agent discounts the future more steeply and therefore concedes more, consistent with Definition 7.

---

## 3. Related Work

**Automated negotiation and bargaining theory.** Rubinstein's alternating-offers model [3] and Nash's axiomatic bargaining solution [4] are the canonical non-cooperative and cooperative treatments of two-party bargaining; both furnish the theoretical baselines in this paper. Classical automated-negotiation research established agent architectures and the role of deadlines and concession strategies. Our contribution is to measure how *learning* and *language-model* agents deviate from these equilibria in a compute market.

**LLM agents in games and negotiation.** NegotiationArena [5] benchmarked LLMs across ultimatum, trading, and buy/sell games, showing that LLMs deviate from Nash play and can be swayed by behavioral tactics (e.g., a "desperate" persona improves payoff ~20%). The Game-Theoretic LLM Agent Workflow [6] introduced structured reasoning workflows and Bayesian belief modeling for negotiation games, and probed LLM rationality via payoff-matrix and message-order perturbations. EconEvals [8] and recent fairness studies [9] document systematic behavioral biases in LLM economic agents. A 2025 IJCAI survey [7] synthesizes the game-theory–LLM intersection. We extend this literature in three ways: a *compute-specific* market with real resource stakes; a *cross-tier* comparison including rule-based and RL agents in the same environment; and a demonstration that LLMs reconstruct *scheduling* (not just bargaining) policy from first principles.

**Trust and reputation in multi-agent systems.** The FIRE model [10] integrates direct experience, witness (gossip) reputation, role-based trust, and certified reputation. Evolutionary game theory studies the survival of partial defectors under image-scoring reputation [11]. Our agents implement direct-experience and gossip reputation; our 30% detection-threshold result quantifies precisely where such mechanisms fail and connects to the partial-defector literature.

**Decentralized compute markets.** Akash, Render, and io.net implement peer-to-peer compute markets with centralized matching and oracle pricing; work on autonomous agents on blockchains [12] formalizes execution and settlement but not negotiation dynamics. We study the regime in which matching and pricing themselves are decentralized, and we provide a real-hardware grounding absent from purely economic treatments.

---

## 4. Problem Formalization

**Definition 8 (Decentralized compute market).** A decentralized compute market is a tuple
$$
M = \langle N, R, \{U_i\}, \{B_i\}, \{u_i\}, \{\theta_i\}, P, \{\mathrm{Rep}_i\} \rangle,
$$
where:
- $N = \{1, \dots, n\}$ is the set of agents;
- $R = \{\mathrm{GPU}, \mathrm{CPU}, \mathrm{MEM}\}$ is the set of resource types, measured in resource-hours; a *bundle* is a vector $L \in \mathbb{R}_{\ge 0}^{|R|}$;
- $U_i$ is agent $i$'s quasi-linear utility (Definition 9);
- $B_i \in \mathbb{R}_{\ge 0}$ is agent $i$'s currency budget;
- $u_i \in [0,1]$ is agent $i$'s urgency (impatience);
- $\theta_i \in \{\textsf{rule}, \textsf{RL}, \textsf{LLM}\}$ is agent $i$'s *intelligence tier*;
- $P$ is the bilateral negotiation protocol (Section 4.1);
- $\mathrm{Rep}_i : N \to [0,1]$ is agent $i$'s *local* reputation function over peers. Crucially there is one $\mathrm{Rep}_i$ per agent — no global reputation oracle exists. This encodes our decentralized-trust design commitment.

**Definition 9 (Quasi-linear utility).** For an allocated bundle $L$ acquired at price $p$, agent $i$'s realized utility is $U_i(L, p) = v_i(L) - p$, where $v_i(L) = \sum_{k \in R} v_i^k L_k$ is an additive valuation with private per-unit values $v_i^k$. The disagreement payoff is $d_i = 0$ (no trade, no surplus).

Because raw utility conflates negotiation skill with starting endowment (a construct-validity concern; Section 9), we additionally define a normalized **needs-based utility**:
$$
\widehat{U}_i = w_f \cdot \underbrace{\frac{\text{units acquired}}{\text{units requested}}}_{\text{fulfillment}} + w_e \cdot \underbrace{\min\!\Big(\frac{\text{fair value acquired}}{\text{price paid}}, 1\Big)}_{\text{efficiency}},
\quad (w_f, w_e) = (0.7, 0.3).
$$
$\widehat{U}_i$ rewards satisfying real demand and penalizes overpayment, and unlike wealth it does not reward non-participation.

### 4.1 The Negotiation Protocol $P$

$P$ is an alternating-offers protocol realized through ten message types, of which five are core: **REQUEST** (buyer declares bundle and reservation price), **OFFER** (seller proposes terms), **COUNTER** (either party revises), **ACCEPT** (agreement; execution follows), and **REJECT** (termination). Four execution/trust messages — **QUERY_REPUTATION**, **REPUTATION_RESPONSE**, **ALLOCATE**, **RELEASE** — and **DEFAULT** (a party fails to deliver, i.e., cheats) complete the vocabulary. A negotiation is a finite alternating exchange capped at $T_{\max}$ turns; reaching $T_{\max}$ without ACCEPT is a disagreement. This realizes Definition 7's extensive-form structure with a finite horizon.

**Assumption 1 (Bounded rationality).** Tier-1 agents follow fixed policies; Tier-2 agents maximize expected discounted reward under tabular Q-learning; Tier-3 agents approximate utility maximization through language-model reasoning. None is assumed to compute the exact SPE; the degree of deviation is an empirical object of study (RQ2).

**Assumption 2 (Local information).** Each agent observes only its own resources, budget, urgency, message history, and local reputation table. Private valuations $v_i^k$ are not disclosed; this makes the full game one of incomplete information.

---

## 5. Agent Intelligence Tiers and Strategies

Each strategy is a policy $\pi_i$ mapping an agent's observable state and an incoming message to an outgoing message. Algorithm 1 gives the shared negotiation loop executed by the simulator.

```
Algorithm 1: Bilateral Negotiation (one pairing)
Input: buyer b, seller s, need L, max turns T_max
1:  m ← b.initiate(s, L)                       # REQUEST
2:  for t = 1 to T_max do
3:      r ← s.decide(m)                         # seller responds
4:      if r.type = ACCEPT: execute(b,s,r); return DEAL
5:      if r.type = REJECT: return NO_DEAL
6:      q ← b.decide(r)                          # buyer responds to counter
7:      if q.type = ACCEPT: execute(b,s,q); return DEAL
8:      if q.type = REJECT: return NO_DEAL
9:      m ← q
10: return NO_DEAL                               # turn limit reached
```

### 5.1 Tier 1: Rule-Based Strategies

Five hand-coded policies (default parameters in parentheses):

- **Greedy** ($g = 0.7$): offers at $g \cdot p_{\text{fair}}$ when selling, demands $(1-g)\cdot p_{\text{fair}}$ when buying; accepts only beyond a favorable threshold.
- **Fair** ($\tau = 0.15$): targets market price; accepts within tolerance $\tau$; splits differences.
- **Patient** ($\rho = 0.8$): waits for bargains with patience decaying over rounds.
- **Adaptive** ($\alpha = 0.2$): maintains a price belief updated by exponential moving average toward observed deal prices — the only Tier-1 strategy that learns within a run.
- **Broker** ($c = 0.1$): a middleman taking commission $c$ on matched deals.

### 5.2 Tier 2: Tabular Q-Learning

A Q-learning agent over a 60-state space — the product of message type $\{$REQUEST, COUNTER$\}$, partner reputation $\{$low, med, high$\}$, price ratio $\{$low, fair, high$\}$, and urgency $\{$low, high$\}$ — with actions $\{$ACCEPT, COUNTER\_LOW, COUNTER\_MID, COUNTER\_HIGH, REJECT$\}$.

```
Algorithm 2: Q-Learning Update (per negotiation step)
1:  s ← encode(message, reputation, price_ratio, urgency)
2:  a ← ε-greedy(Q[s])                           # ε decays 0.40 → 0.05
3:  observe reward r, next state s'
4:  Q[s,a] ← Q[s,a] + α (r + γ max_a' Q[s',a'] − Q[s,a])
```
Reward: $+1$ for a deal at or below fair price, $-0.5$ for overpaying, $-0.1$ for failed negotiation; $\alpha = 0.1$, $\gamma = 0.9$.

### 5.3 Tier 3: LLM Agents

We deploy OpenAI models (default GPT-4o-mini; GPT-4o in the cross-model study) as negotiation and bidding agents. An agent receives a structured description of its context and returns a decision. We study two prompt regimes: a **naive** prompt with minimal instruction, and an **engineered** prompt encoding a pricing framework (reference price, BATNA, surplus), a decision tree, and explicit anti-overpayment guidance. Temperature is fixed at 0.3; output is capped (10 tokens for bid tasks, structured JSON for negotiation). Full prompt texts appear in Appendix C.

### 5.4 Real Compute Execution Layer

To discharge the external-validity threat of abstract resources, we implement a workload layer in which a *job* performs genuine computation: `cpu_burn` chains SHA-256 hashes (returning the final digest so the work cannot be optimized away), and `mem_burn` allocates and page-touches real memory. Jobs run in a `ProcessPoolExecutor` with $K = (\text{cores} - 1)$ real slots, creating genuine contention; we record wall time, CPU time (`getrusage`), peak RSS, and OS process IDs. Work is auto-calibrated to a target per-job duration so results are comparable across hardware.

---

## 6. Experimental Methodology

**Statistical protocol.** Unless stated otherwise, each reported quantity is a mean over 200–1000 independent seeded trials with a 95% confidence interval. We use Welch's *t*-test (unequal variances) for two-group comparisons and Cohen's *d* for effect sizes; significance markers denote $p < 0.001$. Inequality is measured by the Gini coefficient. All statistics route through a single audited module (`agents/stats.py`) to ensure consistency (a conclusion-validity safeguard).

**Metrics (formally named).** *Deal-closure rate*: fraction of negotiations ending in ACCEPT. *Mean payoff / wealth* $\Delta$: end-minus-start net worth. *Needs-based utility* $\widehat{U}$ (Section 4). *Price deviation*: realized price relative to the Rubinstein SPE (Definition 7). *Social welfare*: $\sum_i u_i / (\text{completion time}_i)$ over real jobs, rewarding low-latency service of urgent work. *Urgent latency*: mean completion time of jobs with $u_i \ge 0.7$. *Makespan*: time to finish all jobs. *Fairness*: standard deviation of wait times. *Detection latency / rate*: rounds until, and probability that, a cheater's reputation falls below 0.1.

**Environment.** Simulations run on a 4-core host (3 execution slots) under Linux; LLM calls target OpenAI GPT-4o-mini and GPT-4o. Every LLM experiment is wrapped in a hard call-budget guard that raises an exception before exceeding a preset number of API calls, bounding cost deterministically. Total API expenditure for all LLM experiments in this paper was under \$0.03.

**Reproducibility.** All randomness is seeded; non-LLM experiments are bit-reproducible across runs. Per-trial artifacts (including real PIDs, CPU-seconds, and peak memory for the compute experiments, and per-job LLM bids for the allocation experiments) are dumped as JSON/CSV under `results/`. Appendix B lists the full experiment index and invocation commands.

---

## 7. Results

### 7.1 RQ1 — Strategy Compatibility and Deadlocks

Testing all 16 buyer × seller pairs of the four core strategies over 1000 trials each yields a fully deterministic compatibility matrix:

| Buyer \ Seller | Greedy | Fair | Patient | Adaptive |
|---|---|---|---|---|
| **Greedy** | 0% | 0% | 0% | 0% |
| **Fair** | 0% | 100% | 0% | 100% |
| **Patient** | 0% | 0% | 0% | 0% |
| **Adaptive** | 0% | 100% | 0% | 100% |

Only 4 of 16 pairs trade at all, and those trade with certainty (zero variance over 1000 trials). Deal formation is therefore *not* an emergent statistical tendency but a structural property of strategy compatibility: Greedy and Patient agents deadlock against everyone, including themselves. A 9×9 parameter-variant sweep (greed factor, tolerance, patience) confirms the structure is robust, not a knife-edge of specific constants (Section 7.8).

### 7.2 RQ2 — Distance from Equilibrium

For the canonical configuration (buyer urgency 0.7, seller urgency 0.2, surplus $S = 5$ GPU-hours), Definition 7 yields $\delta_b = 0.93$, $\delta_s = 0.98$, and an SPE price of **3.871** (77.4% of surplus to the seller). Realized prices deviate above equilibrium for every strategy:

| Strategy | Avg Price | Deviation from SPE |
|---|---|---|
| Greedy (0.7 / 0.85) | 4.250 | **+9.8%** |
| Fair / Adaptive / Patient / RL | 5.000 | +29.2% |

Greedy's rigid reservation price happens to sit closest to the SPE, foreshadowing its wealth dominance (RQ3). An equilibrium-sensitivity sweep across all urgency pairs reproduces the comparative-statics prediction of Definition 7: the less patient party's share falls monotonically (buyer share ranges 10.1%–90.8% as urgencies vary), confirming our urgency→discount mapping behaves as theory requires.

### 7.3 RQ3 — Intelligence Tier Inversion

In a heterogeneous 5-agent market over 300 rounds (100–200 trials), ranking by wealth $\Delta$:

| Rank | Agent | Tier | Wealth $\Delta$ |
|---|---|---|---|
| 1 | Greedy | 1 | **+48.7 ± 0.6** |
| 2 | Patient | 1 | −0.9 ± 0.2 |
| 3 | RL (fresh) | 2 | −2.3 ± 0.5 |
| 4 | RL (pretrained) | 2 | −2.9 ± 0.1 |
| 5 | Fair | 1 | −5.1 ± 0.3 |
| 6 | Adaptive | 1 | −15.2 ± 0.8 |

Greedy dominates both RL variants (RL-pretrained vs Greedy: $t = -166.7$, $p < 0.001$, $d = -23.6$). Because wealth rewards abstention (a construct threat), we re-rank by needs-based utility $\widehat{U}$:

| Rank | Agent | Tier | $\widehat{U}$ |
|---|---|---|---|
| 1 | Greedy | 1 | 0.4152 ± 0.0003 |
| 2 | RL (fresh) | 2 | 0.3874 ± 0.0010 |
| 3 | RL (pretrained) | 2 | 0.3867 ± 0.0001 |
| 4 | Patient | 1 | 0.3861 ± 0.0006 |
| 5 | Fair | 1 | 0.3821 ± 0.0007 |
| 6 | Adaptive | 1 | 0.3621 ± 0.0015 |

The utility metric promotes RL above Patient — confirming the metric correction is real — but Greedy retains rank 1. **The tier inversion survives both metrics.** Inspecting the learned Q-policy (8 visited state-action pairs) reveals a sensible but conservative reservation-price rule; RL agents transferred across opponent distributions underperform natively trained ones ($d = -0.30$, $p = 0.037$), confirming overfitting. We explain the inversion mechanistically in Section 8.1.

### 7.4 RQ4 — The 30% Detection Threshold

Embedding one cheater among five honest agents (500 trials per defection rate) reveals a sharp phase transition in reputation-based detection:

| Cheat Rate | Detection | Final Reputation | Cheater Wealth |
|---|---|---|---|
| 1% | 0% | 0.941 [0.933, 0.948] | 145.5 |
| 5% | 0% | 0.883 [0.874, 0.893] | 157.2 |
| 10% | 0% | 0.792 [0.781, 0.804] | 171.3 |
| 20% | 5% | 0.570 [0.555, 0.585] | 196.1 |
| **30%** | **39%** | 0.348 [0.335, 0.362] | 216.5 |
| 50% | 97% | 0.109 [0.102, 0.116] | 246.6 |
| 100% | 100% | 0.001 [0.000, 0.001] | 310.3 |

The detection curve's maximum slope is **4.70**, and the 50%-detection crossover sits at a 30% cheat rate. Below 20%, a cheater's reputation actually *rises* over time because successful deals outvote rare defaults. **Collaborative (gossip) reputation** helps but does not close the gap: at a 10% cheat rate, detection rises from 0% to 14% ($d = 0.90$, $p < 0.001$); at 5%, only 0%→3% ($d = 0.52$). An *adaptive* cheater that modulates its rate by rejection feedback is detected just 7% of the time, converging to an 8.0% effective rate — safely under the threshold. With 2 (resp. 3) cheaters among 6, cheaters out-earn honest agents 166.8 vs 116.6 (resp. 160.8 vs 109.4), both $p < 0.001$. We formalize why additive reputation has this blind spot in Section 8.2.

### 7.5 RQ5a — Real-Compute Allocation

We schedule 12 heterogeneous real jobs (four archetypes spanning 258K–1.29M SHA-256 iterations and 8–128 MB) through 3 real slots under four regimes (averaged over trials):

| Regime | Makespan (s) | Urgent Latency (s) | Social Welfare |
|---|---|---|---|
| FIFO | 1.67 | 0.83 | 12.3 |
| SJF | 1.58 | 0.23 | 25.5 |
| **Urgency** | **1.47** | **0.18** | **25.7** |
| Market (WTP×urgency) | 1.55 | 0.25 | 22.4 |

Urgency-priority dominates. Adding an **LLM regime** — each agent makes one LLM call to bid 0–100 for its own job — yields welfare 25.0 vs the centralized urgency scheduler's 25.3 and identical urgent latency (0.18s). The LLM was given only job descriptions; it assigned high bids (70–98) to urgent jobs and low bids (1–15) to batch jobs, *reconstructing* urgency-priority scheduling without being told the policy. Cost: 36 calls, \$0.0014.

### 7.6 RQ5b — Adversarial Prompt Manipulation

Assigning one low-urgency job ($u \le 0.14$) an adversarial system prompt ("always bid maximum") while 11 peers bid honestly:

| Trial | Adversary Urgency | Honest Bid | Adversarial Bid | Queue Jump |
|---|---|---|---|---|
| 0 | 0.00 | 0 | 100 | +11 (last→first) |
| 1 | 0.14 | 10 | 100 | +10 (last→first) |

The adversary reaches first position in every trial, dropping social welfare 8.3% (19.6→17.9) by displacing genuinely urgent jobs. The vulnerability is structural: the LLM trusts its own system prompt, and no peer can verify another's. Defense requires mechanisms external to the LLM (bid bonds, audit logs, stake slashing). Cost: 60 calls, \$0.0022.

### 7.7 RQ on Prompt Engineering — A Non-Linear Paradox

Comparing prompt regimes in bilateral negotiation:

- **Naive prompt:** LLM achieves a **100% deal rate against Greedy** (where all rule-based agents get 0%) — breaking deadlocks — but **overpays 67.5%** against Adaptive opponents who learn to exploit its generosity.
- **Engineered prompt:** fixes pricing (≈+5% over fair) but causes a **0% deal rate in LLM-vs-LLM** play: two individually rational agents both refuse to concede.

Thus prompt quality is non-monotonic in collective welfare: optimizing each agent's individual rationality re-creates the very deadlock the LLM was meant to dissolve. There is no prompt in our study that simultaneously maximizes individual protection and collective deal-formation.

### 7.8 Robustness

Parameter sweeps over greed factor (0.50–0.95), fairness tolerance (0.05–0.40), and patience (0.30–0.95) leave the qualitative findings intact: Patient deadlocks for $\rho \ge 0.4$ regardless of opponent; Fair's deal rate and price are invariant to $\tau$. The compatibility structure of Section 7.1 is therefore not an artifact of specific constants.

### 7.9 RQ6 — Scaling and Cross-Model Generalization

**Population scaling (4→40 agents, 50 rounds, 30 trials each; no API cost).**

| N | Deal Rate | Gini (wealth) |
|---|---|---|
| 4 | 0.125 ± 0.005 | 0.100 |
| 10 | 0.167 ± 0.009 | 0.093 |
| 16 | 0.192 ± 0.008 | 0.096 |
| 40 | 0.191 ± 0.006 | 0.096 |

Deal rate rises 53% from N=4 to N=16 and then *plateaus* — more agents create more compatible pairings until saturation — while wealth inequality remains essentially constant (Gini ≈ 0.10), showing inequality is structural, not a small-N artifact. Critically, the **strategy ranking shifts with scale**: at N=40 the utility order is Adaptive (0.243) > Fair (0.233) > Greedy (0.206) > Patient (0.184). The small-population "Fair dominates" result is partly an artifact; at scale, the *learning* Adaptive strategy finds compatible partners and converges to better prices, overtaking Fair. This both bounds and enriches the tier story: simple Greedy still wins on raw wealth, but adaptivity pays off precisely when the market is large enough to find counterparties.

**Cross-model allocation (GPT-4o-mini vs GPT-4o on identical real workloads, 3 trials).**

| Regime | Social Welfare | Urgent Latency (s) |
|---|---|---|
| Centralized Urgency | 23.3 | 0.25 |
| FIFO | 12.6 | 0.74 |
| GPT-4o-mini | 22.6 | 0.20 |
| **GPT-4o** | **24.7** | **0.18** |

The LLM-reconstructs-scheduling finding is **not model-specific**: both tiers approximate centralized urgency scheduling, with inter-model bid-order agreement of 0.91. The larger model is strictly better — GPT-4o exceeds the centralized baseline by 6% and beats GPT-4o-mini by 9.3% — indicating allocation quality scales with model capability. Total cost for the cross-model study: \$0.0238.

### 7.10 Secondary Findings: Coalitions, Futures, Arbitrage

Buyer coalitions *hurt* members (wealth $\Delta$ −12.4 vs −4.7 solo) by exhausting pooled budgets; Greedy seller cartels extract *lower* prices (0.62) than a solo Fair provider (0.98) because internal greedy friction undermines cartel pricing; free-riders are detected with 100% accuracy by round 10. Futures and spot prices do not differ ($p = 0.655$); arbitrage is unprofitable in stable markets ($d = -1.06$) but profitable under strategy-induced volatility ($d = 0.92$), consistent with efficient-market intuition that arbitrage profit requires price variance.

---

## 8. Discussion

### 8.1 Why Intelligence Tiers Invert

In bilateral negotiation without repeated-play memory or third-party observation, the (near-)optimal policy is simple: set a reservation price near the Rubinstein SPE and reject everything worse. Greedy approximates this by construction (Section 7.2), extracting maximal surplus per trade while abstaining otherwise. RL must *learn* this rule and overfits to its training opponents; LLMs *reason* about it but either over-cooperate (naive) or over-restrict (engineered). Sophistication is wasted where the environment rewards a fixed threshold. The scaling result (Section 7.9) sharpens this: sophistication begins to pay (Adaptive overtakes Fair) precisely when the market is large enough that *finding* a counterparty — not just pricing against a known one — becomes the binding constraint. We therefore predict the inversion reverses in environments with repeated interaction, multi-party deals, and reputation consequences — a testable hypothesis (Section 12).

### 8.2 The Structural Weakness of Additive Reputation

Our reputation rule updates additively (+0.1 on fulfillment, −0.3 on default, clamped to $[0,1]$). For a cheater defecting with probability $q$, the expected per-deal drift is $0.1(1-q) - 0.3q = 0.1 - 0.4q$, which is *positive* for $q < 0.25$. This closed-form drift explains the empirical threshold: below roughly a 25% defection rate, successful deals rebuild reputation faster than defaults erode it, so reputation trends *up* regardless of dishonesty. The exact crossover in simulation (≈20–30%, Section 7.4) matches this prediction once detection latency and clamping are accounted for. Gossip raises effective sample size and thus sharpens estimates, but cannot change the sign of the drift — which is why collaborative reputation helps only marginally. The implication for market design is strong: *any* additively-updated reputation system has a defection rate below which subtle cheaters are mathematically invisible; robust markets need stake-based or cryptographic commitment, not reputation alone.

### 8.3 LLMs as Zero-Shot Schedulers

That GPT-4o-mini and GPT-4o both reconstruct urgency-priority scheduling from job descriptions — with the larger model *beating* a hand-built centralized scheduler — suggests LLMs have internalized enough resource-management prior to serve as competent, decentralized schedulers at negligible cost (\$10⁻³ per decision). This is a constructive counterpoint to the tier-inversion result: LLMs are weak at adversarial *bilateral bargaining* but strong at *one-shot prioritization reasoning*. The task structure, not the agent's sophistication per se, determines whether intelligence helps.

### 8.4 Practical Implications

For **market designers**: pair reputation with stake-based commitment or verifiable computation; reputation alone is provably insufficient below the defection threshold (Section 8.2). For **LLM-agent deployers**: evaluate prompts at the *system* level under competition, not just individually — the prompt-engineering paradox (Section 7.7) shows agent-level optimization can be anti-correlated with collective welfare, and the adversarial result (Section 7.6) shows LLM allocation needs tamper-resistant wrappers. For **schedulers**: LLM-driven bidding is a viable, near-optimal, low-cost path to decentralized allocation of heterogeneous workloads — provided manipulation is externally constrained.

---

## 9. Threats to Validity

**Construct validity.** Our primary risk is that raw *wealth* conflates negotiation skill with starting endowment and rewards non-participation. We mitigate this by introducing the needs-based utility $\widehat{U}$ (Section 4) and re-deriving the tier ranking under it (Section 7.3); the inversion persists under both. A second construct risk — that abstract resource units do not represent real compute — is discharged by the real-compute experiments (Sections 7.5–7.6, 7.9), which consume genuine CPU and memory. A third risk is that "reputation rises" may be a metric artifact rather than true non-detection; the closed-form drift analysis (Section 8.2) confirms it is a genuine property of additive updating, not an artifact.

**Internal validity.** Role and turn-order could confound deal outcomes; the protocol fixes the buyer-initiates structure and we test both strategies in each role across the full matrix (Section 7.1). Random-seed effects are controlled by averaging over 200–1000 seeds with CIs. Tier is not confounded with budget or endowment because all tiers are evaluated under identical initial conditions in the head-to-head market (Section 7.3). Deal execution (resource/currency transfer) is handled by the simulator, not the agents, isolating negotiation behavior from accounting bugs.

**External validity.** Generalization is bounded along several axes, which we partially discharge: from small to larger populations (scaling study, 4→40 agents, Section 7.9); from one LLM to two (cross-model study, Section 7.9); and from abstract to real compute (Sections 7.5–7.6). Remaining gaps — markets of $10^3+$ agents, additional model families (Claude, Gemini), and production workloads beyond synthetic CPU/memory burns — are stated as limitations (Section 11).

**Conclusion validity.** All inferential claims use Welch's *t*-test with reported effect sizes and 95% CIs, routed through a single audited statistics module to avoid inconsistent procedures. Sample sizes (200–1000 trials) yield narrow intervals; where an effect is statistically significant but practically small (e.g., gossip detection +14 points), we say so explicitly rather than overclaiming.

---

## 10. Broader Impact and Ethical Considerations

This work studies autonomous agents that transact for compute — a capability with dual-use character. **Positive impacts:** decentralized negotiation could lower the cost and improve the utilization of compute, democratizing access for smaller actors otherwise priced out of centralized clouds; our LLM-scheduler result suggests near-optimal allocation is achievable cheaply and without a trusted central operator. **Negative impacts and risks:** (i) our cheater-detection results show that decentralized markets are exploitable by subtle dishonest agents below a 30% defection rate, which could enable fraud if such markets are deployed without commitment mechanisms; (ii) the adversarial-prompt result demonstrates that LLM-mediated allocation can be gamed by a single bad actor, a manipulation vector relevant to any agentic marketplace; (iii) coalition and cartel dynamics could be weaponized for collusive price manipulation. We publish these vulnerabilities precisely so that market designers can defend against them *before* deployment, and we recommend that any real compute market built on agent negotiation incorporate stake-based or cryptographic commitment from the outset. Our experiments consume only our own machine's compute and a few cents of API budget; no human subjects or personal data are involved.

---

## 11. Limitations

We state limitations as bounds on present claims, distinct from the forward-looking agenda of Section 12.

1. **Bilateral protocol.** Negotiation is strictly two-party; we do not model auctions, multilateral bargaining, or combinatorial allocation. Our tier-inversion claim is therefore scoped to bilateral markets.
2. **Population scale.** We evaluate up to 40 agents; behavior in markets of $10^3$+ agents, where network topology and partial observability dominate, is untested.
3. **Model coverage.** LLM results cover two OpenAI models; we do not yet test Claude, Gemini, or open-weight models, so model-family effects are unknown.
4. **Workload realism.** "Real compute" means genuine SHA-256 and memory burns, not the diverse production workloads (inference, training, ETL) of a real cluster; archetypes are representative but not exhaustive.
5. **Reputation rule specificity.** The 30% threshold's exact value depends on the additive update constants; the *qualitative* blind spot is general (Section 8.2), but the precise crossover is rule-dependent.
6. **Static strategy sets.** Apart from Adaptive and RL, agents do not co-evolve their strategies within a run.

---

## 12. Future Work

We frame each direction as a research question with a hypothesis and a concrete method, tied to a specific finding above.

- **FW1 — Commitment mechanisms vs the detection threshold.** *RQ:* Does adding a stake-based deposit with slashing eliminate the sub-30% cheater blind spot (Section 7.4), and at what stake-to-trade ratio? *Hypothesis:* a deposit exceeding the expected per-trade gain from defection makes cheating unprofitable at all rates, collapsing the threshold. *Method:* extend the protocol with an escrow message and slashing on DEFAULT; sweep stake ratios and measure cheater wealth.
- **FW2 — Reversing the tier inversion.** *RQ:* Do RL and LLM agents overtake Greedy when the market has repeated interaction and reputation consequences (Section 8.1)? *Hypothesis:* with memory and reputation, sophisticated agents exploit long-horizon structure that a fixed threshold cannot. *Method:* enable persistent reputation and repeated pairings; re-run the tier comparison and the scaling study jointly.
- **FW3 — Multi-party negotiation.** *RQ:* Does the 4/16 compatibility structure (Section 7.1) generalize, collapse, or transform under three-party bargaining? *Method:* generalize Algorithm 1 to $k$-party rounds and recompute the compatibility tensor.
- **FW4 — Cross-family LLM study.** *RQ:* Is the LLM-as-scheduler result (Section 7.9) invariant across Claude and Gemini, or OpenAI-specific? *Method:* re-run the cross-model allocation harness across families under identical seeds.
- **FW5 — Production workloads.** *RQ:* Do allocation conclusions hold when jobs are real inference/training tasks with heterogeneous GPU/memory profiles rather than synthetic burns? *Method:* replace the workload layer with containerized real jobs and re-measure welfare and urgent latency.
- **FW6 — Manipulation-resistant LLM allocation.** *RQ:* Can a verifiable bidding wrapper (commit-reveal bids with bond forfeiture) neutralize the adversarial-prompt attack (Section 7.6) without sacrificing the near-optimal allocation? *Method:* add commit-reveal to the bidding regime; measure welfare with and without an adversarial agent present.

---

## 13. Conclusion

We presented a formal framework and empirical study of decentralized compute negotiation across three agent intelligence tiers, grounded in real hardware contention and benchmarked against bargaining theory. Five findings challenge intuitive assumptions: subtle dishonesty is mathematically invisible to additive reputation below a 30% defection rate; simpler agents out-negotiate sophisticated ones in bilateral markets, an inversion that only begins to reverse at scale; LLM agents nonetheless reconstruct near-optimal scheduling from first principles, model-generally and at negligible cost; prompt optimization can destroy collective welfare; and LLM allocation is manipulable by a single adversarial prompt. Together these results map where decentralized agent markets are efficient, where they are fragile, and what mechanisms — commitment, verification, scale, and the right matching of task to intelligence — are required to make them robust. The central open question is whether sophistication's advantage, glimpsed in our scaling results, becomes decisive in the richer markets — repeated, multi-party, reputation-bearing — that real deployments will demand.

---

## References

[1] 0G Foundation. "Agentic AI Market at \$7.3B: Infrastructure Gaps Blocking Scale." 2026. https://0g.ai/blog/agentic-ai-market-infra-2026

[2] "AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions." arXiv:2602.06008, 2025.

[3] Rubinstein, A. "Perfect Equilibrium in a Bargaining Model." *Econometrica* 50(1):97–109, 1982.

[4] Nash, J.F. "The Bargaining Problem." *Econometrica* 18(2):155–162, 1950.

[5] Bianchi, F., Chia, P.J., Yuksekgonul, M., Tagliabue, J., Jurafsky, D., Zou, J. "How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis." ICML 2024. arXiv:2402.05863.

[6] "Game-Theoretic LLM: Agent Workflow for Negotiation Games." arXiv:2411.05990, 2024.

[7] "Game Theory Meets Large Language Models: A Systematic Survey." IJCAI 2025.

[8] "EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents." arXiv:2503.18825, 2025.

[9] "Evaluating Fairness in LLM Negotiator Agents via Economic Games Using Multi-Agent Systems." *Mathematics* 14(3):458, 2025.

[10] Huynh, T.D., Jennings, N.R., Shadbolt, N. "An Integrated Trust and Reputation Model for Open Multi-Agent Systems." *AAMAS* 13(2):119–154, 2006.

[11] Nowak, M.A., Sigmund, K. "Evolution of Indirect Reciprocity by Image Scoring." *Nature* 393:573–577, 1998.

[12] "Autonomous Agents on Blockchains: Standards, Execution, and Deployment." arXiv:2601.04583, 2025.

---

## Appendix A: Experiment Index

| Exp | Description | Trials | Result Artifact |
|---|---|---|---|
| 1 | Handshake & strategy matrix | 1,000 | results/exp1_handshake_output.txt |
| 2 | Scarcity games & tournament | 20 | results/exp2_scarcity_output.txt |
| 3 | Cheater detection (always/subtle) | 60 | results/exp3_trust_output.txt |
| 4 | Statistical rigor | 6,200 | results/exp4_statistical_output.txt |
| 5 | LLM negotiations & prompt paradox | 15+ | (API) |
| 6 | Deep cheater analysis | 2,800 | results/exp6_cheater_depth_output.txt |
| 7 | Futures & arbitrage | 200 | results/exp7_futures_output.txt |
| 8 | Coalition formation | 1,600 | results/exp8_coalitions_output.txt |
| 9 | RL learning & tier comparison | 2,000 | results/exp9_learning_output.txt |
| 10 | Robustness & Rubinstein baseline | 500+ | results/exp10_robustness_output.txt |
| 11 | Real compute contention | 3 | results/exp11_real_compute_detail.json |
| 12 | LLM bidding for real compute | 3 | results/exp12_llm_real_compute_detail.json |
| 13 | Adversarial LLM bidder | 3 | results/exp13_adversarial_llm.json |
| 14 | Population scaling | 30/size | results/exp14_scaling.json |
| 15 | Cross-model LLM allocation | 3 | results/exp15_cross_model.json |

## Appendix B: Reproduction

```bash
git clone <repository_url>
cd Agent-compute-negotiations
pip install matplotlib openai
export OPENAI_API_KEY=your_key_here   # only for exp 5, 12, 13, 15
python run_all.py            # all experiments
python run_all.py 4          # statistical rigor (no API)
python run_all.py 11         # real compute (no API)
python run_all.py 12         # LLM + real compute (API)
python run_all.py 14         # scaling (no API)
python run_all.py 15         # cross-model (API)
```
All randomness is seeded; non-LLM experiments are bit-reproducible. Hard call-budget guards bound API cost; the entire LLM suite in this paper cost under \$0.03.

## Appendix C: LLM Prompts

**Bid prompt (system).** "You are an autonomous compute-scheduling agent. You own ONE job that needs to run on a shared server with limited execution slots… Output a single integer bid from 0 to 100… Reason about urgency, job size, budget, and fairness." (Full text in `experiments/exp12_llm_real_compute.py`.)

**Adversarial bid prompt (system).** "You are a selfish compute-scheduling agent. Your ONLY goal is to get your job executed FIRST… Always bid the MAXIMUM priority (100) regardless of your job's actual urgency…" (Full text in `experiments/exp13_adversarial_llm.py`.)

**Negotiation prompts (naive and engineered).** Full texts in `agents/llm_strategy.py` (`NAIVE_SYSTEM_PROMPT`, `ENGINEERED_SYSTEM_PROMPT`).

## Appendix D: Reproducibility Statement

Hyperparameters: RL uses $\alpha = 0.1$, $\gamma = 0.9$, $\varepsilon$ decaying 0.40→0.05 over 1000 rounds; LLM calls use temperature 0.3 and capped output tokens. Hardware: 4-core Linux host, 3 execution slots; work auto-calibrated to ~0.3 s/job. Statistics: Welch's *t*-test, Cohen's *d*, Gini, 95% CIs via `agents/stats.py`. Number of runs per result: 200–1000 (simulation), 3 (real-compute, with full per-job artifacts). All seeds fixed; all code, prompts, and per-trial result files are released in the repository.
