# Paper Outline: Decentralized Compute Negotiation

## Working Title
"When Intelligence Tiers Invert: Emergent Market Dynamics in Decentralized Agent Compute Negotiation"

## Target
Preprint (arXiv: cs.MA / cs.AI) — no venue deadline pressure. Quality over speed.

---

## Structure

### 1. Introduction (1.5 pages)
- Motivation: GPU scarcity, decentralized compute markets (Akash, io.net, Render)
- Gap: no systematic study of how agent intelligence level affects market outcomes
- Contribution: 3 tiers (rule-based, RL, LLM) in same market, 12 experiments, real compute validation
- Paper map

### 2. Related Work (1 page)
- Rubinstein alternating-offers (1982) — our theoretical baseline
- NegotiationArena (2024) — LLM negotiation benchmarks
- AgenticPay (2025), x402-RAM — agent payment protocols
- FIRE trust model — reputation comparison
- Decentralized compute platforms — Akash, io.net technical architecture

### 3. Framework Design (2 pages)
- Protocol: REQUEST → OFFER → COUNTER → ACCEPT/REJECT (message types, turn limits)
- Agent architecture: resources, budget, urgency, strategy, local reputation table
- Five Tier-1 strategies: Greedy, Fair, Patient, Adaptive, Broker (equations)
- Tier-2: Q-learning agent (state space, reward function, hyperparameters)
- Tier-3: LLM agent (system prompt, structured JSON output, two prompt variants)
- Simulator: round-based, deal execution, cheating injection
- Utility metric: 0.7 × fulfillment_ratio + 0.3 × min(fair_value/price_paid, 1.0)

### 4. Experiments & Results (4-5 pages)

#### 4.1 Strategy Compatibility (Exp 1, 4)
- 4×4 strategy matrix, 1000 trials per pair, 100% deterministic
- Only 4/16 pairs can close deals — strategy deadlocks are the default
- Price convergence between Adaptive agents

**Key data:** `results/exp4_statistical_output.txt` — strategy matrix with CIs, tournament rankings

#### 4.2 Scarcity & Tournament Dynamics (Exp 2)
- 5-agent scarcity simulation, rush hour scenario
- Fair strategy dominates tournaments (wealth 173.5 vs next-best 144.7)
- Rush hour increases prices 6% but cuts deal rate 29%

**Key data:** `results/exp2_scarcity_output.txt`

#### 4.3 Cheater Detection Threshold (Exp 3, 4, 6)
- **THE HEADLINE FINDING**: Sharp phase transition at ~30% cheat rate
- Below 20%: cheaters maintain positive reputation (0.57-0.94)
- 5% cheaters: reputation IMPROVES over time (0.665 → 0.999)
- Collaborative reputation helps (0% → 14% detection at 10% rate, d=0.90) but insufficient
- Adaptive cheaters overshoot and get caught more than fixed-rate

**Key data:** `results/exp4_statistical_output.txt` (cheater sweep), `results/exp6_cheater_depth_output.txt`

#### 4.4 Intelligence Tier Comparison (Exp 9, 5)
- Tier inversion: Greedy (Tier 1) > RL (Tier 2) > LLM (Tier 3) on wealth
- Utility metric partially corrects: RL rises above Patient
- RL overfits to training opponents — transfer penalty confirmed (d=0.295)
- LLM breaks deadlocks (100% deal rate vs Greedy) but overpays 67.5%

**Key data:** `results/exp9_learning_output.txt`, `results/exp4_statistical_output.txt`

#### 4.5 Prompt Engineering Paradox (Exp 5)
- Naive prompts: break deadlocks, overpay 67.5%
- Engineered prompts: fix pricing (+5%), create NEW deadlocks (0% LLM-vs-LLM)
- Individual rationality optimization → collective failure
- Non-linear effect of prompt quality on negotiation outcomes

**Key data:** `results/exp12_llm_real_compute_detail.json` (LLM bids)

#### 4.6 Coalition Dynamics (Exp 8)
- Buyer coalitions HURT members (Δ -12.4 vs -4.7 solo)
- Greedy cartels extract lower prices than Fair solo providers (0.62 vs 0.98)
- Free-rider detection: 100% rate, round 10
- Coalition size has minimal effect on outcomes

**Key data:** `results/exp8_coalitions_output.txt`

#### 4.7 Futures & Arbitrage (Exp 7)
- Futures vs spot: no significant price difference (p=0.655)
- Arbitrage unprofitable in stable markets (d=-1.057)
- Arbitrage profitable in volatile markets (d=0.923) — requires price variance

**Key data:** `results/exp7_futures_output.txt`

#### 4.8 Real Compute Validation (Exp 11, 12)
- SHA-256 CPU burns + memory allocation through ProcessPoolExecutor
- 4 allocation regimes: FIFO, SJF, Urgency, Market
- Urgency-priority dominates (welfare 25.7, urgent latency 0.18s)
- LLM bidding matches centralized urgency scheduler (welfare 25.0 vs 25.3)
- LLM independently reconstructed near-optimal scheduling intuition
- Cost: 36 API calls, $0.0014

**Key data:** `results/exp11_real_compute_detail.json`, `results/exp12_llm_real_compute_detail.json`

#### 4.9 Robustness (Exp 10)
- Parameter sweeps: greed_factor, fairness_tolerance, patience
- Findings stable across parameter ranges — not knife-edge artifacts
- Rubinstein equilibrium comparison: all strategies overpay 9.8-29.2%

**Key data:** `results/exp10_robustness_output.txt`

### 5. Discussion (1.5 pages)
- Why tier inversion happens (Greedy's simplicity is its advantage)
- The prompt engineering trap (Nash vs social optimality)
- Implications for decentralized market design (reputation systems are fundamentally weak below 20% cheat rate)
- Why LLM agents reconstruct scheduling intuition

### 6. Limitations & Future Work (0.5 page)
- Bilateral only — no multi-party negotiation
- Single LLM model (GPT-4o-mini) — needs cross-model comparison
- Synthetic workloads even in "real compute" experiments (SHA-256, not diverse real tasks)
- Small agent populations (5-12 agents)
- No network topology effects
- No dynamic entry/exit of agents

### 7. Conclusion (0.5 page)

---

## Figures (planned)

1. Strategy matrix heatmap (4×4, color = deal rate)
2. Cheater detection threshold curve (cheat rate vs detection %, with phase transition annotation)
3. Intelligence tier ranking under wealth vs utility metrics (dual bar chart)
4. LLM bid distribution vs job urgency scatter plot (from exp12 data)
5. Price convergence time series (two Adaptive agents)
6. Real compute regime comparison (grouped bar: makespan, urgent latency, welfare)
7. Rubinstein equilibrium deviation by strategy (bar chart with equilibrium line)

## Tables (planned)

1. Strategy matrix with confidence intervals (from exp4)
2. Tournament rankings with significance tests
3. Cheater detection sweep (cheat rate × detection rate × reputation × wealth)
4. LLM allocation vs mechanical baselines (from exp12)
5. Parameter robustness summary (stable/unstable per sweep)

---

## Data Files for Paper

All raw results stored in `results/`:
- `exp{N}_*_output.txt` — terminal output from each experiment run
- `exp11_real_compute_detail.json` — per-job real compute records (PID, CPU-seconds, memory)
- `exp11_real_compute_jobs.csv` — flat CSV for analysis
- `exp12_llm_real_compute_detail.json` — LLM bids + allocation + cost breakdown

## Remaining Experiments Before Writing

1. **Multi-model LLM comparison** — GPT-4o vs GPT-4o-mini bidding (addresses single-model weakness)
2. **Adversarial LLM bidder** — one agent prompted to game the system (connects trust + LLM work)
3. **Higher trial counts for exp11/12** — more statistical power on real compute results
