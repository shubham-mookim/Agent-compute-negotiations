# Agent Compute Negotiation

Autonomous agents that negotiate with each other for compute resources (GPU time, memory, CPU cycles) through a decentralized protocol. No central authority — just agents making deals based on their needs, resources, and trust relationships.

## Quick Start

```bash
pip install matplotlib

# Run all experiments
python run_all.py

# Run individually
python run_all.py 1   # Basic Handshake
python run_all.py 2   # Scarcity Games
python run_all.py 3   # Trust Fall
```

## Project Structure

```
agents/
  protocol.py    - Message types (REQUEST, OFFER, COUNTER, ACCEPT, REJECT, ...)
  resource.py    - Abstract compute units (GPU hours, CPU hours, memory)
  agent.py       - Core Agent class with reputation tracking
  strategies.py  - Negotiation strategies (Greedy, Fair, Patient, Adaptive, Broker)
  simulator.py   - Simulation engine that runs negotiation rounds

experiments/
  exp1_handshake.py  - Two agents, one negotiation
  exp2_scarcity.py   - 5 agents competing for scarce resources
  exp3_trust.py      - Reputation and cheater detection
  visualize.py       - Matplotlib plots for results

run_all.py           - Run everything
```

## Experiments

### Experiment 1: The Basic Handshake

Two agents negotiate compute. Tests every strategy combination and tracks price convergence between adaptive agents.

**Key finding:** Fair and Adaptive strategies are the only ones that successfully close deals with each other. Greedy and Patient strategies deadlock — neither is willing to meet in the middle. Two Adaptive agents converge to a stable price within ~30 rounds (volatility drops to near zero).

### Experiment 2: Scarcity Games

5 agents (3 providers, 2 seekers) compete over 100 rounds. Includes a "rush hour" scenario where everyone suddenly needs compute, and a tournament ranking strategies by accumulated wealth.

**Key finding:** Fair providers dominate the tournament — they close the most deals and accumulate the most wealth. Greedy providers barely trade at all. The Adaptive seeker outperforms the Greedy seeker by learning market prices. During rush hour, deal success rate drops and prices spike.

### Experiment 3: Trust Fall

6 agents including one cheater. Tests whether the network can isolate dishonest actors using decentralized reputation tracking.

**Key finding:** Always-cheating agents get detected and isolated within 10-15 rounds. BUT they still profit short-term (higher final wealth from stolen resources). Subtle cheaters (5% cheat rate) fly under the radar — their reputation actually *improves* over time because successful deals outweigh occasional defaults. This is the interesting case.

## Strategies

| Strategy | Approach | When it wins |
|----------|----------|--------------|
| **Greedy** | Lowballs buyers, demands premium from sellers | Rarely — deadlocks too often |
| **Fair** | Targets market-rate prices, splits differences | Most situations — closes deals consistently |
| **Patient** | Waits for bargains, sells at premium | Low-demand environments |
| **Adaptive** | Learns price from deal history, adjusts beliefs | Markets with stable pricing |
| **Broker** | Middleman that takes a commission | When supply and demand are separated |

## Experiments 4-7 (Phase A-D)

### Experiment 4: Statistical Rigor
Re-runs experiments 1-3 with 1000 trials. All strategy differences confirmed significant (p < 0.001). Fair provider dominance holds with Cohen's d > 15. Gini coefficient = 0.094.

### Experiment 5: LLM Agents (needs `ANTHROPIC_API_KEY`)
Claude-powered agents negotiate in natural language. Tests LLM vs LLM, LLM vs rule-based, and mixed populations. Includes fallback mode for running without API key.

### Experiment 6: Deep Cheater Analysis
**Key finding:** Detection threshold is ~30% cheat rate with a sharp phase transition. Collaborative reputation (gossip) improves detection from 0% to 16% at 10% cheat rate (p < 0.001, d = 0.94). Adaptive cheaters paradoxically get caught more — simple rate-adjustment overshoots.

### Experiment 7: Futures Market
Spot vs futures market comparison. In Fair-priced markets, futures add no efficiency benefit and arbitrage is unprofitable. Price-volatile markets (Greedy/mixed) likely needed for futures to matter.

## Documentation

- `CLAUDE.md` — Full project context for AI agents picking up this work
- `docs/PROGRESS.md` — End-to-end report: results, roadmap, research landscape, bibliography

## What's Next

- **LLM-powered experiments** with real Claude API (needs key)
- **Coalition formation** — agents pooling resources, free-rider detection
- **Reputation-aware adaptive cheater** — can it stay below detection threshold?
- **Mixed-strategy markets** — introduce price volatility to test arbitrage viability
- **Paper writing** — target AAMAS, AAAI, or JASSS
