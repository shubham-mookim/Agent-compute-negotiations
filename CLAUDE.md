# CLAUDE.md — Agent Compute Negotiation Project

## Project Overview

Research project exploring decentralized negotiation for compute resources (GPU, CPU, memory) between autonomous agents. No central scheduler — agents make bilateral deals based on needs, resources, budgets, and trust. The goal is novel research findings suitable for academic publication.

## Repository Structure

```
agents/                    # Core framework
  __init__.py              # Public API exports
  protocol.py              # Message types: REQUEST, OFFER, COUNTER, ACCEPT, REJECT, etc.
  resource.py              # Resource abstraction (GPU/CPU/memory hours), ResourcePool
  agent.py                 # Agent class: resources, budget, strategy, reputation table
  strategies.py            # 5 strategies: Greedy, Fair, Patient, Adaptive, Broker
  simulator.py             # Simulation engine: runs rounds, executes deals, logs metrics
  llm_strategy.py          # [Phase B] LLM-powered negotiation via Claude API
  stats.py                 # [Phase A] Statistical analysis utilities

experiments/               # Runnable experiments
  __init__.py
  exp1_handshake.py        # Two agents, strategy matrix, price convergence
  exp2_scarcity.py         # 5 agents, scarce resources, rush hour, tournament
  exp3_trust.py            # Cheater detection, reputation evolution, subtle cheaters
  exp4_statistical.py      # [Phase A] Rigorous re-runs with confidence intervals
  exp5_llm_agents.py       # [Phase B] LLM vs rule-based vs mixed negotiations
  exp6_cheater_depth.py    # [Phase C] Detection thresholds, collaborative reputation
  exp7_futures.py          # [Phase D] Predictive negotiation, compute futures
  visualize.py             # Matplotlib plots for all experiments

docs/                      # Research documentation
  PROGRESS.md              # End-to-end report: what's done, results, roadmap

run_all.py                 # Entry point: python run_all.py [1-7]
requirements.txt           # matplotlib, anthropic (when API key available)
```

## Key Design Decisions

- **Resources are abstract units** — not real GPUs. Lets us iterate fast on protocol design without infrastructure complexity.
- **Strategies are pluggable** — any class implementing `initiate()` and `decide()` works. This makes it trivial to drop in new strategies (LLM, RL, etc.).
- **Reputation is local** — each agent maintains its own view of peers. No central reputation database. This is a deliberate design choice to study decentralized trust.
- **Simulator handles deal execution** — agents negotiate, the simulator transfers resources and currency. This separation lets us inject cheating behavior, add latency, simulate failures.

## Current State (as of session)

### Completed
- Core framework (protocol, resources, agents, strategies, simulator)
- Experiment 1: Basic Handshake — strategy matrix across 4 strategies
- Experiment 2: Scarcity Games — 5-agent tournament, rush hour scenario
- Experiment 3: Trust Fall — cheater detection, subtle cheater analysis
- Basic visualization (matplotlib)
- README with findings

### In Progress
- Phase A: Statistical rigor (confidence intervals, 1000-trial runs, formal metrics)
- Phase B: LLM agent layer (scaffolding without API key)
- Phase C: Deep cheater analysis (detection thresholds, collaborative reputation)
- Phase D: Futures/predictive negotiation market

### Not Yet Started
- Phase E: Coalition formation
- Phase F: Paper writing
- Real resource allocation (Docker containers, GPU slices)

## Key Findings So Far

1. **Strategy deadlocks**: Greedy and Patient strategies cannot close deals with anyone. Only Fair and Adaptive successfully negotiate.
2. **Price convergence**: Two Adaptive agents with different starting beliefs converge to stable price within ~30 rounds. Volatility drops to near-zero.
3. **Fair dominates tournaments**: Fair providers accumulate the most wealth in 20-trial tournaments — they close more deals than greedy ones.
4. **Always-cheaters get caught fast** (~15 rounds) but still profit because they steal aggressively before isolation.
5. **CRITICAL FINDING: 5% cheaters are undetectable** — their reputation *improves* over time because successful deals outweigh rare defaults.

## Research Direction

**Core question:** How do different agent intelligence tiers (rule-based, RL, LLM) affect negotiation outcomes, market efficiency, and stability in decentralized compute markets?

**Target venues:** AAMAS, AAAI, JASSS

**Related work to cite:** Rubinstein (1982), NegotiationArena (2024), AgenticPay (2025), Game-theoretic LLM (2024), x402-RAM, FIRE trust model. See docs/PROGRESS.md for full bibliography.

## Running Experiments

```bash
pip install matplotlib
python run_all.py           # all experiments
python run_all.py 1         # individual experiment (1-7)
```

For LLM experiments (Phase B):
```bash
export ANTHROPIC_API_KEY=your_key_here
python run_all.py 5
```

## Code Conventions

- Python 3.10+, type hints everywhere
- `from __future__ import annotations` for forward refs
- Dataclasses over dicts for structured data
- No external dependencies beyond matplotlib and anthropic SDK
- Experiments are self-contained scripts with `if __name__ == "__main__"` blocks
- All randomness goes through `random.seed()` for reproducibility
- Log everything to `logs/` directory

## For Future Agents

If picking up this project:
1. Read this file and `docs/PROGRESS.md` first
2. Check the todo state in the current experiment files
3. The `agents/` package is stable — extend it, don't rewrite
4. New strategies go in `agents/strategies.py` (or new files for complex ones like `agents/llm_strategy.py`)
5. New experiments go in `experiments/exp{N}_{name}.py` and get wired into `run_all.py`
6. Run `python run_all.py` to verify nothing is broken before pushing
7. Statistical claims need 1000+ trials with confidence intervals (see `agents/stats.py`)
