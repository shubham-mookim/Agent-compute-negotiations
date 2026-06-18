#!/usr/bin/env python3
"""
Experiment 14: Scaling Analysis

Addresses the "small populations" threat to validity. All prior experiments
use 2-12 agents. Here we sweep population size from 4 to 48 agents and test
whether the key qualitative findings survive:

  - Does Fair-strategy dominance hold as N grows?
  - Does the strategy-deadlock structure persist (only some pairs trade)?
  - How does market efficiency (realized vs possible deals) scale?
  - How does wealth inequality (Gini) evolve with population size?
  - Does price-per-unit stabilize or drift with more participants?

Pure simulation — costs $0. Each population is a balanced mix of the four
canonical Tier-1 strategies, split evenly into seekers (buyers) and providers
(sellers). We run many independent trials per size for confidence intervals.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    Agent, Resource,
    GreedyStrategy, FairStrategy, PatientStrategy, AdaptiveStrategy,
)
from agents.simulator import Simulator
from agents.stats import describe, gini_coefficient, welch_t_test, cohens_d

RESULTS_DIR = Path(__file__).parent.parent / "results"

STRATEGY_CYCLE = ["greedy", "fair", "patient", "adaptive"]


def _make_strategy(name: str):
    return {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "patient": lambda: PatientStrategy(),
        "adaptive": lambda: AdaptiveStrategy(price_belief=1.0),
    }[name]()


def build_population(n_agents: int):
    """
    Half seekers (buyers, low resources, high budget), half providers
    (sellers, high resources, low urgency). Strategies cycle through the
    four canonical Tier-1 strategies so each is equally represented.
    """
    agents = []
    n_seekers = n_agents // 2
    n_providers = n_agents - n_seekers

    for i in range(n_seekers):
        strat = STRATEGY_CYCLE[i % len(STRATEGY_CYCLE)]
        agents.append(Agent(
            agent_id=f"seeker_{strat}_{i}",
            resources=Resource(gpu_hours=5),
            budget=100.0,
            strategy=_make_strategy(strat),
            urgency=0.8,
        ))
    for i in range(n_providers):
        strat = STRATEGY_CYCLE[i % len(STRATEGY_CYCLE)]
        agents.append(Agent(
            agent_id=f"provider_{strat}_{i}",
            resources=Resource(gpu_hours=100, cpu_hours=50),
            budget=10.0,
            strategy=_make_strategy(strat),
            urgency=0.15,
        ))
    return agents, [a.agent_id for a in agents if a.agent_id.startswith("seeker")]


def run_one_trial(n_agents: int, rounds: int, seed: int):
    """Run a single population trial and return aggregate metrics."""
    agents, seeker_ids = build_population(n_agents)
    sim = Simulator(agents, max_negotiation_turns=6, seed=seed)

    # Record starting net worth to compute deltas
    start_worth = {a.agent_id: a.net_worth() for a in agents}

    deals = 0
    negotiations = 0
    prices = []
    for _ in range(rounds):
        needs = {sid: Resource(gpu_hours=5, cpu_hours=3) for sid in seeker_ids}
        for sid in seeker_ids:
            sim.agents[sid].pending_needs = needs[sid]
        m = sim.run_round(needs=needs)
        negotiations += m.negotiations
        deals += m.deals_made
        if m.avg_price_per_unit > 0:
            prices.append(m.avg_price_per_unit)

    # Per-strategy aggregation
    strat_wealth = defaultdict(list)
    strat_utility = defaultdict(list)
    for a in agents:
        strat = a.agent_id.split("_")[1]
        strat_wealth[strat].append(a.net_worth() - start_worth[a.agent_id])
        strat_utility[strat].append(a.utility())

    worths = [a.net_worth() for a in agents]
    utilities = [a.utility() for a in agents if a.agent_id.startswith("seeker")]

    return {
        "deal_rate": deals / negotiations if negotiations else 0.0,
        "gini_wealth": gini_coefficient(worths),
        "gini_utility": gini_coefficient(utilities) if utilities else 0.0,
        "avg_price": statistics.mean(prices) if prices else 0.0,
        "strat_wealth": {k: statistics.mean(v) for k, v in strat_wealth.items()},
        "strat_utility": {k: statistics.mean(v) for k, v in strat_utility.items()},
    }


def run_scaling(sizes=(4, 6, 10, 16, 24, 40), rounds=50, trials=30):
    print("=" * 64)
    print("  EXPERIMENT 14: SCALING ANALYSIS")
    print("=" * 64)
    print(f"\nPopulation sizes: {list(sizes)}")
    print(f"Rounds/trial: {rounds}, trials/size: {trials}\n")

    per_size = {}
    for n in sizes:
        trial_metrics = [run_one_trial(n, rounds, seed=t) for t in range(trials)]

        deal_rates = [m["deal_rate"] for m in trial_metrics]
        ginis_w = [m["gini_wealth"] for m in trial_metrics]
        ginis_u = [m["gini_utility"] for m in trial_metrics]
        prices = [m["avg_price"] for m in trial_metrics if m["avg_price"] > 0]

        # Per-strategy averages across trials
        strat_wealth = defaultdict(list)
        strat_util = defaultdict(list)
        for m in trial_metrics:
            for k, v in m["strat_wealth"].items():
                strat_wealth[k].append(v)
            for k, v in m["strat_utility"].items():
                strat_util[k].append(v)

        per_size[n] = {
            "deal_rate": describe(deal_rates),
            "gini_wealth": describe(ginis_w),
            "gini_utility": describe(ginis_u),
            "avg_price": describe(prices) if prices else None,
            "strat_wealth": {k: statistics.mean(v) for k, v in strat_wealth.items()},
            "strat_utility": {k: statistics.mean(v) for k, v in strat_util.items()},
        }

    # ── Report: aggregate metrics vs N ────────────────────────────────────
    print(f"{'N':>4} {'DealRate':>18} {'GiniWealth':>18} {'AvgPrice':>16}")
    print("-" * 60)
    for n in sizes:
        d = per_size[n]
        dr = d["deal_rate"]
        gw = d["gini_wealth"]
        ap = d["avg_price"]
        ap_str = f"{ap.mean:.3f}±{ap.std:.3f}" if ap else "n/a"
        print(f"{n:>4} {dr.mean:>10.3f}±{dr.std:.3f} "
              f"{gw.mean:>10.4f}±{gw.std:.4f} {ap_str:>16}")

    # ── Report: per-strategy wealth delta vs N ────────────────────────────
    print(f"\n--- Per-strategy wealth Δ vs population size ---")
    print(f"{'N':>4} " + " ".join(f"{s:>10}" for s in STRATEGY_CYCLE))
    print("-" * 50)
    for n in sizes:
        sw = per_size[n]["strat_wealth"]
        print(f"{n:>4} " + " ".join(f"{sw.get(s, 0):>10.1f}" for s in STRATEGY_CYCLE))

    # ── Report: per-strategy utility vs N ─────────────────────────────────
    print(f"\n--- Per-strategy utility vs population size ---")
    print(f"{'N':>4} " + " ".join(f"{s:>10}" for s in STRATEGY_CYCLE))
    print("-" * 50)
    for n in sizes:
        su = per_size[n]["strat_utility"]
        print(f"{n:>4} " + " ".join(f"{su.get(s, 0):>10.3f}" for s in STRATEGY_CYCLE))

    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n--- Analysis ---")
    small_dr = per_size[sizes[0]]["deal_rate"].mean
    large_dr = per_size[sizes[-1]]["deal_rate"].mean
    print(f"  Deal rate {sizes[0]}→{sizes[-1]} agents: "
          f"{small_dr:.3f} → {large_dr:.3f} ({(large_dr/small_dr-1)*100:+.0f}%)")
    small_g = per_size[sizes[0]]["gini_wealth"].mean
    large_g = per_size[sizes[-1]]["gini_wealth"].mean
    print(f"  Gini {sizes[0]}→{sizes[-1]} agents: {small_g:.4f} → {large_g:.4f}")

    # Does Fair stay on top in utility at the largest size?
    largest = per_size[sizes[-1]]["strat_utility"]
    ranked = sorted(largest.items(), key=lambda x: -x[1])
    print(f"  Strategy utility ranking at N={sizes[-1]}: "
          + " > ".join(f"{k}({v:.3f})" for k, v in ranked))

    # ── Dump ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "exp14_scaling.json"
    serializable = {}
    for n, d in per_size.items():
        serializable[str(n)] = {
            "deal_rate": {"mean": d["deal_rate"].mean, "std": d["deal_rate"].std,
                          "ci": [d["deal_rate"].ci_lower, d["deal_rate"].ci_upper]},
            "gini_wealth": {"mean": d["gini_wealth"].mean, "std": d["gini_wealth"].std,
                            "ci": [d["gini_wealth"].ci_lower, d["gini_wealth"].ci_upper]},
            "gini_utility": {"mean": d["gini_utility"].mean, "std": d["gini_utility"].std},
            "avg_price": ({"mean": d["avg_price"].mean, "std": d["avg_price"].std}
                          if d["avg_price"] else None),
            "strat_wealth": d["strat_wealth"],
            "strat_utility": d["strat_utility"],
        }
    with open(out, "w") as f:
        json.dump({"config": {"sizes": list(sizes), "rounds": rounds, "trials": trials},
                   "results": serializable}, f, indent=2)
    print(f"\n  Results dumped to: {out}")

    return per_size


if __name__ == "__main__":
    run_scaling()
