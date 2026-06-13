#!/usr/bin/env python3
"""
Experiment 5: LLM Agent Negotiations

Tests Claude-powered agents against rule-based agents.
Requires ANTHROPIC_API_KEY environment variable.

Key experiments:
1. LLM vs LLM: Do they find equilibrium?
2. LLM vs Rule-based: Who exploits whom?
3. Mixed populations: What dynamics emerge?
4. Incomplete info: Can LLMs handle private information?
5. Bluffing: Can an LLM bluff about urgency? Can another detect it?

Without API key, runs in fallback mode (rule-based with same interface).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Resource, FairStrategy, AdaptiveStrategy, GreedyStrategy
from agents.llm_strategy import LLMStrategy
from agents.simulator import Simulator
from agents.stats import describe, welch_t_test, cohens_d


def check_api_available() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("WARNING: ANTHROPIC_API_KEY not set.")
        print("Running in FALLBACK mode (rule-based with LLM interface).")
        print("Set the key to run actual LLM experiments.\n")
        return False
    print("API key found. Running with Claude-powered agents.\n")
    return True


def llm_vs_llm(num_trials=10, seed_offset=0):
    """Two LLM agents negotiate. Do they find fair prices?"""
    print("=== LLM vs LLM ===\n")

    prices = []
    deal_count = 0

    for trial in range(num_trials):
        buyer = Agent(
            agent_id="llm_buyer",
            resources=Resource(),
            budget=50.0,
            strategy=LLMStrategy(temperature=0.5),
            urgency=0.7,
            pending_needs=Resource(gpu_hours=10),
        )
        seller = Agent(
            agent_id="llm_seller",
            resources=Resource(gpu_hours=50),
            budget=10.0,
            strategy=LLMStrategy(temperature=0.5),
            urgency=0.2,
        )
        sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial + seed_offset)
        sim.run_round(pairings=[("llm_buyer", "llm_seller")])

        result = sim.results[0]
        if result.agreed:
            deal_count += 1
            prices.append(result.price)
            print(f"  Trial {trial}: DEAL at price {result.price:.2f} ({result.rounds} rounds)")
        else:
            print(f"  Trial {trial}: NO DEAL ({result.rounds} rounds)")

    print(f"\nDeals: {deal_count}/{num_trials}")
    if prices:
        print(f"Prices: {describe(prices)}")
    return prices


def llm_vs_rule_based(num_trials=10, seed_offset=0):
    """LLM agent vs each rule-based strategy."""
    print("\n=== LLM vs Rule-Based Strategies ===\n")

    strategies = {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "adaptive": lambda: AdaptiveStrategy(),
    }

    results = {}
    for strat_name, strat_factory in strategies.items():
        prices_llm_buys = []
        prices_llm_sells = []

        # LLM as buyer
        for trial in range(num_trials):
            buyer = Agent(
                "llm_buyer", Resource(), 50.0, LLMStrategy(temperature=0.5),
                urgency=0.7, pending_needs=Resource(gpu_hours=10),
            )
            seller = Agent(
                f"{strat_name}_seller", Resource(gpu_hours=50), 10.0,
                strat_factory(), urgency=0.2,
            )
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial + seed_offset)
            sim.run_round(pairings=[("llm_buyer", f"{strat_name}_seller")])
            if sim.results[0].agreed:
                prices_llm_buys.append(sim.results[0].price)

        # LLM as seller
        for trial in range(num_trials):
            buyer = Agent(
                f"{strat_name}_buyer", Resource(), 50.0, strat_factory(),
                urgency=0.7, pending_needs=Resource(gpu_hours=10),
            )
            seller = Agent(
                "llm_seller", Resource(gpu_hours=50), 10.0,
                LLMStrategy(temperature=0.5), urgency=0.2,
            )
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial + seed_offset + 1000)
            sim.run_round(pairings=[(f"{strat_name}_buyer", "llm_seller")])
            if sim.results[0].agreed:
                prices_llm_sells.append(sim.results[0].price)

        results[strat_name] = {
            "llm_buys": prices_llm_buys,
            "llm_sells": prices_llm_sells,
        }

        buy_rate = len(prices_llm_buys) / num_trials
        sell_rate = len(prices_llm_sells) / num_trials
        print(f"  vs {strat_name}:")
        print(f"    LLM buying:  {buy_rate:.0%} deals", end="")
        if prices_llm_buys:
            print(f", avg price {sum(prices_llm_buys)/len(prices_llm_buys):.2f}")
        else:
            print()
        print(f"    LLM selling: {sell_rate:.0%} deals", end="")
        if prices_llm_sells:
            print(f", avg price {sum(prices_llm_sells)/len(prices_llm_sells):.2f}")
        else:
            print()

    return results


def mixed_population(num_trials=5, rounds=50, seed_offset=0):
    """5 agents: 2 LLM + 3 rule-based. Track wealth over time."""
    print("\n=== Mixed Population (2 LLM + 3 Rule-Based) ===\n")

    llm_wealth = []
    rule_wealth = []

    for trial in range(num_trials):
        agents = [
            Agent("llm_seeker", Resource(gpu_hours=5), 100.0, LLMStrategy(temperature=0.5), 0.7),
            Agent("llm_provider", Resource(gpu_hours=100), 20.0, LLMStrategy(temperature=0.5), 0.2),
            Agent("fair_provider", Resource(gpu_hours=80), 25.0, FairStrategy(), 0.2),
            Agent("adaptive_seeker", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.7),
            Agent("greedy_provider", Resource(gpu_hours=90), 15.0, GreedyStrategy(), 0.1),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial + seed_offset)

        for _ in range(rounds):
            for sid in ["llm_seeker", "adaptive_seeker"]:
                sim.agents[sid].pending_needs = Resource(gpu_hours=5)
            sim.run_round(needs={
                "llm_seeker": Resource(gpu_hours=5),
                "adaptive_seeker": Resource(gpu_hours=5),
            })

        for agent in sim.agents.values():
            if "llm" in agent.agent_id:
                llm_wealth.append(agent.net_worth())
            else:
                rule_wealth.append(agent.net_worth())

    print(f"LLM agents wealth:       {describe(llm_wealth)}")
    print(f"Rule-based agents wealth: {describe(rule_wealth)}")
    if len(llm_wealth) >= 2 and len(rule_wealth) >= 2:
        t, p = welch_t_test(llm_wealth, rule_wealth)
        print(f"Difference: t={t:.3f}, p={p:.2e}")

    return {"llm": llm_wealth, "rule": rule_wealth}


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 5: LLM AGENT NEGOTIATIONS")
    print("=" * 60)
    print()

    api_ok = check_api_available()
    num = 20 if api_ok else 10

    llm_vs_llm(num_trials=num)
    llm_vs_rule_based(num_trials=num)
    mixed_population(num_trials=5 if api_ok else 3)
