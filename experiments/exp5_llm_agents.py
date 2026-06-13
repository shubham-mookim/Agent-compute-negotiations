#!/usr/bin/env python3
"""
Experiment 5: LLM Agent Negotiations

Tests LLM-powered agents (OpenAI or Anthropic) against rule-based agents.
Requires either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.

Key experiments:
1. LLM vs LLM: Do they find equilibrium?
2. LLM vs Rule-based: Who exploits whom?
3. Mixed populations: What dynamics emerge?

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
    if os.environ.get("OPENAI_API_KEY"):
        print("OpenAI API key found. Running with GPT-powered agents.\n")
        return True
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Anthropic API key found. Running with Claude-powered agents.\n")
        return True
    print("WARNING: No API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY).")
    print("Running in FALLBACK mode (rule-based with LLM interface).")
    print("Set a key to run actual LLM experiments.\n")
    return False


def _collect_usage(strategies: list[LLMStrategy]) -> dict:
    total_calls = sum(s.api_calls for s in strategies)
    total_in = sum(s.total_input_tokens for s in strategies)
    total_out = sum(s.total_output_tokens for s in strategies)
    cost_in = total_in * 0.15 / 1_000_000
    cost_out = total_out * 0.60 / 1_000_000
    return {
        "calls": total_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost": cost_in + cost_out,
    }


def llm_vs_llm(num_trials=10, seed_offset=0):
    """Two LLM agents negotiate. Do they find fair prices?"""
    print("=== LLM vs LLM ===\n")

    prices = []
    deal_count = 0
    all_strategies = []

    for trial in range(num_trials):
        buyer_strat = LLMStrategy(temperature=0.5)
        seller_strat = LLMStrategy(temperature=0.5)
        all_strategies.extend([buyer_strat, seller_strat])

        buyer = Agent(
            agent_id="llm_buyer",
            resources=Resource(),
            budget=50.0,
            strategy=buyer_strat,
            urgency=0.7,
            pending_needs=Resource(gpu_hours=10),
        )
        seller = Agent(
            agent_id="llm_seller",
            resources=Resource(gpu_hours=50),
            budget=10.0,
            strategy=seller_strat,
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
        stat = describe(prices)
        print(f"Prices: {stat}")
        fair_price = 10.0  # 10 GPU hours = 10 units
        avg = sum(prices) / len(prices)
        premium = (avg - fair_price) / fair_price * 100
        print(f"Fair price: {fair_price:.1f}, Avg LLM price: {avg:.2f} ({premium:+.1f}%)")

    usage = _collect_usage(all_strategies)
    print(f"API usage: {usage['calls']} calls, {usage['input_tokens']} in / {usage['output_tokens']} out, est. ${usage['cost']:.4f}")
    return prices, usage


def llm_vs_rule_based(num_trials=10, seed_offset=0):
    """LLM agent vs each rule-based strategy."""
    print("\n=== LLM vs Rule-Based Strategies ===\n")

    strategies = {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "adaptive": lambda: AdaptiveStrategy(),
    }

    results = {}
    all_strategies = []

    for strat_name, strat_factory in strategies.items():
        prices_llm_buys = []
        prices_llm_sells = []

        for trial in range(num_trials):
            buyer_strat = LLMStrategy(temperature=0.5)
            all_strategies.append(buyer_strat)
            buyer = Agent(
                "llm_buyer", Resource(), 50.0, buyer_strat,
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

        for trial in range(num_trials):
            seller_strat = LLMStrategy(temperature=0.5)
            all_strategies.append(seller_strat)
            buyer = Agent(
                f"{strat_name}_buyer", Resource(), 50.0, strat_factory(),
                urgency=0.7, pending_needs=Resource(gpu_hours=10),
            )
            seller = Agent(
                "llm_seller", Resource(gpu_hours=50), 10.0,
                seller_strat, urgency=0.2,
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
            avg_buy = sum(prices_llm_buys) / len(prices_llm_buys)
            print(f", avg price {avg_buy:.2f}")
        else:
            print()
        print(f"    LLM selling: {sell_rate:.0%} deals", end="")
        if prices_llm_sells:
            avg_sell = sum(prices_llm_sells) / len(prices_llm_sells)
            print(f", avg price {avg_sell:.2f}")
        else:
            print()

    usage = _collect_usage(all_strategies)
    print(f"\nAPI usage: {usage['calls']} calls, {usage['input_tokens']} in / {usage['output_tokens']} out, est. ${usage['cost']:.4f}")
    return results, usage


def mixed_population(num_trials=3, rounds=20, seed_offset=0):
    """5 agents: 2 LLM + 3 rule-based. Track wealth over time."""
    print("\n=== Mixed Population (2 LLM + 3 Rule-Based) ===\n")

    llm_wealth = []
    rule_wealth = []
    all_strategies = []

    for trial in range(num_trials):
        llm_strat_1 = LLMStrategy(temperature=0.5)
        llm_strat_2 = LLMStrategy(temperature=0.5)
        all_strategies.extend([llm_strat_1, llm_strat_2])

        agents = [
            Agent("llm_seeker", Resource(gpu_hours=5), 100.0, llm_strat_1, 0.7),
            Agent("llm_provider", Resource(gpu_hours=100), 20.0, llm_strat_2, 0.2),
            Agent("fair_provider", Resource(gpu_hours=80), 25.0, FairStrategy(), 0.2),
            Agent("adaptive_seeker", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.7),
            Agent("greedy_provider", Resource(gpu_hours=90), 15.0, GreedyStrategy(), 0.1),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial + seed_offset)

        for _ in range(rounds):
            sim.run_round(needs={
                "llm_seeker": Resource(gpu_hours=5),
                "adaptive_seeker": Resource(gpu_hours=5),
            })

        for agent in sim.agents.values():
            nw = agent.net_worth()
            if "llm" in agent.agent_id:
                llm_wealth.append(nw)
            else:
                rule_wealth.append(nw)

        print(f"  Trial {trial}:")
        for agent in sim.agents.values():
            print(f"    {agent.agent_id}: wealth={agent.net_worth():.1f}, deals={len(agent.deals)}")

    print(f"\nLLM agents wealth:       {describe(llm_wealth)}")
    print(f"Rule-based agents wealth: {describe(rule_wealth)}")
    if len(llm_wealth) >= 2 and len(rule_wealth) >= 2:
        t, p = welch_t_test(llm_wealth, rule_wealth)
        d = cohens_d(llm_wealth, rule_wealth)
        print(f"Difference: t={t:.3f}, p={p:.2e}, d={d:.2f}")

    usage = _collect_usage(all_strategies)
    print(f"\nAPI usage: {usage['calls']} calls, {usage['input_tokens']} in / {usage['output_tokens']} out, est. ${usage['cost']:.4f}")
    return {"llm": llm_wealth, "rule": rule_wealth}, usage


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 5: LLM AGENT NEGOTIATIONS")
    print("=" * 60)
    print()

    api_ok = check_api_available()
    num = 10 if api_ok else 5

    llm_vs_llm(num_trials=num)
    llm_vs_rule_based(num_trials=num)
    mixed_population(num_trials=3, rounds=20)
