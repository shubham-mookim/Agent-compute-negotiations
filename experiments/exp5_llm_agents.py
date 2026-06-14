#!/usr/bin/env python3
"""
Experiment 5: LLM Agent Negotiations (Expanded)

Tests LLM-powered agents (OpenAI or Anthropic) with two prompt styles:
  - NAIVE: minimal instructions, tests raw LLM ability
  - ENGINEERED: production-grade prompt with pricing framework

Key experiments:
1. Prompt quality comparison: does a better prompt produce better outcomes?
2. LLM vs LLM: do they find equilibrium?
3. LLM vs Rule-based: who exploits whom?
4. Mixed populations: what dynamics emerge?
5. Scaled trials (30+) for statistical validity

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
    print("Running in FALLBACK mode (rule-based with LLM interface).\n")
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


# ── Part 1: Prompt Quality Comparison ─────────────────────────────────────

def prompt_comparison(num_trials=15):
    """Compare naive vs engineered prompts head-to-head."""
    print("=== Part 1: Prompt Quality Comparison ===\n")

    results = {"naive": {"prices": [], "deals": 0, "utilities": []},
               "engineered": {"prices": [], "deals": 0, "utilities": []}}
    all_strategies = []

    for style in ["naive", "engineered"]:
        for trial in range(num_trials):
            buyer_strat = LLMStrategy(temperature=0.5, prompt_style=style)
            all_strategies.append(buyer_strat)
            buyer = Agent(
                "llm_buyer", Resource(), 50.0, buyer_strat,
                urgency=0.7, pending_needs=Resource(gpu_hours=10),
            )
            seller = Agent(
                "fair_seller", Resource(gpu_hours=50), 10.0, FairStrategy(), 0.2,
            )
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
            sim.run_round(pairings=[("llm_buyer", "fair_seller")])

            if sim.results[0].agreed:
                results[style]["deals"] += 1
                results[style]["prices"].append(sim.results[0].price)
            results[style]["utilities"].append(buyer.utility())

    for style in ["naive", "engineered"]:
        r = results[style]
        dr = r["deals"] / num_trials
        print(f"  {style.upper():>12}:")
        print(f"    Deal rate: {dr:.0%} ({r['deals']}/{num_trials})")
        if r["prices"]:
            ps = describe(r["prices"])
            print(f"    Avg price: {ps.mean:.2f} (fair=10.0, deviation={((ps.mean/10)-1)*100:+.1f}%)")
        if r["utilities"]:
            us = describe(r["utilities"])
            print(f"    Utility:   {us.mean:.4f}")

    # Compare
    if results["naive"]["prices"] and results["engineered"]["prices"]:
        t, p = welch_t_test(results["naive"]["prices"], results["engineered"]["prices"])
        d = cohens_d(results["naive"]["prices"], results["engineered"]["prices"])
        print(f"\n  Price difference: t={t:.2f}, p={p:.2e}, d={d:.2f}")

    usage = _collect_usage(all_strategies)
    print(f"\n  API: {usage['calls']} calls, est. ${usage['cost']:.4f}")
    return results


# ── Part 2: LLM vs LLM ───────────────────────────────────────────────────

def llm_vs_llm(num_trials=15):
    """Two LLM agents negotiate with engineered prompts."""
    print("\n=== Part 2: LLM vs LLM (Engineered Prompts) ===\n")

    prices = []
    rounds_to_deal = []
    all_strategies = []

    for trial in range(num_trials):
        buyer_strat = LLMStrategy(temperature=0.5, prompt_style="engineered")
        seller_strat = LLMStrategy(temperature=0.5, prompt_style="engineered")
        all_strategies.extend([buyer_strat, seller_strat])

        buyer = Agent("llm_buyer", Resource(), 50.0, buyer_strat,
                       urgency=0.7, pending_needs=Resource(gpu_hours=10))
        seller = Agent("llm_seller", Resource(gpu_hours=50), 10.0,
                       seller_strat, urgency=0.2)
        sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
        sim.run_round(pairings=[("llm_buyer", "llm_seller")])

        result = sim.results[0]
        if result.agreed:
            prices.append(result.price)
            rounds_to_deal.append(result.rounds)
            print(f"  Trial {trial}: DEAL at {result.price:.2f} in {result.rounds} rounds")
        else:
            print(f"  Trial {trial}: NO DEAL ({result.rounds} rounds)")

    deal_rate = len(prices) / num_trials
    print(f"\n  Deal rate: {deal_rate:.0%}")
    if prices:
        stat = describe(prices)
        print(f"  Prices: {stat}")
        avg_rounds = sum(rounds_to_deal) / len(rounds_to_deal)
        print(f"  Avg rounds to deal: {avg_rounds:.1f}")
        deviation = (stat.mean - 10.0) / 10.0 * 100
        print(f"  vs fair price: {deviation:+.1f}%")
        rubinstein = 3.871  # from exp10
        deviation_r = (stat.mean - rubinstein) / rubinstein * 100
        print(f"  vs Rubinstein equilibrium (3.87): {deviation_r:+.1f}%")

    usage = _collect_usage(all_strategies)
    print(f"  API: {usage['calls']} calls, est. ${usage['cost']:.4f}")
    return prices


# ── Part 3: LLM vs Rule-Based ────────────────────────────────────────────

def llm_vs_rule_based(num_trials=15):
    """Engineered LLM vs each rule-based strategy."""
    print("\n=== Part 3: LLM vs Rule-Based (Engineered Prompts) ===\n")

    strategies = {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "adaptive": lambda: AdaptiveStrategy(),
    }

    all_strategies = []
    results = {}

    for strat_name, strat_factory in strategies.items():
        prices_buy = []
        prices_sell = []
        utilities_buy = []
        utilities_sell = []

        # LLM as buyer
        for trial in range(num_trials):
            buyer_strat = LLMStrategy(temperature=0.5, prompt_style="engineered")
            all_strategies.append(buyer_strat)
            buyer = Agent("llm_buyer", Resource(), 50.0, buyer_strat,
                          urgency=0.7, pending_needs=Resource(gpu_hours=10))
            seller = Agent(f"{strat_name}_seller", Resource(gpu_hours=50), 10.0,
                           strat_factory(), urgency=0.2)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
            sim.run_round(pairings=[("llm_buyer", f"{strat_name}_seller")])
            if sim.results[0].agreed:
                prices_buy.append(sim.results[0].price)
            utilities_buy.append(buyer.utility())

        # LLM as seller
        for trial in range(num_trials):
            seller_strat = LLMStrategy(temperature=0.5, prompt_style="engineered")
            all_strategies.append(seller_strat)
            buyer = Agent(f"{strat_name}_buyer", Resource(), 50.0, strat_factory(),
                          urgency=0.7, pending_needs=Resource(gpu_hours=10))
            seller = Agent("llm_seller", Resource(gpu_hours=50), 10.0,
                           seller_strat, urgency=0.2)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial + 1000)
            sim.run_round(pairings=[(f"{strat_name}_buyer", "llm_seller")])
            if sim.results[0].agreed:
                prices_sell.append(sim.results[0].price)
            utilities_sell.append(seller.utility())

        buy_rate = len(prices_buy) / num_trials
        sell_rate = len(prices_sell) / num_trials
        results[strat_name] = {
            "buy_prices": prices_buy, "sell_prices": prices_sell,
            "buy_rate": buy_rate, "sell_rate": sell_rate,
            "buy_utility": utilities_buy, "sell_utility": utilities_sell,
        }

        print(f"  vs {strat_name}:")
        print(f"    LLM buying:  {buy_rate:.0%} deals", end="")
        if prices_buy:
            avg = sum(prices_buy) / len(prices_buy)
            print(f", avg price {avg:.2f} (vs fair 10.0: {(avg/10-1)*100:+.1f}%)")
        else:
            print()
        print(f"    LLM selling: {sell_rate:.0%} deals", end="")
        if prices_sell:
            avg = sum(prices_sell) / len(prices_sell)
            print(f", avg price {avg:.2f}")
        else:
            print()

    usage = _collect_usage(all_strategies)
    print(f"\n  API: {usage['calls']} calls, est. ${usage['cost']:.4f}")
    return results


# ── Part 4: Mixed Population ─────────────────────────────────────────────

def mixed_population(num_trials=3, rounds=20):
    """5 agents: 2 LLM + 3 rule-based."""
    print("\n=== Part 4: Mixed Population (2 LLM + 3 Rule-Based) ===\n")

    llm_wealth = []
    rule_wealth = []
    llm_utility = []
    rule_utility = []
    all_strategies = []

    for trial in range(num_trials):
        s1 = LLMStrategy(temperature=0.5, prompt_style="engineered")
        s2 = LLMStrategy(temperature=0.5, prompt_style="engineered")
        all_strategies.extend([s1, s2])

        agents = [
            Agent("llm_seeker", Resource(gpu_hours=5), 100.0, s1, 0.7),
            Agent("llm_provider", Resource(gpu_hours=100), 20.0, s2, 0.2),
            Agent("fair_provider", Resource(gpu_hours=80), 25.0, FairStrategy(), 0.2),
            Agent("adaptive_seeker", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.7),
            Agent("greedy_provider", Resource(gpu_hours=90), 15.0, GreedyStrategy(), 0.1),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial)

        for _ in range(rounds):
            sim.run_round(needs={
                "llm_seeker": Resource(gpu_hours=5),
                "adaptive_seeker": Resource(gpu_hours=5),
            })

        print(f"  Trial {trial}:")
        for agent in sim.agents.values():
            nw = agent.net_worth()
            u = agent.utility()
            is_llm = "llm" in agent.agent_id
            if is_llm:
                llm_wealth.append(nw)
                llm_utility.append(u)
            else:
                rule_wealth.append(nw)
                rule_utility.append(u)
            print(f"    {agent.agent_id}: wealth={nw:.1f}, utility={u:.4f}, deals={len(agent.deals)}")

    print(f"\nLLM wealth:       {describe(llm_wealth)}")
    print(f"Rule-based wealth: {describe(rule_wealth)}")
    if len(llm_utility) >= 2 and len(rule_utility) >= 2:
        t, p = welch_t_test(llm_utility, rule_utility)
        d = cohens_d(llm_utility, rule_utility)
        print(f"Utility difference: t={t:.3f}, p={p:.2e}, d={d:.2f}")

    usage = _collect_usage(all_strategies)
    print(f"\n  API: {usage['calls']} calls, est. ${usage['cost']:.4f}")
    return {"llm_wealth": llm_wealth, "rule_wealth": rule_wealth,
            "llm_utility": llm_utility, "rule_utility": rule_utility}


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 5: LLM AGENT NEGOTIATIONS (EXPANDED)")
    print("=" * 60)
    print()

    api_ok = check_api_available()
    n = 15 if api_ok else 5

    prompt_comparison(num_trials=n)
    llm_vs_llm(num_trials=n)
    llm_vs_rule_based(num_trials=n)
    mixed_population(num_trials=3, rounds=20)
