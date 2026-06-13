#!/usr/bin/env python3
"""
Experiment 4: Statistical Rigor

Re-runs experiments 1-3 with 1000 trials each, proper confidence
intervals, effect sizes, and significance tests.

This turns our initial observations into defensible claims.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    Agent, Resource,
    GreedyStrategy, FairStrategy, PatientStrategy, AdaptiveStrategy,
)
from agents.simulator import Simulator
from agents.stats import describe, cohens_d, welch_t_test, gini_coefficient, bootstrap_ci


NUM_TRIALS = 1000
NUM_ROUNDS_PER_TRIAL = 100


def stat_strategy_matrix(num_trials=NUM_TRIALS):
    """Rigorous strategy matrix with CIs."""
    print(f"=== Strategy Matrix ({num_trials} trials per pair) ===\n")

    strategies = {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "patient": lambda: PatientStrategy(),
        "adaptive": lambda: AdaptiveStrategy(),
    }

    results = {}
    for buyer_name, buyer_factory in strategies.items():
        for seller_name, seller_factory in strategies.items():
            successes = []
            prices = []
            rounds_to_deal = []

            for seed in range(num_trials):
                buyer = Agent(
                    agent_id="buyer", resources=Resource(),
                    budget=20.0, strategy=buyer_factory(),
                    urgency=0.7, pending_needs=Resource(gpu_hours=10),
                )
                seller = Agent(
                    agent_id="seller", resources=Resource(gpu_hours=50),
                    budget=5.0, strategy=seller_factory(), urgency=0.2,
                )
                sim = Simulator([buyer, seller], max_negotiation_turns=8, seed=seed)
                sim.run_round(pairings=[("buyer", "seller")])

                result = sim.results[0]
                successes.append(1.0 if result.agreed else 0.0)
                if result.agreed:
                    prices.append(result.price)
                    rounds_to_deal.append(result.rounds)

            rate = describe(successes)
            results[(buyer_name, seller_name)] = {
                "success_rate": rate,
                "prices": describe(prices) if prices else None,
                "rounds": describe(rounds_to_deal) if rounds_to_deal else None,
            }

    # Print matrix
    names = list(strategies.keys())
    label = "buyer \\ seller"
    header = f"{label:<16}" + "".join(f"{n:<20}" for n in names)
    print(header)
    print("-" * len(header))
    for bn in names:
        row = f"{bn:<16}"
        for sn in names:
            r = results[(bn, sn)]
            rate = r["success_rate"]
            row += f"{rate.mean:>5.0%} ±{rate.ci_upper - rate.mean:>4.1%}       "
        print(row)

    print("\n--- Price Analysis (where deals occur) ---\n")
    for (bn, sn), r in results.items():
        if r["prices"] and r["prices"].n > 0:
            p = r["prices"]
            rd = r["rounds"]
            print(f"  {bn} vs {sn}:")
            print(f"    Price: {p}")
            print(f"    Rounds to deal: {rd}")

    return results


def stat_convergence(num_trials=200):
    """
    Statistical test: do Adaptive agents' prices converge?
    Compare volatility of first 20 vs last 20 deals across many trials.
    """
    print(f"\n=== Price Convergence Test ({num_trials} trials) ===\n")

    early_vols = []
    late_vols = []
    final_prices = []

    for seed in range(num_trials):
        buyer = Agent(
            agent_id="buyer", resources=Resource(),
            budget=200.0, strategy=AdaptiveStrategy(price_belief=0.5),
            urgency=0.6,
        )
        seller = Agent(
            agent_id="seller", resources=Resource(gpu_hours=500),
            budget=10.0, strategy=AdaptiveStrategy(price_belief=1.5),
            urgency=0.3,
        )
        sim = Simulator([buyer, seller], max_negotiation_turns=8, seed=seed)

        prices = []
        for _ in range(NUM_ROUNDS_PER_TRIAL):
            buyer.pending_needs = Resource(gpu_hours=5)
            sim.run_round(pairings=[("buyer", "seller")])
            last = sim.results[-1]
            if last.agreed:
                prices.append(last.price)

        if len(prices) >= 40:
            early = prices[:20]
            late = prices[-20:]
            e_mean = sum(early) / len(early)
            l_mean = sum(late) / len(late)
            e_vol = (sum((v - e_mean) ** 2 for v in early) / len(early)) ** 0.5
            l_vol = (sum((v - l_mean) ** 2 for v in late) / len(late)) ** 0.5
            early_vols.append(e_vol)
            late_vols.append(l_vol)
            if late:
                final_prices.append(late[-1])

    e_stat = describe(early_vols)
    l_stat = describe(late_vols)
    t_stat, p_val = welch_t_test(early_vols, late_vols)
    d = cohens_d(early_vols, late_vols)

    print(f"Early volatility:  {e_stat}")
    print(f"Late volatility:   {l_stat}")
    print(f"Welch's t-test:    t={t_stat:.3f}, p={p_val:.2e}")
    print(f"Effect size (d):   {d:.3f}")
    print(f"Final prices:      {describe(final_prices)}")
    print()

    if p_val < 0.001:
        print("Result: STRONG evidence that prices converge (p < 0.001)")
    elif p_val < 0.05:
        print("Result: Significant evidence of convergence (p < 0.05)")
    else:
        print("Result: No significant evidence of convergence")

    return {"early": e_stat, "late": l_stat, "t": t_stat, "p": p_val, "d": d}


def stat_tournament(num_trials=NUM_TRIALS):
    """Tournament with statistical comparison between strategies."""
    print(f"\n=== Strategy Tournament ({num_trials} trials) ===\n")

    wealth_by_agent: dict[str, list[float]] = {}

    for trial in range(num_trials):
        agents = [
            Agent("greedy_provider", Resource(gpu_hours=100, cpu_hours=50), 10.0, GreedyStrategy(0.6), 0.1),
            Agent("fair_provider", Resource(gpu_hours=80, cpu_hours=80), 15.0, FairStrategy(), 0.2),
            Agent("patient_provider", Resource(gpu_hours=120, cpu_hours=30), 5.0, PatientStrategy(0.9), 0.1),
            Agent("adaptive_seeker", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.8),
            Agent("greedy_seeker", Resource(gpu_hours=5), 80.0, GreedyStrategy(0.5), 0.9),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial)

        for _ in range(50):
            for sid in ["adaptive_seeker", "greedy_seeker"]:
                sim.agents[sid].pending_needs = Resource(gpu_hours=5, cpu_hours=3)
            sim.run_round(needs={
                "adaptive_seeker": Resource(gpu_hours=5, cpu_hours=3),
                "greedy_seeker": Resource(gpu_hours=5, cpu_hours=3),
            })

        for agent in sim.agents.values():
            wealth_by_agent.setdefault(agent.agent_id, []).append(agent.net_worth())

    # Rankings with CIs
    rankings = sorted(
        wealth_by_agent.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True,
    )

    print(f"{'Agent':<22} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
    print("-" * 62)
    for agent_id, worths in rankings:
        stat = describe(worths)
        print(f"{agent_id:<22} {stat.mean:>8.1f} {stat.std:>8.1f} [{stat.ci_lower:>7.1f}, {stat.ci_upper:>7.1f}]")

    # Pairwise comparisons between top agents
    print("\n--- Pairwise Significance Tests ---\n")
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            name_a, vals_a = rankings[i]
            name_b, vals_b = rankings[j]
            t, p = welch_t_test(vals_a, vals_b)
            d = cohens_d(vals_a, vals_b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {name_a} vs {name_b}: t={t:.2f}, p={p:.2e}, d={d:.2f} {sig}")

    # Resource inequality
    print("\n--- Resource Inequality (Gini Coefficient) ---\n")
    # Compute Gini across final wealth distributions per trial
    ginis = []
    for trial in range(min(num_trials, 500)):
        trial_wealth = [wealth_by_agent[aid][trial] for aid in wealth_by_agent]
        ginis.append(gini_coefficient(trial_wealth))
    print(f"  Gini coefficient: {describe(ginis)}")

    return rankings


def stat_cheater_detection(num_trials=500):
    """Statistical analysis of cheater detection across many trials."""
    print(f"\n=== Cheater Detection ({num_trials} trials) ===\n")

    from experiments.exp3_trust import run_cheater_detection

    cheat_rates = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]

    print(f"{'Cheat Rate':>10} {'Detected?':>10} {'Final Rep':>20} {'Cheater Wealth':>22} {'Incidents':>16}")
    print("-" * 82)

    for rate in cheat_rates:
        reps = []
        wealths = []
        incidents = []
        detected = 0

        for seed in range(num_trials):
            sim, rep_history, deals = run_cheater_detection(
                cheater_prob=rate, rounds=60, seed=seed,
            )
            final_rep = rep_history[-1] if rep_history else 0.5
            reps.append(final_rep)
            wealths.append(sim.agents["cheater_mallory"].net_worth())
            incidents.append(len(sim.default_log))
            if final_rep < 0.3:
                detected += 1

        rep_stat = describe(reps)
        wealth_stat = describe(wealths)
        inc_stat = describe([float(i) for i in incidents])
        detect_pct = detected / num_trials

        print(
            f"{rate:>9.0%} "
            f"{detect_pct:>9.0%} "
            f"{rep_stat.mean:>7.3f} [{rep_stat.ci_lower:.3f},{rep_stat.ci_upper:.3f}] "
            f"{wealth_stat.mean:>7.1f} [{wealth_stat.ci_lower:.1f},{wealth_stat.ci_upper:.1f}] "
            f"{inc_stat.mean:>5.1f} [{inc_stat.ci_lower:.1f},{inc_stat.ci_upper:.1f}]"
        )

    return cheat_rates


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 4: STATISTICAL RIGOR")
    print("=" * 60)
    print()

    stat_strategy_matrix()
    stat_convergence()
    stat_tournament()
    stat_cheater_detection()
