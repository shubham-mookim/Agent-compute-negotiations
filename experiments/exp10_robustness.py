#!/usr/bin/env python3
"""
Experiment 10: Robustness & Theoretical Grounding

Step 2: Parameter sweep — show that key findings are stable across strategy
        parameter ranges, not knife-edge artifacts of default values.
Step 4: Rubinstein equilibrium baseline — compare observed prices to the
        game-theoretic prediction for alternating-offers bargaining.

These address the two most important "Open Issues" from the abstract.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Resource
from agents.strategies import (
    GreedyStrategy, FairStrategy, PatientStrategy, AdaptiveStrategy,
)
from agents.rl_strategy import QLearningStrategy
from agents.simulator import Simulator
from agents.stats import describe, welch_t_test, cohens_d


# ── Step 2: Parameter Sweep for Robustness ────────────────────────────────

def sweep_greedy_factor(num_trials=50, rounds=100):
    """
    Vary greed_factor from 0.5 to 0.95 and measure deal rate + wealth.
    Key question: is 0% deal rate a knife-edge at greed_factor=0.7,
    or does it hold across the range?
    """
    print("=== Parameter Sweep: Greedy Factor ===\n")
    print(f"{'greed_factor':>14} {'deal_rate':>10} {'wealth_Δ':>10} {'utility':>10}")
    print("-" * 48)

    for gf in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        deal_rates = []
        wealth_deltas = []
        utilities = []

        for trial in range(num_trials):
            buyer = Agent("buyer", Resource(), 100.0, GreedyStrategy(greed_factor=gf), 0.7,
                          pending_needs=Resource(gpu_hours=5))
            seller = Agent("seller", Resource(gpu_hours=200), 10.0, FairStrategy(), 0.2)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
            start = buyer.net_worth()

            deals = 0
            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                sim.run_round(pairings=[("buyer", "seller")])
                if sim.results and sim.results[-1].agreed:
                    deals += 1

            deal_rates.append(deals / rounds)
            wealth_deltas.append(buyer.net_worth() - start)
            utilities.append(buyer.utility())

        avg_dr = sum(deal_rates) / len(deal_rates)
        avg_wd = sum(wealth_deltas) / len(wealth_deltas)
        avg_u = sum(utilities) / len(utilities)
        print(f"{gf:>14.2f} {avg_dr:>10.0%} {avg_wd:>+10.1f} {avg_u:>10.4f}")


def sweep_fairness_tolerance(num_trials=50, rounds=100):
    """
    Vary fairness_tolerance from 0.05 to 0.40.
    Key question: does Fair always deal at 100% regardless of tolerance?
    """
    print("\n=== Parameter Sweep: Fair Tolerance ===\n")
    print(f"{'tolerance':>14} {'deal_rate':>10} {'avg_price':>10} {'utility':>10}")
    print("-" * 48)

    for ft in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        deal_rates = []
        prices = []
        utilities = []

        for trial in range(num_trials):
            buyer = Agent("buyer", Resource(), 100.0, FairStrategy(fairness_tolerance=ft), 0.7,
                          pending_needs=Resource(gpu_hours=5))
            seller = Agent("seller", Resource(gpu_hours=200), 10.0, FairStrategy(fairness_tolerance=ft), 0.2)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)

            trial_prices = []
            deals = 0
            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                sim.run_round(pairings=[("buyer", "seller")])
                if sim.results and sim.results[-1].agreed:
                    deals += 1
                    trial_prices.append(sim.results[-1].price)

            deal_rates.append(deals / rounds)
            if trial_prices:
                prices.append(sum(trial_prices) / len(trial_prices))
            utilities.append(buyer.utility())

        avg_dr = sum(deal_rates) / len(deal_rates)
        avg_p = sum(prices) / len(prices) if prices else 0
        avg_u = sum(utilities) / len(utilities)
        print(f"{ft:>14.2f} {avg_dr:>10.0%} {avg_p:>10.2f} {avg_u:>10.4f}")


def sweep_patience(num_trials=50, rounds=100):
    """
    Vary patience from 0.3 to 0.95.
    Key question: at what patience level do deals start failing?
    """
    print("\n=== Parameter Sweep: Patience ===\n")
    print(f"{'patience':>14} {'deal_rate':>10} {'wealth_Δ':>10} {'utility':>10}")
    print("-" * 48)

    for p in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        deal_rates = []
        wealth_deltas = []
        utilities = []

        for trial in range(num_trials):
            buyer = Agent("buyer", Resource(), 100.0, PatientStrategy(patience=p), 0.7,
                          pending_needs=Resource(gpu_hours=5))
            seller = Agent("seller", Resource(gpu_hours=200), 10.0, FairStrategy(), 0.2)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
            start = buyer.net_worth()

            deals = 0
            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                sim.run_round(pairings=[("buyer", "seller")])
                if sim.results and sim.results[-1].agreed:
                    deals += 1

            deal_rates.append(deals / rounds)
            wealth_deltas.append(buyer.net_worth() - start)
            utilities.append(buyer.utility())

        avg_dr = sum(deal_rates) / len(deal_rates)
        avg_wd = sum(wealth_deltas) / len(wealth_deltas)
        avg_u = sum(utilities) / len(utilities)
        print(f"{p:>14.2f} {avg_dr:>10.0%} {avg_wd:>+10.1f} {avg_u:>10.4f}")


def sweep_cross_strategy(num_trials=50, rounds=100):
    """
    Full cross-strategy matrix at varied parameters.
    Shows whether the 0% deadlock finding is robust.
    """
    print("\n=== Cross-Strategy Deal Rates (parameter variants) ===\n")

    configs = [
        ("Greedy(0.6)", lambda: GreedyStrategy(0.6)),
        ("Greedy(0.7)", lambda: GreedyStrategy(0.7)),
        ("Greedy(0.85)", lambda: GreedyStrategy(0.85)),
        ("Fair(0.10)", lambda: FairStrategy(0.10)),
        ("Fair(0.15)", lambda: FairStrategy(0.15)),
        ("Fair(0.30)", lambda: FairStrategy(0.30)),
        ("Patient(0.5)", lambda: PatientStrategy(0.5)),
        ("Patient(0.8)", lambda: PatientStrategy(0.8)),
        ("Adaptive", lambda: AdaptiveStrategy()),
    ]

    label = "buyer \\ seller"
    header = f"{label:<16}" + "".join(f"{n:<16}" for n, _ in configs)
    print(header)
    print("-" * len(header))

    for buyer_name, buyer_factory in configs:
        row = f"{buyer_name:<16}"
        for seller_name, seller_factory in configs:
            deals = 0
            total = 0
            for trial in range(num_trials):
                buyer = Agent("b", Resource(), 100.0, buyer_factory(), 0.7,
                              pending_needs=Resource(gpu_hours=5))
                seller = Agent("s", Resource(gpu_hours=200), 10.0, seller_factory(), 0.2)
                sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)
                sim.run_round(pairings=[("b", "s")])
                total += 1
                if sim.results[0].agreed:
                    deals += 1
            rate = deals / total
            row += f"{rate:<16.0%}"
        print(row)


# ── Step 4: Rubinstein Equilibrium Baseline ───────────────────────────────

def rubinstein_equilibrium(discount_buyer=0.95, discount_seller=0.95, surplus=10.0):
    """
    Compute the Rubinstein alternating-offers equilibrium price.

    In Rubinstein's model, two players split a surplus. The unique
    subgame-perfect equilibrium gives:
        buyer_share = (1 - δ_s) / (1 - δ_b * δ_s)
        seller_share = δ_s * (1 - δ_b) / (1 - δ_b * δ_s)

    The equilibrium price is surplus * seller_share (what the buyer pays).
    When δ_b = δ_s, the split is ~50/50.
    """
    buyer_share = (1 - discount_seller) / (1 - discount_buyer * discount_seller)
    price = surplus * (1 - buyer_share)
    return price


def theoretical_baseline(num_trials=100, rounds=100):
    """
    Compare observed negotiation prices to the Rubinstein equilibrium.
    Use discount factors derived from agent urgency.
    """
    print("\n=== Step 4: Rubinstein Equilibrium Comparison ===\n")

    surplus = 5.0  # 5 GPU hours at 1.0/unit = 5.0 total value

    # Map urgency to discount factors: high urgency = low patience = low discount
    buyer_urgency = 0.7
    seller_urgency = 0.2
    delta_buyer = 1 - buyer_urgency * 0.1  # 0.93
    delta_seller = 1 - seller_urgency * 0.1  # 0.98
    eq_price = rubinstein_equilibrium(delta_buyer, delta_seller, surplus)

    print(f"Rubinstein equilibrium parameters:")
    print(f"  Surplus (5 GPU-hrs): {surplus:.1f}")
    print(f"  δ_buyer (urgency {buyer_urgency}): {delta_buyer:.2f}")
    print(f"  δ_seller (urgency {seller_urgency}): {delta_seller:.2f}")
    print(f"  Equilibrium price: {eq_price:.3f}")
    print(f"  As fraction of surplus: {eq_price / surplus:.3f}")
    print()

    strategies = {
        "Fair": lambda: FairStrategy(),
        "Adaptive": lambda: AdaptiveStrategy(),
        "Greedy(0.7)": lambda: GreedyStrategy(0.7),
        "Greedy(0.85)": lambda: GreedyStrategy(0.85),
        "Patient(0.5)": lambda: PatientStrategy(0.5),
        "RL": lambda: QLearningStrategy(epsilon=0.40),
    }

    print(f"{'Strategy':<16} {'Avg Price':>10} {'vs Equil':>10} {'Deviation%':>12} {'Deal Rate':>10}")
    print("-" * 62)

    for name, factory in strategies.items():
        trial_prices = []
        deal_count = 0
        total = 0

        for trial in range(num_trials):
            buyer = Agent("b", Resource(), 100.0, factory(), buyer_urgency,
                          pending_needs=Resource(gpu_hours=5))
            seller = Agent("s", Resource(gpu_hours=500), 10.0, FairStrategy(0.15), seller_urgency)
            sim = Simulator([buyer, seller], max_negotiation_turns=6, seed=trial)

            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                sim.run_round(pairings=[("b", "s")])

            for r in sim.results:
                total += 1
                if r.agreed:
                    deal_count += 1
                    trial_prices.append(r.price)

        if trial_prices:
            avg_p = sum(trial_prices) / len(trial_prices)
            deviation = (avg_p - eq_price) / eq_price * 100
            dr = deal_count / total if total > 0 else 0
            print(f"{name:<16} {avg_p:>10.3f} {eq_price:>10.3f} {deviation:>+11.1f}% {dr:>10.0%}")
        else:
            dr = deal_count / total if total > 0 else 0
            print(f"{name:<16} {'N/A':>10} {eq_price:>10.3f} {'N/A':>12} {dr:>10.0%}")

    # Sensitivity: how does equilibrium change with urgency?
    print(f"\n--- Equilibrium Sensitivity to Urgency ---\n")
    print(f"{'buyer_urg':>10} {'seller_urg':>11} {'δ_b':>6} {'δ_s':>6} {'eq_price':>10} {'buyer_share%':>13}")
    print("-" * 60)

    for bu in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for su in [0.1, 0.3, 0.5, 0.7, 0.9]:
            db = 1 - bu * 0.1
            ds = 1 - su * 0.1
            ep = rubinstein_equilibrium(db, ds, surplus)
            bs = (1 - ds) / (1 - db * ds) * 100
            print(f"{bu:>10.1f} {su:>11.1f} {db:>6.2f} {ds:>6.2f} {ep:>10.3f} {bs:>12.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 60)
    print("  EXPERIMENT 10: ROBUSTNESS & THEORETICAL GROUNDING")
    print("=" * 60)
    print()

    sweep_greedy_factor()
    sweep_fairness_tolerance()
    sweep_patience()
    sweep_cross_strategy()
    theoretical_baseline()


if __name__ == "__main__":
    run_all()
