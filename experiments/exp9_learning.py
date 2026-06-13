#!/usr/bin/env python3
"""
Experiment 9: Reinforcement Learning Agents — Tier 2 Intelligence

Places Q-learning agents in the negotiation market and studies:

  1. Learning curve       — performance vs number of rounds played
  2. Convergence          — does the RL agent find a stable strategy?
  3. RL vs rule-based     — head-to-head wealth comparison
  4. Strategy transfer    — RL trained on Fair market, tested on Greedy
  5. Emergent policy      — what strategy did the agent actually learn?
  6. Intelligence tier    — Rule-based < RL < (LLM placeholder) ranking

These results form the "intelligence tier" pillar of the paper.
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


# ── Part 1: Learning Curve ─────────────────────────────────────────────────

def run_learning_curve(total_rounds=2000, window=50, seed=42):
    """
    Single RL agent learning against a mix of opponents.
    Track rolling average reward to see learning curve shape.
    """
    print("=== Part 1: RL Learning Curve ===\n")

    rl_strategy = QLearningStrategy(epsilon=0.40)
    buyer = Agent(
        "rl_buyer", Resource(), 500.0, rl_strategy, urgency=0.6,
    )
    providers = [
        Agent("fair_prov", Resource(gpu_hours=1000), 10.0, FairStrategy(), 0.2),
        Agent("adaptive_prov", Resource(gpu_hours=1000), 10.0, AdaptiveStrategy(), 0.2),
        Agent("greedy_prov", Resource(gpu_hours=1000), 10.0, GreedyStrategy(), 0.1),
    ]
    all_agents = providers + [buyer]
    sim = Simulator(all_agents, max_negotiation_turns=6, seed=seed)

    milestones = {}
    for r in range(total_rounds):
        provider = random.choice(providers)
        buyer.pending_needs = Resource(gpu_hours=5)
        sim.run_round(pairings=[("rl_buyer", provider.agent_id)])

        if r in (0, 49, 99, 199, 499, 999, 1999):
            milestones[r + 1] = {
                "deals": rl_strategy.deals_made,
                "epsilon": rl_strategy.epsilon,
                "avg_reward": rl_strategy.avg_reward_window(window),
                "q_states": len(rl_strategy.q_table),
            }

    print(f"{'Round':>8} {'Deals':>7} {'Epsilon':>8} {'Avg Reward':>11} {'Q-States':>9}")
    print("-" * 47)
    for r, m in milestones.items():
        print(
            f"{r:>8} {m['deals']:>7} {m['epsilon']:>8.3f} "
            f"{m['avg_reward']:>11.3f} {m['q_states']:>9}"
        )

    print(f"\nFinal Q-table entries: {len(rl_strategy.q_table)}")
    print(f"Final deals made: {rl_strategy.deals_made}")
    print(f"Final epsilon: {rl_strategy.epsilon:.4f}")

    return rl_strategy


def run_convergence_test(num_trials=100, rounds_per_trial=500):
    """
    Across many trials, measure when the RL agent's average reward stabilizes.
    """
    print("\n=== Part 1b: Convergence Test ===\n")

    early_rewards = []   # first 50 deals
    late_rewards = []    # last 50 deals

    for trial in range(num_trials):
        rl = QLearningStrategy(epsilon=0.40)
        buyer = Agent("rl", Resource(), 300.0, rl, 0.6)
        providers = [
            Agent("fp", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2),
            Agent("ap", Resource(gpu_hours=2000), 5.0, AdaptiveStrategy(), 0.2),
        ]
        sim = Simulator(providers + [buyer], max_negotiation_turns=6, seed=trial)

        for _ in range(rounds_per_trial):
            buyer.pending_needs = Resource(gpu_hours=5)
            provider = random.choice(providers)
            sim.run_round(pairings=[("rl", provider.agent_id)])

        if len(rl.reward_history) >= 100:
            early_rewards.append(sum(rl.reward_history[:50]) / 50)
            late_rewards.append(sum(rl.reward_history[-50:]) / 50)

    e_stat = describe(early_rewards)
    l_stat = describe(late_rewards)
    t, p = welch_t_test(early_rewards, late_rewards)
    d = cohens_d(early_rewards, late_rewards)

    print(f"Early rewards (first 50):  {e_stat}")
    print(f"Late rewards (last 50):    {l_stat}")
    print(f"Welch's t={t:.3f}, p={p:.2e}, Cohen's d={d:.3f}")
    if p < 0.05 and l_stat.mean > e_stat.mean:
        print("Result: RL agent SIGNIFICANTLY improves over time")
    elif p < 0.05:
        print("Result: Significant change, but not an improvement")
    else:
        print("Result: No significant improvement detected")

    return {"early": e_stat, "late": l_stat, "p": p, "d": d}


# ── Part 2: RL vs Rule-Based Head-to-Head ─────────────────────────────────

def run_rl_vs_rules(num_trials=200, rounds=300, seed_offset=0):
    """
    Compare final wealth: RL agent vs each rule-based strategy.
    RL needs warmup rounds so we give it plenty of time.
    """
    print("\n=== Part 2: RL vs Rule-Based Strategies ===\n")

    strategies = {
        "greedy": lambda: GreedyStrategy(),
        "fair": lambda: FairStrategy(),
        "patient": lambda: PatientStrategy(),
        "adaptive": lambda: AdaptiveStrategy(),
        "rl": lambda: QLearningStrategy(epsilon=0.40),
    }

    wealth_results: dict[str, list[float]] = {k: [] for k in strategies}

    for trial in range(num_trials):
        for strat_name, strat_factory in strategies.items():
            buyer = Agent(
                f"{strat_name}_buyer", Resource(),
                200.0, strat_factory(), 0.7,
            )
            providers = [
                Agent("prov_1", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2),
                Agent("prov_2", Resource(gpu_hours=2000), 5.0, AdaptiveStrategy(), 0.2),
            ]
            sim = Simulator(providers + [buyer], max_negotiation_turns=6,
                            seed=trial + seed_offset + hash(strat_name) % 10000)
            start = buyer.net_worth()

            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                provider = random.choice(providers)
                sim.run_round(pairings=[(f"{strat_name}_buyer", provider.agent_id)])

            wealth_results[strat_name].append(buyer.net_worth() - start)

    # Rankings
    print(f"{'Strategy':<12} {'Wealth Δ':>20} {'Std':>8}")
    print("-" * 44)
    ranked = sorted(
        wealth_results.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True,
    )
    for name, deltas in ranked:
        stat = describe(deltas)
        print(f"{name:<12} {stat.mean:>+10.1f} ±{stat.ci_upper - stat.mean:.1f}   {stat.std:>8.1f}")

    # Statistical comparisons vs RL
    print("\n--- Pairwise vs RL ---")
    rl_vals = wealth_results["rl"]
    for name, vals in ranked:
        if name == "rl":
            continue
        t, p = welch_t_test(rl_vals, vals)
        d = cohens_d(rl_vals, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        direction = ">" if sum(rl_vals) / len(rl_vals) > sum(vals) / len(vals) else "<"
        print(f"  rl {direction} {name:<12}: t={t:.2f}, p={p:.2e}, d={d:.2f} {sig}")

    return wealth_results


# ── Part 3: RL in Mixed Population ────────────────────────────────────────

def run_rl_mixed_population(num_trials=100, rounds=300, seed_offset=5000):
    """
    RL agent competes in a mixed market with all strategy types.
    Does it adapt and exploit weaknesses in rule-based agents?
    """
    print("\n=== Part 3: RL in Mixed Population ===\n")

    rl_deltas = []
    other_deltas: dict[str, list[float]] = {
        "greedy": [], "fair": [], "patient": [], "adaptive": [],
    }

    for trial in range(num_trials):
        rl_agent = Agent("rl", Resource(gpu_hours=5), 200.0, QLearningStrategy(epsilon=0.40), 0.7)
        other_seekers = [
            Agent("greedy", Resource(gpu_hours=5), 200.0, GreedyStrategy(), 0.7),
            Agent("fair", Resource(gpu_hours=5), 200.0, FairStrategy(), 0.7),
            Agent("patient", Resource(gpu_hours=5), 200.0, PatientStrategy(), 0.7),
            Agent("adaptive", Resource(gpu_hours=5), 200.0, AdaptiveStrategy(), 0.7),
        ]
        providers = [
            Agent("prov_1", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2),
            Agent("prov_2", Resource(gpu_hours=2000), 5.0, AdaptiveStrategy(), 0.2),
            Agent("prov_3", Resource(gpu_hours=2000), 5.0, GreedyStrategy(), 0.1),
        ]
        all_agents = providers + other_seekers + [rl_agent]
        start = {a.agent_id: a.net_worth() for a in all_agents}
        sim = Simulator(all_agents, max_negotiation_turns=6, seed=trial + seed_offset)

        seekers = other_seekers + [rl_agent]
        for _ in range(rounds):
            for s in seekers:
                s.pending_needs = Resource(gpu_hours=5)
                provider = random.choice(providers)
                sim.run_round(pairings=[(s.agent_id, provider.agent_id)])

        rl_deltas.append(rl_agent.net_worth() - start["rl"])
        for s in other_seekers:
            other_deltas[s.agent_id].append(s.net_worth() - start[s.agent_id])

    print(f"{'Agent':<12} {'Wealth Δ':>20}")
    print("-" * 34)
    all_results = {"rl": rl_deltas, **other_deltas}
    ranked = sorted(all_results.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    for name, deltas in ranked:
        stat = describe(deltas)
        print(f"{name:<12} {stat.mean:>+10.1f} ±{stat.ci_upper - stat.mean:.1f}")

    return {"rl": describe(rl_deltas), **{k: describe(v) for k, v in other_deltas.items()}}


# ── Part 4: Strategy Transfer ──────────────────────────────────────────────

def run_strategy_transfer(rounds_train=500, rounds_test=200, num_trials=100):
    """
    Train RL on a "fair market" (only Fair/Adaptive opponents).
    Then test on a "hostile market" (Greedy opponents).
    Compare vs RL that trained on the hostile market directly.
    Hypothesis: RL overfits to training environment.
    """
    print("\n=== Part 4: Strategy Transfer ===\n")

    transferred_deltas = []  # trained on fair, tested on greedy
    native_deltas = []       # trained on greedy from start

    for trial in range(num_trials):
        # --- Transferred agent ---
        rl_transferred = QLearningStrategy(epsilon=0.40)
        buyer_t = Agent("buyer", Resource(), 300.0, rl_transferred, 0.6)
        fair_providers = [
            Agent(f"fp_{i}", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2)
            for i in range(2)
        ]
        sim_train = Simulator(fair_providers + [buyer_t], max_negotiation_turns=6, seed=trial)
        for _ in range(rounds_train):
            buyer_t.pending_needs = Resource(gpu_hours=5)
            p = random.choice(fair_providers)
            sim_train.run_round(pairings=[("buyer", p.agent_id)])

        # Now test on greedy market
        greedy_providers = [
            Agent(f"gp_{i}", Resource(gpu_hours=2000), 5.0, GreedyStrategy(), 0.1)
            for i in range(2)
        ]
        buyer_t.budget = 200.0
        buyer_t.resources = Resource()
        start_t = buyer_t.net_worth()
        rl_transferred.epsilon = 0.05  # mostly exploit learned policy
        sim_test = Simulator(greedy_providers + [buyer_t], max_negotiation_turns=6, seed=trial + 10000)
        for _ in range(rounds_test):
            buyer_t.pending_needs = Resource(gpu_hours=5)
            p = random.choice(greedy_providers)
            sim_test.run_round(pairings=[("buyer", p.agent_id)])
        transferred_deltas.append(buyer_t.net_worth() - start_t)

        # --- Native agent (trained on greedy from start) ---
        rl_native = QLearningStrategy(epsilon=0.40)
        buyer_n = Agent("buyer_n", Resource(), 200.0, rl_native, 0.6)
        greedy_train = [
            Agent(f"gp_tr_{i}", Resource(gpu_hours=2000), 5.0, GreedyStrategy(), 0.1)
            for i in range(2)
        ]
        sim_n = Simulator(greedy_train + [buyer_n], max_negotiation_turns=6, seed=trial + 20000)
        for _ in range(rounds_train + rounds_test):
            buyer_n.pending_needs = Resource(gpu_hours=5)
            p = random.choice(greedy_train)
            sim_n.run_round(pairings=[("buyer_n", p.agent_id)])
        native_deltas.append(buyer_n.net_worth() - 200.0)

    t_stat = describe(transferred_deltas)
    n_stat = describe(native_deltas)
    t, p = welch_t_test(transferred_deltas, native_deltas)
    d = cohens_d(transferred_deltas, native_deltas)

    print(f"Transferred (fair→greedy): {t_stat}")
    print(f"Native (greedy from start): {n_stat}")
    print(f"Welch's t={t:.3f}, p={p:.2e}, d={d:.3f}")
    if p < 0.05 and n_stat.mean > t_stat.mean:
        print("Result: Native RL SIGNIFICANTLY outperforms transferred — overfitting confirmed")
    elif p < 0.05:
        print("Result: Transferred agent surprisingly better — robust policy")
    else:
        print("Result: No significant difference — policy transfers well")

    return {"transferred": t_stat, "native": n_stat, "p": p}


# ── Part 5: Emergent Policy Inspection ────────────────────────────────────

def inspect_emergent_policy(rounds=1000, seed=42):
    """
    Train an RL agent for 1000 rounds, then inspect what policy it learned.
    Shows which states map to which actions.
    """
    print("\n=== Part 5: Emergent Policy Inspection ===\n")

    rl = QLearningStrategy(epsilon=0.40)
    buyer = Agent("rl", Resource(), 400.0, rl, 0.6)
    providers = [
        Agent("fair_p", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2),
        Agent("adaptive_p", Resource(gpu_hours=2000), 5.0, AdaptiveStrategy(), 0.2),
        Agent("greedy_p", Resource(gpu_hours=2000), 5.0, GreedyStrategy(), 0.1),
    ]
    sim = Simulator(providers + [buyer], max_negotiation_turns=6, seed=seed)

    for _ in range(rounds):
        buyer.pending_needs = Resource(gpu_hours=5)
        p = random.choice(providers)
        sim.run_round(pairings=[("rl", p.agent_id)])

    policy = rl.policy_summary()
    if policy:
        print(f"Learned policy ({len(policy)} state-action pairs):\n")
        # Group by message type
        for context, action in sorted(policy.items()):
            print(f"  {context:<55} → {action}")
    else:
        print("  Policy is empty (agent needs more training rounds)")

    print(f"\nTotal Q-table entries: {len(rl.q_table)}")
    print(f"Deals made: {rl.deals_made}")
    print(f"Avg reward (last 50): {rl.avg_reward_window(50):.4f}")

    return policy


# ── Part 6: Intelligence Tier Summary ─────────────────────────────────────

def run_tier_comparison(num_trials=100, rounds=300):
    """
    Full intelligence tier comparison.
    Tier 1 = rule-based, Tier 2 = RL.
    (Tier 3 = LLM, placeholder for when API key is available.)
    """
    print("\n=== Part 6: Intelligence Tier Comparison ===\n")

    results: dict[str, list[float]] = {}

    tier1_strategies = {
        "T1-Greedy": lambda: GreedyStrategy(),
        "T1-Fair": lambda: FairStrategy(),
        "T1-Patient": lambda: PatientStrategy(),
        "T1-Adaptive": lambda: AdaptiveStrategy(),
    }
    tier2_strategies = {
        "T2-RL(fresh)": lambda: QLearningStrategy(epsilon=0.40),
        "T2-RL(pretrained)": None,  # handle below
    }

    # Pretrain RL agent
    print("  Pre-training RL agent (500 rounds)...")
    pretrained_rl = QLearningStrategy(epsilon=0.40)
    _buyer = Agent("pretrain", Resource(), 500.0, pretrained_rl, 0.6)
    _provs = [
        Agent(f"pp_{i}", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2)
        for i in range(2)
    ]
    _sim = Simulator(_provs + [_buyer], max_negotiation_turns=6, seed=999)
    for _ in range(500):
        _buyer.pending_needs = Resource(gpu_hours=5)
        _sim.run_round(pairings=[("pretrain", random.choice(_provs).agent_id)])
    pretrained_rl.epsilon = 0.05  # mostly exploit after pretraining

    for trial in range(num_trials):
        providers = [
            Agent("prov_1", Resource(gpu_hours=2000), 5.0, FairStrategy(), 0.2),
            Agent("prov_2", Resource(gpu_hours=2000), 5.0, AdaptiveStrategy(), 0.2),
        ]

        for strat_name, strat_factory in tier1_strategies.items():
            buyer = Agent("b", Resource(), 200.0, strat_factory(), 0.7)
            sim = Simulator(providers + [buyer], max_negotiation_turns=6,
                            seed=trial + hash(strat_name) % 10000)
            start = buyer.net_worth()
            for _ in range(rounds):
                buyer.pending_needs = Resource(gpu_hours=5)
                sim.run_round(pairings=[("b", random.choice(providers).agent_id)])
            results.setdefault(strat_name, []).append(buyer.net_worth() - start)

        # Fresh RL
        fresh_rl = QLearningStrategy(epsilon=0.40)
        buyer_fresh = Agent("b_fresh", Resource(), 200.0, fresh_rl, 0.7)
        sim_fresh = Simulator(providers + [buyer_fresh], max_negotiation_turns=6, seed=trial + 50000)
        start_fresh = buyer_fresh.net_worth()
        for _ in range(rounds):
            buyer_fresh.pending_needs = Resource(gpu_hours=5)
            sim_fresh.run_round(pairings=[("b_fresh", random.choice(providers).agent_id)])
        results.setdefault("T2-RL(fresh)", []).append(buyer_fresh.net_worth() - start_fresh)

        # Pretrained RL (reuse shared weights but fresh agent state)
        buyer_pre = Agent("b_pre", Resource(), 200.0, pretrained_rl, 0.7)
        sim_pre = Simulator(providers + [buyer_pre], max_negotiation_turns=6, seed=trial + 60000)
        start_pre = buyer_pre.net_worth()
        for _ in range(rounds):
            buyer_pre.pending_needs = Resource(gpu_hours=5)
            sim_pre.run_round(pairings=[("b_pre", random.choice(providers).agent_id)])
        results.setdefault("T2-RL(pretrained)", []).append(buyer_pre.net_worth() - start_pre)

    print(f"\n{'Agent':<22} {'Tier':<8} {'Wealth Δ':>20}")
    print("-" * 54)
    ranked = sorted(results.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    for name, vals in ranked:
        tier = "Tier 1" if name.startswith("T1") else "Tier 2"
        stat = describe(vals)
        print(f"{name:<22} {tier:<8} {stat.mean:>+10.1f} ±{stat.ci_upper - stat.mean:.1f}")

    print("\n  [Tier 3 - LLM agents: pending ANTHROPIC_API_KEY]")

    # Check if Tier 2 > Tier 1 best
    t2_pre = results.get("T2-RL(pretrained)", [])
    t1_best_name = max(
        [k for k in results if k.startswith("T1")],
        key=lambda k: sum(results[k]) / len(results[k]),
    )
    t1_best = results[t1_best_name]
    t, p = welch_t_test(t2_pre, t1_best)
    d = cohens_d(t2_pre, t1_best)
    direction = ">" if sum(t2_pre) / len(t2_pre) > sum(t1_best) / len(t1_best) else "<"
    print(f"\n  T2-RL(pretrained) {direction} {t1_best_name}: t={t:.2f}, p={p:.2e}, d={d:.2f}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 9: RL LEARNING AGENTS")
    print("=" * 60)
    print()

    run_learning_curve(total_rounds=1000)
    run_convergence_test(num_trials=100, rounds_per_trial=500)
    run_rl_vs_rules(num_trials=200, rounds=300)
    run_rl_mixed_population(num_trials=100, rounds=300)
    run_strategy_transfer()
    inspect_emergent_policy(rounds=1000)
    run_tier_comparison(num_trials=100, rounds=300)
