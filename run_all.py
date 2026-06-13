#!/usr/bin/env python3
"""
Run all experiments and generate visualizations.

Usage:
    python run_all.py              # run everything
    python run_all.py 1            # run only experiment 1
    python run_all.py 2            # run only experiment 2
    python run_all.py 3            # run only experiment 3
    python run_all.py 4            # statistical rigor
    python run_all.py 5            # LLM agents (needs ANTHROPIC_API_KEY)
    python run_all.py 6            # deep cheater analysis
    python run_all.py 7            # futures market
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_exp1():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: THE BASIC HANDSHAKE")
    print("=" * 60 + "\n")

    from experiments.exp1_handshake import message_trace, strategy_matrix, convergence_test
    from experiments.visualize import plot_price_convergence

    message_trace()
    strategy_matrix()
    prices = convergence_test()
    plot_price_convergence(prices, "logs/exp1_price_convergence.png")


def run_exp2():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: SCARCITY GAMES")
    print("=" * 60 + "\n")

    from experiments.exp2_scarcity import run_scarcity_sim, run_rush_hour, strategy_tournament
    from experiments.visualize import plot_strategy_comparison

    print("--- 100-Round Scarcity Simulation ---\n")
    sim, deal_counts, total_spent, resources_acquired, prices = run_scarcity_sim()

    print("Seeker performance:")
    for sid in ["adaptive_seeker", "greedy_seeker"]:
        print(f"  {sid}:")
        print(f"    Deals made: {deal_counts.get(sid, 0)}")
        print(f"    Total spent: {total_spent.get(sid, 0):.1f}")
        acquired = resources_acquired.get(sid, 0)
        spent = total_spent.get(sid, 0)
        print(f"    Resources acquired: {acquired:.1f}")
        if acquired > 0:
            print(f"    Avg cost/unit: {spent / acquired:.2f}")
    print()
    print(sim.summary())
    print()

    run_rush_hour()
    rankings = strategy_tournament()
    plot_strategy_comparison(rankings, "logs/exp2_strategy_tournament.png")


def run_exp3():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: TRUST FALL")
    print("=" * 60 + "\n")

    from experiments.exp3_trust import run_cheater_detection
    from experiments.visualize import plot_reputation_evolution

    print("--- Always-Cheating Mallory ---\n")
    sim, rep_history, deals_history = run_cheater_detection(cheater_prob=1.0)

    print("Cheater reputation over time:")
    for phase, s, e in [("Early (1-10)", 0, 10), ("Mid (25-35)", 24, 35), ("Late (50-60)", 49, 60)]:
        avg = sum(rep_history[s:e]) / len(rep_history[s:e])
        deals = sum(deals_history[s:e])
        print(f"  {phase}: rep={avg:.3f}, deals={deals}")

    print(f"\nCheating incidents: {len(sim.default_log)}")
    print(f"Final state: budget={sim.agents['cheater_mallory'].budget:.1f}, "
          f"resources={sim.agents['cheater_mallory'].resources}")

    print("\nHonest agents' view of Mallory:")
    for aid in sorted(sim.agents):
        if "cheater" not in aid:
            rep = sim.agents[aid].reputation_of("cheater_mallory")
            print(f"  {aid}: {rep:.3f}")

    plot_reputation_evolution(rep_history, deals_history, "logs/exp3_always_cheat.png")

    print("\n--- Subtle Mallory (5% cheat rate) ---\n")
    sim2, rep2, deals2 = run_cheater_detection(cheater_prob=0.05, rounds=100, seed=123)

    for phase, s, e in [("Early", 0, 20), ("Mid", 39, 60), ("Late", 79, 100)]:
        sl = rep2[s:e]
        if sl:
            print(f"  {phase}: rep={sum(sl)/len(sl):.3f}, deals={sum(deals2[s:e])}")

    plot_reputation_evolution(rep2, deals2, "logs/exp3_subtle_cheat.png")

    print(f"\n  Always cheat wealth: {sim.agents['cheater_mallory'].net_worth():.1f}")
    print(f"  5% cheat wealth:    {sim2.agents['cheater_mallory'].net_worth():.1f}")


def run_exp4():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: STATISTICAL RIGOR")
    print("=" * 60 + "\n")

    from experiments.exp4_statistical import (
        stat_strategy_matrix, stat_convergence,
        stat_tournament, stat_cheater_detection,
    )

    stat_strategy_matrix()
    stat_convergence()
    stat_tournament()
    stat_cheater_detection()


def run_exp5():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: LLM AGENT NEGOTIATIONS")
    print("=" * 60 + "\n")

    from experiments.exp5_llm_agents import (
        check_api_available, llm_vs_llm,
        llm_vs_rule_based, mixed_population,
    )

    api_ok = check_api_available()
    num = 20 if api_ok else 10
    llm_vs_llm(num_trials=num)
    llm_vs_rule_based(num_trials=num)
    mixed_population(num_trials=5 if api_ok else 3)


def run_exp6():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 6: DEEP CHEATER ANALYSIS")
    print("=" * 60 + "\n")

    from experiments.exp6_cheater_depth import (
        detection_threshold_sweep,
        run_collaborative_vs_isolated,
        run_adaptive_cheater,
        run_multiple_cheaters,
    )

    detection_threshold_sweep()
    run_collaborative_vs_isolated(cheat_rate=0.10)
    run_collaborative_vs_isolated(cheat_rate=0.05)
    run_adaptive_cheater()
    run_multiple_cheaters(num_cheaters=2)
    run_multiple_cheaters(num_cheaters=3)


def run_exp7():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 7: PREDICTIVE NEGOTIATION & FUTURES")
    print("=" * 60 + "\n")

    from experiments.exp7_futures import (
        spot_vs_futures, demand_pattern_analysis,
        arbitrage_experiment, volatile_market_arbitrage,
    )

    spot_vs_futures()
    demand_pattern_analysis()
    arbitrage_experiment()
    volatile_market_arbitrage()


def run_exp8():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 8: COALITION FORMATION")
    print("=" * 60 + "\n")

    from experiments.exp8_coalitions import (
        run_solo_baseline, run_buyer_coalition, run_seller_coalition,
        run_free_rider_test, run_size_sweep, run_stability_test,
    )

    print("--- Part 1: Solo Baseline ---\n")
    solo = run_solo_baseline()
    print(f"  Price/unit:       {solo['prices']}")
    print(f"  Deal rate:        {solo['deal_rate']}")
    print(f"  Seeker wealth Δ:  {solo['seeker_delta']}")

    print("\n--- Part 2: Buyer Coalition ---\n")
    buyer = run_buyer_coalition()
    print(f"  Price/unit:         {buyer['prices']}")
    print(f"  Coalition Δ:        {buyer['coalition_delta']}")
    print(f"  Solo seeker Δ:      {buyer['solo_seeker_delta']}")

    print("\n--- Part 3: Seller Cartel ---\n")
    seller = run_seller_coalition()
    if seller["cartel_price_per_unit"]:
        print(f"  Cartel price/unit:   {seller['cartel_price_per_unit']}")
    if seller["solo_price_per_unit"]:
        print(f"  Solo provider price: {seller['solo_price_per_unit']}")
    print(f"  Seeker total cost:   {seller['seeker_total_cost']}")

    print("\n--- Part 4: Free-Rider Detection ---\n")
    fr = run_free_rider_test()
    print(f"  Detection rate:       {fr['detected_pct']:.0%}")
    print(f"  Detection round:      {fr['detection_round']}")
    print(f"  Free-rider Δ:         {fr['freerider_wealth_delta']}")
    print(f"  Honest member Δ:      {fr['honest_member_wealth_delta']}")

    print("\n--- Part 5: Coalition Size Sweep ---\n")
    run_size_sweep()

    print("\n--- Part 6: Stability Under Scarcity ---\n")
    stab = run_stability_test()
    print(f"  Stability rate:         {stab['stability_rate']:.0%}")
    print(f"  Avg end size:           {stab['avg_end_size']:.1f}")
    if stab["avg_defection_round"]:
        print(f"  Avg defection round:    {stab['avg_defection_round']:.1f}")


def run_exp9():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 9: RL LEARNING AGENTS")
    print("=" * 60 + "\n")

    from experiments.exp9_learning import (
        run_learning_curve, run_convergence_test,
        run_rl_vs_rules, run_rl_mixed_population,
        run_strategy_transfer, inspect_emergent_policy,
        run_tier_comparison,
    )

    run_learning_curve(total_rounds=1000)
    run_convergence_test(num_trials=100, rounds_per_trial=500)
    run_rl_vs_rules(num_trials=200, rounds=300)
    run_rl_mixed_population(num_trials=100, rounds=300)
    run_strategy_transfer()
    inspect_emergent_policy(rounds=1000)
    run_tier_comparison(num_trials=100, rounds=300)


def main():
    Path("logs").mkdir(exist_ok=True)

    runners = {
        "1": run_exp1,
        "2": run_exp2,
        "3": run_exp3,
        "4": run_exp4,
        "5": run_exp5,
        "6": run_exp6,
        "7": run_exp7,
        "8": run_exp8,
        "9": run_exp9,
    }

    if len(sys.argv) > 1:
        exp = sys.argv[1]
        if exp in runners:
            runners[exp]()
        else:
            print(f"Unknown experiment: {exp}. Use 1-7.")
    else:
        for key in sorted(runners):
            runners[key]()

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("\nCheck logs/ for plots and data.")


if __name__ == "__main__":
    main()
