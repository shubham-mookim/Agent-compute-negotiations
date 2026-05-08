#!/usr/bin/env python3
"""
Experiment 8: Coalition Formation

First study of coalition dynamics in decentralized compute markets.

Scenarios:
  1. Solo baseline     — no coalitions, pure individual negotiation
  2. Buyer coalition   — seekers pool together for collective bargaining
  3. Seller coalition  — providers form a cartel
  4. Free-rider test   — an agent exploits coalition membership
  5. Coalition size    — sweep from 2 to 5 members; find optimal size
  6. Stability         — do coalitions hold under resource scarcity?

Key metrics:
  - Price paid / received per resource unit vs solo baseline
  - Deal success rate
  - Free-rider detection rounds
  - Member wealth vs solo-equivalent wealth
"""

import sys
import random
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Resource
from agents.strategies import FairStrategy, AdaptiveStrategy, GreedyStrategy, PatientStrategy
from agents.simulator import Simulator, NegotiationResult
from agents.coalition import Coalition, CoalitionAgent, MemberRecord
from agents.stats import describe, welch_t_test, cohens_d


# ── Market factory ─────────────────────────────────────────────────────────

def make_solo_market(seed=0):
    """Baseline: 4 providers, 4 seekers, all individual."""
    return [
        Agent("prov_1", Resource(gpu_hours=150), 20.0, FairStrategy(), 0.2),
        Agent("prov_2", Resource(gpu_hours=120), 25.0, AdaptiveStrategy(), 0.2),
        Agent("prov_3", Resource(gpu_hours=100), 15.0, FairStrategy(), 0.3),
        Agent("prov_4", Resource(gpu_hours=130), 18.0, AdaptiveStrategy(), 0.2),
        Agent("seek_1", Resource(gpu_hours=5), 120.0, AdaptiveStrategy(), 0.7),
        Agent("seek_2", Resource(gpu_hours=5), 110.0, FairStrategy(), 0.8),
        Agent("seek_3", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.6),
        Agent("seek_4", Resource(gpu_hours=5), 130.0, FairStrategy(), 0.7),
    ]


# ── Part 1: Solo baseline ──────────────────────────────────────────────────

def run_solo_baseline(num_trials=200, rounds=60, seed_offset=0):
    """All agents act individually. Records per-unit prices and deal rates."""
    prices = []
    deal_rates = []
    seeker_wealth_delta = []

    for trial in range(num_trials):
        agents = make_solo_market(seed=trial)
        start_worth = {a.agent_id: a.net_worth() for a in agents}
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial + seed_offset)

        seeker_ids = [a.agent_id for a in agents if "seek" in a.agent_id]

        for _ in range(rounds):
            needs = {sid: Resource(gpu_hours=8) for sid in seeker_ids}
            for sid in seeker_ids:
                sim.agents[sid].pending_needs = needs[sid]
            m = sim.run_round(needs=needs)
            if m.avg_price_per_unit > 0:
                prices.append(m.avg_price_per_unit)
            deal_rates.append(m.deals_made / max(m.negotiations, 1))

        for sid in seeker_ids:
            delta = sim.agents[sid].net_worth() - start_worth[sid]
            seeker_wealth_delta.append(delta)

    return {
        "prices": describe(prices),
        "deal_rate": describe(deal_rates),
        "seeker_delta": describe(seeker_wealth_delta),
    }


# ── Part 2: Buyer coalition ────────────────────────────────────────────────

def run_buyer_coalition(num_trials=200, rounds=60, seed_offset=1000):
    """
    Two seekers form a coalition. They negotiate as a unit.
    Hypothesis: pooled buying power improves deal rate and lowers price paid.
    """
    prices = []
    deal_rates = []
    coalition_wealth_delta = []
    solo_seeker_wealth_delta = []

    for trial in range(num_trials):
        # Build coalition members
        s1 = Agent("seek_1", Resource(gpu_hours=5), 120.0, FairStrategy(), 0.7)
        s2 = Agent("seek_2", Resource(gpu_hours=5), 110.0, AdaptiveStrategy(), 0.8)
        coalition = Coalition("buyer_coalition", [s1, s2])
        ca = CoalitionAgent(coalition, strategy=FairStrategy())

        # Solo seekers (unaffiliated)
        solo_agents = [
            Agent("prov_1", Resource(gpu_hours=150), 20.0, FairStrategy(), 0.2),
            Agent("prov_2", Resource(gpu_hours=120), 25.0, AdaptiveStrategy(), 0.2),
            Agent("prov_3", Resource(gpu_hours=100), 15.0, FairStrategy(), 0.3),
            Agent("prov_4", Resource(gpu_hours=130), 18.0, AdaptiveStrategy(), 0.2),
            Agent("seek_3", Resource(gpu_hours=5), 100.0, AdaptiveStrategy(), 0.6),
            Agent("seek_4", Resource(gpu_hours=5), 130.0, FairStrategy(), 0.7),
        ]
        all_agents = solo_agents + [ca]
        start_worth = {a.agent_id: a.net_worth() for a in all_agents}
        start_worth["seek_1"] = s1.net_worth()
        start_worth["seek_2"] = s2.net_worth()

        sim = Simulator(all_agents, max_negotiation_turns=6, seed=trial + seed_offset)
        seeker_ids = ["buyer_coalition", "seek_3", "seek_4"]

        for _ in range(rounds):
            ca.sync_from_members()
            for sid in seeker_ids:
                sim.agents[sid].pending_needs = Resource(gpu_hours=8)
            sim.run_round(needs={sid: Resource(gpu_hours=8) for sid in seeker_ids})

            last = sim.results[-1] if sim.results else None
            if last and last.agreed and last.resource:
                prices.append(last.price / last.resource.total_units())
                deal_rates.append(1.0)
            else:
                deal_rates.append(0.0)

        coalition_wealth_delta.append(
            (s1.net_worth() + s2.net_worth()) - (start_worth["seek_1"] + start_worth["seek_2"])
        )
        for sid in ["seek_3", "seek_4"]:
            solo_seeker_wealth_delta.append(
                sim.agents[sid].net_worth() - start_worth[sid]
            )

    return {
        "prices": describe(prices),
        "deal_rate": describe(deal_rates),
        "coalition_delta": describe(coalition_wealth_delta),
        "solo_seeker_delta": describe(solo_seeker_wealth_delta),
    }


# ── Part 3: Seller coalition (cartel) ─────────────────────────────────────

def run_seller_coalition(num_trials=200, rounds=60, seed_offset=2000):
    """
    Two providers form a cartel. They pool resources and raise prices.
    Hypothesis: cartel extracts higher prices from individual seekers.
    """
    cartel_prices = []
    solo_prices = []
    seeker_costs = []

    for trial in range(num_trials):
        p1 = Agent("prov_1", Resource(gpu_hours=150), 20.0, GreedyStrategy(greed_factor=0.6), 0.2)
        p2 = Agent("prov_2", Resource(gpu_hours=120), 25.0, GreedyStrategy(greed_factor=0.6), 0.2)
        cartel = Coalition("seller_cartel", [p1, p2])
        ca = CoalitionAgent(cartel, strategy=GreedyStrategy(greed_factor=0.5))

        other_agents = [
            Agent("prov_solo", Resource(gpu_hours=100), 15.0, FairStrategy(), 0.3),
            Agent("seek_1", Resource(gpu_hours=5), 200.0, AdaptiveStrategy(), 0.7),
            Agent("seek_2", Resource(gpu_hours=5), 180.0, FairStrategy(), 0.8),
            Agent("seek_3", Resource(gpu_hours=5), 160.0, AdaptiveStrategy(), 0.6),
        ]
        all_agents = other_agents + [ca]
        sim = Simulator(all_agents, max_negotiation_turns=6, seed=trial + seed_offset)
        seeker_ids = ["seek_1", "seek_2", "seek_3"]
        seeker_start = {sid: sim.agents[sid].budget for sid in seeker_ids}

        for _ in range(rounds):
            ca.sync_from_members()
            for sid in seeker_ids:
                sim.agents[sid].pending_needs = Resource(gpu_hours=6)
            sim.run_round(needs={sid: Resource(gpu_hours=6) for sid in seeker_ids})

        # Measure prices paid to cartel vs solo provider
        for r in sim.results:
            if r.agreed and r.resource:
                price_unit = r.price / r.resource.total_units()
                if r.seller_id == "seller_cartel":
                    cartel_prices.append(price_unit)
                elif r.seller_id == "prov_solo":
                    solo_prices.append(price_unit)

        for sid in seeker_ids:
            seeker_costs.append(seeker_start[sid] - sim.agents[sid].budget)

    return {
        "cartel_price_per_unit": describe(cartel_prices) if cartel_prices else None,
        "solo_price_per_unit": describe(solo_prices) if solo_prices else None,
        "seeker_total_cost": describe(seeker_costs),
    }


# ── Part 4: Free-rider detection ──────────────────────────────────────────

def run_free_rider_test(num_trials=200, rounds=80, seed_offset=3000):
    """
    3 agents in a coalition, one is a free-rider: contributes minimal
    resources but consumes maximum.
    Measures: detection rounds, wealth comparison, expulsion effectiveness.
    """
    detection_rounds = []
    freerider_wealth = []
    honest_member_wealth = []
    expelled = 0

    for trial in range(num_trials):
        # Honest members
        honest_1 = Agent("honest_1", Resource(gpu_hours=100), 50.0, FairStrategy(), 0.3)
        honest_2 = Agent("honest_2", Resource(gpu_hours=90), 45.0, FairStrategy(), 0.3)
        # Free-rider: low resources but high urgency (will consume more)
        freerider = Agent("freerider", Resource(gpu_hours=10), 80.0, FairStrategy(), 0.9)

        coalition = Coalition(
            "test_coalition",
            [honest_1, honest_2, freerider],
            free_rider_threshold=1.8,
        )
        ca = CoalitionAgent(coalition, strategy=AdaptiveStrategy())

        # Update initial contribution records correctly
        coalition.records["honest_1"].contributed_resources = 100.0
        coalition.records["honest_2"].contributed_resources = 90.0
        coalition.records["freerider"].contributed_resources = 10.0

        other_agents = [
            Agent("prov_1", Resource(gpu_hours=300), 10.0, FairStrategy(), 0.2),
            Agent("prov_2", Resource(gpu_hours=250), 15.0, AdaptiveStrategy(), 0.2),
        ]
        all_agents = other_agents + [ca]
        start_worth_fr = freerider.net_worth()
        start_worth_h = (honest_1.net_worth() + honest_2.net_worth()) / 2

        sim = Simulator(all_agents, max_negotiation_turns=6, seed=trial + seed_offset)

        detected_round = None
        for r in range(rounds):
            ca.sync_from_members()
            sim.agents["test_coalition"].pending_needs = Resource(gpu_hours=6)
            sim.run_round(needs={"test_coalition": Resource(gpu_hours=6)})

            # Free-rider over-consumes relative to contribution
            if "freerider" in coalition.members:
                fr_rec = coalition.records["freerider"]
                # Simulate that freerider consumes 3x its weight
                fr_rec.consumed_resources += 3.0 * (6.0 / max(coalition.size(), 1))
                # Honest members consume their fair share
                for mid in ["honest_1", "honest_2"]:
                    if mid in coalition.members:
                        coalition.records[mid].consumed_resources += (6.0 / max(coalition.size(), 1))

            # Check for free-riders every 10 rounds
            if r > 0 and r % 10 == 0:
                expulsions = coalition.expel_free_riders()
                if "freerider" in expulsions and detected_round is None:
                    detected_round = r
                    expelled += 1

        detection_rounds.append(detected_round if detected_round is not None else rounds + 1)
        freerider_wealth.append(freerider.net_worth() - start_worth_fr)
        honest_member_wealth.append(
            ((honest_1.net_worth() + honest_2.net_worth()) / 2) - start_worth_h
        )

    detection_stat = describe(detection_rounds)
    detected_pct = expelled / num_trials

    return {
        "detected_pct": detected_pct,
        "detection_round": detection_stat,
        "freerider_wealth_delta": describe(freerider_wealth),
        "honest_member_wealth_delta": describe(honest_member_wealth),
    }


# ── Part 5: Coalition size sweep ──────────────────────────────────────────

def run_size_sweep(num_trials=200, rounds=50, seed_offset=4000):
    """
    Test coalition sizes from 2 to 5 members.
    Measures per-member wealth gain vs solo equivalent.
    """
    print("  Coalition Size | Avg Member Wealth Delta | Deal Rate | vs Solo")
    print("  " + "-" * 62)

    solo_result = run_solo_baseline(num_trials=100, rounds=rounds, seed_offset=seed_offset + 9000)
    solo_delta = solo_result["seeker_delta"].mean

    size_results = {}
    for size in [2, 3, 4, 5]:
        per_member_deltas = []
        deal_rates = []

        for trial in range(num_trials):
            members = [
                Agent(f"cm_{i}", Resource(gpu_hours=80), 100.0, FairStrategy(), 0.7)
                for i in range(size)
            ]
            coalition = Coalition(f"coal_{size}", members)
            for m in members:
                coalition.records[m.agent_id].contributed_resources = 80.0
            ca = CoalitionAgent(coalition, strategy=AdaptiveStrategy())

            providers = [
                Agent(f"prov_{j}", Resource(gpu_hours=200), 10.0, FairStrategy(), 0.2)
                for j in range(3)
            ]
            start_worths = {m.agent_id: m.net_worth() for m in members}
            sim = Simulator(providers + [ca], max_negotiation_turns=6, seed=trial + seed_offset + size * 100)

            for _ in range(rounds):
                ca.sync_from_members()
                sim.agents[ca.agent_id].pending_needs = Resource(gpu_hours=6 * size)
                m_obj = sim.run_round(needs={ca.agent_id: Resource(gpu_hours=6 * size)})
                deal_rates.append(m_obj.deals_made / max(m_obj.negotiations, 1))

            for member in members:
                per_member_deltas.append(member.net_worth() - start_worths[member.agent_id])

        stat = describe(per_member_deltas)
        dr = describe(deal_rates)
        vs_solo = stat.mean - solo_delta
        print(
            f"  {size:<15} {stat.mean:>+8.1f} ±{stat.std:.1f}"
            f"        {dr.mean:>5.0%}      {vs_solo:>+7.1f}"
        )
        size_results[size] = {"wealth_delta": stat, "deal_rate": dr, "vs_solo": vs_solo}

    return size_results


# ── Part 6: Coalition stability under scarcity ────────────────────────────

def run_stability_test(num_trials=200, rounds=80, seed_offset=5000):
    """
    Start with a coalition. Introduce resource scarcity mid-simulation.
    Measure whether the coalition holds together.
    """
    stable_counts = 0
    defection_rounds = []
    sizes_at_end = []

    for trial in range(num_trials):
        members = [
            Agent(f"cm_{i}", Resource(gpu_hours=100), 80.0, AdaptiveStrategy(), 0.5)
            for i in range(3)
        ]
        coalition = Coalition("stability_test", members, stability_threshold=0.75)
        for m in members:
            coalition.records[m.agent_id].contributed_resources = 100.0

        ca = CoalitionAgent(coalition, strategy=AdaptiveStrategy())
        providers = [
            Agent("prov_rich", Resource(gpu_hours=500), 10.0, FairStrategy(), 0.2),
            Agent("prov_poor", Resource(gpu_hours=30), 20.0, AdaptiveStrategy(), 0.2),
        ]
        sim = Simulator(providers + [ca], max_negotiation_turns=6, seed=trial + seed_offset)

        first_defection = None
        for r in range(rounds):
            ca.sync_from_members()

            # Phase 1 (rounds 0-29): normal market
            # Phase 2 (rounds 30-59): scarcity — cut provider resources
            if r == 30:
                sim.agents["prov_rich"].resources = Resource(gpu_hours=30)
                sim.agents["prov_poor"].resources = Resource(gpu_hours=10)

            sim.agents[ca.agent_id].pending_needs = Resource(gpu_hours=5)
            sim.run_round(needs={ca.agent_id: Resource(gpu_hours=5)})

            # Simple stability check: if a member's share is very poor, they leave
            if coalition.size() > 1:
                total_budget = coalition.pooled_budget()
                for m in list(coalition.members.values()):
                    weight = coalition.contribution_weights().get(m.agent_id, 0)
                    coalition_share = weight * total_budget
                    solo_est = m.budget * 0.9  # would keep 90% if solo
                    if coalition_share < coalition.stability_threshold * solo_est:
                        if m.agent_id in coalition.members:
                            del coalition.members[m.agent_id]
                            coalition.defected.append(m.agent_id)
                            if first_defection is None:
                                first_defection = r

        if first_defection is None:
            stable_counts += 1
        else:
            defection_rounds.append(first_defection)
        sizes_at_end.append(coalition.size())

    stability_rate = stable_counts / num_trials
    avg_end_size = sum(sizes_at_end) / len(sizes_at_end)
    avg_defection = sum(defection_rounds) / len(defection_rounds) if defection_rounds else None

    return {
        "stability_rate": stability_rate,
        "avg_end_size": avg_end_size,
        "avg_defection_round": avg_defection,
        "defection_rounds": describe(defection_rounds) if defection_rounds else None,
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 8: COALITION FORMATION")
    print("=" * 60)

    print("\n--- Part 1: Solo Baseline (200 trials, 60 rounds) ---\n")
    solo = run_solo_baseline()
    print(f"  Price/unit:       {solo['prices']}")
    print(f"  Deal rate:        {solo['deal_rate']}")
    print(f"  Seeker wealth Δ:  {solo['seeker_delta']}")

    print("\n--- Part 2: Buyer Coalition vs Solo Seekers ---\n")
    buyer = run_buyer_coalition()
    print(f"  Price/unit:               {buyer['prices']}")
    print(f"  Deal rate:                {buyer['deal_rate']}")
    print(f"  Coalition member Δ:       {buyer['coalition_delta']}")
    print(f"  Solo seeker Δ:            {buyer['solo_seeker_delta']}")
    if buyer["coalition_delta"].n > 0 and buyer["solo_seeker_delta"].n > 0:
        t, p = welch_t_test(
            [buyer["coalition_delta"].mean] * buyer["coalition_delta"].n,
            [buyer["solo_seeker_delta"].mean] * buyer["solo_seeker_delta"].n,
        )
        print(f"  Coalition vs Solo: p={p:.2e}")

    print("\n--- Part 3: Seller Coalition (Cartel) ---\n")
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
    print(f"  Free-rider wealth Δ:  {fr['freerider_wealth_delta']}")
    print(f"  Honest member Δ:      {fr['honest_member_wealth_delta']}")

    print("\n--- Part 5: Coalition Size Sweep ---\n")
    size_results = run_size_sweep()

    print("\n--- Part 6: Stability Under Scarcity ---\n")
    stab = run_stability_test()
    print(f"  Stability rate (no defection):  {stab['stability_rate']:.0%}")
    print(f"  Avg coalition size at end:       {stab['avg_end_size']:.1f}")
    if stab["avg_defection_round"]:
        print(f"  Avg defection round:             {stab['avg_defection_round']:.1f}")
    if stab["defection_rounds"]:
        print(f"  Defection round stats:           {stab['defection_rounds']}")
