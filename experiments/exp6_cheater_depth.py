#!/usr/bin/env python3
"""
Experiment 6: Deep Cheater Analysis

Goes beyond Experiment 3 to answer specific research questions:

1. Detection threshold: At what cheat rate does the network reliably
   detect a bad actor? Is there a sharp phase transition?

2. Collaborative reputation: If agents share bad experiences with
   peers, does it close the detection gap?

3. Adaptive cheater: An agent that adjusts its cheat rate based on
   its own reputation — can it game the system indefinitely?

4. Multiple cheaters: What happens when 2-3 cheaters coordinate?

5. Network topology: Dense vs sparse reputation sharing.
"""

import sys
import random
import math
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Resource, MessageType
from agents.agent import Deal
from agents.protocol import Message
from agents.strategies import FairStrategy, AdaptiveStrategy
from agents.stats import describe, welch_t_test, cohens_d
from experiments.exp3_trust import (
    CheaterStrategy, TrustSimulator, ReputationAwareStrategy,
    setup_trust_agents, run_cheater_detection,
)


# ──────────────────────────────────────────────────────────
# Part 1: Detection Threshold Sweep
# ──────────────────────────────────────────────────────────

def detection_threshold_sweep(num_trials=200, rounds=80):
    """
    Sweep cheat rate from 1% to 100% and find the detection threshold.
    Detection = cheater's average reputation drops below 0.3.
    """
    print("=== Part 1: Detection Threshold Sweep ===\n")

    cheat_rates = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    print(f"{'Rate':>6} {'Detection%':>10} {'Avg Final Rep':>14} {'Avg Wealth':>11} {'Avg Incidents':>14}")
    print("-" * 59)

    detection_curve = []

    for rate in cheat_rates:
        detected = 0
        final_reps = []
        wealths = []
        incident_counts = []

        for seed in range(num_trials):
            sim, rep_history, deals = run_cheater_detection(
                cheater_prob=rate, rounds=rounds, seed=seed,
            )
            final_rep = rep_history[-1] if rep_history else 0.5
            final_reps.append(final_rep)
            wealths.append(sim.agents["cheater_mallory"].net_worth())
            incident_counts.append(len(sim.default_log))
            if final_rep < 0.3:
                detected += 1

        detect_rate = detected / num_trials
        detection_curve.append((rate, detect_rate))
        rep_stat = describe(final_reps)
        w_stat = describe(wealths)
        i_stat = describe([float(x) for x in incident_counts])

        print(f"{rate:>5.0%} {detect_rate:>9.0%} {rep_stat.mean:>13.3f} {w_stat.mean:>10.1f} {i_stat.mean:>13.1f}")

    # Find threshold (first rate where detection > 50%)
    threshold = None
    for rate, detect in detection_curve:
        if detect > 0.50:
            threshold = rate
            break

    print(f"\nDetection threshold (>50% detection): {threshold if threshold else 'Not reached'}")

    # Check for phase transition sharpness
    if len(detection_curve) >= 3:
        diffs = []
        for i in range(1, len(detection_curve)):
            rate_diff = detection_curve[i][0] - detection_curve[i-1][0]
            detect_diff = detection_curve[i][1] - detection_curve[i-1][1]
            if rate_diff > 0:
                diffs.append(detect_diff / rate_diff)
        max_slope = max(diffs) if diffs else 0
        print(f"Max slope of detection curve: {max_slope:.2f} (higher = sharper transition)")

    return detection_curve


# ──────────────────────────────────────────────────────────
# Part 2: Collaborative Reputation
# ──────────────────────────────────────────────────────────

class CollaborativeReputationStrategy:
    """
    Like ReputationAwareStrategy but agents gossip — they share
    negative reputation info with peers they trust.
    """

    def __init__(self, inner_strategy, trust_threshold=0.3, gossip_rate=0.5):
        self.inner = inner_strategy
        self.trust_threshold = trust_threshold
        self.gossip_rate = gossip_rate

    def initiate(self, agent, target_id, need):
        return self.inner.initiate(agent, target_id, need)

    def decide(self, agent, msg):
        rep = agent.reputation_of(msg.sender_id)
        if rep < self.trust_threshold:
            return msg.reply(MessageType.REJECT, {"reason": "low_reputation", "your_reputation": rep})
        return self.inner.decide(agent, msg)


class CollaborativeTrustSimulator(TrustSimulator):
    """Simulator where honest agents share reputation info after each round."""

    def __init__(self, *args, gossip_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gossip_rate = gossip_rate

    def run_round(self, *args, **kwargs):
        metrics = super().run_round(*args, **kwargs)
        self._gossip_round()
        return metrics

    def _gossip_round(self):
        """Agents share negative reputation info with random peers."""
        honest = [a for aid, a in self.agents.items() if aid not in self.cheater_ids]
        for agent in honest:
            for peer_id, rep in list(agent.reputation_table.items()):
                if rep < 0.4:
                    # Share this bad reputation with a random honest peer
                    if random.random() < self.gossip_rate:
                        target = random.choice(honest)
                        if target.agent_id != agent.agent_id:
                            their_rep = target.reputation_of(peer_id)
                            # Weighted average — don't blindly trust gossip
                            blended = 0.7 * their_rep + 0.3 * rep
                            target.reputation_table[peer_id] = blended


def run_collaborative_vs_isolated(cheat_rate=0.10, num_trials=200, rounds=60):
    """Compare cheater detection with and without gossip."""
    print(f"\n=== Part 2: Collaborative Reputation (cheat rate={cheat_rate:.0%}) ===\n")

    isolated_reps = []
    collab_reps = []
    isolated_detected = 0
    collab_detected = 0

    for seed in range(num_trials):
        # Isolated (standard)
        sim_iso, rep_iso, _ = run_cheater_detection(cheater_prob=cheat_rate, rounds=rounds, seed=seed)
        final_iso = rep_iso[-1] if rep_iso else 0.5
        isolated_reps.append(final_iso)
        if final_iso < 0.3:
            isolated_detected += 1

        # Collaborative
        agents = setup_trust_agents(cheat_rate)
        # Replace strategies with collaborative versions
        for a in agents:
            if "cheater" not in a.agent_id and isinstance(a.strategy, ReputationAwareStrategy):
                a.strategy = CollaborativeReputationStrategy(a.strategy.inner, gossip_rate=0.5)

        sim_col = CollaborativeTrustSimulator(agents, max_negotiation_turns=6, seed=seed, gossip_rate=0.5)
        sim_col.mark_cheater("cheater_mallory")

        honest_ids = [a.agent_id for a in agents if "cheater" not in a.agent_id]
        all_ids = [a.agent_id for a in agents]

        for _ in range(rounds):
            seekers = random.sample(honest_ids, k=min(3, len(honest_ids)))
            needs = {sid: Resource(gpu_hours=5) for sid in seekers}
            if random.random() < 0.5:
                needs["cheater_mallory"] = Resource(gpu_hours=5)
            for aid in all_ids:
                sim_col.agents[aid].pending_needs = needs.get(aid, Resource())
            sim_col.run_round(needs=needs)

        col_reps_round = []
        for aid in honest_ids:
            col_reps_round.append(sim_col.agents[aid].reputation_of("cheater_mallory"))
        final_col = sum(col_reps_round) / len(col_reps_round) if col_reps_round else 0.5
        collab_reps.append(final_col)
        if final_col < 0.3:
            collab_detected += 1

    iso_stat = describe(isolated_reps)
    col_stat = describe(collab_reps)
    t, p = welch_t_test(isolated_reps, collab_reps)
    d = cohens_d(isolated_reps, collab_reps)

    print(f"Isolated:      detection={isolated_detected/num_trials:.0%}, final rep={iso_stat}")
    print(f"Collaborative: detection={collab_detected/num_trials:.0%}, final rep={col_stat}")
    print(f"Welch's t={t:.3f}, p={p:.2e}, Cohen's d={d:.3f}")
    if p < 0.05:
        print("Result: Collaborative reputation SIGNIFICANTLY improves detection")
    else:
        print("Result: No significant improvement from collaboration")

    return {"isolated": iso_stat, "collaborative": col_stat, "p": p, "d": d}


# ──────────────────────────────────────────────────────────
# Part 3: Adaptive Cheater
# ──────────────────────────────────────────────────────────

@dataclass
class AdaptiveCheaterStrategy:
    """
    A cheater that monitors its own reputation (inferred from
    deal success rate) and adjusts cheat rate to stay undetected.
    """
    base_cheat_rate: float = 0.3
    current_cheat_rate: float = 0.3
    min_cheat_rate: float = 0.02
    max_cheat_rate: float = 0.5
    # Track recent outcomes
    recent_deals: int = 0
    recent_rejections: int = 0
    adjustment_window: int = 10
    _total_interactions: int = 0
    _accepted_count: int = 0
    _cheated_count: int = 0
    cheat_probability: float = 0.3  # for TrustSimulator compatibility

    def _adjust_rate(self):
        """If getting rejected more, reduce cheat rate. If not, increase."""
        if self._total_interactions < self.adjustment_window:
            return
        rejection_rate = self.recent_rejections / max(self.recent_rejections + self.recent_deals, 1)
        if rejection_rate > 0.5:
            # Being detected — reduce cheating
            self.current_cheat_rate = max(self.min_cheat_rate, self.current_cheat_rate * 0.7)
        elif rejection_rate < 0.2:
            # Flying under radar — increase cheating
            self.current_cheat_rate = min(self.max_cheat_rate, self.current_cheat_rate * 1.1)
        self.cheat_probability = self.current_cheat_rate
        self.recent_deals = 0
        self.recent_rejections = 0

    def initiate(self, agent, target_id, need):
        price = need.total_units()
        return Message(
            msg_type=MessageType.REQUEST,
            sender_id=agent.agent_id,
            receiver_id=target_id,
            payload={"resource": need.to_dict(), "max_price": price * 1.2, "urgency": 0.7},
        )

    def decide(self, agent, msg):
        self._total_interactions += 1

        if msg.msg_type == MessageType.REQUEST:
            requested = Resource.from_dict(msg.payload["resource"])
            their_max = msg.payload.get("max_price", 0)
            price = requested.total_units()
            if their_max >= price * 0.8:
                self.recent_deals += 1
                self._adjust_rate()
                return msg.reply(MessageType.ACCEPT, {
                    "resource": requested.to_dict(),
                    "price": min(price, their_max),
                })
            return msg.reply(MessageType.REJECT, {"reason": "not_enough"})

        elif msg.msg_type in (MessageType.OFFER, MessageType.COUNTER):
            price = msg.payload.get("price", float("inf"))
            resource = Resource.from_dict(msg.payload["resource"])
            if price <= resource.total_units() * 1.3:
                self.recent_deals += 1
                self._adjust_rate()
                return msg.reply(MessageType.ACCEPT, {"resource": resource.to_dict(), "price": price})
            return msg.reply(MessageType.REJECT, {"reason": "too_much"})

        elif msg.msg_type == MessageType.REJECT:
            self.recent_rejections += 1
            self._adjust_rate()
            return None

        return None


def run_adaptive_cheater(num_trials=200, rounds=100):
    """Test whether an adaptive cheater can game the system indefinitely."""
    print(f"\n=== Part 3: Adaptive Cheater ({num_trials} trials, {rounds} rounds) ===\n")

    final_reps = []
    final_rates = []
    final_wealths = []
    total_cheats = []
    detected = 0

    for seed in range(num_trials):
        random.seed(seed)
        agents_list = [
            Agent("honest_alice", Resource(gpu_hours=100), 50.0, ReputationAwareStrategy(FairStrategy()), 0.3),
            Agent("honest_bob", Resource(gpu_hours=80), 60.0, ReputationAwareStrategy(AdaptiveStrategy()), 0.4),
            Agent("honest_carol", Resource(gpu_hours=90), 45.0, ReputationAwareStrategy(FairStrategy()), 0.2),
            Agent("honest_dave", Resource(gpu_hours=70), 70.0, ReputationAwareStrategy(AdaptiveStrategy()), 0.5),
            Agent("seeker_eve", Resource(gpu_hours=5), 100.0, ReputationAwareStrategy(FairStrategy()), 0.7),
            Agent("adaptive_cheater", Resource(gpu_hours=100), 50.0, AdaptiveCheaterStrategy(base_cheat_rate=0.3), 0.3),
        ]

        sim = TrustSimulator(agents_list, max_negotiation_turns=6, seed=seed)
        sim.mark_cheater("adaptive_cheater")

        honest_ids = [a.agent_id for a in agents_list if "cheater" not in a.agent_id and "adaptive_cheater" not in a.agent_id]
        all_ids = [a.agent_id for a in agents_list]

        for _ in range(rounds):
            seekers = random.sample(honest_ids, k=min(3, len(honest_ids)))
            needs = {sid: Resource(gpu_hours=5) for sid in seekers}
            if random.random() < 0.5:
                needs["adaptive_cheater"] = Resource(gpu_hours=5)
            for aid in all_ids:
                sim.agents[aid].pending_needs = needs.get(aid, Resource())
            sim.run_round(needs=needs)

        # Final reputation of adaptive cheater
        reps = [sim.agents[aid].reputation_of("adaptive_cheater") for aid in honest_ids]
        avg_rep = sum(reps) / len(reps) if reps else 0.5
        final_reps.append(avg_rep)

        cheater_strat = sim.agents["adaptive_cheater"].strategy
        final_rates.append(cheater_strat.current_cheat_rate)
        final_wealths.append(sim.agents["adaptive_cheater"].net_worth())
        total_cheats.append(cheater_strat._cheated_count)

        if avg_rep < 0.3:
            detected += 1

    print(f"Detection rate: {detected/num_trials:.0%}")
    print(f"Final reputation:  {describe(final_reps)}")
    print(f"Final cheat rate:  {describe(final_rates)}")
    print(f"Final wealth:      {describe(final_wealths)}")
    print(f"Total cheats:      {describe([float(c) for c in total_cheats])}")

    return {
        "detection_rate": detected / num_trials,
        "final_reps": describe(final_reps),
        "final_rates": describe(final_rates),
        "final_wealths": describe(final_wealths),
    }


# ──────────────────────────────────────────────────────────
# Part 4: Multiple Cheaters
# ──────────────────────────────────────────────────────────

def run_multiple_cheaters(num_cheaters=2, num_trials=200, rounds=60):
    """What happens when multiple cheaters are in the same market?"""
    print(f"\n=== Part 4: {num_cheaters} Cheaters in Market ({num_trials} trials) ===\n")

    honest_final_wealth = []
    cheater_final_wealth = []
    cheater_reps = []

    for seed in range(num_trials):
        random.seed(seed)
        agents_list = [
            Agent("honest_1", Resource(gpu_hours=100), 60.0, ReputationAwareStrategy(FairStrategy()), 0.3),
            Agent("honest_2", Resource(gpu_hours=80), 70.0, ReputationAwareStrategy(AdaptiveStrategy()), 0.4),
            Agent("honest_3", Resource(gpu_hours=90), 55.0, ReputationAwareStrategy(FairStrategy()), 0.5),
            Agent("seeker_1", Resource(gpu_hours=5), 100.0, ReputationAwareStrategy(FairStrategy()), 0.7),
        ]

        for ci in range(num_cheaters):
            agents_list.append(
                Agent(f"cheater_{ci}", Resource(gpu_hours=80), 40.0, CheaterStrategy(cheat_probability=0.3), 0.3)
            )

        sim = TrustSimulator(agents_list, max_negotiation_turns=6, seed=seed)
        for ci in range(num_cheaters):
            sim.mark_cheater(f"cheater_{ci}")

        honest_ids = [a.agent_id for a in agents_list if "cheater" not in a.agent_id]
        all_ids = [a.agent_id for a in agents_list]

        for _ in range(rounds):
            seekers = random.sample(honest_ids, k=min(2, len(honest_ids)))
            needs = {sid: Resource(gpu_hours=5) for sid in seekers}
            for ci in range(num_cheaters):
                if random.random() < 0.4:
                    needs[f"cheater_{ci}"] = Resource(gpu_hours=5)
            for aid in all_ids:
                sim.agents[aid].pending_needs = needs.get(aid, Resource())
            sim.run_round(needs=needs)

        for aid in honest_ids:
            honest_final_wealth.append(sim.agents[aid].net_worth())
        for ci in range(num_cheaters):
            cid = f"cheater_{ci}"
            cheater_final_wealth.append(sim.agents[cid].net_worth())
            reps = [sim.agents[aid].reputation_of(cid) for aid in honest_ids]
            cheater_reps.append(sum(reps) / len(reps) if reps else 0.5)

    print(f"Honest agents wealth:  {describe(honest_final_wealth)}")
    print(f"Cheater agents wealth: {describe(cheater_final_wealth)}")
    print(f"Cheater reputations:   {describe(cheater_reps)}")

    t, p = welch_t_test(honest_final_wealth, cheater_final_wealth)
    print(f"Wealth difference: t={t:.3f}, p={p:.2e}")

    return {"honest_wealth": describe(honest_final_wealth), "cheater_wealth": describe(cheater_final_wealth)}


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 6: DEEP CHEATER ANALYSIS")
    print("=" * 60)
    print()

    detection_threshold_sweep()
    run_collaborative_vs_isolated(cheat_rate=0.10)
    run_collaborative_vs_isolated(cheat_rate=0.05)
    run_adaptive_cheater()
    run_multiple_cheaters(num_cheaters=2)
    run_multiple_cheaters(num_cheaters=3)
