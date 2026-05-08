"""
Coalition Formation for Multi-Agent Compute Negotiation.

A coalition is a group of agents that pool resources and negotiate
collectively. Key dynamics modelled:

  - Pooled bargaining power (larger coalition → stronger position)
  - Profit sharing by contribution weight (Shapley-inspired)
  - Free-rider detection: agents that consume > their contribution
  - Stability: members defect if coalition share < solo expected value
  - Expulsion: persistent free-riders get kicked out

Design: a Coalition wraps a list of Agents. The simulator sees it as a
single Agent (CoalitionAgent). After each deal, gains/costs are
distributed back to members. Each round, members re-evaluate staying.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent

from agents.agent import Agent as _Agent, Deal
from agents.protocol import Message, MessageType
from agents.resource import Resource
from agents.strategies import FairStrategy, AdaptiveStrategy


@dataclass
class MemberRecord:
    agent_id: str
    contributed_resources: float = 0.0   # total units ever contributed to pool
    consumed_resources: float = 0.0      # total units ever received from pool
    contributed_budget: float = 0.0      # total currency contributed
    received_income: float = 0.0         # total currency received back

    @property
    def free_rider_score(self) -> float:
        """consumption/contribution ratio. >1.5 = exploiting the coalition."""
        if self.contributed_resources == 0:
            return 2.0 if self.consumed_resources > 0 else 1.0
        return self.consumed_resources / self.contributed_resources


class Coalition:
    """
    Manages coalition membership, resource pooling, and profit distribution.
    """

    def __init__(
        self,
        coalition_id: str,
        members: list[_Agent],
        inner_strategy=None,
        free_rider_threshold: float = 2.0,
        stability_threshold: float = 0.80,
    ):
        self.coalition_id = coalition_id
        self.members: dict[str, _Agent] = {a.agent_id: a for a in members}
        self.inner_strategy = inner_strategy or FairStrategy()
        self.free_rider_threshold = free_rider_threshold
        self.stability_threshold = stability_threshold
        self.records: dict[str, MemberRecord] = {
            aid: MemberRecord(aid) for aid in self.members
        }
        self.expelled: list[str] = []
        self.defected: list[str] = []
        self.rounds_tracked = 0

        # Initial contribution snapshot
        for agent in members:
            self.records[agent.agent_id].contributed_resources = (
                agent.resources.total_units()
            )
            self.records[agent.agent_id].contributed_budget = agent.budget

    # ── Resource / budget pooling ──────────────────────────────────

    def pooled_resources(self) -> Resource:
        total = Resource()
        for a in self.members.values():
            total = total + a.resources
        return total

    def pooled_budget(self) -> float:
        return sum(a.budget for a in self.members.values())

    def contribution_weights(self) -> dict[str, float]:
        """Share of each member based on contributed resources + budget."""
        scores = {}
        for aid, rec in self.records.items():
            scores[aid] = rec.contributed_resources + rec.contributed_budget * 0.1
        total = sum(scores.values()) or 1.0
        return {aid: s / total for aid, s in scores.items()}

    # ── Deal distribution ──────────────────────────────────────────

    def distribute_purchase(self, resource: Resource, price: float) -> None:
        """Coalition bought compute — split resources gained, costs paid proportionally."""
        weights = self.contribution_weights()
        for aid, weight in weights.items():
            if aid not in self.members:
                continue
            agent = self.members[aid]
            share_res = resource * weight
            share_cost = price * weight
            agent.resources = agent.resources + share_res
            agent.budget = max(0.0, agent.budget - share_cost)
            self.records[aid].consumed_resources += share_res.total_units()

    def distribute_sale(self, resource: Resource, price: float) -> None:
        """Coalition sold compute — split resources given up, income gained proportionally."""
        weights = self.contribution_weights()
        for aid, weight in weights.items():
            if aid not in self.members:
                continue
            agent = self.members[aid]
            share_res = resource * weight
            share_income = price * weight
            agent.resources = agent.resources - share_res
            agent.budget += share_income
            self.records[aid].received_income += share_income
            self.records[aid].contributed_resources += share_res.total_units()

    # ── Free-rider detection ───────────────────────────────────────

    def detect_free_riders(self) -> list[str]:
        """Return agent IDs with free_rider_score above threshold."""
        return [
            aid for aid, rec in self.records.items()
            if rec.free_rider_score > self.free_rider_threshold
            and self.records[aid].consumed_resources > 2.0  # enough history
        ]

    def expel_free_riders(self) -> list[str]:
        """Expel free-riders from the coalition."""
        to_expel = self.detect_free_riders()
        for aid in to_expel:
            if aid in self.members:
                del self.members[aid]
                self.expelled.append(aid)
        return to_expel

    # ── Stability check ────────────────────────────────────────────

    def stability_check(self, solo_estimates: dict[str, float]) -> list[str]:
        """
        Members defect if their expected coalition share < threshold * solo value.
        solo_estimates: {agent_id: expected_solo_net_worth_gain_per_round}
        Returns list of defecting agent IDs.
        """
        weights = self.contribution_weights()
        defectors = []

        for aid, solo_est in solo_estimates.items():
            if aid not in self.members:
                continue
            coalition_share = weights.get(aid, 0) * self._estimate_coalition_value()
            if solo_est > 0 and coalition_share < self.stability_threshold * solo_est:
                defectors.append(aid)

        for aid in defectors:
            if aid in self.members:
                del self.members[aid]
                self.defected.append(aid)

        return defectors

    def _estimate_coalition_value(self) -> float:
        """Rough estimate of coalition's total value per round (average income)."""
        total_income = sum(r.received_income for r in self.records.values())
        rounds = max(self.rounds_tracked, 1)
        return total_income / rounds

    # ── Info ───────────────────────────────────────────────────────

    def size(self) -> int:
        return len(self.members)

    def member_stats(self) -> str:
        lines = [f"Coalition {self.coalition_id} ({self.size()} members):"]
        for aid, rec in self.records.items():
            status = "ACTIVE" if aid in self.members else ("EXPELLED" if aid in self.expelled else "DEFECTED")
            lines.append(
                f"  {aid}: contrib={rec.contributed_resources:.1f}, "
                f"consumed={rec.consumed_resources:.1f}, "
                f"FRS={rec.free_rider_score:.2f} [{status}]"
            )
        return "\n".join(lines)


class CoalitionAgent(_Agent):
    """
    A virtual agent that represents a coalition in the simulator.
    The simulator treats it like any other agent.
    After deals, the coalition distributes gains/costs to real members.
    """

    def __init__(self, coalition: Coalition, strategy=None):
        # Snapshot pooled resources/budget at creation
        pooled_res = coalition.pooled_resources()
        pooled_bud = coalition.pooled_budget()

        super().__init__(
            agent_id=coalition.coalition_id,
            resources=pooled_res,
            budget=pooled_bud,
            strategy=strategy or coalition.inner_strategy,
            urgency=max(a.urgency for a in coalition.members.values()),
        )
        self.coalition = coalition

    def sync_from_members(self) -> None:
        """Re-sync pooled resources/budget from current member state."""
        self.resources = self.coalition.pooled_resources()
        self.budget = self.coalition.pooled_budget()

    def record_deal(self, deal: Deal) -> None:
        """Override to distribute deal back to members."""
        super().record_deal(deal)
        if deal.fulfilled and deal.resource is not None:
            if deal.price > 0:  # we paid (bought)
                self.coalition.distribute_purchase(deal.resource, deal.price)
            else:               # we received (sold)
                self.coalition.distribute_sale(deal.resource, abs(deal.price))
        self.sync_from_members()
