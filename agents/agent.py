"""
Core Agent class.

An agent has resources, needs, a strategy for negotiation,
and a local view of peer reputations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from agents.protocol import Message, MessageType
from agents.resource import Resource


class NegotiationStrategy(Protocol):
    """Interface for pluggable negotiation strategies."""

    def decide(self, agent: Agent, incoming: Message) -> Message | None:
        """Given an incoming message, decide what to do."""
        ...

    def initiate(self, agent: Agent, target_id: str, need: Resource) -> Message:
        """Create an opening request for resources."""
        ...


@dataclass
class Deal:
    """Record of a completed deal."""
    partner_id: str
    resource: Resource
    price: float  # abstract currency
    round_num: int
    fulfilled: bool = True


@dataclass
class Agent:
    agent_id: str
    resources: Resource
    budget: float  # abstract currency for paying for compute
    strategy: NegotiationStrategy | None = None
    urgency: float = 0.5  # 0=can wait, 1=need it now

    # Internal state
    pending_needs: Resource = field(default_factory=Resource)
    deals: list[Deal] = field(default_factory=list)
    reputation_table: dict[str, float] = field(default_factory=dict)
    message_log: list[Message] = field(default_factory=list)
    _active_negotiations: dict[str, dict] = field(default_factory=dict)

    # Utility tracking — cumulative across rounds
    total_needs_requested: Resource = field(default_factory=Resource)
    total_needs_fulfilled: Resource = field(default_factory=Resource)
    total_fair_value_acquired: float = 0.0
    total_price_paid: float = 0.0

    def reputation_of(self, other_id: str) -> float:
        """Get our local reputation score for another agent. Default 0.5 (neutral)."""
        return self.reputation_table.get(other_id, 0.5)

    def update_reputation(self, other_id: str, delta: float) -> None:
        """Adjust reputation score for a peer. Clamp to [0, 1]."""
        current = self.reputation_of(other_id)
        self.reputation_table[other_id] = max(0.0, min(1.0, current + delta))

    def record_deal(self, deal: Deal) -> None:
        self.deals.append(deal)
        # Fulfilled deals boost reputation, defaults hurt it
        delta = 0.1 if deal.fulfilled else -0.3
        self.update_reputation(deal.partner_id, delta)

    def receive(self, msg: Message) -> Message | None:
        """Process an incoming message using our strategy."""
        self.message_log.append(msg)
        if self.strategy is None:
            return None
        return self.strategy.decide(self, msg)

    def initiate_negotiation(self, target_id: str, need: Resource) -> Message:
        """Start a negotiation with another agent."""
        if self.strategy is None:
            raise RuntimeError(f"Agent {self.agent_id} has no strategy")
        return self.strategy.initiate(self, target_id, need)

    def net_worth(self) -> float:
        """Simple metric: budget + resource value."""
        return self.budget + self.resources.total_units()

    def utility(self, w_fulfill: float = 0.7, w_efficiency: float = 0.3) -> float:
        """Needs-based utility: rewards acquiring compute, penalizes overpaying."""
        requested = self.total_needs_requested.total_units()
        fulfilled = self.total_needs_fulfilled.total_units()
        fulfill_ratio = fulfilled / requested if requested > 0 else 0.0
        efficiency = (self.total_fair_value_acquired / self.total_price_paid
                      if self.total_price_paid > 0 else 0.0)
        efficiency = min(efficiency, 1.0)
        return w_fulfill * fulfill_ratio + w_efficiency * efficiency

    def __repr__(self) -> str:
        return (
            f"Agent({self.agent_id}, resources={self.resources}, "
            f"budget={self.budget:.1f}, urgency={self.urgency:.2f})"
        )
