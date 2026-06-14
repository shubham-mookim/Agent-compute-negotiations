"""
Simulation engine.

Runs negotiation rounds between agents, logs everything,
tracks metrics. The core loop is simple:

  1. Each round, agents with needs pick a target and negotiate
  2. Negotiations run for up to max_turns back-and-forth
  3. Accepted deals transfer resources and currency
  4. Log everything for later analysis
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.agent import Agent, Deal
from agents.protocol import Message, MessageType
from agents.resource import Resource


@dataclass
class NegotiationResult:
    buyer_id: str
    seller_id: str
    resource: Resource | None
    price: float
    agreed: bool
    rounds: int
    messages: list[Message]


@dataclass
class RoundMetrics:
    round_num: int
    negotiations: int
    deals_made: int
    total_volume: float  # total resource units traded
    total_currency: float  # total currency exchanged
    avg_price_per_unit: float
    agent_budgets: dict[str, float]
    agent_resources: dict[str, float]


class Simulator:
    def __init__(
        self,
        agents: list[Agent],
        max_negotiation_turns: int = 6,
        log_dir: str | None = None,
        seed: int | None = None,
    ):
        self.agents = {a.agent_id: a for a in agents}
        self.max_turns = max_negotiation_turns
        self.round_num = 0
        self.results: list[NegotiationResult] = []
        self.metrics: list[RoundMetrics] = []
        self.full_log: list[dict[str, Any]] = []

        if seed is not None:
            random.seed(seed)

        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _negotiate(self, buyer: Agent, seller: Agent, need: Resource) -> NegotiationResult:
        """Run a single negotiation between two agents."""
        messages: list[Message] = []

        # Buyer opens
        msg = buyer.initiate_negotiation(seller.agent_id, need)
        messages.append(msg)

        current_msg = msg
        turns = 0

        while turns < self.max_turns:
            # Seller responds
            response = seller.receive(current_msg)
            if response is None:
                break
            messages.append(response)

            if response.msg_type == MessageType.ACCEPT:
                # Deal made — notify buyer so learning strategies get their reward signal
                resource = Resource.from_dict(response.payload.get("resource", need.to_dict()))
                price = response.payload.get("price", 0)
                buyer.receive(response)
                return NegotiationResult(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    resource=resource,
                    price=price,
                    agreed=True,
                    rounds=turns + 1,
                    messages=messages,
                )

            if response.msg_type == MessageType.REJECT:
                buyer.receive(response)
                return NegotiationResult(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    resource=None,
                    price=0,
                    agreed=False,
                    rounds=turns + 1,
                    messages=messages,
                )

            # Buyer responds to counter
            buyer_response = buyer.receive(response)
            if buyer_response is None:
                break
            messages.append(buyer_response)

            if buyer_response.msg_type == MessageType.ACCEPT:
                resource = Resource.from_dict(buyer_response.payload.get("resource", need.to_dict()))
                price = buyer_response.payload.get("price", 0)
                seller.receive(buyer_response)  # notify seller of buyer's accept
                return NegotiationResult(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    resource=resource,
                    price=price,
                    agreed=True,
                    rounds=turns + 1,
                    messages=messages,
                )

            if buyer_response.msg_type == MessageType.REJECT:
                return NegotiationResult(
                    buyer_id=buyer.agent_id,
                    seller_id=seller.agent_id,
                    resource=None,
                    price=0,
                    agreed=False,
                    rounds=turns + 1,
                    messages=messages,
                )

            current_msg = buyer_response
            turns += 1

        # Timed out - no deal
        return NegotiationResult(
            buyer_id=buyer.agent_id,
            seller_id=seller.agent_id,
            resource=None,
            price=0,
            agreed=False,
            rounds=turns,
            messages=messages,
        )

    def _execute_deal(self, result: NegotiationResult) -> None:
        """Transfer resources and currency for an accepted deal."""
        if not result.agreed or result.resource is None:
            return

        buyer = self.agents[result.buyer_id]
        seller = self.agents[result.seller_id]

        # Transfer resources: seller -> buyer
        if seller.resources.can_afford(result.resource) and buyer.budget >= result.price:
            seller.resources = seller.resources - result.resource
            buyer.resources = buyer.resources + result.resource
            buyer.budget -= result.price
            seller.budget += result.price

            # Track utility: what the buyer acquired and at what cost
            buyer.total_needs_fulfilled = buyer.total_needs_fulfilled + result.resource
            buyer.total_fair_value_acquired += result.resource.total_units()
            buyer.total_price_paid += result.price

            # Record deals on both sides
            buyer.record_deal(Deal(
                partner_id=seller.agent_id,
                resource=result.resource,
                price=result.price,
                round_num=self.round_num,
                fulfilled=True,
            ))
            seller.record_deal(Deal(
                partner_id=buyer.agent_id,
                resource=result.resource,
                price=result.price,
                round_num=self.round_num,
                fulfilled=True,
            ))
        else:
            # Deal fell through at execution - mark as default
            result.agreed = False
            buyer.record_deal(Deal(
                partner_id=seller.agent_id,
                resource=result.resource,
                price=result.price,
                round_num=self.round_num,
                fulfilled=False,
            ))

    def run_round(
        self,
        needs: dict[str, Resource] | None = None,
        pairings: list[tuple[str, str]] | None = None,
    ) -> RoundMetrics:
        """
        Run one round of negotiations.

        Args:
            needs: {agent_id: Resource they want}. If None, agents with
                   pending_needs will try to negotiate.
            pairings: Explicit (buyer_id, seller_id) pairs. If None,
                      buyers randomly pick a seller.
        """
        self.round_num += 1

        # Determine who needs what
        if needs:
            for agent_id, need in needs.items():
                self.agents[agent_id].pending_needs = need

        # Track what each agent requested this round (for utility metric)
        for a in self.agents.values():
            if a.pending_needs.total_units() > 0:
                a.total_needs_requested = a.total_needs_requested + a.pending_needs

        buyers = [a for a in self.agents.values() if a.pending_needs.total_units() > 0]
        sellers = [a for a in self.agents.values() if a.pending_needs.total_units() == 0]

        if not sellers:
            # Everyone is a buyer, pair them randomly
            sellers = list(self.agents.values())

        round_results: list[NegotiationResult] = []

        if pairings:
            for buyer_id, seller_id in pairings:
                buyer = self.agents[buyer_id]
                seller = self.agents[seller_id]
                result = self._negotiate(buyer, seller, buyer.pending_needs)
                self._execute_deal(result)
                round_results.append(result)
        else:
            for buyer in buyers:
                # Pick a random seller (not self)
                available = [s for s in sellers if s.agent_id != buyer.agent_id]
                if not available:
                    continue
                seller = random.choice(available)
                result = self._negotiate(buyer, seller, buyer.pending_needs)
                self._execute_deal(result)
                round_results.append(result)

        self.results.extend(round_results)

        # Compute metrics
        deals = [r for r in round_results if r.agreed]
        total_volume = sum(r.resource.total_units() for r in deals if r.resource)
        total_currency = sum(r.price for r in deals)
        avg_price = total_currency / total_volume if total_volume > 0 else 0

        metrics = RoundMetrics(
            round_num=self.round_num,
            negotiations=len(round_results),
            deals_made=len(deals),
            total_volume=total_volume,
            total_currency=total_currency,
            avg_price_per_unit=avg_price,
            agent_budgets={a.agent_id: a.budget for a in self.agents.values()},
            agent_resources={a.agent_id: a.resources.total_units() for a in self.agents.values()},
        )
        self.metrics.append(metrics)

        # Log
        self.full_log.append({
            "round": self.round_num,
            "negotiations": len(round_results),
            "deals": len(deals),
            "volume": total_volume,
            "currency": total_currency,
            "avg_price": avg_price,
        })

        return metrics

    def run(self, rounds: int, needs_per_round: dict[str, Resource] | None = None) -> list[RoundMetrics]:
        """Run multiple rounds."""
        all_metrics = []
        for _ in range(rounds):
            m = self.run_round(needs=needs_per_round)
            all_metrics.append(m)
        return all_metrics

    def summary(self) -> str:
        """Human-readable summary of the simulation."""
        total_deals = sum(1 for r in self.results if r.agreed)
        total_negotiations = len(self.results)
        deal_rate = total_deals / total_negotiations if total_negotiations else 0

        lines = [
            f"=== Simulation Summary ({self.round_num} rounds) ===",
            f"Total negotiations: {total_negotiations}",
            f"Deals made: {total_deals} ({deal_rate:.0%} success rate)",
            "",
            "Agent states:",
        ]
        for agent in self.agents.values():
            lines.append(
                f"  {agent.agent_id}: budget={agent.budget:.1f}, "
                f"resources={agent.resources.total_units():.1f}, "
                f"deals={len(agent.deals)}"
            )
            if agent.reputation_table:
                rep_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in agent.reputation_table.items()
                )
                lines.append(f"    reputation: {rep_str}")
        return "\n".join(lines)

    def save_log(self, filename: str = "simulation_log.json") -> str:
        """Save full log to JSON file."""
        path = self.log_dir / filename if self.log_dir else Path(filename)
        with open(path, "w") as f:
            json.dump(self.full_log, f, indent=2)
        return str(path)
