"""
Reinforcement Learning (Q-Learning) Negotiation Strategy — Tier 2 Intelligence.

Sits between rule-based (Tier 1) and LLM (Tier 3). Uses tabular Q-learning
to learn a negotiation policy purely from experience across many rounds.

State space (60 states):
  msg_type     : request | counter  (2)
  reputation   : low | medium | high  (3)
  price_ratio  : very_low | low | fair | high | very_high  (5)  — their price / fair price
  urgency      : low | high  (2)

Actions (5):
  0 = ACCEPT
  1 = REJECT
  2 = COUNTER at 0.75 * fair  (aggressive)
  3 = COUNTER at 1.00 * fair  (fair)
  4 = COUNTER at 1.25 * fair  (premium)

Rewards (episodic — assigned when negotiation concludes):
  Deal as buyer  : fair_price - paid_price  (positive = saved money)
  Deal as seller : paid_price - fair_price  (positive = charged premium)
  No deal        : -0.5  (opportunity cost)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent

from agents.protocol import Message, MessageType
from agents.resource import Resource


@dataclass
class QLearningStrategy:
    alpha: float = 0.15        # learning rate
    gamma: float = 0.90        # discount factor
    epsilon: float = 0.40      # initial exploration probability
    epsilon_min: float = 0.05  # floor for exploration
    epsilon_decay: float = 0.998

    q_table: dict[tuple, float] = field(default_factory=dict)

    # Per-negotiation tracking
    _last_state: tuple | None = None
    _last_action: int | None = None
    _current_fair: float = 0.0
    _role: str = "buyer"  # "buyer" or "seller" for reward sign
    _step: int = 0

    # Diagnostics
    deals_made: int = 0
    total_reward: float = 0.0
    reward_history: list[float] = field(default_factory=list)

    # ── State encoding ─────────────────────────────────────────────

    def _rep_bucket(self, rep: float) -> int:
        return 0 if rep < 0.40 else (1 if rep < 0.70 else 2)

    def _price_bucket(self, their_price: float, fair: float) -> int:
        ratio = their_price / max(fair, 0.001)
        if ratio < 0.70:  return 0  # very low
        if ratio < 0.90:  return 1  # low
        if ratio < 1.10:  return 2  # fair zone
        if ratio < 1.35:  return 3  # high
        return 4                     # very high

    def _state(self, agent: "Agent", msg: Message) -> tuple:
        rep = agent.reputation_of(msg.sender_id)
        resource = Resource.from_dict(msg.payload.get("resource", {}))
        fair = resource.total_units() if resource.total_units() > 0 else self._current_fair

        their_price = (
            msg.payload.get("price")
            or msg.payload.get("max_price")
            or fair
        )

        type_idx = 0 if msg.msg_type == MessageType.REQUEST else 1
        return (
            type_idx,
            self._rep_bucket(rep),
            self._price_bucket(float(their_price), fair),
            0 if agent.urgency < 0.5 else 1,
        )

    # ── Q-table operations ─────────────────────────────────────────

    def _q(self, state: tuple, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def _best_action(self, state: tuple) -> int:
        return max(range(5), key=lambda a: self._q(state, a))

    def _pick_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return self._best_action(state)

    def _update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        current = self._q(state, action)
        best_next = max(self._q(next_state, a) for a in range(5))
        self.q_table[(state, action)] = (
            current + self.alpha * (reward + self.gamma * best_next - current)
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._step += 1
        self.total_reward += reward
        self.reward_history.append(reward)

    # ── Action execution ───────────────────────────────────────────

    def _counter_message(
        self, agent: "Agent", msg: Message, price_factor: float
    ) -> Message:
        resource = Resource.from_dict(msg.payload.get("resource", {}))
        fair = resource.total_units() if resource.total_units() > 0 else self._current_fair
        price = fair * price_factor
        return msg.reply(MessageType.COUNTER, {
            "resource": resource.to_dict(),
            "price": price,
        })

    def _execute(self, agent: "Agent", msg: Message, action: int) -> Message | None:
        resource = Resource.from_dict(msg.payload.get("resource", {}))
        fair = resource.total_units()
        their_price = msg.payload.get("price") or msg.payload.get("max_price") or fair

        if action == 0:  # ACCEPT
            if agent.budget >= float(their_price):
                # Self-reward immediately — don't wait for a callback that may never come
                reward = (fair - float(their_price)) if self._role == "buyer" else (float(their_price) - fair)
                if self._last_state is not None:
                    self._update(self._last_state, self._last_action, reward, ())
                    self._last_state = None
                    self._last_action = None
                self.deals_made += 1
                return msg.reply(MessageType.ACCEPT, {
                    "resource": resource.to_dict(),
                    "price": their_price,
                })
            action = 3  # fall back to fair counter if can't afford

        if action == 1:  # REJECT
            return msg.reply(MessageType.REJECT, {"reason": "rl_reject"})

        factors = {2: 0.75, 3: 1.00, 4: 1.25}
        return self._counter_message(agent, msg, factors[action])

    # ── Main interface ─────────────────────────────────────────────

    def initiate(self, agent: "Agent", target_id: str, need: Resource) -> Message:
        self._current_fair = need.total_units()
        self._role = "buyer"
        price = need.total_units() * (1.0 + 0.1 * (1 - agent.urgency))
        # Track initial state so direct-accept rewards get logged in reward_history
        self._last_state = (0, self._rep_bucket(agent.reputation_of(target_id)), 2,
                            0 if agent.urgency < 0.5 else 1)
        self._last_action = 3  # treat as "sent fair-price request"
        return Message(
            msg_type=MessageType.REQUEST,
            sender_id=agent.agent_id,
            receiver_id=target_id,
            payload={
                "resource": need.to_dict(),
                "max_price": price,
                "urgency": agent.urgency,
            },
        )

    def decide(self, agent: "Agent", msg: Message) -> Message | None:
        # ── Terminal messages: compute reward, update Q-table ──
        if msg.msg_type == MessageType.ACCEPT:
            resource = Resource.from_dict(msg.payload.get("resource", {}))
            price = float(msg.payload.get("price", 0))
            fair = resource.total_units() if resource.total_units() > 0 else self._current_fair

            if self._role == "buyer":
                reward = fair - price        # saved money → positive
            else:
                reward = price - fair        # charged premium → positive

            if self._last_state is not None:
                self._update(self._last_state, self._last_action, reward, ())
            self.deals_made += 1
            self._last_state = None
            self._last_action = None
            return None

        if msg.msg_type == MessageType.REJECT:
            reward = -0.5
            if self._last_state is not None:
                self._update(self._last_state, self._last_action, reward, ())
            self._last_state = None
            self._last_action = None
            return None

        # ── Decide how to respond ──
        if msg.msg_type == MessageType.REQUEST:
            self._role = "seller"
            resource = Resource.from_dict(msg.payload.get("resource", {}))
            self._current_fair = resource.total_units()
            if not agent.resources.can_afford(resource):
                return msg.reply(MessageType.REJECT, {"reason": "no_resources"})

        state = self._state(agent, msg)

        # Intermediate update: reward for continuing the negotiation
        if self._last_state is not None:
            interim_reward = -0.05  # small cost for each extra round
            self._update(self._last_state, self._last_action, interim_reward, state)

        action = self._pick_action(state)
        self._last_state = state
        self._last_action = action

        return self._execute(agent, msg, action)

    # ── Analytics ─────────────────────────────────────────────────

    def policy_summary(self) -> dict[str, str]:
        """Return the dominant action per state bucket for inspection."""
        summary = {}
        type_names = ["request", "counter"]
        rep_names = ["low_rep", "med_rep", "high_rep"]
        price_names = ["price_vlow", "price_low", "price_fair", "price_high", "price_vhigh"]
        urg_names = ["low_urg", "high_urg"]
        action_names = ["ACCEPT", "REJECT", "COUNTER_LOW", "COUNTER_FAIR", "COUNTER_HIGH"]

        for t in range(2):
            for r in range(3):
                for p in range(5):
                    for u in range(2):
                        state = (t, r, p, u)
                        qs = [self._q(state, a) for a in range(5)]
                        if any(v != 0 for v in qs):
                            best = max(range(5), key=lambda a: qs[a])
                            key = f"{type_names[t]}|{rep_names[r]}|{price_names[p]}|{urg_names[u]}"
                            summary[key] = action_names[best]
        return summary

    def avg_reward_window(self, window: int = 50) -> float:
        if not self.reward_history:
            return 0.0
        recent = self.reward_history[-window:]
        return sum(recent) / len(recent)
