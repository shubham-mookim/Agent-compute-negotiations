"""
LLM-powered negotiation strategy using Claude API.

Agents negotiate in natural language, with structured message parsing.
Requires ANTHROPIC_API_KEY environment variable.

This is the Tier 3 intelligence level — agents that can reason about
context, detect bluffs, and strategize in ways rule-based agents can't.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent

from agents.protocol import Message, MessageType
from agents.resource import Resource

# Lazy import — only fail when actually used without the key
_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it to your Anthropic API key to use LLM agents."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are an autonomous agent negotiating for compute resources in a decentralized market.

You have:
- A budget (abstract currency) to spend on compute
- Resources you currently hold (GPU hours, CPU hours, memory)
- A reputation table of how much you trust other agents
- Knowledge of your past deals

Your goal is to maximize your utility: get the compute you need at the best price, or sell your excess compute profitably.

When you receive a negotiation message, respond with a JSON object:
{
    "action": "accept" | "reject" | "counter",
    "reasoning": "brief explanation of your thinking",
    "price": <number if countering>,
    "resource": {"gpu_hours": <n>, "cpu_hours": <n>, "memory_gb_hours": <n>}
}

Be strategic but not adversarial. Consider:
- Your urgency (how badly you need this deal)
- The other agent's reputation
- Whether this price is fair based on your history
- Whether you can get a better deal elsewhere

Keep reasoning concise (1-2 sentences)."""


@dataclass
class LLMStrategy:
    """
    Uses Claude to make negotiation decisions.
    Falls back to a simple rule-based approach if API is unavailable.
    """
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.7
    max_tokens: int = 300
    negotiation_history: list[dict] = field(default_factory=list)
    _api_available: bool | None = None

    def _check_api(self) -> bool:
        if self._api_available is None:
            try:
                _get_client()
                self._api_available = True
            except RuntimeError:
                self._api_available = False
        return self._api_available

    def _build_context(self, agent: Agent, msg: Message) -> str:
        """Build a context string describing the agent's current state."""
        lines = [
            f"Your ID: {agent.agent_id}",
            f"Your budget: {agent.budget:.1f} currency",
            f"Your resources: {agent.resources}",
            f"Your urgency: {agent.urgency:.2f} (0=can wait, 1=desperate)",
            f"Your pending needs: {agent.pending_needs}",
            "",
            f"Incoming message from {msg.sender_id}:",
            f"  Type: {msg.msg_type.value}",
            f"  Payload: {json.dumps(msg.payload, indent=2)}",
        ]

        # Add reputation info
        rep = agent.reputation_of(msg.sender_id)
        lines.append(f"\nYour trust in {msg.sender_id}: {rep:.2f} (0=untrusted, 1=fully trusted)")

        # Add recent deal history
        if agent.deals:
            lines.append(f"\nRecent deals ({len(agent.deals)} total):")
            for deal in agent.deals[-5:]:
                status = "fulfilled" if deal.fulfilled else "DEFAULTED"
                lines.append(f"  {deal.partner_id}: {deal.resource} at {deal.price:.1f} [{status}]")

        return "\n".join(lines)

    def _call_llm(self, agent: Agent, msg: Message) -> dict:
        """Call Claude API and parse the response."""
        client = _get_client()
        context = self._build_context(agent, msg)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )

        text = response.content[0].text.strip()

        # Parse JSON from response
        try:
            # Handle cases where LLM wraps JSON in markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract action from text
            text_lower = text.lower()
            if "accept" in text_lower:
                result = {"action": "accept", "reasoning": text}
            elif "reject" in text_lower:
                result = {"action": "reject", "reasoning": text}
            else:
                result = {"action": "reject", "reasoning": f"Failed to parse: {text[:100]}"}

        self.negotiation_history.append({
            "round": len(self.negotiation_history),
            "incoming": {"type": msg.msg_type.value, "from": msg.sender_id, "payload": msg.payload},
            "response": result,
        })

        return result

    def _fallback_decide(self, agent: Agent, msg: Message) -> Message | None:
        """Simple rule-based fallback when API is unavailable."""
        if msg.msg_type == MessageType.REQUEST:
            requested = Resource.from_dict(msg.payload.get("resource", {}))
            if not agent.resources.can_afford(requested):
                return msg.reply(MessageType.REJECT, {"reason": "insufficient_resources"})
            price = requested.total_units()
            their_max = msg.payload.get("max_price", 0)
            if their_max >= price:
                return msg.reply(MessageType.ACCEPT, {"resource": requested.to_dict(), "price": price})
            return msg.reply(MessageType.COUNTER, {"resource": requested.to_dict(), "price": price})

        elif msg.msg_type in (MessageType.OFFER, MessageType.COUNTER):
            price = msg.payload.get("price", float("inf"))
            resource = Resource.from_dict(msg.payload.get("resource", {}))
            if price <= resource.total_units() * 1.15 and agent.budget >= price:
                return msg.reply(MessageType.ACCEPT, {"resource": resource.to_dict(), "price": price})
            return msg.reply(MessageType.REJECT, {"reason": "too_expensive"})

        return None

    def initiate(self, agent: Agent, target_id: str, need: Resource) -> Message:
        """Start a negotiation. Uses LLM to set price if available."""
        if self._check_api():
            context = (
                f"You need to request compute from {target_id}.\n"
                f"You need: {need}\n"
                f"Your budget: {agent.budget:.1f}\n"
                f"Your urgency: {agent.urgency:.2f}\n"
                f"Your trust in {target_id}: {agent.reputation_of(target_id):.2f}\n"
                f"What max_price should you offer?"
            )
            try:
                client = _get_client()
                response = client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": context}],
                )
                text = response.content[0].text.strip()
                # Try to extract a number
                import re
                numbers = re.findall(r"[\d.]+", text)
                if numbers:
                    max_price = float(numbers[0])
                    max_price = min(max_price, agent.budget)
                else:
                    max_price = need.total_units() * 1.1
            except Exception:
                max_price = need.total_units() * 1.1
        else:
            max_price = need.total_units() * 1.1

        return Message(
            msg_type=MessageType.REQUEST,
            sender_id=agent.agent_id,
            receiver_id=target_id,
            payload={
                "resource": need.to_dict(),
                "max_price": max_price,
                "urgency": agent.urgency,
            },
        )

    def decide(self, agent: Agent, msg: Message) -> Message | None:
        """Process incoming message using LLM or fallback."""
        if not self._check_api():
            return self._fallback_decide(agent, msg)

        if msg.msg_type in (MessageType.ACCEPT, MessageType.REJECT):
            return None

        try:
            result = self._call_llm(agent, msg)
        except Exception:
            return self._fallback_decide(agent, msg)

        action = result.get("action", "reject").lower()
        resource_dict = result.get("resource", msg.payload.get("resource", {}))
        price = result.get("price", msg.payload.get("price", 0))

        if action == "accept":
            return msg.reply(MessageType.ACCEPT, {
                "resource": resource_dict,
                "price": price or msg.payload.get("price", 0),
            })
        elif action == "counter":
            return msg.reply(MessageType.COUNTER, {
                "resource": resource_dict,
                "price": price,
            })
        else:
            return msg.reply(MessageType.REJECT, {
                "reason": result.get("reasoning", "rejected by LLM agent"),
            })
