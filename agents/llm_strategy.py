"""
LLM-powered negotiation strategy — supports OpenAI and Anthropic.

Agents negotiate in natural language, with structured message parsing.
Requires either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.

This is the Tier 3 intelligence level — agents that can reason about
context, detect bluffs, and strategize in ways rule-based agents can't.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent

from agents.protocol import Message, MessageType
from agents.resource import Resource

_openai_client = None
_anthropic_client = None
_provider: str | None = None


def _detect_provider() -> str:
    """Detect which API is available. Prefers OpenAI if both are set."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


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

Keep reasoning concise (1-2 sentences). Always respond with valid JSON only."""


MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


@dataclass
class LLMStrategy:
    """
    Uses an LLM (OpenAI or Anthropic) to make negotiation decisions.
    Falls back to a simple rule-based approach if no API is available.
    """
    model: str | None = None
    temperature: float = 0.5
    max_tokens: int = 300
    negotiation_history: list[dict] = field(default_factory=list)
    _api_available: bool | None = None
    _provider: str | None = None
    api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def _check_api(self) -> bool:
        if self._api_available is None:
            self._provider = _detect_provider()
            if self._provider == "none":
                self._api_available = False
            else:
                try:
                    if self._provider == "openai":
                        _get_openai_client()
                    else:
                        _get_anthropic_client()
                    if self.model is None:
                        self.model = MODELS[self._provider]
                    self._api_available = True
                except (RuntimeError, KeyError):
                    self._api_available = False
        return self._api_available

    def _build_context(self, agent: "Agent", msg: Message) -> str:
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
        rep = agent.reputation_of(msg.sender_id)
        lines.append(f"\nYour trust in {msg.sender_id}: {rep:.2f} (0=untrusted, 1=fully trusted)")
        if agent.deals:
            lines.append(f"\nRecent deals ({len(agent.deals)} total):")
            for deal in agent.deals[-5:]:
                status = "fulfilled" if deal.fulfilled else "DEFAULTED"
                lines.append(f"  {deal.partner_id}: {deal.resource} at {deal.price:.1f} [{status}]")
        return "\n".join(lines)

    def _call_llm(self, agent: "Agent", msg: Message) -> dict:
        context = self._build_context(agent, msg)
        self.api_calls += 1

        if self._provider == "openai":
            return self._call_openai(context)
        else:
            return self._call_anthropic(context)

    def _call_openai(self, context: str) -> dict:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
        )
        text = response.choices[0].message.content.strip()
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
        return self._parse_response(text)

    def _call_anthropic(self, context: str) -> dict:
        client = _get_anthropic_client()
        response = client.messages.create(
            model=self.model or "claude-haiku-4-5-20251001",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
        return self._parse_response(text)

    def _parse_response(self, text: str) -> dict:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
        except json.JSONDecodeError:
            text_lower = text.lower()
            if "accept" in text_lower:
                result = {"action": "accept", "reasoning": text}
            elif "counter" in text_lower:
                numbers = re.findall(r"[\d.]+", text)
                price = float(numbers[0]) if numbers else 10.0
                result = {"action": "counter", "reasoning": text, "price": price}
            elif "reject" in text_lower:
                result = {"action": "reject", "reasoning": text}
            else:
                result = {"action": "reject", "reasoning": f"Unparseable: {text[:100]}"}

        self.negotiation_history.append({"response": result})
        return result

    def _call_llm_simple(self, prompt: str) -> str:
        self.api_calls += 1
        if self._provider == "openai":
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                max_tokens=150,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content.strip()
        else:
            client = _get_anthropic_client()
            response = client.messages.create(
                model=self.model or "claude-haiku-4-5-20251001",
                max_tokens=150,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if response.usage:
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
            return response.content[0].text.strip()

    def _fallback_decide(self, agent: "Agent", msg: Message) -> Message | None:
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

    def initiate(self, agent: "Agent", target_id: str, need: Resource) -> Message:
        if self._check_api():
            prompt = (
                f"You need to request compute from {target_id}.\n"
                f"You need: {need}\n"
                f"Your budget: {agent.budget:.1f}\n"
                f"Your urgency: {agent.urgency:.2f}\n"
                f"Your trust in {target_id}: {agent.reputation_of(target_id):.2f}\n"
                f"What max_price should you offer? Reply with just the number."
            )
            try:
                text = self._call_llm_simple(prompt)
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

    def decide(self, agent: "Agent", msg: Message) -> Message | None:
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

    def usage_summary(self) -> str:
        cost_in = self.total_input_tokens * 0.15 / 1_000_000
        cost_out = self.total_output_tokens * 0.60 / 1_000_000
        return (
            f"API calls: {self.api_calls}, "
            f"tokens: {self.total_input_tokens} in / {self.total_output_tokens} out, "
            f"est. cost: ${cost_in + cost_out:.4f}"
        )
