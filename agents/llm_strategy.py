"""
LLM-powered negotiation strategy — supports OpenAI and Anthropic.

Agents negotiate in natural language, with structured message parsing.
Requires either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.

This is the Tier 3 intelligence level — agents that can reason about
context, detect bluffs, and strategize in ways rule-based agents can't.

Two prompt variants:
  - NAIVE:      minimal instructions, tests raw LLM negotiation ability
  - ENGINEERED: production-grade prompt with pricing framework, strategy
                guidelines, and structured reasoning — what you'd actually
                ship in a real agentic product
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


def _detect_provider() -> str:
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


NAIVE_SYSTEM_PROMPT = """You are an autonomous agent negotiating for compute resources.

Respond with a JSON object:
{
    "action": "accept" | "reject" | "counter",
    "reasoning": "brief explanation",
    "price": <number if countering>,
    "resource": {"gpu_hours": <n>, "cpu_hours": <n>, "memory_gb_hours": <n>}
}

Always respond with valid JSON only."""


ENGINEERED_SYSTEM_PROMPT = """You are an autonomous compute-resource negotiation agent deployed in a decentralized marketplace. You represent a single participant (buyer or seller) and must maximize your principal's utility over multiple negotiation rounds.

## Your Objective
Maximize: (value_of_compute_acquired - price_paid) if buying, or (price_received - cost_of_compute_sold) if selling. A deal at fair price is better than no deal. No deal is better than a bad deal.

## Pricing Framework
- Fair market price is approximately 1.0 currency per compute-unit (1 GPU-hour = 1 unit).
- A "good" buy is anything below 1.0/unit. A "bad" buy is above 1.15/unit.
- A "good" sell is anything above 1.0/unit. A "bad" sell is below 0.85/unit.
- Your walk-away price (BATNA) depends on your urgency: high urgency = accept wider range.

## Decision Framework
For each incoming message, evaluate:
1. PRICE ANALYSIS: Is the offered price above or below fair? By how much?
2. PARTNER TRUST: Check their reputation score. Below 0.3 = risky, avoid. Above 0.7 = reliable.
3. BUDGET CHECK: Can you afford this? Never accept if price > your remaining budget.
4. URGENCY ADJUSTMENT: If urgency > 0.7, accept up to 1.2x fair price. If urgency < 0.3, hold out for below 0.9x fair.
5. COUNTER STRATEGY: When countering, move 30-50% toward their price from your ideal. Never counter at a price worse than your previous position.

## Actions
- ACCEPT: Price is within your acceptable range AND you can afford it AND partner is trustworthy.
- COUNTER: Price is close but not quite right. Propose a specific price with reasoning.
- REJECT: Price is far from acceptable, budget insufficient, or partner untrusted.

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "action": "accept" | "reject" | "counter",
    "reasoning": "1-2 sentences explaining your price analysis and decision",
    "price": <number — required for accept and counter>,
    "resource": {"gpu_hours": <n>, "cpu_hours": <n>, "memory_gb_hours": <n>}
}"""


INITIATE_NAIVE_PROMPT = """You need compute resources from {target_id}.
You need: {need}
Your budget: {budget:.1f}
Your urgency: {urgency:.2f}
What max_price should you offer? Reply with just the number."""

INITIATE_ENGINEERED_PROMPT = """You are initiating a purchase request for compute resources.

Target seller: {target_id}
Resources needed: {need}
Your available budget: {budget:.1f} currency
Your urgency: {urgency:.2f} (0=can wait forever, 1=need immediately)
Your trust in this seller: {trust:.2f}

Determine your opening max_price offer. Guidelines:
- Fair price is approximately {fair_price:.1f} (1.0 per compute-unit)
- If urgency > 0.7: offer up to 1.1x fair to close quickly
- If urgency < 0.3: offer 0.9x fair to get a bargain
- Never offer more than your budget
- A reasonable opening is fair_price * (1.0 + 0.1 * urgency)

Reply with just the number (the max_price you want to offer)."""


MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


@dataclass
class LLMStrategy:
    model: str | None = None
    temperature: float = 0.5
    max_tokens: int = 300
    prompt_style: str = "engineered"  # "naive" or "engineered"
    negotiation_history: list[dict] = field(default_factory=list)
    _api_available: bool | None = None
    _provider: str | None = None
    api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def _system_prompt(self) -> str:
        if self.prompt_style == "naive":
            return NAIVE_SYSTEM_PROMPT
        return ENGINEERED_SYSTEM_PROMPT

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
        ]
        resource = msg.payload.get("resource", {})
        fair = Resource.from_dict(resource).total_units() if resource else 0
        if fair > 0 and self.prompt_style == "engineered":
            lines.append(f"Fair market price for this resource: ~{fair:.1f} currency (1.0/unit)")

        lines.extend([
            "",
            f"Incoming message from {msg.sender_id}:",
            f"  Type: {msg.msg_type.value}",
            f"  Payload: {json.dumps(msg.payload, indent=2)}",
        ])
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
        return self._call_anthropic(context)

    def _call_openai(self, context: str) -> dict:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self._system_prompt},
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
            system=self._system_prompt,
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
                    {"role": "system", "content": self._system_prompt},
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
                system=self._system_prompt,
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
            fair_price = need.total_units()
            if self.prompt_style == "engineered":
                prompt = INITIATE_ENGINEERED_PROMPT.format(
                    target_id=target_id, need=need, budget=agent.budget,
                    urgency=agent.urgency, trust=agent.reputation_of(target_id),
                    fair_price=fair_price,
                )
            else:
                prompt = INITIATE_NAIVE_PROMPT.format(
                    target_id=target_id, need=need, budget=agent.budget,
                    urgency=agent.urgency,
                )
            try:
                text = self._call_llm_simple(prompt)
                numbers = re.findall(r"[\d.]+", text)
                if numbers:
                    max_price = float(numbers[0])
                    max_price = min(max_price, agent.budget)
                else:
                    max_price = fair_price * (1.0 + 0.1 * agent.urgency)
            except Exception:
                max_price = fair_price * (1.0 + 0.1 * agent.urgency)
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
