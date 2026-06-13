#!/usr/bin/env python3
"""
Experiment 7: Predictive Negotiation & Futures Market

Agents that learn their own usage patterns and negotiate preemptively.

Key questions:
1. Can agents predict their future compute needs?
2. Do futures contracts improve overall market efficiency?
3. What happens when predictions are wrong (default risk)?
4. Can agents arbitrage (buy cheap off-peak, sell dear at peak)?
5. Does a futures market reduce price volatility vs spot-only?
"""

import sys
import random
import math
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Resource, MessageType
from agents.protocol import Message
from agents.strategies import FairStrategy, AdaptiveStrategy, GreedyStrategy
from agents.simulator import Simulator
from agents.agent import Deal
from agents.stats import describe, welch_t_test, cohens_d, gini_coefficient


# ──────────────────────────────────────────────────────────
# Demand Patterns
# ──────────────────────────────────────────────────────────

def cyclic_demand(round_num: int, period: int = 20, base: float = 5.0, amplitude: float = 8.0) -> float:
    """Sinusoidal demand pattern — models daily/weekly cycles."""
    return base + amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * round_num / period))


def bursty_demand(round_num: int, burst_prob: float = 0.15, base: float = 3.0, burst_size: float = 20.0, rng=None) -> float:
    """Mostly low demand with random spikes."""
    if rng is None:
        rng = random
    if rng.random() < burst_prob:
        return base + burst_size
    return base


def trending_demand(round_num: int, start: float = 3.0, growth_rate: float = 0.05) -> float:
    """Linearly increasing demand over time."""
    return start + growth_rate * round_num


# ──────────────────────────────────────────────────────────
# Futures Contract
# ──────────────────────────────────────────────────────────

@dataclass
class FuturesContract:
    """A promise to deliver resources at a future time for a locked-in price."""
    contract_id: str
    seller_id: str
    buyer_id: str
    resource: Resource
    price: float
    delivery_round: int
    created_round: int
    fulfilled: bool = False
    defaulted: bool = False


# ──────────────────────────────────────────────────────────
# Demand Predictor
# ──────────────────────────────────────────────────────────

class DemandPredictor:
    """Simple exponential moving average predictor for demand."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.history: list[float] = []
        self.prediction: float = 5.0

    def observe(self, demand: float) -> None:
        self.history.append(demand)
        self.prediction = self.alpha * demand + (1 - self.alpha) * self.prediction

    def predict(self, steps_ahead: int = 1) -> float:
        return self.prediction

    def accuracy(self, last_n: int = 20) -> float:
        """Mean absolute percentage error over recent predictions."""
        if len(self.history) < 2:
            return 0.0
        errors = []
        pred = self.history[0]
        for actual in self.history[1:min(len(self.history), last_n + 1)]:
            if actual > 0:
                errors.append(abs(pred - actual) / actual)
            pred = self.alpha * actual + (1 - self.alpha) * pred
        return 1.0 - (sum(errors) / len(errors)) if errors else 0.0


# ──────────────────────────────────────────────────────────
# Futures-Aware Strategy
# ──────────────────────────────────────────────────────────

@dataclass
class FuturesStrategy:
    """
    An agent that uses demand prediction to negotiate futures contracts.
    Buys futures when predicted demand > current price.
    Sells futures when it has excess capacity.
    """
    inner_strategy: object  # fallback for spot market
    predictor: DemandPredictor = field(default_factory=DemandPredictor)
    futures_premium: float = 0.15  # willing to pay 15% above spot for certainty
    held_contracts: list[FuturesContract] = field(default_factory=list)

    def initiate(self, agent, target_id, need):
        predicted = self.predictor.predict(steps_ahead=5)
        spot_price = need.total_units()

        # If we predict high demand, try to lock in futures
        if predicted > spot_price * 1.2:
            return Message(
                msg_type=MessageType.REQUEST,
                sender_id=agent.agent_id,
                receiver_id=target_id,
                payload={
                    "resource": need.to_dict(),
                    "max_price": spot_price * (1 + self.futures_premium),
                    "urgency": agent.urgency,
                    "contract_type": "futures",
                    "delivery_round": "current+5",
                },
            )
        return self.inner_strategy.initiate(agent, target_id, need)

    def decide(self, agent, msg):
        return self.inner_strategy.decide(agent, msg)


# ──────────────────────────────────────────────────────────
# Futures Market Simulator
# ──────────────────────────────────────────────────────────

class FuturesMarketSimulator(Simulator):
    """Extends Simulator with a futures contract layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contracts: list[FuturesContract] = []
        self.contract_counter = 0
        self.spot_prices: list[float] = []
        self.futures_prices: list[float] = []

    def create_contract(self, seller_id, buyer_id, resource, price, delivery_round):
        self.contract_counter += 1
        contract = FuturesContract(
            contract_id=f"FC-{self.contract_counter:04d}",
            seller_id=seller_id,
            buyer_id=buyer_id,
            resource=resource,
            price=price,
            delivery_round=delivery_round,
            created_round=self.round_num,
        )
        self.contracts.append(contract)
        # Buyer pays upfront
        self.agents[buyer_id].budget -= price
        self.agents[seller_id].budget += price
        self.futures_prices.append(price / resource.total_units())
        return contract

    def settle_contracts(self):
        """Deliver on any contracts due this round."""
        defaults = 0
        deliveries = 0
        for contract in self.contracts:
            if contract.fulfilled or contract.defaulted:
                continue
            if contract.delivery_round <= self.round_num:
                seller = self.agents[contract.seller_id]
                buyer = self.agents[contract.buyer_id]
                if seller.resources.can_afford(contract.resource):
                    seller.resources = seller.resources - contract.resource
                    buyer.resources = buyer.resources + contract.resource
                    contract.fulfilled = True
                    deliveries += 1
                    seller.record_deal(Deal(buyer.agent_id, contract.resource, contract.price, self.round_num, True))
                    buyer.record_deal(Deal(seller.agent_id, contract.resource, contract.price, self.round_num, True))
                else:
                    contract.defaulted = True
                    defaults += 1
                    # Penalty: buyer gets refund from seller's budget
                    refund = min(contract.price * 1.5, seller.budget)
                    seller.budget -= refund
                    buyer.budget += refund
                    seller.record_deal(Deal(buyer.agent_id, contract.resource, contract.price, self.round_num, False))
                    buyer.record_deal(Deal(seller.agent_id, contract.resource, contract.price, self.round_num, False))
        return deliveries, defaults


# ──────────────────────────────────────────────────────────
# Experiment: Spot Market vs Futures Market
# ──────────────────────────────────────────────────────────

def spot_vs_futures(num_trials=100, rounds=100, seed_offset=0):
    """
    Compare market efficiency and price volatility between:
    1. Spot-only market (agents negotiate each round)
    2. Spot + futures market (agents can lock in future deals)
    """
    print("=== Spot Market vs Futures Market ===\n")

    spot_metrics = {"prices": [], "deal_rates": [], "ginis": [], "unmet_demand": []}
    futures_metrics = {"prices": [], "deal_rates": [], "ginis": [], "unmet_demand": [], "defaults": []}

    for trial in range(num_trials):
        seed = trial + seed_offset

        # --- Spot-only simulation ---
        spot_agents = [
            Agent("provider_1", Resource(gpu_hours=200), 20.0, FairStrategy(), 0.2),
            Agent("provider_2", Resource(gpu_hours=150), 25.0, AdaptiveStrategy(), 0.2),
            Agent("seeker_1", Resource(gpu_hours=5), 150.0, AdaptiveStrategy(price_belief=1.0), 0.7),
            Agent("seeker_2", Resource(gpu_hours=5), 120.0, FairStrategy(), 0.8),
        ]
        spot_sim = Simulator(spot_agents, max_negotiation_turns=6, seed=seed)

        trial_prices_spot = []
        trial_deals_spot = 0
        trial_attempts_spot = 0

        for r in range(rounds):
            demand = cyclic_demand(r)
            for sid in ["seeker_1", "seeker_2"]:
                spot_sim.agents[sid].pending_needs = Resource(gpu_hours=demand)
            metrics = spot_sim.run_round(needs={
                "seeker_1": Resource(gpu_hours=demand),
                "seeker_2": Resource(gpu_hours=demand),
            })
            trial_deals_spot += metrics.deals_made
            trial_attempts_spot += metrics.negotiations
            if metrics.avg_price_per_unit > 0:
                trial_prices_spot.append(metrics.avg_price_per_unit)

        if trial_prices_spot:
            spot_metrics["prices"].append(sum(trial_prices_spot) / len(trial_prices_spot))
        spot_metrics["deal_rates"].append(trial_deals_spot / max(trial_attempts_spot, 1))

        # --- Futures simulation ---
        futures_agents = [
            Agent("provider_1", Resource(gpu_hours=200), 20.0, FairStrategy(), 0.2),
            Agent("provider_2", Resource(gpu_hours=150), 25.0, AdaptiveStrategy(), 0.2),
            Agent("seeker_1", Resource(gpu_hours=5), 150.0, AdaptiveStrategy(price_belief=1.0), 0.7),
            Agent("seeker_2", Resource(gpu_hours=5), 120.0, FairStrategy(), 0.8),
        ]
        fut_sim = FuturesMarketSimulator(futures_agents, max_negotiation_turns=6, seed=seed)

        predictor_1 = DemandPredictor(alpha=0.3)
        predictor_2 = DemandPredictor(alpha=0.3)

        trial_prices_fut = []
        trial_deals_fut = 0
        trial_attempts_fut = 0
        trial_defaults = 0

        for r in range(rounds):
            demand = cyclic_demand(r)
            predictor_1.observe(demand)
            predictor_2.observe(demand)

            # Futures: if predicted demand > current demand, buy futures
            pred_1 = predictor_1.predict(steps_ahead=5)
            pred_2 = predictor_2.predict(steps_ahead=5)

            if pred_1 > demand * 1.3 and r < rounds - 5:
                future_resource = Resource(gpu_hours=pred_1 * 0.5)
                future_price = future_resource.total_units() * 1.1
                if fut_sim.agents["seeker_1"].budget >= future_price:
                    # Pick a provider
                    provider = "provider_1" if random.random() < 0.5 else "provider_2"
                    fut_sim.create_contract("provider_1", "seeker_1", future_resource, future_price, r + 5)

            # Settle any due contracts
            deliveries, defaults = fut_sim.settle_contracts()
            trial_defaults += defaults

            # Spot market for remaining needs
            for sid in ["seeker_1", "seeker_2"]:
                fut_sim.agents[sid].pending_needs = Resource(gpu_hours=demand)
            metrics = fut_sim.run_round(needs={
                "seeker_1": Resource(gpu_hours=demand),
                "seeker_2": Resource(gpu_hours=demand),
            })
            trial_deals_fut += metrics.deals_made + deliveries
            trial_attempts_fut += metrics.negotiations + deliveries + defaults
            if metrics.avg_price_per_unit > 0:
                trial_prices_fut.append(metrics.avg_price_per_unit)

        if trial_prices_fut:
            futures_metrics["prices"].append(sum(trial_prices_fut) / len(trial_prices_fut))
        futures_metrics["deal_rates"].append(trial_deals_fut / max(trial_attempts_fut, 1))
        futures_metrics["defaults"].append(trial_defaults)

    # --- Analysis ---
    print(f"{'Metric':<25} {'Spot Market':>25} {'Futures Market':>25}")
    print("-" * 77)

    spot_price_stat = describe(spot_metrics["prices"])
    fut_price_stat = describe(futures_metrics["prices"])
    print(f"{'Avg Price/Unit':<25} {spot_price_stat.mean:>10.3f} ±{spot_price_stat.std:.3f}      {fut_price_stat.mean:>10.3f} ±{fut_price_stat.std:.3f}")

    spot_deal_stat = describe(spot_metrics["deal_rates"])
    fut_deal_stat = describe(futures_metrics["deal_rates"])
    print(f"{'Deal Success Rate':<25} {spot_deal_stat.mean:>10.1%} ±{spot_deal_stat.std:.1%}      {fut_deal_stat.mean:>10.1%} ±{fut_deal_stat.std:.1%}")

    default_stat = describe([float(d) for d in futures_metrics["defaults"]])
    print(f"{'Contract Defaults':<25} {'N/A':>16}      {default_stat.mean:>10.1f} ±{default_stat.std:.1f}")

    # Price volatility comparison
    price_t, price_p = welch_t_test(spot_metrics["prices"], futures_metrics["prices"])
    deal_t, deal_p = welch_t_test(spot_metrics["deal_rates"], futures_metrics["deal_rates"])

    print(f"\nPrice difference: t={price_t:.3f}, p={price_p:.2e}")
    print(f"Deal rate difference: t={deal_t:.3f}, p={deal_p:.2e}")

    return {"spot": spot_metrics, "futures": futures_metrics}


# ──────────────────────────────────────────────────────────
# Experiment: Demand Pattern Effects
# ──────────────────────────────────────────────────────────

def demand_pattern_analysis(num_trials=100, rounds=100):
    """How do different demand patterns affect market dynamics?"""
    print("\n=== Demand Pattern Analysis ===\n")

    patterns = {
        "cyclic": lambda r, rng: cyclic_demand(r, period=20),
        "bursty": lambda r, rng: bursty_demand(r, rng=rng),
        "trending": lambda r, rng: trending_demand(r),
        "constant": lambda r, rng: 8.0,
    }

    for pattern_name, demand_fn in patterns.items():
        prices = []
        deal_rates = []
        volatilities = []

        for trial in range(num_trials):
            rng = random.Random(trial)
            agents = [
                Agent("provider", Resource(gpu_hours=500), 20.0, FairStrategy(), 0.2),
                Agent("seeker", Resource(gpu_hours=5), 200.0, AdaptiveStrategy(), 0.7),
            ]
            sim = Simulator(agents, max_negotiation_turns=6, seed=trial)
            trial_prices = []

            for r in range(rounds):
                demand = demand_fn(r, rng)
                sim.agents["seeker"].pending_needs = Resource(gpu_hours=demand)
                m = sim.run_round(needs={"seeker": Resource(gpu_hours=demand)})
                if m.avg_price_per_unit > 0:
                    trial_prices.append(m.avg_price_per_unit)

            if trial_prices:
                prices.append(sum(trial_prices) / len(trial_prices))
                mean_p = prices[-1]
                vol = (sum((p - mean_p) ** 2 for p in trial_prices) / len(trial_prices)) ** 0.5
                volatilities.append(vol)
            deal_rates.append(sum(1 for m in sim.metrics if m.deals_made > 0) / rounds)

        p_stat = describe(prices)
        d_stat = describe(deal_rates)
        v_stat = describe(volatilities) if volatilities else describe([0.0])
        print(f"  {pattern_name:<12}: price={p_stat.mean:.3f}±{p_stat.std:.3f}, "
              f"deals={d_stat.mean:.0%}, volatility={v_stat.mean:.4f}")


# ──────────────────────────────────────────────────────────
# Experiment: Arbitrage Agent
# ──────────────────────────────────────────────────────────

def arbitrage_experiment(num_trials=100, rounds=100):
    """
    Can an agent profit by buying cheap compute during low demand
    and reselling during peak demand?
    """
    print("\n=== Arbitrage Experiment ===\n")

    arb_profits = []
    normal_profits = []

    for trial in range(num_trials):
        # Agent with arbitrage strategy: buys when demand is low, sells when high
        agents = [
            Agent("provider", Resource(gpu_hours=500), 30.0, FairStrategy(), 0.2),
            Agent("normal_seeker", Resource(gpu_hours=5), 200.0, AdaptiveStrategy(price_belief=1.0), 0.7),
            Agent("arbitrageur", Resource(gpu_hours=50), 200.0, AdaptiveStrategy(price_belief=0.8), 0.5),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial)
        arb_start_worth = sim.agents["arbitrageur"].net_worth()
        normal_start_worth = sim.agents["normal_seeker"].net_worth()

        for r in range(rounds):
            demand = cyclic_demand(r, period=20)

            # Arbitrageur buys during low demand (demand < 7)
            if demand < 7:
                sim.agents["arbitrageur"].pending_needs = Resource(gpu_hours=10)
                sim.agents["arbitrageur"].urgency = 0.3  # patient when buying cheap
                sim.agents["normal_seeker"].pending_needs = Resource(gpu_hours=demand)
            else:
                # During high demand, arbitrageur sells (acts as provider)
                sim.agents["arbitrageur"].pending_needs = Resource()
                sim.agents["arbitrageur"].urgency = 0.2
                sim.agents["normal_seeker"].pending_needs = Resource(gpu_hours=demand)

            sim.run_round(needs={
                "normal_seeker": sim.agents["normal_seeker"].pending_needs,
                "arbitrageur": sim.agents["arbitrageur"].pending_needs,
            })

        arb_end_worth = sim.agents["arbitrageur"].net_worth()
        normal_end_worth = sim.agents["normal_seeker"].net_worth()
        arb_profits.append(arb_end_worth - arb_start_worth)
        normal_profits.append(normal_end_worth - normal_start_worth)

    arb_stat = describe(arb_profits)
    norm_stat = describe(normal_profits)
    t, p = welch_t_test(arb_profits, normal_profits)
    d = cohens_d(arb_profits, normal_profits)

    print(f"Arbitrageur profit:  {arb_stat}")
    print(f"Normal seeker profit: {norm_stat}")
    print(f"Difference: t={t:.3f}, p={p:.2e}, d={d:.3f}")
    if arb_stat.mean > norm_stat.mean and p < 0.05:
        print("Result: Arbitrage IS significantly profitable")
    elif p < 0.05:
        print("Result: Significant difference but arbitrage is LESS profitable")
    else:
        print("Result: No significant difference — arbitrage doesn't help")

    return {"arb": arb_stat, "normal": norm_stat, "p": p}


def volatile_market_arbitrage(num_trials=100, rounds=100):
    """
    Re-test arbitrage in a volatile-priced market where Greedy providers
    create real price variation. The prior test used Fair providers only
    (stable prices), so arbitrage had nothing to exploit.
    """
    print("\n=== Volatile Market Arbitrage ===\n")
    print("(Mixed Greedy + Fair providers — real price variation)\n")

    arb_profits = []
    normal_profits = []
    price_series_arb = []

    for trial in range(num_trials):
        agents = [
            # Mix of greedy and fair — creates genuine price volatility
            Agent("greedy_prov_1", Resource(gpu_hours=300), 20.0, GreedyStrategy(greed_factor=0.5), 0.1),
            Agent("greedy_prov_2", Resource(gpu_hours=250), 15.0, GreedyStrategy(greed_factor=0.6), 0.1),
            Agent("fair_prov",     Resource(gpu_hours=200), 20.0, FairStrategy(), 0.2),
            Agent("normal_seeker", Resource(gpu_hours=5), 200.0, AdaptiveStrategy(), 0.7),
            Agent("arbitrageur",   Resource(gpu_hours=50), 200.0, AdaptiveStrategy(price_belief=0.6), 0.5),
        ]
        sim = Simulator(agents, max_negotiation_turns=6, seed=trial)
        arb_start = sim.agents["arbitrageur"].net_worth()
        norm_start = sim.agents["normal_seeker"].net_worth()

        trial_prices = []
        for r in range(rounds):
            demand = cyclic_demand(r, period=20)

            if demand < 7:
                # Low demand: buy from greedy providers (they're cheap now)
                sim.agents["arbitrageur"].pending_needs = Resource(gpu_hours=12)
                sim.agents["arbitrageur"].urgency = 0.2
            else:
                # High demand: sell inventory back into market (act as provider)
                sim.agents["arbitrageur"].pending_needs = Resource()
                sim.agents["arbitrageur"].urgency = 0.1

            sim.agents["normal_seeker"].pending_needs = Resource(gpu_hours=demand)

            m = sim.run_round(needs={
                "normal_seeker": sim.agents["normal_seeker"].pending_needs,
                "arbitrageur": sim.agents["arbitrageur"].pending_needs,
            })
            if m.avg_price_per_unit > 0:
                trial_prices.append(m.avg_price_per_unit)

        arb_profits.append(sim.agents["arbitrageur"].net_worth() - arb_start)
        normal_profits.append(sim.agents["normal_seeker"].net_worth() - norm_start)
        if trial_prices:
            price_std = (sum((p - sum(trial_prices) / len(trial_prices)) ** 2
                             for p in trial_prices) / len(trial_prices)) ** 0.5
            price_series_arb.append(price_std)

    arb_stat = describe(arb_profits)
    norm_stat = describe(normal_profits)
    vol_stat = describe(price_series_arb)
    t, p = welch_t_test(arb_profits, normal_profits)
    d = cohens_d(arb_profits, normal_profits)

    print(f"Price volatility (std):   {vol_stat}")
    print(f"Arbitrageur profit:       {arb_stat}")
    print(f"Normal seeker profit:     {norm_stat}")
    print(f"Difference: t={t:.3f}, p={p:.2e}, d={d:.3f}")

    if arb_stat.mean > norm_stat.mean and p < 0.05:
        print("Result: Arbitrage IS significantly profitable in volatile market")
    elif p < 0.05:
        print("Result: Still LESS profitable even with price volatility — arbitrage costs outweigh gains")
    else:
        print("Result: No significant difference")

    return {"arb": arb_stat, "normal": norm_stat, "volatility": vol_stat, "p": p}


if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT 7: PREDICTIVE NEGOTIATION & FUTURES")
    print("=" * 60)
    print()

    spot_vs_futures()
    demand_pattern_analysis()
    arbitrage_experiment()
    volatile_market_arbitrage()
