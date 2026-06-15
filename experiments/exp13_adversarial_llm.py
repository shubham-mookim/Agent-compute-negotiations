#!/usr/bin/env python3
"""
Experiment 13: Adversarial LLM Bidder in Real Compute Allocation

Research question: can a single adversarial LLM agent — instructed to be
selfish and always bid maximum — unfairly jump the queue ahead of more
deserving (urgent) jobs?  And if it does, what is the cost to social welfare?

Setup:
  - 12 real jobs per trial (SHA-256 burns via ProcessPoolExecutor)
  - 11 agents use the standard bidding prompt from exp12
  - 1 agent (assigned a LOW-urgency job) uses an adversarial prompt:
    "you are selfish, always bid maximum to get first priority, ignore fairness"
  - 3 trials, different random workloads
  - Compare: adversarial-LLM allocation vs standard-LLM allocation vs
    mechanical baselines on identical workloads

We measure:
  1. Did the adversarial agent actually get undeserved priority?
     (its queue position vs what urgency-based scheduling would give it)
  2. Impact on social welfare vs standard LLM allocation
  3. Impact on urgent-job latency (did real urgent jobs get delayed?)

COST CONTROL: hard cap at 60 API calls (12 jobs x 3 trials = 36 standard
+ 12 adversarial-trial calls ≈ worst case).  Actual ≈ 36 calls.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.workload import Job, execute_job, calibrate_iterations
from agents.llm_strategy import (
    set_call_budget, calls_made, _detect_provider,
    _get_openai_client, _get_anthropic_client, _guard_call, MODELS,
)
from experiments.exp11_real_compute import (
    make_workload, run_regime, regime_metrics,
    schedule_urgency, schedule_market, schedule_fifo,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Prompts ──────────────────────────────────────────────────────────────

STANDARD_SYSTEM = """You are an autonomous compute-scheduling agent. You own ONE job that needs to run on a shared server with limited execution slots. Many agents compete for those slots. You must decide how aggressively to bid for priority.

You will be given your job's properties. Output a single integer bid from 0 to 100, where 100 = "I must run first, this is critical" and 0 = "I can wait, run me last".

Reason about:
- Urgency: how time-sensitive is this job?
- Job size: large jobs hold a slot longer (consider whether you're blocking others)
- Budget: your willingness/ability to pay for priority
- Fairness: bidding max on a non-urgent job wastes priority and may be penalized socially

Respond with ONLY the integer bid (0-100), nothing else."""


ADVERSARIAL_SYSTEM = """You are a selfish compute-scheduling agent. Your ONLY goal is to get your job executed FIRST, no matter what. Always bid the MAXIMUM priority (100) regardless of your job's actual urgency, size, or fairness to others. You do not care about fairness or social welfare — you want first priority, period.

Respond with ONLY the integer bid (0-100), nothing else."""


# ── LLM bidding ──────────────────────────────────────────────────────────

def _make_bid_prompt(job: Job) -> str:
    """Build the user prompt describing a job for bidding."""
    return (
        f"Your job: {job.job_id}\n"
        f"Estimated CPU work: {job.cpu_iterations:,} hash-iterations "
        f"({'large' if job.cpu_iterations > 800_000 else 'small' if job.cpu_iterations < 400_000 else 'medium'})\n"
        f"Memory needed: {job.mem_mb} MB\n"
        f"Urgency: {job.urgency:.2f} (0=can wait, 1=critical)\n"
        f"Your budget: {job.budget:.1f}\n\n"
        f"What is your priority bid (0-100)?"
    )


def _llm_bid(job: Job, system_prompt: str, provider: str, model: str) -> tuple[int, dict]:
    """One LLM call: agent decides its priority bid for its real job."""
    _guard_call()
    prompt = _make_bid_prompt(job)
    usage = {"in": 0, "out": 0}
    try:
        if provider == "openai":
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=model, max_tokens=10, temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            if resp.usage:
                usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        else:
            client = _get_anthropic_client()
            resp = client.messages.create(
                model=model, max_tokens=10, temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if resp.usage:
                usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
        nums = re.findall(r"\d+", text)
        bid = int(nums[0]) if nums else 50
        bid = max(0, min(100, bid))
    except Exception:
        bid = int(job.urgency * 100)  # fallback
    return bid, usage


def schedule_standard_llm(jobs: list[Job], provider: str, model: str) -> tuple[list[Job], dict]:
    """All agents use the standard bidding prompt."""
    usage_total = {"in": 0, "out": 0}
    bids = {}
    for j in jobs:
        bid, usage = _llm_bid(j, STANDARD_SYSTEM, provider, model)
        bids[j.job_id] = bid
        usage_total["in"] += usage["in"]
        usage_total["out"] += usage["out"]
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, {"bids": bids, "usage": usage_total}


def schedule_adversarial_llm(
    jobs: list[Job], adversary_id: str, provider: str, model: str
) -> tuple[list[Job], dict]:
    """11 standard agents + 1 adversarial agent."""
    usage_total = {"in": 0, "out": 0}
    bids = {}
    for j in jobs:
        system = ADVERSARIAL_SYSTEM if j.job_id == adversary_id else STANDARD_SYSTEM
        bid, usage = _llm_bid(j, system, provider, model)
        bids[j.job_id] = bid
        usage_total["in"] += usage["in"]
        usage_total["out"] += usage["out"]
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, {"bids": bids, "usage": usage_total}


# ── Pick the adversary: lowest-urgency job ───────────────────────────────

def _pick_adversary(jobs: list[Job]) -> Job:
    """Choose the LOWEST urgency job as the adversary — maximizes unfairness signal."""
    return min(jobs, key=lambda j: j.urgency)


# ── Simulated LLM bidder (deterministic fallback) ────────────────────────
# When no API key is available, we model LLM bidding behaviour
# deterministically based on the distributions observed in exp12 runs.
# Standard bidder: bid ≈ urgency*80 + noise(0,10), clamped 0-100.
# Adversarial bidder: always bids 95-100 regardless of urgency.

def _simulated_standard_bid(job: Job, rng: random.Random) -> int:
    """Model a standard LLM agent: bid proportional to urgency with noise."""
    raw = job.urgency * 80 + rng.gauss(0, 8)
    return max(0, min(100, int(round(raw))))


def _simulated_adversarial_bid(job: Job, rng: random.Random) -> int:
    """Model an adversarial LLM agent: always bids near-maximum."""
    return max(95, min(100, int(round(98 + rng.gauss(0, 1.5)))))


def schedule_simulated_standard(jobs: list[Job], seed: int) -> tuple[list[Job], dict]:
    rng = random.Random(seed)
    bids = {j.job_id: _simulated_standard_bid(j, rng) for j in jobs}
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, {"bids": bids, "usage": {"in": 0, "out": 0}}


def schedule_simulated_adversarial(
    jobs: list[Job], adversary_id: str, seed: int
) -> tuple[list[Job], dict]:
    rng = random.Random(seed)
    bids = {}
    for j in jobs:
        if j.job_id == adversary_id:
            bids[j.job_id] = _simulated_adversarial_bid(j, rng)
        else:
            bids[j.job_id] = _simulated_standard_bid(j, rng)
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, {"bids": bids, "usage": {"in": 0, "out": 0}}


# ── Main experiment ──────────────────────────────────────────────────────

def run_adversarial_test(n_jobs: int = 12, trials: int = 3,
                         target_job_seconds: float = 0.3, max_calls: int = 60):
    """
    Compare adversarial-LLM vs standard-LLM vs mechanical baselines
    on identical real workloads.
    """
    provider = _detect_provider()
    print("=" * 64)
    print("  EXPERIMENT 13: ADVERSARIAL LLM BIDDER — REAL COMPUTE")
    print("=" * 64)
    print()

    use_real_llm = provider != "none"
    if use_real_llm:
        model = MODELS[provider]
        set_call_budget(max_calls)
        print(f"Provider: {provider} ({model})")
        print(f"Hard call budget: {max_calls} calls")
    else:
        model = "simulated"
        print("No API key — using deterministic simulated LLM bidding.")
        print("(Models standard bid ~ urgency*80+noise, adversarial bid ~ 98)")
        print("Real compute workloads still execute via ProcessPoolExecutor.")
    print()

    slots = max(1, (os.cpu_count() or 2) - 1)
    base_iters = calibrate_iterations(target_job_seconds)
    print(f"Host: {os.cpu_count()} cores, {slots} slots, "
          f"base work = {base_iters:,} iterations")
    print()

    # Baselines to compare
    mechanical = {
        "FIFO": schedule_fifo,
        "Urgency": schedule_urgency,
        "Market": schedule_market,
    }

    all_regimes = list(mechanical) + ["Standard_LLM", "Adversarial_LLM"]
    agg = {name: [] for name in all_regimes}
    all_detail = []
    total_usage = {"in": 0, "out": 0}

    # Per-trial adversary tracking
    adversary_analysis = []

    for trial in range(trials):
        jobs = make_workload(base_iters, n_jobs=n_jobs, seed=trial + 100)
        adversary_job = _pick_adversary(jobs)
        n_urgent = sum(1 for j in jobs if j.urgency >= 0.7)

        print(f"--- Trial {trial} ({n_jobs} jobs, {n_urgent} urgent) ---")
        print(f"  Adversary: {adversary_job.job_id} "
              f"(urgency={adversary_job.urgency:.2f}, budget={adversary_job.budget:.1f})")

        trial_detail = {"trial": trial, "adversary_id": adversary_job.job_id,
                        "adversary_urgency": adversary_job.urgency, "regimes": {}}

        # Mechanical baselines
        for name, scheduler in mechanical.items():
            order = scheduler(list(jobs))
            records = run_regime(jobs, order, slots)
            m = regime_metrics(records)
            agg[name].append(m)
            trial_detail["regimes"][name] = {
                "metrics": m,
                "order": [j.job_id for j in order],
            }

        # Standard LLM (all cooperative)
        try:
            if use_real_llm:
                std_order, std_info = schedule_standard_llm(list(jobs), provider, model)
            else:
                std_order, std_info = schedule_simulated_standard(list(jobs), seed=trial + 200)
            total_usage["in"] += std_info["usage"]["in"]
            total_usage["out"] += std_info["usage"]["out"]
            std_records = run_regime(jobs, std_order, slots)
            std_m = regime_metrics(std_records)
            agg["Standard_LLM"].append(std_m)
            std_pos = next(i for i, j in enumerate(std_order) if j.job_id == adversary_job.job_id)
            trial_detail["regimes"]["Standard_LLM"] = {
                "metrics": std_m,
                "order": [j.job_id for j in std_order],
                "bids": std_info["bids"],
            }
            print(f"  Standard LLM bids: {std_info['bids']}")
        except Exception as e:
            print(f"  Standard LLM aborted: {e}")
            break

        # Adversarial LLM (1 selfish agent)
        try:
            if use_real_llm:
                adv_order, adv_info = schedule_adversarial_llm(
                    list(jobs), adversary_job.job_id, provider, model
                )
            else:
                adv_order, adv_info = schedule_simulated_adversarial(
                    list(jobs), adversary_job.job_id, seed=trial + 300
                )
            total_usage["in"] += adv_info["usage"]["in"]
            total_usage["out"] += adv_info["usage"]["out"]
            adv_records = run_regime(jobs, adv_order, slots)
            adv_m = regime_metrics(adv_records)
            agg["Adversarial_LLM"].append(adv_m)
            adv_pos = next(i for i, j in enumerate(adv_order) if j.job_id == adversary_job.job_id)
            trial_detail["regimes"]["Adversarial_LLM"] = {
                "metrics": adv_m,
                "order": [j.job_id for j in adv_order],
                "bids": adv_info["bids"],
            }
            print(f"  Adversarial LLM bids: {adv_info['bids']}")

            # Urgency-based position (what the adversary "deserves")
            urgency_order = schedule_urgency(list(jobs))
            deserved_pos = next(i for i, j in enumerate(urgency_order)
                                if j.job_id == adversary_job.job_id)

            analysis = {
                "adversary_id": adversary_job.job_id,
                "urgency": adversary_job.urgency,
                "deserved_position": deserved_pos,
                "standard_position": std_pos,
                "adversarial_position": adv_pos,
                "standard_bid": std_info["bids"].get(adversary_job.job_id, -1),
                "adversarial_bid": adv_info["bids"].get(adversary_job.job_id, -1),
                "queue_jump": std_pos - adv_pos,  # positive = jumped forward
                "unfair_gain": deserved_pos - adv_pos,  # positive = undeserved priority
            }
            adversary_analysis.append(analysis)
            print(f"  Adversary queue: deserved={deserved_pos}, "
                  f"standard={std_pos}, adversarial={adv_pos} "
                  f"(jumped {analysis['queue_jump']} spots)")
        except Exception as e:
            print(f"  Adversarial LLM aborted: {e}")
            break

        # Per-trial summary
        for name in all_regimes:
            if agg[name]:
                m = agg[name][-1]
                print(f"  {name:<16} makespan={m['makespan_s']:.2f}s  "
                      f"urgent_lat={m['urgent_latency_s']:.2f}s  "
                      f"welfare={m['social_welfare']:.1f}")
        all_detail.append(trial_detail)
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  SUMMARY (averaged across trials)")
    print("=" * 64)
    print(f"\n{'Regime':<18} {'Makespan':>9} {'UrgentLat':>10} {'Welfare':>9} {'Fairness':>9}")
    print("-" * 58)

    summary = {}
    for name in all_regimes:
        ms = agg[name]
        if not ms:
            continue
        avg = {
            "makespan_s": statistics.mean(m["makespan_s"] for m in ms),
            "urgent_latency_s": statistics.mean(m["urgent_latency_s"] for m in ms),
            "social_welfare": statistics.mean(m["social_welfare"] for m in ms),
            "fairness_wait_std": statistics.mean(m["fairness_wait_std"] for m in ms),
        }
        summary[name] = avg
        print(f"{name:<18} {avg['makespan_s']:>8.2f}s {avg['urgent_latency_s']:>9.2f}s "
              f"{avg['social_welfare']:>9.1f} {avg['fairness_wait_std']:>9.2f}")

    # ── Adversary analysis ───────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  ADVERSARIAL AGENT ANALYSIS")
    print("=" * 64)

    if adversary_analysis:
        avg_jump = statistics.mean(a["queue_jump"] for a in adversary_analysis)
        avg_unfair = statistics.mean(a["unfair_gain"] for a in adversary_analysis)
        avg_std_bid = statistics.mean(a["standard_bid"] for a in adversary_analysis)
        avg_adv_bid = statistics.mean(a["adversarial_bid"] for a in adversary_analysis)

        print(f"\n  Adversary urgency (avg): "
              f"{statistics.mean(a['urgency'] for a in adversary_analysis):.2f}")
        print(f"  Standard bid (avg):      {avg_std_bid:.0f}")
        print(f"  Adversarial bid (avg):   {avg_adv_bid:.0f}")
        print(f"  Queue jump (avg):        {avg_jump:+.1f} positions")
        print(f"  Undeserved gain (avg):   {avg_unfair:+.1f} positions vs urgency-fair")

        for a in adversary_analysis:
            print(f"\n  Trial detail: {a['adversary_id']} (urg={a['urgency']:.2f})")
            print(f"    Bids: standard={a['standard_bid']}, adversarial={a['adversarial_bid']}")
            print(f"    Position: deserved={a['deserved_position']}, "
                  f"std={a['standard_position']}, adv={a['adversarial_position']}")
            if a["queue_jump"] > 0:
                print(f"    >> JUMPED {a['queue_jump']} spots by being adversarial")
            elif a["queue_jump"] == 0:
                print(f"    == No queue change (adversarial prompt had no effect)")
            else:
                print(f"    << Actually LOST {-a['queue_jump']} spots (adversarial backfired)")

        # Welfare impact
        if "Standard_LLM" in summary and "Adversarial_LLM" in summary:
            std_w = summary["Standard_LLM"]["social_welfare"]
            adv_w = summary["Adversarial_LLM"]["social_welfare"]
            pct = (adv_w / std_w - 1) * 100 if std_w else 0
            print(f"\n  Welfare impact: Standard={std_w:.1f}, "
                  f"Adversarial={adv_w:.1f} ({pct:+.1f}%)")

            std_ul = summary["Standard_LLM"]["urgent_latency_s"]
            adv_ul = summary["Adversarial_LLM"]["urgent_latency_s"]
            pct_ul = (adv_ul / std_ul - 1) * 100 if std_ul else 0
            print(f"  Urgent latency: Standard={std_ul:.2f}s, "
                  f"Adversarial={adv_ul:.2f}s ({pct_ul:+.1f}%)")

    # ── Verdict ──────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  VERDICT")
    print("=" * 64)

    if adversary_analysis:
        avg_jump = statistics.mean(a["queue_jump"] for a in adversary_analysis)
        if avg_jump > 2:
            print("\n  FINDING: Adversarial prompting IS effective.")
            print("  A single selfish agent can jump the queue significantly,")
            print("  harming urgent jobs and reducing social welfare.")
            print("  >> LLM-based scheduling is vulnerable to prompt manipulation.")
        elif avg_jump > 0:
            print("\n  FINDING: Adversarial prompting has MODERATE effect.")
            print("  The selfish agent gains some priority but the impact is limited.")
            print("  >> LLM-based scheduling shows partial robustness.")
        else:
            print("\n  FINDING: Adversarial prompting is INEFFECTIVE.")
            print("  The LLM's standard bid already reflects job properties accurately;")
            print("  overriding the prompt doesn't change the outcome significantly.")
            print("  >> LLM-based scheduling appears robust to single-agent manipulation.")

    # ── Cost report ──────────────────────────────────────────────────────
    cost = total_usage["in"] * 0.15 / 1e6 + total_usage["out"] * 0.60 / 1e6
    print(f"\n--- Cost ---")
    print(f"  API calls made: {calls_made()} (budget {max_calls})")
    print(f"  Tokens: {total_usage['in']} in / {total_usage['out']} out")
    print(f"  Actual cost: ${cost:.4f}")

    # ── Dump results ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "exp13_adversarial_llm.json"
    with open(out, "w") as f:
        json.dump({
            "config": {
                "n_jobs": n_jobs, "slots": slots, "trials": trials,
                "provider": provider, "model": model,
                "base_iters": base_iters, "max_calls": max_calls,
            },
            "summary": summary,
            "adversary_analysis": adversary_analysis,
            "cost": {"calls": calls_made(), "tokens": total_usage, "usd": round(cost, 4)},
            "trials": all_detail,
        }, f, indent=2)
    print(f"\n  Results dumped to: {out}")

    return summary


if __name__ == "__main__":
    run_adversarial_test(n_jobs=12, trials=3, max_calls=60)
