#!/usr/bin/env python3
"""
Experiment 12: LLM Agents Bidding for REAL Compute

The full integration: real LLM agents (OpenAI) reason about their own job's
importance and bid for REAL execution slots running REAL CPU/memory workloads.

This is the realistic agentic scenario the project was always reaching for:
  - Each agent owns a real job (real CPU iterations, real memory)
  - Each agent uses an LLM to decide how aggressively to bid, reasoning about
    its urgency, deadline pressure, budget, and job size
  - Limited real slots → contention → allocation order determines real outcomes
  - We compare LLM-driven allocation against mechanical baselines on the
    IDENTICAL real workload

Research question: does an LLM agent reasoning about its job produce better
allocation (real social welfare, urgent-job latency) than a mechanical
urgency×budget formula — or does LLM idiosyncrasy hurt?

COST CONTROL: hard call-budget guard. One LLM call per job per trial.
Default 12 jobs × 3 trials = 36 calls ≈ $0.005. Guard aborts if exceeded.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
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


LLM_BID_SYSTEM = """You are an autonomous compute-scheduling agent. You own ONE job that needs to run on a shared server with limited execution slots. Many agents compete for those slots. You must decide how aggressively to bid for priority.

You will be given your job's properties. Output a single integer bid from 0 to 100, where 100 = "I must run first, this is critical" and 0 = "I can wait, run me last".

Reason about:
- Urgency: how time-sensitive is this job?
- Job size: large jobs hold a slot longer (consider whether you're blocking others)
- Budget: your willingness/ability to pay for priority
- Fairness: bidding max on a non-urgent job wastes priority and may be penalized socially

Respond with ONLY the integer bid (0-100), nothing else."""


def llm_bid_for_job(job: Job, provider: str, model: str) -> tuple[int, dict]:
    """One LLM call: agent decides its priority bid for its real job."""
    _guard_call()
    prompt = (
        f"Your job: {job.job_id}\n"
        f"Estimated CPU work: {job.cpu_iterations:,} hash-iterations "
        f"({'large' if job.cpu_iterations > 800_000 else 'small' if job.cpu_iterations < 400_000 else 'medium'})\n"
        f"Memory needed: {job.mem_mb} MB\n"
        f"Urgency: {job.urgency:.2f} (0=can wait, 1=critical)\n"
        f"Your budget: {job.budget:.1f}\n\n"
        f"What is your priority bid (0-100)?"
    )
    usage = {"in": 0, "out": 0}
    try:
        if provider == "openai":
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model=model, max_tokens=10, temperature=0.3,
                messages=[
                    {"role": "system", "content": LLM_BID_SYSTEM},
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
                system=LLM_BID_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if resp.usage:
                usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
        nums = re.findall(r"\d+", text)
        bid = int(nums[0]) if nums else 50
        bid = max(0, min(100, bid))
    except Exception:
        bid = int(job.urgency * 100)  # fallback to urgency
    return bid, usage


def schedule_llm(jobs: list[Job], provider: str, model: str, bid_cache: dict) -> tuple[list[Job], dict]:
    """Order jobs by LLM-decided bids. Caches bids per job_id to avoid re-calling."""
    usage_total = {"in": 0, "out": 0}
    bids = {}
    for j in jobs:
        if j.job_id in bid_cache:
            bids[j.job_id] = bid_cache[j.job_id]
        else:
            bid, usage = llm_bid_for_job(j, provider, model)
            bids[j.job_id] = bid
            bid_cache[j.job_id] = bid
            usage_total["in"] += usage["in"]
            usage_total["out"] += usage["out"]
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, {"bids": bids, "usage": usage_total}


def run_llm_real_compute(n_jobs=12, trials=3, target_job_seconds=0.3, max_calls=60):
    """
    Compare LLM-driven allocation vs mechanical baselines on identical real
    workloads. Hard cost guard via max_calls.
    """
    provider = _detect_provider()
    print("=" * 60)
    print("  EXPERIMENT 12: LLM AGENTS BIDDING FOR REAL COMPUTE")
    print("=" * 60)
    print()

    if provider == "none":
        print("No API key set — cannot run LLM allocation. Set OPENAI_API_KEY.")
        return None

    model = MODELS[provider]
    set_call_budget(max_calls)  # HARD CAP
    print(f"Provider: {provider} ({model})")
    print(f"Hard call budget: {max_calls} calls\n")

    slots = max(1, (os.cpu_count() or 2) - 1)
    base_iters = calibrate_iterations(target_job_seconds)
    print(f"Host: {os.cpu_count()} cores, {slots} slots, "
          f"base work = {base_iters:,} iterations\n")

    regimes = {
        "FIFO": schedule_fifo,
        "Urgency": schedule_urgency,
        "Market": schedule_market,
    }
    agg = {name: [] for name in list(regimes) + ["LLM"]}
    all_detail = []
    total_usage = {"in": 0, "out": 0}

    for trial in range(trials):
        jobs = make_workload(base_iters, n_jobs=n_jobs, seed=trial)
        n_urgent = sum(1 for j in jobs if j.urgency >= 0.7)
        print(f"--- Trial {trial} ({n_jobs} jobs, {n_urgent} urgent) ---")

        trial_detail = {"trial": trial, "regimes": {}}

        # Mechanical baselines
        for name, scheduler in regimes.items():
            order = scheduler(list(jobs))
            records = run_regime(jobs, order, slots)
            m = regime_metrics(records)
            agg[name].append(m)
            trial_detail["regimes"][name] = {"metrics": m, "order": [j.job_id for j in order]}

        # LLM allocation
        bid_cache = {}
        try:
            order, info = schedule_llm(list(jobs), provider, model, bid_cache)
            total_usage["in"] += info["usage"]["in"]
            total_usage["out"] += info["usage"]["out"]
            records = run_regime(jobs, order, slots)
            m = regime_metrics(records)
            agg["LLM"].append(m)
            trial_detail["regimes"]["LLM"] = {
                "metrics": m, "order": [j.job_id for j in order],
                "bids": info["bids"],
                "jobs": records,
            }
            print(f"  LLM bids: {info['bids']}")
        except Exception as e:
            print(f"  LLM allocation aborted: {e}")
            break

        for name in ["FIFO", "Urgency", "Market", "LLM"]:
            if agg[name]:
                m = agg[name][-1]
                print(f"  {name:<9} makespan={m['makespan_s']:.2f}s  "
                      f"urgent_lat={m['urgent_latency_s']:.2f}s  welfare={m['social_welfare']:.1f}")
        all_detail.append(trial_detail)
        print()

    # Summary
    print("=" * 60)
    print("  SUMMARY (averaged across trials)")
    print("=" * 60)
    print(f"\n{'Regime':<10} {'Makespan':>9} {'UrgentLat':>10} {'Welfare':>9} {'Fairness':>9}")
    print("-" * 50)
    summary = {}
    for name in ["FIFO", "Urgency", "Market", "LLM"]:
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
        print(f"{name:<10} {avg['makespan_s']:>8.2f}s {avg['urgent_latency_s']:>9.2f}s "
              f"{avg['social_welfare']:>9.1f} {avg['fairness_wait_std']:>9.2f}")

    # Cost report
    cost = total_usage["in"] * 0.15 / 1e6 + total_usage["out"] * 0.60 / 1e6
    print(f"\n--- Cost ---")
    print(f"  API calls made: {calls_made()} (budget {max_calls})")
    print(f"  Tokens: {total_usage['in']} in / {total_usage['out']} out")
    print(f"  Actual cost: ${cost:.4f}")

    # Analysis
    if "LLM" in summary:
        print(f"\n--- Analysis ---")
        best_welfare = max(summary, key=lambda n: summary[n]["social_welfare"])
        best_urgent = min(summary, key=lambda n: summary[n]["urgent_latency_s"])
        print(f"  Best welfare: {best_welfare}, Best urgent latency: {best_urgent}")
        llm = summary["LLM"]
        urg = summary["Urgency"]
        print(f"  LLM vs Urgency welfare: {llm['social_welfare']:.1f} vs "
              f"{urg['social_welfare']:.1f} "
              f"({(llm['social_welfare']/urg['social_welfare']-1)*100:+.0f}%)")
        print(f"  LLM vs Urgency urgent latency: {llm['urgent_latency_s']:.2f}s vs "
              f"{urg['urgent_latency_s']:.2f}s")

    # Dump
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "exp12_llm_real_compute_detail.json"
    with open(out, "w") as f:
        json.dump({
            "config": {"n_jobs": n_jobs, "slots": slots, "trials": trials,
                       "provider": provider, "model": model, "base_iters": base_iters},
            "summary": summary,
            "cost": {"calls": calls_made(), "tokens": total_usage, "usd": round(cost, 4)},
            "trials": all_detail,
        }, f, indent=2)
    print(f"\n  Detailed results dumped to: {out}")

    return summary


if __name__ == "__main__":
    run_llm_real_compute(n_jobs=12, trials=3, max_calls=60)
