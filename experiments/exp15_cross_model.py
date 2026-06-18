#!/usr/bin/env python3
"""
Experiment 15: Cross-Model LLM Allocation Comparison

Addresses the "single LLM model" threat to validity. Experiments 12-13 use
only GPT-4o-mini. Here we run the IDENTICAL real-compute bidding task across
two model tiers and ask:

  - Do larger models produce better allocation (social welfare, urgent latency)?
  - Do models agree on bids, or does model choice change the queue order?
  - Is the "LLM reconstructs near-optimal scheduling" finding model-specific
    or general across model capability?

Each agent owns one REAL job (SHA-256 burn + memory) and makes one LLM call
to choose a 0-100 priority bid. We run the same seeded workloads through each
model and compare against the centralized urgency baseline.

COST CONTROL: hard call-budget guard. n_jobs x trials x n_models calls.
Default 12 x 2 x 2 = 48 calls. gpt-4o-mini ~ $0.002, gpt-4o ~ $0.03. Guard
aborts before exceeding max_calls.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_strategy import (
    set_call_budget, calls_made, _detect_provider, _guard_call,
)
from experiments.exp11_real_compute import (
    make_workload, run_regime, regime_metrics,
    schedule_urgency, schedule_fifo,
)
from experiments.exp12_llm_real_compute import llm_bid_for_job
from agents.workload import calibrate_iterations

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Model tiers to compare. Pricing per 1M tokens (input, output) for cost report.
MODEL_TIERS = {
    "gpt-4o-mini": {"in_price": 0.15, "out_price": 0.60},
    "gpt-4o":      {"in_price": 2.50, "out_price": 10.00},
}


def schedule_with_model(jobs, model, usage_accum):
    """Order jobs by bids from a specific model. One call per job."""
    bids = {}
    for j in jobs:
        bid, usage = llm_bid_for_job(j, "openai", model)
        bids[j.job_id] = bid
        usage_accum["in"] += usage["in"]
        usage_accum["out"] += usage["out"]
    ordered = sorted(jobs, key=lambda j: -bids[j.job_id])
    return ordered, bids


def _spearman_like_agreement(bids_a: dict, bids_b: dict) -> float:
    """
    Fraction of job pairs ordered the same way by both models (Kendall-tau-ish
    concordance). 1.0 = identical ordering, 0.5 = random, 0.0 = reversed.
    """
    ids = list(bids_a.keys())
    concordant = 0
    total = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_i, a_j = bids_a[ids[i]], bids_a[ids[j]]
            b_i, b_j = bids_b[ids[i]], bids_b[ids[j]]
            if a_i == a_j or b_i == b_j:
                continue  # ties don't count
            total += 1
            if (a_i > a_j) == (b_i > b_j):
                concordant += 1
    return concordant / total if total else 1.0


def run_cross_model(n_jobs=12, trials=2, target_job_seconds=0.3, max_calls=60):
    provider = _detect_provider()
    print("=" * 64)
    print("  EXPERIMENT 15: CROSS-MODEL LLM ALLOCATION")
    print("=" * 64)
    print()

    if provider != "openai":
        print("Requires OPENAI_API_KEY (cross-model test uses OpenAI tiers). Skipping.")
        return None

    models = list(MODEL_TIERS.keys())
    set_call_budget(max_calls)
    print(f"Models: {models}")
    print(f"Hard call budget: {max_calls} calls\n")

    slots = max(1, (os.cpu_count() or 2) - 1)
    base_iters = calibrate_iterations(target_job_seconds)
    print(f"Host: {os.cpu_count()} cores, {slots} slots, "
          f"base work = {base_iters:,} iterations\n")

    # Aggregate metrics per model + baselines
    agg = {m: [] for m in models}
    agg["Urgency"] = []
    agg["FIFO"] = []
    usage_by_model = {m: {"in": 0, "out": 0} for m in models}
    agreement_scores = []
    all_detail = []

    aborted = False
    for trial in range(trials):
        jobs = make_workload(base_iters, n_jobs=n_jobs, seed=trial)
        n_urgent = sum(1 for j in jobs if j.urgency >= 0.7)
        print(f"--- Trial {trial} ({n_jobs} jobs, {n_urgent} urgent) ---")

        trial_detail = {"trial": trial, "models": {}}

        # Centralized baselines (free)
        for name, sched in [("Urgency", schedule_urgency), ("FIFO", schedule_fifo)]:
            order = sched(list(jobs))
            m = regime_metrics(run_regime(jobs, order, slots))
            agg[name].append(m)
            trial_detail["models"][name] = {"metrics": m}

        # Each model bids on the identical workload
        model_bids = {}
        try:
            for model in models:
                order, bids = schedule_with_model(list(jobs), model, usage_by_model[model])
                m = regime_metrics(run_regime(jobs, order, slots))
                agg[model].append(m)
                model_bids[model] = bids
                trial_detail["models"][model] = {
                    "metrics": m, "bids": bids,
                    "order": [j.job_id for j in order],
                }
                print(f"  {model:<14} welfare={m['social_welfare']:.1f} "
                      f"urgent_lat={m['urgent_latency_s']:.2f}s")
        except Exception as e:
            print(f"  Aborted (budget guard): {e}")
            aborted = True
            break

        # Inter-model bid agreement
        if len(model_bids) == 2:
            agr = _spearman_like_agreement(model_bids[models[0]], model_bids[models[1]])
            agreement_scores.append(agr)
            print(f"  Bid-order agreement ({models[0]} vs {models[1]}): {agr:.2f}")

        all_detail.append(trial_detail)
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 64)
    print("  SUMMARY (averaged across trials)")
    print("=" * 64)
    print(f"\n{'Regime':<14} {'Welfare':>9} {'UrgentLat':>10} {'Makespan':>10}")
    print("-" * 46)
    summary = {}
    for name in ["Urgency", "FIFO"] + models:
        ms = agg[name]
        if not ms:
            continue
        avg = {
            "social_welfare": statistics.mean(m["social_welfare"] for m in ms),
            "urgent_latency_s": statistics.mean(m["urgent_latency_s"] for m in ms),
            "makespan_s": statistics.mean(m["makespan_s"] for m in ms),
        }
        summary[name] = avg
        print(f"{name:<14} {avg['social_welfare']:>9.1f} "
              f"{avg['urgent_latency_s']:>9.2f}s {avg['makespan_s']:>9.2f}s")

    # ── Analysis ──────────────────────────────────────────────────────────
    print(f"\n--- Analysis ---")
    if agreement_scores:
        print(f"  Mean inter-model bid agreement: {statistics.mean(agreement_scores):.2f}")
    if all(m in summary for m in models) and "Urgency" in summary:
        for model in models:
            w = summary[model]["social_welfare"]
            uw = summary["Urgency"]["social_welfare"]
            print(f"  {model} welfare vs centralized urgency: "
                  f"{w:.1f} vs {uw:.1f} ({(w/uw-1)*100:+.0f}%)")
        if len(models) == 2:
            w0, w1 = summary[models[0]]["social_welfare"], summary[models[1]]["social_welfare"]
            print(f"  {models[1]} vs {models[0]} welfare: "
                  f"{w1:.1f} vs {w0:.1f} ({(w1/w0-1)*100:+.1f}%)")
            print(f"  >> Larger model {'DID' if w1 > w0 else 'did NOT'} "
                  f"produce better allocation on this task.")

    # ── Cost ──────────────────────────────────────────────────────────────
    print(f"\n--- Cost ---")
    total_cost = 0.0
    for model in models:
        u = usage_by_model[model]
        price = MODEL_TIERS[model]
        c = u["in"] * price["in_price"] / 1e6 + u["out"] * price["out_price"] / 1e6
        total_cost += c
        print(f"  {model:<14} {u['in']} in / {u['out']} out → ${c:.4f}")
    print(f"  API calls made: {calls_made()} (budget {max_calls})")
    print(f"  Total cost: ${total_cost:.4f}")
    if aborted:
        print("  NOTE: run aborted early by budget guard — partial results.")

    # ── Dump ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "exp15_cross_model.json"
    with open(out, "w") as f:
        json.dump({
            "config": {"n_jobs": n_jobs, "trials": trials, "slots": slots,
                       "models": models, "base_iters": base_iters},
            "summary": summary,
            "agreement": agreement_scores,
            "cost": {"by_model": usage_by_model, "usd": round(total_cost, 4),
                     "calls": calls_made()},
            "trials": all_detail,
        }, f, indent=2)
    print(f"\n  Results dumped to: {out}")

    return summary


if __name__ == "__main__":
    run_cross_model(n_jobs=12, trials=2, max_calls=60)
