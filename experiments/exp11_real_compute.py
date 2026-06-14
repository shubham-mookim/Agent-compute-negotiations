#!/usr/bin/env python3
"""
Experiment 11: REAL Compute Contention

This is the experiment that grounds the whole project. Instead of trading
abstract "GPU-hours", agents here contend for REAL execution slots on this
machine, and their jobs consume REAL CPU cycles and REAL memory.

A fixed pool of K real slots (K = real core count) runs jobs concurrently.
There are more jobs than slots, so allocation order matters: long jobs block
slots, urgent jobs may wait behind them.

We compare four allocation regimes on the IDENTICAL real workload:

  1. FIFO              — naive centralized: run in submission order
  2. SJF               — centralized smart: shortest job first (classic optimal
                         for mean completion time)
  3. Urgency-priority  — centralized: highest urgency first
  4. Market/negotiation — decentralized: agents bid their willingness-to-pay
                         (urgency × budget); highest bidders get slots first

Research question: does decentralized willingness-to-pay allocation produce
better REAL outcomes (urgent-job latency, social welfare) than centralized
schedulers — and when does it fail?

Everything here costs $0 — it only burns this machine's own CPU/RAM.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.workload import Job, execute_job, calibrate_iterations

# Detailed result artifacts go to a TRACKED results/ dir (logs/ is gitignored)
LOG_DIR = Path(__file__).parent.parent / "results"


# ── Workload generation ───────────────────────────────────────────────────

def make_workload(base_iters: int, n_jobs: int = 12, seed: int = 0) -> list[Job]:
    """
    Generate a heterogeneous batch of real jobs:
      - some short + urgent  (interactive-like)
      - some long + patient  (batch-like)
      - some memory-heavy
    The mix is what makes scheduling order matter.
    """
    rng = random.Random(seed)
    jobs = []
    archetypes = [
        # (cpu_mult, mem_mb, urgency_range, weight)
        ("short_urgent",  0.4, 8,   (0.7, 1.0)),
        ("long_batch",    2.0, 16,  (0.0, 0.3)),
        ("medium_mixed",  1.0, 32,  (0.3, 0.7)),
        ("mem_heavy",     0.6, 128, (0.4, 0.8)),
    ]
    for i in range(n_jobs):
        name, cpu_mult, mem_mb, (ulo, uhi) = rng.choice(archetypes)
        urgency = round(rng.uniform(ulo, uhi), 2)
        # Budget correlates loosely with urgency but with noise — willingness
        # to pay is not identical to urgency (the interesting tension)
        budget = round(max(1.0, urgency * 10 + rng.uniform(-3, 5)), 1)
        jobs.append(Job(
            job_id=f"{name}_{i}",
            cpu_iterations=int(base_iters * cpu_mult),
            mem_mb=mem_mb,
            urgency=urgency,
            budget=budget,
            owner=f"agent_{i}",
        ))
    return jobs


# ── Schedulers (decide execution ORDER) ───────────────────────────────────

def schedule_fifo(jobs: list[Job]) -> list[Job]:
    return list(jobs)


def schedule_sjf(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: j.cpu_iterations)


def schedule_urgency(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: -j.urgency)


def schedule_market(jobs: list[Job]) -> list[Job]:
    """
    Decentralized: each agent bids willingness-to-pay = urgency × budget.
    Highest bidders get scheduled first. This incorporates BOTH how badly
    the agent needs it (urgency) and how much it values it (budget) — a
    signal a pure central urgency scheduler ignores.
    """
    return sorted(jobs, key=lambda j: -(j.urgency * j.budget))


SCHEDULERS = {
    "FIFO": schedule_fifo,
    "SJF": schedule_sjf,
    "Urgency": schedule_urgency,
    "Market": schedule_market,
}


# ── Real execution under a fixed slot pool ────────────────────────────────

def run_regime(jobs: list[Job], order: list[Job], slots: int) -> list[dict]:
    """
    Execute jobs in the given order through a pool of `slots` real workers.
    The pool runs at most `slots` jobs concurrently — real contention.
    Returns per-job records with real timing relative to batch start.
    """
    t0 = time.time()
    records = {}

    with ProcessPoolExecutor(max_workers=slots) as pool:
        futures = []
        for j in order:
            fut = pool.submit(execute_job, j)
            futures.append((j, fut))
        for j, fut in futures:
            res = fut.result()
            res["urgency"] = j.urgency
            res["budget"] = j.budget
            res["cpu_iterations"] = j.cpu_iterations
            res["mem_mb"] = j.mem_mb
            res["start_rel"] = res["t_start_epoch"] - t0
            res["end_rel"] = res["t_end_epoch"] - t0
            res["completion_time"] = res["end_rel"]  # submit-at-0 batch model
            res["wait_time"] = res["start_rel"]
            records[j.job_id] = res

    return list(records.values())


# ── Metrics ───────────────────────────────────────────────────────────────

def regime_metrics(records: list[dict]) -> dict:
    """Compute real outcome metrics for one regime."""
    makespan = max(r["end_rel"] for r in records)
    mean_completion = statistics.mean(r["completion_time"] for r in records)

    # Urgent jobs (urgency >= 0.7): how fast did they finish?
    urgent = [r for r in records if r["urgency"] >= 0.7]
    urgent_latency = statistics.mean(r["completion_time"] for r in urgent) if urgent else 0.0

    # Social welfare: urgency-weighted speed. Finishing an urgent job fast
    # is worth more. welfare = sum(urgency / completion_time).
    welfare = sum(r["urgency"] / max(r["completion_time"], 0.001) for r in records)

    # Fairness: std of wait times (lower = more even)
    fairness = statistics.pstdev(r["wait_time"] for r in records)

    total_cpu = sum(r["cpu_seconds"] for r in records)
    peak_mem_mb = max(r["peak_mem_kb"] for r in records) / 1024

    return {
        "makespan_s": round(makespan, 3),
        "mean_completion_s": round(mean_completion, 3),
        "urgent_latency_s": round(urgent_latency, 3),
        "social_welfare": round(welfare, 2),
        "fairness_wait_std": round(fairness, 3),
        "total_cpu_s": round(total_cpu, 2),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "n_urgent": len(urgent),
        "n_jobs": len(records),
    }


# ── Main experiment ───────────────────────────────────────────────────────

def run_real_compute(n_jobs=12, slots=None, trials=3, target_job_seconds=0.3):
    """
    Run the four regimes on identical real workloads across several trials.
    Each trial uses a different random workload; all four regimes run the
    SAME workload within a trial for a fair comparison.
    """
    if slots is None:
        slots = max(1, (os.cpu_count() or 2) - 1)  # leave one core for the OS

    print("=" * 60)
    print("  EXPERIMENT 11: REAL COMPUTE CONTENTION")
    print("=" * 60)
    print()
    print(f"Host: {os.cpu_count()} cores, using {slots} real execution slots")
    print(f"Calibrating workload to ~{target_job_seconds}s/job...")
    base_iters = calibrate_iterations(target_job_seconds)
    print(f"  Base CPU work = {base_iters:,} SHA-256 iterations\n")

    # Accumulate metrics per regime across trials
    agg: dict[str, list[dict]] = {name: [] for name in SCHEDULERS}
    all_trial_detail = []

    for trial in range(trials):
        jobs = make_workload(base_iters, n_jobs=n_jobs, seed=trial)
        print(f"--- Trial {trial} ({n_jobs} jobs, "
              f"{sum(1 for j in jobs if j.urgency >= 0.7)} urgent) ---")

        trial_detail = {"trial": trial, "regimes": {}}
        for name, scheduler in SCHEDULERS.items():
            order = scheduler(list(jobs))
            records = run_regime(jobs, order, slots)
            m = regime_metrics(records)
            agg[name].append(m)
            trial_detail["regimes"][name] = {
                "metrics": m,
                "order": [j.job_id for j in order],
                "jobs": records,
            }
            print(f"  {name:<9} makespan={m['makespan_s']:.2f}s  "
                  f"urgent_lat={m['urgent_latency_s']:.2f}s  "
                  f"welfare={m['social_welfare']:.1f}  "
                  f"fairness={m['fairness_wait_std']:.2f}")
        all_trial_detail.append(trial_detail)
        print()

    # Aggregate summary
    print("=" * 60)
    print("  SUMMARY (averaged across trials)")
    print("=" * 60)
    print(f"\n{'Regime':<10} {'Makespan':>9} {'UrgentLat':>10} {'Welfare':>9} "
          f"{'Fairness':>9} {'MeanComp':>9}")
    print("-" * 60)

    summary = {}
    for name in SCHEDULERS:
        ms = agg[name]
        avg = {
            "makespan_s": statistics.mean(m["makespan_s"] for m in ms),
            "urgent_latency_s": statistics.mean(m["urgent_latency_s"] for m in ms),
            "social_welfare": statistics.mean(m["social_welfare"] for m in ms),
            "fairness_wait_std": statistics.mean(m["fairness_wait_std"] for m in ms),
            "mean_completion_s": statistics.mean(m["mean_completion_s"] for m in ms),
        }
        summary[name] = avg
        print(f"{name:<10} {avg['makespan_s']:>8.2f}s {avg['urgent_latency_s']:>9.2f}s "
              f"{avg['social_welfare']:>9.1f} {avg['fairness_wait_std']:>9.2f} "
              f"{avg['mean_completion_s']:>8.2f}s")

    # Verdict
    print("\n--- Analysis ---")
    best_urgent = min(summary, key=lambda n: summary[n]["urgent_latency_s"])
    best_welfare = max(summary, key=lambda n: summary[n]["social_welfare"])
    best_makespan = min(summary, key=lambda n: summary[n]["makespan_s"])
    print(f"  Best urgent-job latency: {best_urgent}")
    print(f"  Best social welfare:     {best_welfare}")
    print(f"  Best makespan:           {best_makespan}")

    market = summary["Market"]
    urgency = summary["Urgency"]
    fifo = summary["FIFO"]
    print(f"\n  Market vs FIFO urgent latency: "
          f"{market['urgent_latency_s']:.2f}s vs {fifo['urgent_latency_s']:.2f}s "
          f"({(market['urgent_latency_s']/fifo['urgent_latency_s']-1)*100:+.0f}%)")
    print(f"  Market vs Urgency welfare: "
          f"{market['social_welfare']:.1f} vs {urgency['social_welfare']:.1f} "
          f"({(market['social_welfare']/urgency['social_welfare']-1)*100:+.0f}%)")

    # Dump detailed results
    LOG_DIR.mkdir(exist_ok=True)
    out_path = LOG_DIR / "exp11_real_compute_detail.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_jobs": n_jobs, "slots": slots, "trials": trials,
                "base_iters": base_iters, "cpu_count": os.cpu_count(),
                "target_job_seconds": target_job_seconds,
            },
            "summary": summary,
            "trials": all_trial_detail,
        }, f, indent=2)
    print(f"\n  Detailed per-job results dumped to: {out_path}")

    # Also dump a flat CSV for easy analysis
    csv_path = LOG_DIR / "exp11_real_compute_jobs.csv"
    with open(csv_path, "w") as f:
        f.write("trial,regime,job_id,urgency,budget,cpu_iterations,mem_mb,"
                "wait_time,wall_seconds,cpu_seconds,completion_time,peak_mem_kb,pid\n")
        for td in all_trial_detail:
            for regime, rd in td["regimes"].items():
                for j in rd["jobs"]:
                    f.write(f"{td['trial']},{regime},{j['job_id']},{j['urgency']},"
                            f"{j['budget']},{j['cpu_iterations']},{j['mem_mb']},"
                            f"{j['wait_time']:.4f},{j['wall_seconds']:.4f},"
                            f"{j['cpu_seconds']:.4f},{j['completion_time']:.4f},"
                            f"{j['peak_mem_kb']},{j['pid']}\n")
    print(f"  Flat CSV dumped to: {csv_path}")

    return summary


if __name__ == "__main__":
    run_real_compute(n_jobs=12, trials=3, target_job_seconds=0.3)
