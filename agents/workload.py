"""
Real compute workloads — actual CPU and memory consumption.

Unlike the abstract Resource units used elsewhere, these functions burn
real CPU cycles and allocate real memory on the host machine. This lets
us study negotiation over compute that agents *actually consume*, not
abstract numbers.

Stdlib only (hashlib, resource, time) — no numpy/psutil dependency.
"""

from __future__ import annotations

import hashlib
import os
import resource
import time
from dataclasses import dataclass, field


# ── Real work primitives ──────────────────────────────────────────────────

def cpu_burn(iterations: int) -> str:
    """
    Burn real CPU by chaining SHA-256 hashes. Returns final digest so the
    optimizer can't elide the work. ~1M iterations ≈ 0.3-0.5s on a modern core.
    """
    h = b"seed"
    for _ in range(iterations):
        h = hashlib.sha256(h).digest()
    return h.hex()


def mem_burn(mb: int, touch_passes: int = 2) -> int:
    """
    Allocate `mb` megabytes of real memory and touch every page so the OS
    actually commits it. Returns bytes touched. Frees on return.
    """
    n_bytes = mb * 1024 * 1024
    buf = bytearray(n_bytes)
    page = 4096
    total = 0
    for _ in range(touch_passes):
        for i in range(0, n_bytes, page):
            buf[i] = (buf[i] + 1) & 0xFF
            total += 1
    return total * page


# ── Job definition ────────────────────────────────────────────────────────

@dataclass
class Job:
    """A real unit of compute work with real resource demands."""
    job_id: str
    cpu_iterations: int          # real SHA-256 iterations to run
    mem_mb: int                  # real memory to allocate
    urgency: float               # 0=can wait, 1=needs to finish ASAP
    budget: float = 10.0         # willingness-to-pay (for negotiation)
    owner: str = ""              # which agent owns this job

    # Filled in after execution
    submit_round: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    cpu_seconds: float = 0.0
    peak_mem_kb: int = 0
    wait_time: float = 0.0       # time between submit and start

    @property
    def wall_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def completion_time(self) -> float:
        """Total time from submission to completion (wait + run)."""
        return self.wait_time + self.wall_seconds


def execute_job(job: Job) -> dict:
    """
    Run a job's real workload in the current process. Measures real CPU
    time and peak memory via getrusage. Designed to run inside a worker
    process (ProcessPoolExecutor), so RUSAGE_SELF is this job alone.
    """
    t0 = time.perf_counter()
    t_start_epoch = time.time()
    cpu0 = _process_cpu_seconds()

    # Real memory first (held during CPU burn for realistic pressure)
    if job.mem_mb > 0:
        buf = bytearray(job.mem_mb * 1024 * 1024)
        page = 4096
        for i in range(0, len(buf), page):
            buf[i] = 1

    # Real CPU work
    digest = cpu_burn(job.cpu_iterations)

    cpu1 = _process_cpu_seconds()
    t1 = time.perf_counter()
    t_end_epoch = time.time()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux

    return {
        "job_id": job.job_id,
        "wall_seconds": t1 - t0,
        "cpu_seconds": cpu1 - cpu0,
        "peak_mem_kb": peak_kb,
        "t_start_epoch": t_start_epoch,
        "t_end_epoch": t_end_epoch,
        "digest": digest[:8],
        "pid": os.getpid(),
    }


def _process_cpu_seconds() -> float:
    """User + system CPU time for the current process."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# ── Calibration ───────────────────────────────────────────────────────────

def calibrate_iterations(target_seconds: float = 0.3) -> int:
    """
    Empirically find how many SHA-256 iterations take roughly target_seconds
    on this machine. Used to make jobs take meaningful, comparable real time.
    """
    probe = 100_000
    t0 = time.perf_counter()
    cpu_burn(probe)
    elapsed = time.perf_counter() - t0
    rate = probe / elapsed  # iterations per second
    return max(10_000, int(rate * target_seconds))
