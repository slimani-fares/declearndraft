"""
Quick profiler overhead check (baseline vs cProfile).
Usage:
    python profiling/overhead_check.py --tool cprofile --runs 5
    python profiling/overhead_check.py --tool none --runs 5   # baseline only
"""

import argparse
import asyncio
import cProfile
import os
import statistics
import tempfile
import time

from declearn.quickrun._run import quickrun


CONFIG_PATH = "examples/mnist_quickrun/config.toml"


def run_baseline():
    """Run quickrun with no profiler attached."""
    asyncio.run(quickrun(CONFIG_PATH))


def run_cprofile():
    """Run quickrun under cProfile, dumping stats to a temp file."""
    profiler = cProfile.Profile()
    profiler.enable()
    asyncio.run(quickrun(CONFIG_PATH))
    profiler.disable()
    # Dump to a temp file so disk I/O is included in the overhead.
    with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
        profiler.dump_stats(f.name)
        os.unlink(f.name)


PROFILERS = {
    "none": run_baseline,
    "cprofile": run_cprofile,
}


def time_run(run_fn):
    """Run once, return elapsed wall-clock seconds."""
    start = time.perf_counter()
    run_fn()
    return time.perf_counter() - start


def summarize(label, times):
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"\n--- {label} ---")
    print(f"  runs:   {len(times)}")
    print(f"  mean:   {mean:.3f}s")
    print(f"  median: {median:.3f}s")
    print(f"  stdev:  {stdev:.3f}s")
    print(f"  min:    {min(times):.3f}s")
    print(f"  max:    {max(times):.3f}s")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=list(PROFILERS.keys()), required=True,
                        help="Which profiler to compare against baseline")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of timed runs per condition")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the untimed warm-up run")
    args = parser.parse_args()

    # Warm-up: first run is always slower (imports, caches, etc.)
    if not args.no_warmup:
        print("Warm-up run (untimed)...")
        run_baseline()

    # Interleave baseline and profiled runs to average out any drift.
    baseline_times = []
    profiled_times = []
    profiled_fn = PROFILERS[args.tool]

    if args.tool == "none":
        # Baseline-only mode: just run N times and report.
        for i in range(args.runs):
            elapsed = time_run(run_baseline)
            print(f"  baseline run {i+1}/{args.runs}: {elapsed:.3f}s")
            baseline_times.append(elapsed)
        summarize("baseline", baseline_times)
        return

    for i in range(args.runs):
        # Baseline
        elapsed = time_run(run_baseline)
        print(f"  baseline run {i+1}/{args.runs}: {elapsed:.3f}s")
        baseline_times.append(elapsed)

        # Profiled
        elapsed = time_run(profiled_fn)
        print(f"  {args.tool} run {i+1}/{args.runs}: {elapsed:.3f}s")
        profiled_times.append(elapsed)

    baseline_mean = summarize("baseline", baseline_times)
    profiled_mean = summarize(args.tool, profiled_times)

    abs_overhead = profiled_mean - baseline_mean
    rel_overhead = abs_overhead / baseline_mean * 100
    print(f"\n=== Overhead of {args.tool} ===")
    print(f"  absolute: {abs_overhead:+.3f}s")
    print(f"  relative: {rel_overhead:+.1f}%")


if __name__ == "__main__":
    main()