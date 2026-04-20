"""
Quick profiler overhead check (baseline vs cProfile vs py-spy).

Every run is executed as a subprocess so that all three conditions are
measured consistently (py-spy can only be used by launching its own process).

Usage:
    python profiling/overhead_check.py --tool cprofile --runs 5
    python profiling/overhead_check.py --tool pyspy --runs 5
    python profiling/overhead_check.py --tool none --runs 5   # baseline only
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time


CONFIG_PATH = "examples/mnist_quickrun/config.toml"
RUNNER = "profiling/quickrun_runner.py"


def baseline_cmd():
    """Plain subprocess: python runner.py config."""
    return [sys.executable, RUNNER, CONFIG_PATH]


def cprofile_cmd(outfile):
    """cProfile via -m, dumping stats to outfile."""
    return [sys.executable, "-m", "cProfile", "-o", outfile,
            RUNNER, CONFIG_PATH]


def pyspy_cmd(outfile):
    """py-spy record in speedscope format, sampling the subprocess."""
    return ["py-spy", "record",
            "--subprocesses",
            "--format", "speedscope",
            "-o", outfile,
            "--", sys.executable, RUNNER, CONFIG_PATH]


TOOLS = ("none", "cprofile", "pyspy")


def time_run(tool):
    """Run once as a subprocess, return elapsed wall-clock seconds."""
    # Each profiler writes output to a temp file that we discard.
    if tool == "none":
        cmd = baseline_cmd()
        tmp = None
    elif tool == "cprofile":
        tmp = tempfile.NamedTemporaryFile(suffix=".prof", delete=False)
        tmp.close()
        cmd = cprofile_cmd(tmp.name)
    elif tool == "pyspy":
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        cmd = pyspy_cmd(tmp.name)
    else:
        raise ValueError(f"Unknown tool: {tool}")

    start = time.perf_counter()
    # Capture output so stdout/stderr don't clutter our timing log.
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if tmp is not None:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"\n[ERROR] Subprocess failed (tool={tool}):")
        print(result.stderr[-1000:])  # last 1000 chars of stderr
        raise RuntimeError(f"{tool} run failed")

    return elapsed


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
    parser.add_argument(
        "--tool",
        choices=list(TOOLS) + ["all"],
        required=True,
        help="Which profiler to compare against baseline. "
             "Use 'all' to compare baseline vs cprofile vs pyspy in one run.",
    )
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of timed runs per condition")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the untimed warm-up run")
    args = parser.parse_args()

    # Warm-up: first run is always slower (imports, data caches, etc.)
    if not args.no_warmup:
        print("Warm-up run (untimed baseline)...")
        time_run("none")

    # Build the list of conditions to measure.
    if args.tool == "all":
        conditions = ["none", "cprofile", "pyspy"]
    elif args.tool == "none":
        conditions = ["none"]
    else:
        conditions = ["none", args.tool]

    # Interleave runs across conditions so any drift is spread evenly.
    # e.g. for conditions=[none, cprofile, pyspy] and runs=3, the order is:
    #   none, cprofile, pyspy, none, cprofile, pyspy, none, cprofile, pyspy
    results = {cond: [] for cond in conditions}
    for i in range(args.runs):
        for cond in conditions:
            elapsed = time_run(cond)
            label = "baseline" if cond == "none" else cond
            print(f"  {label} run {i+1}/{args.runs}: {elapsed:.3f}s")
            results[cond].append(elapsed)

    # Per-condition summaries
    means = {}
    for cond in conditions:
        label = "baseline" if cond == "none" else cond
        means[cond] = summarize(label, results[cond])

    # Overhead vs baseline
    if "none" in means and len(conditions) > 1:
        baseline_mean = means["none"]
        print("\n=== Overhead vs baseline ===")
        for cond in conditions:
            if cond == "none":
                continue
            abs_oh = means[cond] - baseline_mean
            rel_oh = abs_oh / baseline_mean * 100
            print(f"  {cond:10s}  absolute: {abs_oh:+7.3f}s   relative: {rel_oh:+6.1f}%")


if __name__ == "__main__":
    main()