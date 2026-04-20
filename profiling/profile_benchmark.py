"""
Profile a list of TOML benchmark configs with py-spy and memray.

Results layout:
    profiling/results/{parent}_{config_stem}/{timestamp}/
        pyspy_speedscope.json
        memray.bin
        summary.txt

Assumes data has already been split (runner does not re-split).
To split once: `declearn-split --folder examples/mnist_quickrun --seed 0`
"""

import os
import subprocess
import sys
import time
from datetime import datetime


RUNNER = "profiling/quickrun_runner.py"

# Configs to profile, in order. Add/remove entries as needed.
# Common root for all benchmark configs.
BENCHMARKS_ROOT = "profiling/benchmarks"

# Configs to profile, relative to BENCHMARKS_ROOT.
CONFIGS = [
    "backends/config_fedavg_tensorflow.toml",
    "backends/config_fedavg_torch.toml",

    "regularizers/config_fedavg_torch_lasso.toml",
    "regularizers/config_fedavg_torch_ridge.toml",
    "regularizers/config_fedavg_torch_fedprox.toml",

    "scaffold/config_fedavg_torch_scaffold.toml"
]


def config_label(config_path):
    """Build a unique label from the config's parent dir + filename stem.

    Example: 'benchmarks/backends/config_fedavg_torch.toml'
             -> 'backends_config_fedavg_torch'
    """
    parent = os.path.basename(os.path.dirname(config_path))
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return f"{parent}_{stem}"


def setup_results_dir(config_path):
    """Create profiling/results/{label}/{timestamp}/ and return its path."""
    label = config_label(config_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("profiling", "results", label, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def run_pyspy(config_path, results_dir):
    """Run quickrun under py-spy, save speedscope JSON, return wall-clock time."""
    print("\n--- py-spy ---")
    json_path = os.path.join(results_dir, "pyspy_speedscope.json")

    cmd = [
        "py-spy", "record",
        "--subprocesses",
        "--format", "speedscope",
        "-o", json_path,
        "--", sys.executable, RUNNER, config_path,
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"[WARN] py-spy exited with code {result.returncode}")

    print(f"py-spy: {elapsed:.2f}s  -> {json_path}")
    return elapsed


def run_memray(config_path, results_dir):
    """Run quickrun under memray, save .bin, return wall-clock time."""
    print("\n--- memray ---")
    bin_path = os.path.join(results_dir, "memray.bin")

    cmd = [
        "memray", "run",
        "--force",
        "-o", bin_path,
        RUNNER, config_path,
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"[WARN] memray exited with code {result.returncode}")

    print(f"memray: {elapsed:.2f}s  -> {bin_path}")
    return elapsed


def profile_one(config_path):
    """Profile a single config with both tools. Returns True on success."""
    if not os.path.isfile(config_path):
        print(f"[SKIP] config not found: {config_path}")
        return False

    results_dir = setup_results_dir(config_path)
    print(f"\n{'='*60}")
    print(f"Config:      {config_path}")
    print(f"Results dir: {results_dir}")
    print(f"{'='*60}")

    pyspy_time = run_pyspy(config_path, results_dir)
    memray_time = run_memray(config_path, results_dir)

    # Per-run summary file for quick glance-ability
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Config:       {config_path}\n")
        f.write(f"Timestamp:    {datetime.now().isoformat()}\n")
        f.write(f"py-spy time:  {pyspy_time:.2f}s\n")
        f.write(f"memray time:  {memray_time:.2f}s\n")

    return True


def main():
    print(f"Profiling {len(CONFIGS)} config(s)")

    n_done = 0
    for rel_path in CONFIGS:
        config_path = os.path.join(BENCHMARKS_ROOT, rel_path)
        if profile_one(config_path):
            n_done += 1

    print(f"\n{'='*60}")
    print(f"Done. Profiled {n_done}/{len(CONFIGS)} config(s).")
    print(f"Results under profiling/results/")


if __name__ == "__main__":
    main()