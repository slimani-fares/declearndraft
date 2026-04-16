"""
Profiling pipeline runner.
Usage: python profiling/run_profile.py --config examples/mnist_quickrun/config.toml
"""

import argparse
import os
import subprocess # to run the declearn-split
from datetime import datetime
import asyncio
import cProfile


#Creating a timestamped result directory with the config name also 
def setup_results_dir(config_name):
    """Create a timestamped results folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dirname = f"{config_name}_{timestamp}"
    results_dir = os.path.join("profiling", "results", dirname)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

#Running the quick run example (shared among all tools)
def run_quickrun(config_path):
    from declearn.quickrun._run import quickrun
    asyncio.run(quickrun(config_path))


#cProfile
def run_cprofile(config_path, results_dir):
    """Profile the quickrun using cProfile."""
    print("\n=== Running cProfile ===")
    prof_path = os.path.join(results_dir, "cprofile.prof")

    profiler = cProfile.Profile()
    profiler.enable()

    run_quickrun(config_path)

    profiler.disable()
    profiler.dump_stats(prof_path)
    print(f"Saved to {prof_path}")

#Yappi 
def run_yappi(config_path, results_dir):
    """Profile the quickrun using Yappi (async-aware)."""
    print("\n=== Running Yappi ===")
    import yappi

    prof_path = os.path.join(results_dir, "yappi.prof")

    yappi.set_clock_type("cpu")
    yappi.start()

    run_quickrun(config_path)

    yappi.stop()

    func_stats = yappi.get_func_stats()
    func_stats.save(prof_path, type="pstat")

    print("\n--- Thread Stats ---")
    yappi.get_thread_stats().print_all()

    yappi.clear_stats()
    print(f"Saved to {prof_path}")

#Py-spy
def run_pyspy(config_path, results_dir):
    """Profile the quickrun using py-spy (sampling profiler)."""
    print("\n=== Running py-spy ===")

    runner = os.path.join("profiling", "quickrun_runner.py")
    svg_path = os.path.join(results_dir, "pyspy_flamegraph.svg")
    json_path = os.path.join(results_dir, "pyspy_speedscope.json")

    # generate flamegraph (i am keeping this for now just to visualise but it will go later and i will stick with jus the JSON)
    # subprocess.run([
    #     "py-spy", "record",
    #     "--subprocesses",
    #     "-o", svg_path,
    #     "--", "python", runner, config_path
    # ])
    # print(f"Saved flamegraph to {svg_path}")

    # generate speedscope JSON (quantified data)
    subprocess.run([
        "py-spy", "record",
        "--subprocesses",
        "--format", "speedscope",
        "-o", json_path,
        "--", "python", runner, config_path
    ])
    print(f"Saved speedscope to {json_path}")


#Memray 
def run_memray(config_path, results_dir):
    """Profile memory using memray."""
    print("\n=== Running memray ===")

    runner = os.path.join("profiling", "quickrun_runner.py")
    bin_path = os.path.join(results_dir, "memray.bin")
    html_path = os.path.join(results_dir, "memray_flamegraph.html")

    # collect memory data
    subprocess.run([
        "memray", "run",
        "--force",
        "-o", bin_path,
        runner, config_path
    ])

    # generate flamegraph from the data
    # subprocess.run([
    #     "memray", "flamegraph",
    #     bin_path,
    #     "-o", html_path,
    #     "--force"
    # ])

    print(f"Saved to {bin_path} ")


def main():
    parser = argparse.ArgumentParser(description="Run profiling pipeline")
    # the config file path argument 
    parser.add_argument("--config", required=True,
                        help="Path to benchmark TOML config")
    # the profiling tools to use 
    parser.add_argument("--tools", nargs="+",
                        default=["cprofile", "yappi", "pyspy", "memray"],
                        choices=["cprofile", "yappi", "pyspy", "memray"],
                        help="Which profiling tools to run")
    args = parser.parse_args()




    config_name = os.path.splitext(os.path.basename(args.config))[0]
    results_dir = setup_results_dir(config_name)

    print(f"Config: {args.config}")
    print(f"Tools: {args.tools}")
    print(f"Results will be saved to: {results_dir}")

    # prepare data
    subprocess.run(["declearn-split", "--folder", "examples/mnist_quickrun"])

    # tool functions
    if "cprofile" in args.tools:
        run_cprofile(args.config, results_dir)

    if "yappi" in args.tools:
        run_yappi(args.config, results_dir)
        
    if "pyspy" in args.tools:
        run_pyspy(args.config, results_dir)
        
    if "memray" in args.tools:
        run_memray(args.config, results_dir)
    
    print(f"\nDone. Results in: {results_dir}")


if __name__ == "__main__":
    main()
