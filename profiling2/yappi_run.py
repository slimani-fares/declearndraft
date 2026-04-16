import os

# go to the project root (parent of wherever this script lives)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
os.chdir(project_root)

import asyncio
import yappi

yappi.set_clock_type("cpu")
yappi.start()

from declearn.quickrun._run import quickrun
asyncio.run(quickrun("declearn/examples/mnist_quickrun/config.toml"))

yappi.stop()

func_stats = yappi.get_func_stats()
func_stats.save("declearn/profiling/yappi_output.prof", type="pstat")

print("\n=== Thread Stats ===")
thread_stats = yappi.get_thread_stats()
thread_stats.print_all()

print("\nDone. Results saved to profiling/yappi_output.prof")