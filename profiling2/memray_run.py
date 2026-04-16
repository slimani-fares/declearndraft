import os
import asyncio

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
os.chdir(project_root)

from declearn.quickrun._run import quickrun
asyncio.run(quickrun("examples/mnist_quickrun/config.toml"))