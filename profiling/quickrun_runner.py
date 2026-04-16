"""
Minimal runner script for tools that need to launch a subprocess
(py-spy, memray). Called from run_profile.py, not directly.

Expects the config path as the first command line argument.
"""

import asyncio
import sys

config_path = sys.argv[1]

from declearn.quickrun._run import quickrun
asyncio.run(quickrun(config_path))