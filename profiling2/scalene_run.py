import os
os.chdir('..')

import sys
sys.argv = ['run', '--config', 'examples/mnist_quickrun/config.toml']

from declearn.quickrun._run import main
main()