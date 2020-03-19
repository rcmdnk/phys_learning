"""
    Command line interface for core object
"""


import sys
from .core import PhysLearning
import fire


class CliObject(object):
    """CliObject"""

    def __init__(self):
        pass

    def help(self):
        """Show help."""
        PhysLearning.help()

    def version(self):
        """Show version."""
        PhysLearning.version()

    def run(self, cmd=None, nshot=10):
        """Main command"""
        PhysLearning('events/my_z.txt', 'events/my_jj.txt').run(cmd)

    def multishot(self, nshot=10):
        """Multishot"""
        PhysLearning('events/my_z.txt', 'events/my_jj.txt').multishot(nshot)


def cli():
    """Main command line tool function."""
    if len(sys.argv) <= 1 or sys.argv[1] in ['-h', '--help']:
        PhysLearning.help()
    else:
        fire.Fire(CliObject)
